###  Injecting and recovering GWB and CGW  ###


import os
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import corner
import libstempo as LT
import libstempo.toasim as LTsim
from libstempo.toasim import (
        add_efac, 
        add_equad, 
        add_gwb,
        add_cgw
        )
from enterprise.pulsar import Pulsar
from enterprise.signals.gp_signals import TimingModel
from enterprise.signals.signal_base import PTA
from enterprise.signals import (deterministic_signals, parameter, signal_base,
                                utils)
from enterprise_extensions.deterministic import cw_delay, CWSignal
from enterprise_extensions.sampler import JumpProposal
from enterprise_extensions.blocks import (
                                white_noise_block,
                                red_noise_block,
                                dm_noise_block,
                                common_red_noise_block
                                )
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import glob
import scipy
import math
import json


########    CGW model    ########################################################################################    

def cw_block_circ(amp_prior='log-uniform', dist_prior=None,
                  skyloc=None, log10_fgw=None,
                  psrTerm=False, tref=0, name='cw', evolve=False, phase_approx=False):
    if dist_prior is None:
        log10_dist = None

        if amp_prior == 'uniform':
            log10_h = parameter.LinearExp(-18.0, -11.0)('{}_log10_h'.format(name))
        elif amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(-18.0, -11.0)('{}_log10_h'.format(name))

    elif dist_prior == 'log-uniform':
        log10_dist = parameter.Uniform(-2.0, 4.0)('{}_log10_dL'.format(name))
        log10_h = None

    # chirp mass [Msol]
    log10_Mc = parameter.Uniform(6.0, 10.0)('{}_log10_Mc'.format(name))

    # GW frequency [Hz]
    if log10_fgw is None:
        log10_fgw = parameter.Uniform(-9.0, -7.0)('{}_log10_fgw'.format(name))
    else:
        log10_fgw = parameter.Constant(log10_fgw)('{}_log10_fgw'.format(name))
    # orbital inclination angle [radians]
    cosinc = parameter.Uniform(-1.0, 1.0)('{}_cosinc'.format(name))
    # initial GW phase [radians]
    phase0 = parameter.Uniform(0.0, 2*np.pi)('{}_phase0'.format(name))

    # polarization
    psi_name = '{}_psi'.format(name)
    psi = parameter.Uniform(0, np.pi)(psi_name)

    # sky location
    costh_name = '{}_costheta'.format(name)
    phi_name = '{}_phi'.format(name)
    if skyloc is None:
        costh = parameter.Uniform(-1, 1)(costh_name)
        phi = parameter.Uniform(0, 2*np.pi)(phi_name)
    else:
        costh = parameter.Constant(skyloc[0])(costh_name)
        phi = parameter.Constant(skyloc[1])(phi_name)

    if psrTerm:
        # orbital phase
        p_phase = parameter.Uniform(0, np.pi)
        p_dist = parameter.Normal(0, 1)
    else:
        p_phase = None
        p_dist = 0

    # continuous wave signal
    wf = cw_delay(cos_gwtheta=costh, gwphi=phi, cos_inc=cosinc,
                  log10_mc=log10_Mc, log10_fgw=log10_fgw,
                  log10_h=log10_h, log10_dist=log10_dist,
                  phase0=phase0, psi=psi,
                  psrTerm=True, p_dist=p_dist, p_phase=p_phase,
                  evolve=evolve, phase_approx=phase_approx, check=False,
                  tref=tref)
    cw = CWSignal(wf, ecc=False, psrTerm=psrTerm)

    return cw


########    Create model    ########################################################################################    

def define_model(psr, components, noisedict = None, cgw_type = [None], evolve=False, phase_approx=False):
    s = TimingModel()

    if 'wn' in components:
        s += white_noise_block(vary=False, select='backend', tnequad=True) 
    
    if 'rn' in components:
        s += red_noise_block() 
    
    if 'dm' in components:   
        s += dm_noise_block() 
    
    if 'gwb' in components:   
        s += common_red_noise_block(orf='crn')
    
    if 'cgw' in components:
        
        if 'PT' not in cgw_type:
            s += cw_block_circ(amp_prior=None, dist_prior='log-uniform', skyloc=None, log10_fgw=None,
                      psrTerm=False, tref=0, name='cw')
            
        
        if 'PT' in cgw_type:
            s += cw_block_circ(amp_prior=None, dist_prior='log-uniform', skyloc=None, log10_fgw=None,
                  psrTerm=True, tref=0, name='cw', evolve=evolve, phase_approx=phase_approx)
            
    if 'mon' in components:   
        s += common_red_noise_block(orf='gw_monopole', name='mon')
        
    
    # Create PTA object, from which one can compute LH and prior
    pta = PTA([s(p) for p in psr])
    
    if noisedict is not None:
        pta.set_default_params(noisedict)
        
        print('I am using noisefiles for priors')
        
    return pta

########    Run the MCMC    ########################################################################################   

def MCMC(nsamples, filename, pta):
    
    # Write parameters names in a .txt
    outdir = "%s/"%filename
    with open(os.path.join(outdir, "pars.txt"), "w") as fout:
        for pname in pta.param_names:
            fout.write(pname + "\n")
            
    # Initialize parameters
    x0 = np.hstack([p.sample() for p in pta.params])
    
    ndim = len(x0)
    
    # Initialize sampler
    cov = np.diag(np.ones(ndim) * 0.01**2)
    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=outdir, resume=True)

    # Set Jump Proposals
    jp = JumpProposal(pta)

	# always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 5)

    sel_sig = {"gw":30,
	     "cw":30
            }

    # Jump proposals from priors of selected params
    for s, val in sel_sig.items():
     if any([s in p for p in pta.param_names]):
      sampler.addProposalToCycle(jp.draw_from_par_prior(s), val)

    # Sample
    sampler.sample(x0, int(nsamples), SCAMweight=40, DEweight=60, AMweight=20, Tmax=None)



######################################################################################################################################################################################################


# Read .par and .tim files
path = '/home/samwise/Master_thesis/WN_trials/simurr_mib/simulated_data_wm87/MultiF/'
parfiles_all = sorted(glob.glob(path+'*/*.par'))
parfiles_TN = sorted(glob.glob(path+'*/*_TN.par'))
indx = []
for par in parfiles_TN:
    for i in range(len(parfiles_all)):
        if parfiles_all[i] == par:
            indx.append(i)         
parfiles = np.delete(parfiles_all, indx)

timfiles = sorted(glob.glob(path+'*/*_all.tim'))

if len(parfiles) == 0:
	print('Failed reading parameters files')
if len(timfiles) == 0:
	print('Failed reading time of arrival files')


# Read noisefiles
noisefile = '/home/samwise/Master_thesis/WN_trials/simurr_mib/injection_noisefile/noisefile_efacequad.json'

if len(noisefile) == 0:
	print('Failed reading noise files')
	
noisedict = json.load(open(noisefile))    


# Create pulsar array
LTPSR, PSR = [], []
for timfile, parfile in zip(timfiles, parfiles):
    ltpsr = LT.tempopulsar(parfile, timfile)
    LTPSR.append(ltpsr)
    PSR.append(Pulsar(ltpsr))

pta = define_model(PSR, ['wn', 'gwb', 'cgw'], noisedict = noisedict, cgw_type = ['PT'], evolve=True)


# Run MCMC
nsamples = 1e7
filename = '/home/samwise/Master_thesis/WN_trials/Chains/curn_PTevolve'
MCMC(nsamples, filename, pta)


# Pray

