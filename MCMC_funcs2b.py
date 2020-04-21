# These are all the modules and definitions you will need in this notebook
import batman
from astropy.stats import BoxLeastSquares

import os
import matplotlib.pyplot as plt
plt.style.use("default")

from matplotlib import rcParams
rcParams["savefig.dpi"] = 75
rcParams["figure.dpi"] = 75
rcParams["font.size"] = 16
rcParams["text.usetex"] = False
rcParams["font.family"] = ["sans-serif"]
rcParams["font.sans-serif"] = ["cmss10"]
rcParams["axes.unicode_minus"] = False

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)
logger.propagate = False

import corner
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt

import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt
tt.config.optimizer='fast_compile'
# print(tt.config.optimizer)



#stuff for getting FFI data from MAST
import astropy
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

#in case there are WiFi issues, these may help
from urllib.error import HTTPError
import requests

#stuff for detecting periodic transit events
from transitleastsquares import catalog_info

import time as clock
import pandas as pd

def phasefold(T0,time,period,flux):
    phase=(time- T0 + 0.5*period) % period - 0.5*period #centers transit event at phase 0.0
    ind=np.argsort(phase, axis=0) #sorts phases from min to max (helps with visualization)
    return phase[ind],flux[ind]


def build_model(time, flux, error, periods, t0s, depths, qld_a, qld_b,\
                 logg, logg_err, M_star, M_star_err,R_star, R_star_err, ID, Sector, \
                 mask=None, start=None):
    x=time
    y=flux
    yerr=error
    texp = np.median(np.diff(x))

    
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    
    periods = np.atleast_1d(periods)[0]
    t0s = np.atleast_1d(t0s)[0]
    depths = np.atleast_1d(depths)[0]
    r_guess = np.sqrt(depths) # R_Planet / R_Star
    n_planets = len(np.atleast_1d(depths))
    
    with pm.Model() as model:
        
        
        # Extract the un-masked data points
        model.x = x[mask]
        model.y = y[mask]
        model.yerr = (yerr + np.zeros_like(x))[mask]
        model.mask = mask

        # The baseline flux 
        mean = pm.Normal("mean", mu=0.0, sd=10.0)
        u_star = xo.distributions.QuadLimbDark("u_star")
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
        m_star = BoundedNormal("m_star", mu=M_star, sd=M_star_err)
        r_star = BoundedNormal("r_star", mu=R_star, sd=R_star_err)
        
        
#         logP = pm.Normal("logP", mu=np.log(periods), sd=1)
        logP = pm.Normal("logP", mu=np.log(periods), sd=0.1, shape=n_planets)
        # Tracking planet parameters
        period = pm.Deterministic("period", tt.exp(logP))
        
        t0 = pm.Normal("t0", mu=t0s, sd=1)
        
        logr = pm.Normal("logr",sd=1.0,mu=0.5 * np.log(1e-3 * np.array(depths)) +\
                         np.log(R_star))
        # The Espinoza (2018) parameterization for the joint radius ratio and
        # impact parameter distribution
        r, b = xo.distributions.get_joint_radius_impact(
            min_radius=0.001, max_radius=1.0,
            testval_r=np.sqrt(1e-3*np.array(depths)),
            testval_b=0.5+np.zeros(n_planets)
        )
        r_pl = pm.Deterministic("r_pl", r * r_star)
        
         # This is the eccentricity prior from Kipping (2013):
        # https://arxiv.org/abs/1306.4982
#         ecc = xo.distributions.eccentricity.kipping13("ecc", testval=0.0)
        ecc = pm.Beta("ecc", alpha=0.867, beta=3.03, testval=0.0)
        omega = xo.distributions.Angle("omega")#, testval=90.0)
        
        # Transit jitter & GP parameters
        logs2 = pm.Normal("logs2", mu=np.log(np.var(y[mask])), sd=10)
        logw0 = pm.Normal("logw0", mu=0, sd=10)
        logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y[mask])), sd=10)
        
        
        
        # Orbit model
        orbit = xo.orbits.KeplerianOrbit(r_star=r_star,m_star=m_star,\
                                         period=period,t0=t0,b=b,ecc=ecc,omega=omega)
        
        # Compute the model light curve using starry
        model.light_curves = (xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp)* 1e3)
               
        model.light_curve = pm.math.sum(model.light_curves, axis=-1) + mean
        pm.Deterministic("light_curves", model.light_curves)
        
        pm.Normal("obs", mu=model.light_curve, sd=tt.sqrt(model.yerr**2+tt.exp(logs2)),
                  observed=model.y)        
        
        # GP model for the light curve
        kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2))
        gp = xo.gp.GP(kernel, x[mask], tt.exp(logs2) + tt.zeros(mask.sum()), J=2)
        pm.Potential("transit_obs", gp.log_likelihood(y[mask] - model.light_curve))
        pm.Deterministic("gp_pred", gp.predict())
        
        
        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
                        
        map_soln = start 
        map_soln = xo.optimize(start=start, vars=[logs2, logSw4, logw0,mean])
        map_soln = xo.optimize(start=map_soln, vars=[logr])
        map_soln = xo.optimize(start=map_soln, vars=[b])
        map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
#         map_soln = xo.optimize(start=map_soln, vars=[u_star])
        map_soln = xo.optimize(start=map_soln, vars=[logr])
        map_soln = xo.optimize(start=map_soln, vars=[b])
        map_soln = xo.optimize(start=map_soln, vars=[ecc, omega]) #this might not work
#         map_soln = xo.optimize(start=map_soln, vars=[logP, t0])
#         map_soln = xo.optimize(start=map_soln, vars=[mean])
#         map_soln = xo.optimize(start=map_soln, vars=[logs2, logSw4, logw0])
        map_soln = xo.optimize(start=map_soln)
        model.map_soln = map_soln

    return model#, map_soln


def build_model_sigma_clip(ID,Sector,time, flux, error,logg,logg_err,M_star,M_star_err,R_star,R_star_err, qld_a,qld_b, periods, t0s, depths, sigma=5.0, maxiter=10):
#     start_time=clock.time()
    x=time
    y=flux
    yerr=error
    texp = np.median(np.diff(x))
    ntot = len(x)
    for i in range(maxiter):
        print("*** Sigma clipping round {0} ***".format(i+1))
        
        # Build the model
#         print(start)
#         print('Build the model')
        model = build_model(x,y,yerr, periods, t0s, depths, qld_a, qld_b,logg, logg_err,\
                            M_star,M_star_err,R_star, R_star_err, ID, Sector, mask=None, start=None)


        # Compute the map prediction
        with model:
            mod = xo.utils.eval_in_model(model.light_curve, model.map_soln)
            
        # Do sigma clipping
#         print('sigma clipping')
        resid = y - mod
        rms = np.sqrt(np.median(resid**2))
        mask = np.abs(resid) < sigma * rms
        if ntot == mask.sum():
            break
        ntot = mask.sum()
#     print('runtime: ',(clock.time()-start_time)/60.0," minutes")
    return model


MCMCfigpath=os.getcwd()+'/MCMC_figures/'

def MCMC_planet_model(ID, Sector, T,F,E, qld_a, qld_b, logg, logg_err, M_star,M_star_err,R_star,\
                      R_star_err, Periods, T0s, Depths, MCMCfigpath):
    MCMCfigpath=os.getcwd()+'/MCMC_figures/'
    
    texp = np.median(np.diff(T))
    #constants
    Rad_sun = 6.955*10.0**10.0 #cm
    Rad_earth = 6.378*10.0**8.0 #cm
    Mass_Sun = 1.989*10.0**33.0 #grams
    
    model = build_model_sigma_clip(ID, Sector, T,F,E, logg, logg_err, M_star,M_star_err, R_star,\
                                   R_star_err, qld_a, qld_b,Periods, T0s, Depths, sigma=5.0, maxiter=10)
        
    print('plotting initial guesses')
    print(' ')
    #if multiple planets input
    letters = "bcdefghijklmnopqrstuvwxyz"[:len([Periods])]
    with model:
        mean = model.map_soln["mean"]
        light_curves = xo.utils.eval_in_model(model.light_curves, model.map_soln)

    plt.plot(model.x, model.y - mean, "k.", label="data")
    for n, l in enumerate(letters):

    #     print(np.shape(light_curves[:,n]),np.shape(model.x))

        plt.plot(model.x, light_curves, "r.-",label="planet {0}".format(l), zorder=100-n)
        #plt.plot(model.x, synthetic_signal-1, 'r.',label="injected model", zorder=99-n)

    plt.xlabel("time [days] TJD")
    plt.ylabel("norm. flux")
    plt.title("initial fit")
    # plt.xlim(model.x.min(), model.x.max())
    plt.legend(fontsize=10,ncol=3);
    plt.savefig(MCMCfigpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+'_LC_intialguess.png')
    plt.close()
#     plt.show()
    
    t0_init=model.map_soln["t0"]
    per_init=model.map_soln["period"]
    x_fold_init,x_fold_init_f = phasefold(t0_init,model.x,per_init,model.y)
    modelpf,modelff=phasefold(t0_init,model.x,per_init,light_curves)
    plt.plot(x_fold_init,x_fold_init_f-mean,"k.", label="data")
    plt.plot(modelpf,modelff, "r.-",label="planet {0}".format(l), zorder=100-n)
    #plt.plot(x_fold, synthetic_signal-1, 'r.',label="injected model", zorder=99-n)
    plt.xlim(-0.3,0.3)
    plt.xlabel('phase [days]')
    plt.ylabel('norm. flux')
    plt.savefig(MCMCfigpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+'_PFLC_intialguess.png')
    plt.close()
#     plt.show()
    
    print('starting sampling chains')
    print('')
    import time as clock

    start_time=clock.time()

    np.random.seed(123)
    sampler = xo.PyMC3Sampler(window=50, start=50, finish=500)
    with model:
        burnin = sampler.tune(tune=3000, start=model.map_soln,
                              step_kwargs=dict(target_accept=0.9),
                              chains=2, cores=2)
        trace = sampler.sample(draws=1000, chains=2, cores=2)

    print('runtime: ',(clock.time()-start_time)/60.0, ' minutes') #minutes
    print('')
    
    print('trace shape',np.shape(trace))
    print('LC shape', np.shape(model.light_curves.tag.test_value) )
    
    
    with model:
    #     light_curves = np.empty((500, len(model.x), len(Periods)))
    #     light_curves = np.empty((500, len(model.x), 2))
        light_curves = np.empty((500, len(model.x)))#, 1))

        func = xo.utils.get_theano_function_for_var(model.light_curves)
        for i, sample in enumerate(xo.utils.get_samples_from_trace(
                trace, size=len(light_curves.flatten))):
            light_curves[i] = func(*xo.utils.get_args_for_theano_function(sample))

    for n, letter in enumerate(letters):
        plt.figure()

        # Compute the GP prediction
        mean_mod = 0#np.median(trace["mean"])#[:, None])

        # Get the posterior median orbital parameters
        p = np.median(trace["period"])##[:, n]) #<-too many indices
        t0 = np.median(trace["t0"])##[:, n])

        # Compute the median of posterior estimate of the contribution from
        # the other planet. Then we can remove this from the data to plot
        # just the planet we care about.

        inds = np.arange(0) != n
    #     inds = np.arange(2) != n
    #     inds = np.arange(1) != n
        others = np.median(np.sum(light_curves[:, inds], axis=-1), axis=0)
#         others = np.median(np.sum(light_curves, axis=-1), axis=0)

        # Plot the folded data
        x_fold = (model.x - t0 + 0.5*p) % p - 0.5*p
        plt.plot(x_fold, model.y - mean_mod - others, ".k", label="data", zorder=-1000)

        # Plot the folded model
        inds = np.argsort(x_fold)
        inds = inds[np.abs(x_fold)[inds] < 0.3]
    #     pred = light_curves[:, inds, n]
        pred = light_curves[:, inds]#, 0]
        pred = np.percentile(pred, [16, 50, 84], axis=0)
        plt.plot(x_fold[inds], pred[1], color="C1", label="model")
        art = plt.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5,
                               zorder=1000)
        art.set_edgecolor("none")

        # Annotate the plot with the planet's period
        txt = "period = {0:.4f} +/- {1:.4f} d ; TC = {2:.4f}".format(
            np.mean(trace["period"]), np.std(trace["period"]), np.mean(trace["t0"])-2457000)
        plt.annotate(txt, (0, 0), xycoords="axes fraction",
                     xytext=(5, 5), textcoords="offset points",
                     ha="left", va="bottom", fontsize=12)

        plt.legend(fontsize=10, loc='upper center',ncol=2)

        plt.xlabel("time since transit [Hrs]")
        plt.ylabel("relative flux")
        plt.title("TIC {0}-{1} ; Sector {2}".format(ID, letter, Sector));
        plt.xlim(-0.3,0.3)
        plt.savefig(MCMCfigpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+'_PFLC_finalguess.png',\
                    bbox_inches='tight')
#         plt.show()
        plt.close()

    RE = 6.378 * 10**8  #cm
    RS = 6.955 * 10**10 #cm
    print('True Period: ',Periods)
    print('True time start: ',T0s)
    print('True planet radius (RE): ', np.sqrt(1-(1-(Depths/1000)))*R_star*RS/RE) #converting ppt to ppo
    print('True Stellar radius (RS): ',R_star)    

    plt.plot(x_fold[inds], pred[1], "C1")
    art = plt.fill_between(x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5,
                           zorder=1000)
    art.set_edgecolor("none")

    plt.plot(x_fold, model.y - mean_mod - others, ".k", label="data", zorder=-1000)
    #x_fold_synthetic=(T - TLS_TC + 0.5*TLS_P) % TLS_P - 0.5*TLS_P
    #plt.plot(x_fold_synthetic,synthetic_signal-1,'r.')    
    plt.xlim(-0.3,0.3)
#     plt.ylim(-0.005,0.005)
    plt.xlabel('time since transit [Hrs]')
    plt.ylabel('relative flux')
    plt.savefig(MCMCfigpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+'_PFLC_finalguess_comparison.png'\
                ,bbox_inches='tight')
#     plt.show()
    plt.close()

    print('making posterior plots')
    print('')
    
    # Convert to Earth radii
    r_pl = trace["r"]*R_star/Rad_earth  #rp/sun
    samples = np.concatenate((r_pl, trace["b"]), axis=-1)

    # labels = ["$R_{{\mathrm{{Pl}},{0}}}$ [$R_\oplus$]".format(i) for i in letters]
    # labels += ["impact param {0}".format(i) for i in letters]

    # corner.corner(samples, labels=labels, show_titles=True, title_kwargs=dict(fontsize=10));
    # plt.show()

    labels = ["$P_{{{0}}}$ [days]".format(i) for i in letters]
    labels += ["$t0_{{{0}}} $ [TBJD]".format(i) for i in letters]
    # labels += ["$R_{{\mathrm{{Pl}},{0}}}$ ".format(i) for i in letters]
    labels += ["$R_{P}/R_{S}$ "]#.format(i) for i in letters]
    labels += ["impact param {0}".format(i) for i in letters]

    n=len(letters)
    samples = pm.trace_to_dataframe(trace, varnames=["period", "t0","r","b"])
    truth = np.concatenate(xo.eval_in_model([tt.as_tensor_variable([Periods]), tt.as_tensor_variable([T0s-2457000])], model.test_point, model=model))


    ndim=4
    value1= model.map_soln["mean"]
    value2=np.mean(samples, axis=0)
    # Make the base corner plot
    figure = corner.corner(samples,labels=labels, quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs=dict(fontsize=10));

    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(value1, color="g")
        ax.axvline(value2[i], color="r")
    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")

    figure.tight_layout()               
    figure.savefig(MCMCfigpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+'_MCMC_posteriors.png',\
                   bbox_inches='tight')
#     figure.show()
    plt.close()
    
    return model, trace