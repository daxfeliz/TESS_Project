import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import numpy as np



import exoplanet as xo

import pandas as pd
import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#prepare input files

def phasefold(T0,time,period,flux):
    phase=(time- T0 + 0.5*period) % period - 0.5*period
    ind=np.argsort(phase, axis=0)
    return phase[ind],flux[ind]

def SMA_AU_from_Period_to_stellar(Period,R_star,M_star):
    #assumes circular orbit
    #using Kepler's third law, calculate SMA
    #solar units
    RS = 6.955*10.0**10.0 #cm, solar radius
    MS = 1.989*10.0**33.0 #g, solar mass

    G = 6.6743*10.0**(-8.0) #cm^3 per g per s^2

    R = R_star*RS
    M = M_star*MS
    P=Period*60.0*24.0*60.0 #in seconds

    #SMA
    SMA_cm = ((G*M*(P**2))/(4*(np.pi**2)))**(1/3)   

    #note R_star is already in solar units so we need to convert to cm using
    # solar radius as a constant
    Stellar_Radius = R #now in cm

    SMA = SMA_cm / Stellar_Radius #now unitless (cm / cm)
    return SMA, SMA_cm

def convert_rp_in_re_to_rs(RP,R_star):
    R_sun=6.955*10**10 # cm
    R_earth=6.378*10**8 #cm
    
    rp_in_rs_units  = (RP/R_star)*(R_earth/R_sun) #in Stellar units
    
    return rp_in_rs_units

def convert_rp_in_rs_to_re(rp_in_rs_units):
    R_sun=6.955*10**10 # cm
    R_earth=6.378*10**8 #cm
    
    rp_in_re_units = (rp_in_rs_units)*(R_sun/R_earth) #in Earth units
    
    return rp_in_re_units

def Transit_duration(Period, SMA_cm, R_star, R_planet_RE):
    RE = 6.378*10.0**8 #cm
    RS = 6.955 *10.0**10 #cm    
    A = Period/np.pi #in days
    B =(R_star*RS +R_planet_RE*RE)/ SMA_cm #in cm
    
    T_dur = A*np.arcsin(B) #in days
    return T_dur

def BATMAN_MODEL(RP,T0,P,B,qld,R_star,M_star, t,y,yerr):
    import batman
    import math
    
    R_sun=6.955*10**10 # cm
    R_earth=6.378*10**8 #cm

    rp = convert_rp_in_re_to_rs(RP,R_star)
    
    cad = np.nanmedian(np.diff(t))
    
    SMA , SMA_cm = SMA_AU_from_Period_to_stellar(P,R_star,M_star)
    
    I = math.degrees(np.arccos( (B*R_star*R_sun) /SMA_cm )) #assuming e=0, w=90
        
    #Initialize Parameters 
    params = batman.TransitParams()      # object to store transit parameters
    params.t0 = T0                       # time of inferior conjunction
    params.per = P                       # orbital period
    params.rp = rp                       # planet radius (in units of stellar radii)
    params.a = SMA                       # semi-major axis (in units of stellar radii)
    params.inc = I                       # orbital inclination (in degrees)
    params.ecc = 0.                      # eccentricity
    params.w = 90.                       # longitude of periastron (in degrees)
    params.limb_dark = "quadratic"       # limb darkening model ->"uniform", "linear", "quadratic", "nonlinear", etc.
    params.u = [qld[0],qld[1]]           # limb darkening coefficients

    mt = np.linspace(np.min(t), np.max(t), len(t))       # times at which to calculate light curve
    m = batman.TransitModel(params, mt, exp_time=cad)    # initializes model

    flux = m.light_curve(params)
    flux= flux/np.nanmedian(flux)
    
    return mt, flux



def Make_dir(ID,Sector,path=None):
    import os
    
    if path is None:
        path = os.getcwd()+'/Sector_'+str(Sector)+'/TIC_'+str(ID)+'/pymc3/'
    else:
        path = path+'Sector_'+str(Sector)+'/TIC_'+str(ID)+'/pymc3/'
    
    if os.path.exists(path)==True:
        pass
    else:
        os.makedirs(path)
        
    return path

def cornerplot(ID,Sector,samples,map_soln,savepath,use_range=False):
    import corner
    labels = ['P','T0','RP','b']
    
    ndim=4
    mean = ndim*np.random.rand(ndim)
    value1= mean
    value2=np.mean(samples, axis=0)
    #    
    # Make the base corner plot
    if use_range==True:
        figure = corner.corner(samples[["period__0", "t0__0", "r_pl", "b__0"]],\
                               labels=labels, quantiles=[0.16, 0.5, 0.84],smooth=True,\
                               range=[(0.1,0.9), (0.1,0.9), (0.1,0.9), (0.1,0.9)],\
                               show_titles=True, title_fmt=".4f",title_kwargs=dict(fontsize=10));
    if use_range==False:
        figure = corner.corner(samples[["period__0", "t0__0", "r_pl", "b__0"]],\
                               labels=labels, quantiles=[0.16, 0.5, 0.84],smooth=True,\
                               show_titles=True, title_fmt=".4f",title_kwargs=dict(fontsize=10));        
   
    
    # Make the base corner plot
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))
#     axes.ticklabel_format(useOffset=False, style='plain')
    
#     # Loop over the diagonal
#     for i in range(ndim):
#         ax = axes[i, i]
#         ax.axvline(value1[i], color="g")
#         ax.axvline(value2[i], color="r")

#     # Loop over the histograms
#     for yi in range(ndim):
#         for xi in range(yi):
#             ax = axes[yi, xi]            
#             ax.axvline(value1[xi], color="g")
#             ax.axvline(value2[xi], color="r")
#             ax.axhline(value1[yi], color="g")
#             ax.axhline(value2[yi], color="r")
#             ax.plot(value1[xi], value1[yi], "sg")
#             ax.plot(value2[xi], value2[yi], "sr")

    figure.tight_layout()
    figure.savefig(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_Corner.png')
    plt.close()
    
def stellar_cornerplot(ID,Sector,samples,map_soln,savepath,use_range=False):
    import corner
#     print(samples)
    print(' ')    
    labels = ['u1','u2','R_star','M_star']
    
    ndim=4
    mean = ndim*np.random.rand(ndim)
    value1= mean
    value2=np.mean(samples, axis=0)
    #    
    # Make the base corner plot
    if use_range==True:
        figure = corner.corner(samples[["u__0", "u__1", "R_star", "M_star"]],\
                               labels=labels, quantiles=[0.16, 0.5, 0.84],smooth=True,\
                               range=[(0.1,0.9), (0.1,0.9), (0.1,0.9), (0.1,0.9)],\
                               show_titles=True, title_fmt=".4f",title_kwargs=dict(fontsize=10));
    if use_range==False:
        figure = corner.corner(samples[["u__0", "u__1", "R_star", "M_star"]],\
                               labels=labels, quantiles=[0.16, 0.5, 0.84],smooth=True,\
                               show_titles=True, title_fmt=".4f",title_kwargs=dict(fontsize=10));        
    # Make the base corner plot
    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))
    figure.tight_layout()
    figure.savefig(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_Stellar_Corner.png')
    plt.close()    

def plot_initial_model(ID,Sector,t,y,yerr, map_soln,P, T0,Dur,savepath):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig=plt.figure(figsize=(12,6))
    ax1=fig.add_subplot(211)
    ax2=fig.add_subplot(212)
#     ax1.errorbar(t,y,yerr=yerr,fmt='o',alpha=0.5,color='black',markersize=5,label='Data')    
    ax1.plot(t,y,'k.',markersize=5,label='Data',zorder=-100)
    ax1.plot(t, map_soln["light_curves"]+ map_soln["mean"],'r.-',markersize=3,label='Model',zorder=100)
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylabel("Normalized Relative Flux")
    ax1.set_xlabel("Time [BTJD]")
    ax1.legend(loc='best',fontsize=10,ncol=3,handletextpad=0.1)
    ax1.set_title('Initialized Model for TIC '+str(ID)+' in Sector '+str(Sector))

    pfm,ffm = phasefold(T0,t,P,map_soln["light_curves"]+ map_soln["mean"])
    pf,ff = phasefold(T0,t,P,y)

    ax2.errorbar(24*pf,ff,yerr=yerr,fmt='o',alpha=0.5,color='black',markersize=5,label='Data',zorder=-100)    

#     divider = make_axes_locatable(ax2)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cbar=ax2.scatter(24*pf, ff, c=t)#, s=10)
#     plt.colorbar(cbar,label="Time [BTJD]",cax=cax)

    ax2.plot(24*pfm,ffm,'r.-',markersize=3,zorder=100)

    ax2.set_xlim(-5*Dur,5*Dur)
    ax2.set_xlabel('Phase [Hours since '+str(np.round(T0,4))+' BTJD]')
    ax2.set_ylabel('Normalized Relative Flux')    
    ax2.set_xlim(-3.5*Dur,3.5*Dur)

    fig.tight_layout(pad=1)
    fig.savefig(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_initial_model.png',bbox_inches='tight')
    plt.close()
    
def plot_final_model(ID,Sector,trace, t, y, yerr, Dur,savepath):    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Get the posterior median orbital parameters
    p = np.median(trace["period"])
    t0 = np.median(trace["t0"])
    
    x_fold = (t - t0 + 0.5 * p) % p - 0.5*p
    inds = np.argsort(x_fold)
    inds = inds[np.abs(x_fold)[inds] < Dur*24]
    
    depth = (1-np.nanmin(y[inds]))*1e6 #in ppm
    
    pred = trace["light_curves"][:, inds, 0]+ trace["mean"][:, None]
    #print('test mcmc model shape', np.shape(pred))
    
    pred = np.percentile(pred, [16, 50, 84], axis=0)

    best_model = pred[1] #+np.nanmedian(y)       
    #print('test mcmc model shape (post percentile)', np.shape(best_model))
    
    residual = (y[inds]-best_model)/yerr[inds]
    
    #print('model len check: ',len(t[inds]) , len(x_fold[inds]), len(best_model), len(residual) 
    model_df = pd.DataFrame({'model time':t[inds], 'model phase':x_fold[inds], 'model':best_model, 'residuals':residual})
    model_df.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_model.csv')    
    
    f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[4,1]},
                   figsize=(10,7),sharex=True)
    a0.set_title('Best-fit MCMC model for TIC '+str(ID)+' in Sector '+str(Sector),fontsize=20)
    a0.errorbar(24*x_fold[inds], y[inds], yerr=yerr[inds] ,fmt='o',alpha=0.5,color='black',markersize=5,label='Data',zorder=-100)    
    
#     divider = make_axes_locatable(a0)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cbar=a0.scatter(24*x_fold[inds], y[inds], c=t)#, s=10)
#     plt.colorbar(cbar,label="Time [BTJD]",cax=cax)

    a0.plot(24*x_fold[inds], best_model,'r.-',markersize=3,label='Best-fit model',zorder=100)       
#     a0.plot(24*x_fold[inds], best_model[inds],'r.-',markersize=3,label='Best-fit model')       

    art = a0.fill_between(24*x_fold[inds], pred[0], pred[2],color='red',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
    art.set_edgecolor("none")
    a0.set_ylabel('Normalized Flux',fontsize=18)
    a0.set_xlabel('Phase [Hours from '+str(np.round(t0,4))+' BTJD]')
    a0.set_xlim(-3.5*Dur,3.5*Dur) #hrs from mid transit
#     a0.set_ylim(np.nanmean(best_model)-2*np.nanstd(best_model),np.nanmean(best_model)+2*np.nanstd(best_model))
    a0.set_ylim(1-2*depth/1e6,1+2*depth/1e6)
    a0.minorticks_on()
    a0.legend(loc='lower right')
    a0.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                   bottom=True, top=True, left=True, right=True)
    
    a1.scatter(24*x_fold[inds],residual[inds],color='black',alpha=0.5)
    a1.axhline(0,color='k')
#     a1.set_ylim(0-1.5*np.nanmax(np.abs(residual[inds])),0+1.5*np.max(np.abs(residual[inds])))
    a1.set_ylim(0-np.nanmax(np.abs(residual[inds]))-np.nanstd(np.abs(residual[inds])),\
                            0+np.nanmax(np.abs(residual[inds]))+np.std(np.abs(residual[inds])))
    a1.set_xlim(-3.5*Dur,3.5*Dur) #hrs from mid transit
    a1.minorticks_on()
    a1.set_ylabel(r'Residuals ($\rm{\sigma}$)',fontsize=15)
    a1.set_xlabel('Phase [Hours from '+str(np.round(t0,4))+' BTJD]')
    a1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                   bottom=True, top=True, left=True, right=True)
    a1.minorticks_on()
    a1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                   bottom=True, top=True, left=True, right=True)
    f.tight_layout(pad=1)
    f.savefig(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_bestfit_MCMC_model.png',bbox_inches='tight')
    plt.close()
    
def trace_chain_plot(ID,Sector,trace,labels,savepath):
    #this is hardcoded to work for four labels
    fig= plt.figure(figsize=(15, 10))

    ax1=fig.add_subplot(411)
    ax2=fig.add_subplot(412)
    ax3=fig.add_subplot(413)
    ax4=fig.add_subplot(414)
    
    ax1.get_xaxis().get_major_formatter().set_scientific(False)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_scientific(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    
    ax2.get_xaxis().get_major_formatter().set_scientific(False)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_scientific(False)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    
    ax3.get_xaxis().get_major_formatter().set_scientific(False)
    ax3.get_xaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_scientific(False)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    
    ax4.get_xaxis().get_major_formatter().set_scientific(False)
    ax4.get_xaxis().get_major_formatter().set_useOffset(False)
    ax4.get_yaxis().get_major_formatter().set_scientific(False)
    ax4.get_yaxis().get_major_formatter().set_useOffset(False)    
    
    label = labels[0]
    param = (trace[label])
    mean_param = [np.mean(param[:i]) for i in np.arange(1, len(param))]
    ax1.plot(mean_param, lw=2.5,color='black')
    ax1.axhline(y=np.mean(mean_param),color='red',label='mean = '+str(np.round(np.mean(mean_param),3)))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MCMC mean of "+label)
    ax1.set_title("MCMC estimation of "+label);
    ax1.legend(loc='upper right')

    label = labels[1]
    param = (trace[label])
    mean_param = [np.mean(param[:i]) for i in np.arange(1, len(param))]
    ax2.plot(mean_param, lw=2.5,color='black')
    ax2.axhline(y=np.mean(mean_param),color='red',label='mean = '+str(np.round(np.mean(mean_param),3)))
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MCMC mean of "+label)
    ax2.set_title("MCMC estimation of "+label);
    ax2.legend(loc='upper right')

    label = labels[2]
    param = (trace[label])
    mean_param = [np.mean(param[:i]) for i in np.arange(1, len(param))]
    ax3.plot(mean_param, lw=2.5,color='black')
    ax3.axhline(y=np.mean(mean_param),color='red',label='mean = '+str(np.round(np.mean(mean_param),3)))
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("MCMC mean of "+label)
    ax3.set_title("MCMC estimation of "+label);
    ax3.legend(loc='upper right')


    label = labels[3]
    param = (trace[label])
    mean_param = [np.mean(param[:i]) for i in np.arange(1, len(param))]
    ax4.plot(mean_param, lw=2.5,color='black')
    ax4.axhline(y=np.mean(mean_param),color='red',label='mean = '+str(np.round(np.mean(mean_param),3)))
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("MCMC mean of "+label)
    ax4.set_title("MCMC estimation of "+label);
    ax4.legend(loc='upper right')

    fig.tight_layout(pad=1)
    fig.savefig(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_trace_chain.png',bbox_inches='tight')
    plt.close()
    
def MCMC_transit_fit_OLD(ID,Sector,input_LC,input_Transit_params, savedestinationpath,Niters=2500, Ndraws=2000,OS=7):    
    import time as clock
    import pymc3 as pm
    import corner
    start=clock.time()
    
    #first make directory to store summary figures and data
    savepath = Make_dir(ID,Sector,path=savedestinationpath)
    
    # light curve
    t = np.array(input_LC['Time'].to_list())
    y = np.array(input_LC['Flux'].to_list())
    yerr = np.array(input_LC['Flux Error'].to_list())
    texp = np.nanmedian(np.diff(t))
    
    # TLS best fit parameters
    P = input_Transit_params['Period'].item()
    Perr= input_Transit_params['Period Error'].item()
    if np.isfinite(Perr)==False:
        Perr=5/(60*24) # 5 MINUTES
    #
    T0= input_Transit_params['T0'].item()
    # NOTE: TLS' transit time is the 1st transit.
    # We can calculate all the transit times and use the middle-est
    # transit as the MCMC input transit time
    #
    # from transitleastsquares.stats import all_transit_times
    # TLS_all_transit_times = all_transit_times(T0, t, P)
    # T0 = np.nanmedian(TLS_all_transit_times)
    #
    #
    #
    Dur=input_Transit_params['Duration'].item()
    RP=input_Transit_params['Planet Radius'].item()
    RP_RS=input_Transit_params['RP_RS'].item()
    RP_RSerr=input_Transit_params['RP_RSerr'].item()
    
    #stellar params
    qld_a=input_Transit_params['qld_a'].item()
    qld_b=input_Transit_params['qld_b'].item()
    u=[qld_a,qld_b]
    R_star=input_Transit_params['R_star'].item()
    M_star=input_Transit_params['M_star'].item()    
        
    #make sure y is zero centered
#     y = y - np.nanmedian(y)
#     y = y -1
    
    print('initializing MCMC model')
    with pm.Model() as model:
        
        s=1 # <----for 1 planet model? 

        # The baseline flux
#         mean = pm.Normal("mean", mu=0.0, sd=1.0)
        mean = pm.Normal("mean", mu=np.nanmean(y), sd=np.nanstd(y))

        # The time of a reference transit for each planet
#         t0 = pm.Normal("t0", mu=T0, sd=1.0, shape=s)
        t0 = pm.Normal("t0", mu=T0, sd=Dur/24.0, shape=s)

        # The log period; also tracking the period itself
#         logP = pm.Normal("logP", mu=np.log(P), sd=0.1, shape=s)        
        logP = pm.Normal("logP", mu=np.log(P), sd=Perr, shape=s)
        period = pm.Deterministic("period", pm.math.exp(logP))

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        U = xo.distributions.QuadLimbDark("u", testval=np.array( [u[0] , u[1]] ))

        #Rp/ RS
        #
        r = pm.Uniform("r", lower=0.01, upper=RP_RS+RP_RSerr, shape=s, testval=np.array([RP_RS]))
        b = xo.distributions.ImpactParameter("b", ror=r, shape=s)#, testval=np.random.rand(s))
        #r, b = xo.distributions.get_joint_radius_impact(min_radius=0.01, max_radius=RP_RS+RP_RSerr, testval_r=np.array([RP_RS]))

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=P, t0=T0, b=b, m_star=M_star, r_star=R_star)
        
        # Compute the model light curve using starry
        #light_curves = xo.LimbDarkLightCurve(U).get_light_curve(orbit=orbit, r=r, t=t)
#         light_curves = xo.LimbDarkLightCurve(U).get_light_curve(orbit=orbit, r=r, t=t,texp=texp,oversample=10,order=0)
        light_curves = xo.LimbDarkLightCurve(U).get_light_curve(orbit=orbit, r=r, t=t,texp=texp,oversample=OS,\
                                                                order=0,use_in_transit =True)
        light_curve = pm.math.sum(light_curves, axis=-1) + mean

        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", light_curves)

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset
        map_soln = xo.optimize(start=model.test_point)
        
#         txt, bib = xo.citations.get_citations_for_model(model=model)
#         print(txt)
#         print('')
#         print(bib)
        
        
    plot_initial_model(ID,Sector,t,y,yerr, map_soln,P, T0,Dur,savepath)
    
    print(' ')
    if (clock.time()-start)> 60:
        print('Initialization complete. Runtime ',(clock.time()-start)/60,' minutes')
    if (clock.time()-start)< 60:
        print('Initilization complete. Runtime ',(clock.time()-start),' seconds')
    print(' ')
    
    
    # For sampling posteriors from MCMC initial guess
    # Niters = Number of iterations to tune, defaults to 1000. 
    # Samplers adjust the step sizes, scalings or similar during tuning.
    #
    # Ndraws = The number of samples to draw. Defaults to 1000. 
    # The number of tuned samples are discarded by default.
    print('Sampling Posteriors')
    np.random.seed(42)
    with model:
        #trace = pm.sample(tune=Niters,draws=Ndraws,start=map_soln,cores=2,chains=2,init="adapt_full",target_accept=0.9)
        #trace = pm.sample(tune=Niters,draws=Ndraws,start=map_soln, cores=2,chains=2,target_accept=0.9)
        trace = pm.sample(tune=Niters,draws=Ndraws,start=map_soln,init='adapt_diag',target_accept=0.99)
    #
    labels=["period", "t0", "r", "b"]
    trace_chain_plot(ID,Sector,trace,labels,savepath)
    #
    trace_df=pm.summary(trace, varnames=["period", "t0", "r", "b", "u", "mean"])
    
    #model_df=pm.summary(trace, varnames=["light_curves","mean"])
    model_df=pm.trace_to_dataframe(trace, varnames=["light_curves","mean"])
    
    stellar_samples = pm.trace_to_dataframe(trace, varnames=["u","R_star", "M_star"])
    samples = pm.trace_to_dataframe(trace, varnames=["period", "t0","r","b"])
    
    truth = np.concatenate(xo.eval_in_model([period, r], model.test_point, model=model))
    
    from astropy import units as u
    # Convert the radius to Earth radii
    samples["r_pl"] = (np.array(samples["r__0"]) * u.R_sun).to(u.R_earth).value
    
    cornerplot(ID,Sector,samples,map_soln,savepath)
    stellar_cornerplot(ID,Sector,stellar_samples,map_soln,savepath)
    
    
    print(' ')
    print('Finished!')
    print('')
       
    #save trace and samples as csv
    trace_df.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_trace.csv',float_format='%.16f')       
    samples.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_samples.csv',float_format='%.16f')

    #grab best-fit params and errors
    quantiles = [5/100, 50/100, 95/100] #ranges 0 to 1
    quantiles = [16/100, 50/100, 84/100]
    #
    Pmin,Pmed,Pmax=np.quantile(samples['period__0'],q=quantiles)
    T0min,T0med,T0max=np.quantile(samples['t0__0'],q=quantiles)
    RPmin,RPmed,RPmax=np.quantile(samples['r_pl'],q=quantiles)
    Bmin,Bmed,Bmax=np.quantile(samples['b__0'],q=quantiles)
    # calculate differences from 50th percentile
    Pmin = np.abs(Pmin-Pmed)
    Pmax = np.abs(Pmax-Pmed)
    RPmin = np.abs(RPmin-RPmed)
    RPmax = np.abs(RPmax-RPmed)
    T0min = np.abs(T0min-T0med)
    T0max = np.abs(T0max-T0med)
    Bmin = np.abs(Bmin-Bmed)
    Bmax = np.abs(Bmax-Bmed)    
    
    
    print('Best-fit MCMC parameters: ')
    print('P = ',Pmin,Pmed,Pmax)
    print('T0 = ',T0min,T0med,T0max)
    print('RP = ',RPmin,RPmed,RPmax)    
    print('b = ',Bmin,Bmed,Bmax)
    
    
#     #convert to earth radii
#     RPmin = (RPmin * u.R_sun).to(u.R_earth).value
#     RPmed = (RPmed * u.R_sun).to(u.R_earth).value
#     RPmax = (RPmax * u.R_sun).to(u.R_earth).value
    
#     print('Best-fit MCMC parameters: ')
#     print('P = {:.3f}'.format(trace_df.loc['period[0]']['mean'].item()),
#           '+/ {:.3f}'.format(trace_df.loc['period[0]']['sd'].item()))    

#     print('Mid-Transit Time = {:.3f}'.format(trace_df.loc['t0[0]']['mean'].item()),
#           '+/ {:.3f}'.format(trace_df.loc['t0[0]']['sd'].item()))   

#     print('Rp  {:.3f}'.format((trace_df.loc['r[0]']['mean'].item() * u.R_sun).to(u.R_earth).value ),\
#           '+/ {:.3f}'.format((trace_df.loc['r[0]']['sd'].item() * u.R_sun).to(u.R_earth).value))  

#     print('Impact parameter (b) = {:.3f}'.format(trace_df.loc['b[0]']['mean'].item()),
#           '+/ {:.3f}'.format(trace_df.loc['b[0]']['sd'].item()))   
    
#     Pmin = trace_df.loc['period[0]']['mean'].item()-trace_df.loc['period[0]']['sd'].item()
#     Pmed = trace_df.loc['period[0]']['mean'].item()
#     Pmax = trace_df.loc['period[0]']['mean'].item()+trace_df.loc['period[0]']['sd'].item()
    
#     T0min = trace_df.loc['t0[0]']['mean'].item()-trace_df.loc['t0[0]']['sd'].item()
#     T0med = trace_df.loc['t0[0]']['mean'].item()
#     T0max = trace_df.loc['t0[0]']['mean'].item()+trace_df.loc['t0[0]']['sd'].item()
    
#     RPmin = trace_df.loc['r[0]']['mean'].item()-trace_df.loc['r[0]']['sd'].item()
#     RPmed = trace_df.loc['r[0]']['mean'].item()
#     RPmax = trace_df.loc['r[0]']['mean'].item()+trace_df.loc['r[0]']['sd'].item()
    
#     #convert to earth radii
#     RPmin = (RPmin * u.R_sun).to(u.R_earth).value
#     RPmed = (RPmed * u.R_sun).to(u.R_earth).value
#     RPmax = (RPmax * u.R_sun).to(u.R_earth).value    
    
#     Bmin = trace_df.loc['b[0]']['mean'].item()-trace_df.loc['b[0]']['sd'].item()
#     Bmed = trace_df.loc['b[0]']['mean'].item()
#     Bmax = trace_df.loc['b[0]']['mean'].item()+trace_df.loc['b[0]']['sd'].item()
    
    
    
    best_MCMC_results = pd.DataFrame({'P err1':Pmin,'P':Pmed, 'P err2':Pmax,\
                                     'T0 err1':T0min, 'T0':T0med,'T0 err2':T0max,\
                                     'RP err1':RPmin,'RP':RPmed, 'RP err2':RPmax,\
                                     'b err1':Bmin,'b':Bmed, 'b err2':Bmax}, index=[0])
    print(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_bestfit_parameters.csv')
    #
    best_MCMC_results.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_bestfit_parameters.csv',\
                             index=False,float_format='%.16f')    
    
    # Plot the best-fit MCMC model
    plot_final_model(ID,Sector,trace, t, y, yerr, Dur,savepath)
    
    
    if (clock.time()-start)> 60:
        print('Total runtime ',(clock.time()-start)/60,' minutes')
    if (clock.time()-start)< 60:
        print('Total runtime ',(clock.time()-start),' seconds')
    print(' ')
    
#     return samples, trace


def MCMC_transit_fit(ID,Sector,input_LC,input_Transit_params, savedestinationpath,Niters=2500, Ndraws=2000,OS=10):    
    import time as clock
    import pymc3 as pm
    import corner
    import exoplanet as xo
    start=clock.time()
    
    #first make directory to store summary figures and data
    savepath = Make_dir(ID,Sector,path=savedestinationpath)
    
    # light curve
    t = np.array(input_LC['Time'].to_list())
    y = np.array(input_LC['Flux'].to_list())
    yerr = np.array(input_LC['Flux Error'].to_list())
    texp = np.nanmedian(np.diff(t))
    
    # TLS best fit parameters
    P = input_Transit_params['Period'].item()
    Perr= input_Transit_params['Period Error'].item()
    if np.isfinite(Perr)==False:
        Perr=5/(60*24) # 5 MINUTES
    #
    T0= input_Transit_params['T0'].item()
    # NOTE: TLS' transit time is the 1st transit.
    # We can calculate all the transit times and use the middle-est
    # transit as the MCMC input transit time
    # from transitleastsquares.stats import all_transit_times
    # TLS_all_transit_times = all_transit_times(T0, t, P)
    # T0 = np.nanmedian(TLS_all_transit_times)
    #
    #
    #
    Dur=input_Transit_params['Duration'].item()
    RP=input_Transit_params['Planet Radius'].item()
    RP_RS=input_Transit_params['RP_RS'].item()
    RP_RSerr=input_Transit_params['RP_RSerr'].item()
    
    #stellar params
#     qld_a=input_Transit_params['qld_a'].item()
#     qld_b=input_Transit_params['qld_b'].item()
#     u=[qld_a,qld_b]
#     R_star=input_Transit_params['R_star'].item()
#     M_star=input_Transit_params['M_star'].item()    
    
    from transitleastsquares import catalog_info
    qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    R_Star_err = (R_star_min+R_star_max)/2.0
    M_Star_err = (M_star_min+M_star_max)/2.0
    u=qld
        
    #make sure y is zero centered (?)
#     y = y - np.nanmedian(y)
#     y = y -1
    
    print('initializing MCMC model')
    with pm.Model() as model:
        
        s=1 # <----for 1 planet model? 

        # The baseline flux
        mean = pm.Normal("mean", mu=np.nanmean(y), sd=np.nanstd(y))

        # The time of a reference transit for each planet
#         t0 = pm.Normal("t0", mu=T0, sd=1.0, shape=s)
        t0 = pm.Normal("t0", mu=T0, sd=Dur/24.0, shape=s)

        # The log period; also tracking the period itself
#         logP = pm.Normal("logP", mu=np.log(P), sd=0.1, shape=s)        
        logP = pm.Normal("logP", mu=np.log(P), sd=Perr, shape=s)
        period = pm.Deterministic("period", pm.math.exp(logP))

        # The Kipping (2013) parameterization for quadratic limb darkening paramters
        U = xo.distributions.QuadLimbDark("u", testval=np.array( [u[0] , u[1]] ))
        
        #Mstar and Rstar        
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
        m_star = BoundedNormal("M_star", mu=M_star, sd=M_Star_err)
        r_star = BoundedNormal("R_star", mu=R_star, sd=R_Star_err)

        #Rp/ RS
        #
        r = pm.Uniform("r", lower=0.01, upper=1, shape=s, testval=np.array([RP_RS]))
        b = xo.distributions.ImpactParameter("b", ror=r, shape=s)#, testval=np.random.rand(s))
        #r, b = xo.distributions.get_joint_radius_impact(min_radius=0.01, max_radius=RP_RS+RP_RSerr, testval_r=np.array([RP_RS]))

        # Set up a Keplerian orbit for the planets
        orbit = xo.orbits.KeplerianOrbit(period=P, t0=T0, b=b, m_star=m_star, r_star=r_star)
        
        # Compute the model light curve using starry
        #light_curves = xo.LimbDarkLightCurve(U).get_light_curve(orbit=orbit, r=r, t=t)
#         light_curves = xo.LimbDarkLightCurve(U).get_light_curve(orbit=orbit, r=r, t=t,texp=texp,oversample=10,order=0)
        light_curves = xo.LimbDarkLightCurve(U).get_light_curve(orbit=orbit, r=r, t=t,texp=texp,oversample=OS,\
                                                                order=0,use_in_transit =True)
        light_curve = pm.math.sum(light_curves, axis=-1) + mean

        # Here we track the value of the model light curve for plotting
        # purposes
        pm.Deterministic("light_curves", light_curves)

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)

        # Fit for the maximum a posteriori parameters given the simuated
        # dataset
        map_soln = xo.optimize(start=model.test_point)
        
#         txt, bib = xo.citations.get_citations_for_model(model=model)
#         print(txt)
#         print('')
#         print(bib)
        
        
    plot_initial_model(ID,Sector,t,y,yerr, map_soln,P, T0,Dur,savepath)
    
    print(' ')
    if (clock.time()-start)> 60:
        print('Initialization complete. Runtime ',(clock.time()-start)/60,' minutes')
    if (clock.time()-start)< 60:
        print('Initilization complete. Runtime ',(clock.time()-start),' seconds')
    print(' ')
    
    
    # For sampling posteriors from MCMC initial guess
    # Niters = Number of iterations to tune, defaults to 1000. 
    # Samplers adjust the step sizes, scalings or similar during tuning.
    #
    # Ndraws = The number of samples to draw. Defaults to 1000. 
    # The number of tuned samples are discarded by default.
    print('Sampling Posteriors')
    np.random.seed(42)
    with model:
        trace = pm.sample(tune=Niters,draws=Ndraws,start=map_soln,init='adapt_diag',target_accept=0.99)
    #
    labels=["period", "t0", "r", "b"]
    trace_chain_plot(ID,Sector,trace,labels,savepath)
    #
    trace_df=pm.summary(trace, varnames=["period", "t0", "r", "b", "u", "mean"])
    
    #model_df=pm.summary(trace, varnames=["light_curves","mean"])
    model_df=pm.trace_to_dataframe(trace, varnames=["light_curves","mean"])
    
    stellar_samples = pm.trace_to_dataframe(trace, varnames=["u","R_star", "M_star"])
    samples = pm.trace_to_dataframe(trace, varnames=["period", "t0","r","b"])
    
    truth = np.concatenate(xo.eval_in_model([period, r], model.test_point, model=model))
    
    from astropy import units as u
    # Convert the radius to Earth radii
    samples["r_pl"] = (np.array(samples["r__0"]) * u.R_sun).to(u.R_earth).value
    
    cornerplot(ID,Sector,samples,map_soln,savepath)
    stellar_cornerplot(ID,Sector,stellar_samples,map_soln,savepath)      
       
    #save trace and samples as csv
    trace_df.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_trace.csv',float_format='%.16f')       
    samples.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_samples.csv',float_format='%.16f')

    #grab best-fit params and errors
    quantiles = [5/100, 50/100, 95/100] #ranges 0 to 1
    quantiles = [16/100, 50/100, 84/100]
    #
    Pmin,Pmed,Pmax=np.quantile(samples['period__0'],q=quantiles)
    T0min,T0med,T0max=np.quantile(samples['t0__0'],q=quantiles)
    RPmin,RPmed,RPmax=np.quantile(samples['r_pl'],q=quantiles)
    Bmin,Bmed,Bmax=np.quantile(samples['b__0'],q=quantiles)
    # calculate differences from 50th percentile
    Pmin = np.abs(Pmin-Pmed)
    Pmax = np.abs(Pmax-Pmed)
    RPmin = np.abs(RPmin-RPmed)
    RPmax = np.abs(RPmax-RPmed)
    T0min = np.abs(T0min-T0med)
    T0max = np.abs(T0max-T0med)
    Bmin = np.abs(Bmin-Bmed)
    Bmax = np.abs(Bmax-Bmed)    
    #
    #
    print('Best-fit MCMC parameters: ')
    print('P = ',Pmin,Pmed,Pmax)
    print('T0 = ',T0min,T0med,T0max)
    print('RP = ',RPmin,RPmed,RPmax)    
    print('b = ',Bmin,Bmed,Bmax)
    #
    best_MCMC_results = pd.DataFrame({'P err1':Pmin,'P':Pmed, 'P err2':Pmax,\
                                     'T0 err1':T0min, 'T0':T0med,'T0 err2':T0max,\
                                     'RP err1':RPmin,'RP':RPmed, 'RP err2':RPmax,\
                                     'b err1':Bmin,'b':Bmed, 'b err2':Bmax}, index=[0])
    print(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_bestfit_parameters.csv')
    #
    best_MCMC_results.to_csv(savepath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_MCMC_bestfit_parameters.csv',\
                             index=False,float_format='%.16f')    
    
    # Plot the best-fit MCMC model
    plot_final_model(ID,Sector,trace, t, y, yerr, Dur,savepath)
    
    
    if (clock.time()-start)> 60:
        print('Total runtime ',(clock.time()-start)/60,' minutes')
    if (clock.time()-start)< 60:
        print('Total runtime ',(clock.time()-start),' seconds')
    print(' ')
    print('Finished!')
    print('')
