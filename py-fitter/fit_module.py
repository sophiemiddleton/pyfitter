# Fit the data to a product of PDFs defined in PDF_list

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tensorflow as tf
import zfit
from typing import List, Tuple, Optional, Any
from zfit.result import FitResult

from momPDF_module import MomModel, poly58
from timePDF_module import TimeModel
from recoplot_module import plotmom_fit, plottime_fit, plotmom_fit_old, plot_variable
from mom_components import mom_components
from time_components import time_components

def Unbinned_fit_mom(mom_mag, track_cat, count_particle_types, fit_range_low, fit_range_hi, plot_cat=False, verbose=0, minos=False, dio_efficiency=None, dio_resolution=None, plot_NLL= False):
    """
    ----------
    Configures and calls the unbinned maximum likelihood fit for momentum using zfit

    Parameters
    ----------
    mom_mag : awkward array of floats
        magnitude of momenta at chosen SID
    track_cat : awkward array of floats
        gives track catagory (corresponds to index in component list)
    fit_range_low, fit_range_hi : float, float
        min and max of fit range (args in the main function)
    plot_cat: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts
    minos : bool
        set true to evaluate minos errors
    """

    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ initializing fit")

    fit_range = (fit_range_low, fit_range_hi)
    obs_mom = zfit.Space('x', limits=fit_range)

    # PDF components
    pars = []
    pdfs = {}
    norms = {}
    constraints = []
    
    # Renamed 'nlls' to 'aux_nlls' for clarity (stores NLL terms from secondary fits/constraints)
    aux_nlls = [] 
    
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ components", mom_components)
      
    # --- Loop over Components and Build Model ---
    for proc in mom_components:
        comp_config = mom_components[proc]
        pdf = comp_config['pdf']
        pardict = comp_config['pars']
        treat_params = comp_config['treat_params']
        
        # Determine if advanced fit structure is present
        use_advanced_model = 'advanced_pars' in comp_config

        # Call the updated MomModel
        pdfs[proc], norms[proc] = MomModel(
            obs_mom, 
            pars, 
            proc, 
            pdf, 
            pardict, 
            treat_params, 
            fit_range, 
            constraints,
            advanced_config=comp_config,
            use_advanced=use_advanced_model  
        )
        
        if use_advanced_model  and comp_config.get('advanced_pars') and 'nll_sources' in comp_config['advanced_pars']:
            sources = comp_config['advanced_pars']['nll_sources']
            if not isinstance(sources, list): 
                sources = [sources]
            
            for nll_source in sources:
                # Use getattr to safely check for 'simul_source' without crashing
                sim_data = getattr(nll_source, 'simul_source', None)
                
                if sim_data is not None:
                    if verbose > 0:
                        print(f"[py-fitter] 🔗 Activating simultaneous fit for: {proc}")
                    # Only call get_nll if data is actually present to unpack
                    aux_nlls.extend(nll_source.get_nll(pars))
                else:
                    if verbose > 0:
                        print(f"[py-fitter] 📌 Using fixed resolution/loss for: {proc}")

    # --- build combined PDF ---
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))

    # Convert data to zfit Data
    mom_mag_skim = ak.nan_to_none(mom_mag)
    mom_mag_skim = ak.drop_none(mom_mag_skim)
    mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
    mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ running minimizer")

    # --- Loss function creation (MODIFIED) ---
    # Build the main loss from the Extended NLL and initial constraints
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=mom_zfit, constraints=constraints)
    
    # Add the auxiliary NLL terms 
    for nll in aux_nlls:
        loss = loss + nll # zfit overloads the '+' operator for loss addition

    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ finished minimizing")
      
    # --- Minos Error Calculation  ---
    if minos == True:
      try:
          param_errors, _ = result.errors(method='minuit_minos')
      except:
          print('[py-fitter/fit_module/Unbinned_fit_mom] ❌ ERROR! Invalid fit, postfit parameters may not be optimal')

    if result.valid == True:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
      
    # Plot after fit
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ plotting")
    print(pars)
    
    # plotmom_fit (Assuming this is an external function you need to call)
    plotmom_fit(mom_mag,count_particle_types, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in mom_components.keys()], plot_cat)
    plt.show()
    
    # --- NLL Scan Plotting  ---
    if plot_NLL:
      # performs optional scan to draw NLL plot:
      best_nll = result.fmin
      print(f"Best fit nsig: {result.params[pars[1]]['value']:.2f}")
      print(f"Minimum NLL: {best_nll:.2f}")

      scan_range = np.linspace(0,float(pars[1].value())+float(pars[1].value())*0.5, 41)
      nll_values = []

      print("Starting NLL scan...")
      # Loop over the scan range for the signal yield
      for n in scan_range:
          with pars[1].set_value(n):
              pars[1].floating = False
              
              minimizer.minimize(loss )
              nll_values.append(loss.value())  
              pars[1].floating = True

      print("Scan complete...")

      # Find true number:
      data_signal = mom_mag.mask[count_particle_types == 168]
      data_signal = np.array(ak.flatten(data_signal, axis=None))
      
      delta_nll = np.array(nll_values) - best_nll
      fig, ax = plt.subplots()
      ax.plot(scan_range, delta_nll)
      #ax.plot([len(data_signal),len(data_signal)], [min(delta_nll),max(delta_nll)], 'k--')
      true_signal = len(data_signal)
      ax.axvline(true_signal, color='red', linestyle='--', label=f'True $N_{{sig}}$: {true_signal:.1f}')
      ax.legend()
      ax.text(true_signal + 5, 4, f'True $N_{{sig}} = {true_signal:.1f}$',
                verticalalignment='top', horizontalalignment='left', color='red')

      ax.set_xlabel('$N_{sig}$')
      ax.set_ylabel('$-2\Delta \ln(L)$')
      ax.set_title('NLL Scan for $N_{sig}$')
      ax.grid(True)
      plt.show()
      
    return result, pars[1], loss, aux_nlls, combine_pdf, constraints

def Unbinned_fit_time(times, track_cat, count_particle_types, fit_range_low, fit_range_hi, plot_cat=False, verbose=0):
    """
    Configures and calls the unbinned maximum likelihood fit for time using zfit

    Parameters
    ----------
    data : awkward array (with cuts applied)
        your data array post-processing
    fit_range_low, fit_range_hi : float, float
        min and max of fit range (args in the main function)
    plot_cat: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts

    """
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ initializing fit")
    fit_range = (fit_range_low, fit_range_hi)
    obs_time = zfit.Space('time', limits=fit_range)
    
    #build PDF components
    pars = []
    pdfs = {}
    norms = {}
    for proc in time_components:
        pdf = time_components[proc]['pdf']
        pardict = time_components[proc]['pars']
        pdfs[proc], norms[proc] = TimeModel(obs_time, pars, proc, pdf, pardict, fit_range)

    # build combined PDF
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))

    # Convert data to zfit Data
    time_skim = ak.nan_to_none(times)
    time_skim = ak.drop_none(time_skim)
    time_np = ak.to_numpy(ak.flatten(time_skim, axis=None))
    time_zfit = zfit.Data.from_numpy(array=time_np, obs=obs_time)
    
    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=time_zfit)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ finished minimizing")
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except:
        print('[py-fitter/fit_module] ❌ WARNING! Invalid fit, postfit parameters may not be optimal')
    if result.valid == True:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
    # Plot after fit

    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ plotting")
    plottime_fit(times, count_particle_types, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in time_components.keys()], True)
    plt.show()
    return result, pars[1], loss, combine_pdf

def Unbinned_2d_fit_mom_time(mom_mag, times, track_cat, fit_range_mom, fit_range_time, plot_cat=False, verbose=0, dio_efficiency = None, dio_resolution = None):
    """
    Configures and calls the unbinned maximum likelihood fit for momentum and time using zfit

    Parameters
    ----------
    data : awkward array (with cuts applied)
        your data array post-processing
    fit_range_mom, fit_range_time : [float, float] [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    plot_cat: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts

    """

    obs_mom = zfit.Space('mom', limits=fit_range_mom)
    obs_time = zfit.Space('time', limits=fit_range_time)
    obs_2D = obs_mom * obs_time

    mompars = []
    mompdfs = {}
    timepars = []
    timepdfs = {}
    norms = {}
    constraints = []
    nlls = []
    
    
    # loop over mom components
    for proc in mom_components:
        mompdf = mom_components[proc]['pdf']
        pardict = (mom_components[proc]['pars'])
        treat_params = mom_components[proc]['treat_params']
        timepdf = mom_components[proc]['timepdf']

        pdfs[proc], norms[proc] = MomTimeModel(obs_mom, obs_time, mompars,timepars, proc, mompdf, timepdf, pardict, treat_params, fit_range_mom, constraints, dio_efficiency, dio_resolution)
        if 'nll' in mom_components[proc].keys():
            nlls.extend(mom_components[proc]['nll'].get_nll(pars))

    
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))
    # Convert data to zfit Data
    data_np_time = ak.to_numpy(ak.flatten(times, axis=None))
    data_np_mom = ak.to_numpy(ak.flatten(mom_mag, axis=None))
    data_zfit = zfit.Data.from_numpy(array=np.column_stack((data_np_mom, data_np_time)), obs=obs_2D)

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ finished minimizing")
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except:
        print('[py-fitter/fit_module] ❌ WARNING! Invalid fit, postfit parameters may not be optimal')
    if result.valid == True:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
    # Plot after fit

    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_fit_time] ✅ plotting")
      
    # plot time fit
    plottime_fit(times, count_particle_types, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in time_components.keys()], True)
    
    # plot mom fit
    plotmom_fit(data_zfit,track_cat, fit_range_mom, [(proc,pdfs[proc],norms[proc]) for proc in mom_components.keys()], plot_cat) 
    plt.show()
    return result, pars[1], loss, combine_pdf
