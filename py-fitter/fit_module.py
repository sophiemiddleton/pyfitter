# Fit the data to a product of PDFs defined in PDF_list

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tensorflow as tf
import zfit
from typing import List, Tuple, Optional, Any
from zfit.result import FitResult
import traceback
from pyutils.pylogger import Logger
import time

# Module-level logger with fallback
try:
  logger = Logger(print_prefix='[fit_module] ', verbosity=2)
except Exception:
  logger = None

from momPDF_module import MomModel, MomTimeModel, poly58
from timePDF_module import TimeModel
from plot_module import plotmom_fit, plottime_fit, plot_variable, bin_by_bin_mom_confusion
from mom_components import mom_components
from time_components import time_components
from uncertainty_loader import load_constraints_json, build_zfit_constraints_from_specs, load_templates_npz

def Unbinned_fit_mom(mom_mag, track_cat, count_particle_types, fit_range_low, fit_range_hi, plot_cat=False, verbose=0, minos=False, plot_NLL= False, constraints_dir=None):
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
      if logger:
        logger.log("Initializing fit", "info")
      else:
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
      if logger:
        logger.log(f"components {list(mom_components.keys())}", "info")
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ components", mom_components)
      
    # --- Loop over Components and Build Model ---
    for proc in mom_components:
        comp_config = mom_components[proc]
        pdf = comp_config['pdf']
        pardict = comp_config['pars']
        treat_params = comp_config['treat_params']
        
        # Determine if advanced fit structure is present (truthy config required)
        use_advanced_model = bool(comp_config.get('advanced_pars'))

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
                      if logger:
                        logger.log(f"Activating simultaneous fit for: {proc}", "info")
                      else:
                        print(f"[py-fitter] 🔗 Activating simultaneous fit for: {proc}")
                    # Only call get_nll if data is actually present to unpack
                    aux_nlls.extend(nll_source.get_nll(pars))
                else:
                    if verbose > 0:
                      if logger:
                        logger.log(f"Using fixed resolution/loss for: {proc}", "info")
                      else:
                        print(f"[py-fitter] 📌 Using fixed resolution/loss for: {proc}")

    # --- build combined PDF ---
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))

    # --- optional: load external constraints/templates (standalone uncertainty artifacts)
    if constraints_dir is not None:
      try:
        specs = load_constraints_json(constraints_dir)
        extra_constraints = build_zfit_constraints_from_specs(pars, specs, logger=logger)
        # append to existing constraints list used by the loss builder
        constraints.extend(extra_constraints)
        # expose loaded templates (not automatically injected; available for later use)
        templates = load_templates_npz(constraints_dir)
        if logger and templates:
          logger.log(f'Loaded templates: {list(templates.keys())}', 'info')
      except Exception as e:
        if logger:
          logger.log(f'Failed to load constraints from {constraints_dir}: {e}', 'error')
        else:
          print(f'Failed to load constraints from {constraints_dir}: {e}')

    # Convert data to zfit Data
    mom_mag_skim = ak.nan_to_none(mom_mag)
    mom_mag_skim = ak.drop_none(mom_mag_skim)
    mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
    mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

    if verbose > 0:
      if logger:
        logger.log("Running minimizer", "info")
      else:
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
      if logger:
        logger.log("Finished minimizing", "info")
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ finished minimizing")
      
    # --- Minos Error Calculation  ---
    if minos == True:
      try:
        param_errors, _ = result.errors(method='minuit_minos')
      except Exception as e:
        if logger:
          logger.log('Invalid fit, postfit parameters may not be optimal', 'error')
          logger.log(traceback.format_exc(), 'max')
        else:
          print('[py-fitter/fit_module/Unbinned_fit_mom] ❌ ERROR! Invalid fit, postfit parameters may not be optimal')

    if result.valid == True:
      if logger:
        logger.log('fit is valid', 'success')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      if logger:
        logger.log('fit is not valid', 'info')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
      
    # Plot after fit
    if verbose > 0:
      if logger:
        logger.log('Plotting results', 'info')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ plotting")
    if logger:
      logger.log(str(pars), 'max')
    else:
      print(pars)
    
    # plotmom_fit (Assuming this is an external function you need to call)
    plotmom_fit(mom_mag,count_particle_types, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in mom_components.keys()], plot_cat)
    ts = int(time.time())
    fname = f"fit_mom_{ts}.png"
    try:
      plt.savefig(fname)
      if logger:
        logger.log(f"Saved fit figure to {fname}", "info")
      else:
        print(f"Saved fit figure to {fname}")
    except Exception as e:
      if logger:
        logger.log(f"Failed to save fit figure: {e}", "error")
      else:
        print(f"Failed to save fit figure: {e}")
    plt.close()
    
    # --- NLL Scan Plotting  ---
    if plot_NLL:
      # performs optional scan to draw NLL plot:
      best_nll = result.fmin
      if logger:
        logger.log(f"Best fit nsig: {result.params[pars[1]]['value']:.2f}", 'info')
        logger.log(f"Minimum NLL: {best_nll:.2f}", 'info')
      else:
        print(f"Best fit nsig: {result.params[pars[1]]['value']:.2f}")
        print(f"Minimum NLL: {best_nll:.2f}")

      scan_range = np.linspace(0,float(pars[1].value())+float(pars[1].value())*0.5, 41)
      nll_values = []

      if logger:
        logger.log('Starting NLL scan...', 'info')
      else:
        print("Starting NLL scan...")
      # Loop over the scan range for the signal yield
      for n in scan_range:
          with pars[1].set_value(n):
              pars[1].floating = False
              
              minimizer.minimize(loss )
              nll_values.append(loss.value())  
              pars[1].floating = True

      if logger:
        logger.log('Scan complete', 'info')
      else:
        print("Scan complete...")

      # Find true number:
      data_signal = ak.mask(mom_mag, count_particle_types == 168)
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
      ts = int(time.time())
      fname_nll = f"fit_mom_nll_{ts}.png"
      try:
        plt.savefig(fname_nll)
        if logger:
            logger.log(f"Saved NLL scan to {fname_nll}", "info")
        else:
            print(f"Saved NLL scan to {fname_nll}")
      except Exception as e:
        if logger:
            logger.log(f"Failed to save NLL scan: {e}", "error")
        else:
            print(f"Failed to save NLL scan: {e}")
      plt.close()

    # produce bin-by-bin momentum confusion plot (true vs fitted fractions)
    try:
      bin_by_bin_mom_confusion(mom_mag, count_particle_types, [(proc, pdfs[proc], norms[proc]) for proc in mom_components.keys()], fit_range, bin_width=0.5, filename_prefix='mom_confusion_1d')
    except Exception as e:
      if logger:
        logger.log(f'Failed to produce bin-by-bin momentum confusion plot: {e}', 'error')
      else:
        print(f'Failed to produce bin-by-bin momentum confusion plot: {e}')
      
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
      if logger:
        logger.log('Initializing time fit', 'info')
      else:
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
      if logger:
        logger.log('Finished minimizing time fit', 'info')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_time] ✅ finished minimizing")
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except Exception as e:
      if logger:
        logger.log('Invalid fit, postfit parameters may not be optimal', 'error')
        logger.log(traceback.format_exc(), 'max')
      else:
        print('[py-fitter/fit_module] ❌ WARNING! Invalid fit, postfit parameters may not be optimal')
    if result.valid == True:
      if logger:
        logger.log('fit is valid', 'success')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ✅ fit is valid")
    else:
      if logger:
        logger.log('fit is not valid', 'info')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_mom] ⚠️ WARNING! fit is not valid")
    # Plot after fit

    if verbose > 0:
      if logger:
        logger.log('Plotting time fit', 'info')
      else:
        print("[py-fitter/fit_module/Unbinned_fit_time] ✅ plotting")
    plottime_fit(times, count_particle_types, fit_range, [(proc,pdfs[proc],norms[proc]) for proc in time_components.keys()], True)
    ts = int(time.time())
    fname_t = f"fit_time_{ts}.png"
    try:
      plt.savefig(fname_t)
      if logger:
        logger.log(f"Saved time-fit figure to {fname_t}", "info")
      else:
        print(f"Saved time-fit figure to {fname_t}")
    except Exception as e:
      if logger:
        logger.log(f"Failed to save time-fit figure: {e}", "error")
      else:
        print(f"Failed to save time-fit figure: {e}")
    plt.close()

    
    return result, pars[1], loss, combine_pdf

def Unbinned_2d_fit_mom_time(mom_mag, times, track_cat, count_particle_types, fit_range_mom, fit_range_time, plot_cat=False, verbose=0):
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
    pdfs = {}
    timepars = []
    timepdfs = {}
    norms = {}
    constraints = []
    nlls = []
    nll_sources = []
    
    
    # loop over mom components
    for proc in mom_components:
      mompdf = mom_components[proc]['pdf']
      pardict = mom_components[proc]['pars']
      treat_params = mom_components[proc]['treat_params']
      # time model name may be defined in time_components; fall back to None
      timepdf = time_components.get(proc, {}).get('pdf', None)

      # MomTimeModel now returns (pdf_2d, N, mom_pdf, time_pdf)
      try:
        pdf2d, N, mom_subpdf, time_subpdf = MomTimeModel(obs_mom, obs_time, mompars, timepars, proc, mompdf, timepdf, pardict, treat_params, fit_range_mom, constraints)
      except TypeError:
        # backward-compatible: older MomTimeModel returned (pdf_2d, N)
        pdf2d, N = MomTimeModel(obs_mom, obs_time, mompars, timepars, proc, mompdf, timepdf, pardict, treat_params, fit_range_mom, constraints)
        mom_subpdf = mompdf
        time_subpdf = time_components.get(proc, {}).get('pdf', zfit.pdf.Uniform(low=fit_range_time[0], high=fit_range_time[1], obs=obs_time))

      pdfs[proc] = pdf2d
      norms[proc] = N
      mompdfs[proc] = mom_subpdf
      if 'nll' in mom_components[proc].keys():
        nll_sources.append(mom_components[proc]['nll'])

      # also build a time-only (non-extended) PDF for plotting
      if proc in ('DIO', 'CE'):
        timepdfs[proc] = zfit.pdf.Exponential(zfit.Parameter(f'decay_shared_CE_DIO_plot', -1.0/864.0, floating=False), obs=obs_time)
      elif proc == 'RPC':
        timepdfs[proc] = zfit.pdf.Exponential(zfit.Parameter(f'decay_rpc_plot', -1.0/26.0, floating=False), obs=obs_time)
      elif proc == 'Cosmic':
        # use requested cosmic window for plotting
        timepdfs[proc] = zfit.pdf.Uniform(low=fit_range_time[0], high=fit_range_time[1], obs=obs_time)
      else:
        # fallback to a uniform over the time fit range
        timepdfs[proc] = zfit.pdf.Uniform(low=fit_range_time[0], high=fit_range_time[1], obs=obs_time)

    
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))
    # Convert data to zfit Data
    data_np_time = ak.to_numpy(ak.flatten(times, axis=None))
    data_np_mom = ak.to_numpy(ak.flatten(mom_mag, axis=None))
    data_zfit = zfit.Data.from_numpy(array=np.column_stack((data_np_mom, data_np_time)), obs=obs_2D)

    # Combine parameter lists (momentum + time)
    pars = mompars + timepars

    # Now collect auxiliary NLL terms from any sources that needed the full param list
    for src in nll_sources:
      try:
        nlls.extend(src.get_nll(pars))
      except Exception:
        pass

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit, constraints=constraints)
    for nll in nlls:
      loss = loss + nll
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    
    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_2d_fit_mom_time] ✅ finished minimizing")
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except:
        print('[py-fitter/fit_module] ❌ WARNING! Invalid fit, postfit parameters may not be optimal')
    if result.valid == True:
      print("[py-fitter/fit_module/Unbinned_2d_fit_mom_time] ✅ fit is valid")
    else:
      print("[py-fitter/fit_module/Unbinned_2d_fit_mom_time] ⚠️ WARNING! fit is not valid")
    # Plot after fit

    if verbose > 0:
      print("[py-fitter/fit_module/Unbinned_2d_fit_mom_time] ✅ plotting")
      
    # plot time fit — pass the time-only PDFs (projection of 2D) and yields
    plottime_fit(times, count_particle_types, fit_range_time, [(proc, timepdfs[proc], norms[proc]) for proc in mom_components.keys()], plot_cat)

    
    ts = int(time.time())
    fname_time_2d = f"fit_2d_time_{ts}.png"
    try:
      plt.savefig(fname_time_2d)
      if logger:
        logger.log(f"Saved 2D-fit figure to {fname_time_2d}", "info")
      else:
        print(f"Saved 2D-fit figure to {fname_time_2d}")
    except Exception as e:
      if logger:
        logger.log(f"Failed to save 2D-fit figure: {e}", "error")
      else:
        print(f"Failed to save 2D-fit figure: {e}")
    plt.close()

    # plot mom fit using the 1D momentum sub-PDFs (projections)
    plotmom_fit(mom_mag, count_particle_types, fit_range_mom, [(proc, mompdfs[proc], norms[proc]) for proc in mom_components.keys()], plot_cat)
    fname_mom_2d = f"fit_2d_mom_{ts}.png"
    try:
      plt.savefig(fname_mom_2d)
      if logger:
        logger.log(f"Saved 2D-fit figure to {fname_mom_2d}", "info")
      else:
        print(f"Saved 2D-fit figure to {fname_mom_2d}")
    except Exception as e:
      if logger:
        logger.log(f"Failed to save 2D-fit figure: {e}", "error")
      else:
        print(f"Failed to save 2D-fit figure: {e}")
    plt.close()


    # produce bin-by-bin momentum confusion plot (true vs fitted fractions)
    try:
      bin_by_bin_mom_confusion(mom_mag, count_particle_types, [(proc, mompdfs[proc], norms[proc]) for proc in mom_components.keys()], fit_range_mom, bin_width=0.5, filename_prefix='mom_confusion_2d')
    except Exception as e:
      if logger:
        logger.log(f'Failed to produce bin-by-bin momentum confusion plot: {e}', 'error')
      else:
        print(f'Failed to produce bin-by-bin momentum confusion plot: {e}')

    return result, pars[1], loss, combine_pdf, norms

def stability_scan(axis, slices, mom_mag, times, track_cat, count_particle_types, fit_range_mom, fit_range_time, plot_cat=False, verbose=0):
    """Run Unbinned_2d_fit_mom_time across a series of slices and plot fitted yields vs slice.

    axis: 'time' or 'mom' -- which axis to slice
    slices: iterable of (low, high) tuples for the chosen axis
    Other args forwarded to the 2D fitter.
    Saves `yield_stability_{axis}_{timestamp}.png`.
    Returns: dict mapping process -> list of fitted yields (per slice)
    """
    import time as _time
    results = {}
    slice_centers = []
    for low, high in slices:
      if axis == 'time':
        fr_time = [low, high]
        fr_mom = fit_range_mom
      else:
        fr_mom = [low, high]
        fr_time = fit_range_time

      try:
        res, par, loss, combine_pdf, norms = Unbinned_2d_fit_mom_time(
          mom_mag, times, track_cat, count_particle_types, fr_mom, fr_time, plot_cat, verbose
        )
      except Exception:
        # on failure, append zeros
        norms = {p: 0.0 for p in mom_components.keys()}

      # record center
      slice_centers.append(0.5 * (low + high))
      for p in norms:
        try:
          val = norms[p].numpy() if hasattr(norms[p], 'numpy') else float(norms[p])
        except Exception:
          try:
            val = float(norms[p].value())
          except Exception:
            val = 0.0
        results.setdefault(p, []).append(val)

    # plot results
    import matplotlib.pyplot as plt
    ts = int(_time.time())
    fig, ax = plt.subplots(figsize=(8, 4))
    for p, vals in results.items():
      errs = [np.sqrt(abs(v)) for v in vals]
      ax.errorbar(slice_centers, vals, yerr=errs, marker='o', label=p)
    ax.set_xlabel(f'{axis} slice center')
    ax.set_ylabel('Fitted yield')
    ax.set_title(f'Yield stability vs {axis}')
    ax.legend()
    ax.grid(True, linestyle=':')
    fname = f'yield_stability_{axis}_{ts}.png'
    try:
      plt.tight_layout()
      plt.savefig(fname)
      if logger:
        logger.log(f'Saved stability plot to {fname}', 'info')
      else:
        print(f'Saved stability plot to {fname}')
    except Exception as e:
      if logger:
        logger.log(f'Failed to save stability plot: {e}', 'error')
      else:
        print(f'Failed to save stability plot: {e}')
    plt.close(fig)
    return results, slice_centers
