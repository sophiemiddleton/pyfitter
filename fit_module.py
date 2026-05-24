# Fit the data to a product of PDFs defined in PDF_list

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tensorflow as tf
import zfit
from zfit.minimizers.strategy import FailMinimizeNaN
from typing import List, Tuple, Optional, Any
from zfit.result import FitResult
import traceback
from pyutils.pylogger import Logger
from config import GLOBAL_VERBOSITY
import time

# Module-level logger
logger = Logger(print_prefix='[fit_module] ', verbosity=GLOBAL_VERBOSITY)

from momentum_pdf_builder import poly58, MomPDFBuilder, TimePDFBuilder, MomTimePDFBuilder
from data_prep import DataPreparationManager
from plot_module import plotmom_fit, plottime_fit, plot_variable, bin_by_bin_mom_confusion, bin_by_bin_time_confusion
from physics_components import mom_components, time_components
from uncertainty_loader import load_constraints_json, build_zfit_constraints_from_specs, load_templates_npz

def Unbinned_fit_mom(mom_mag, count_particle_types, fit_range_low, fit_range_hi, plot_truth=False, verbose=0, minos=False, plot_NLL=False, plot_results=True, constraints_dir=None):
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
    plot_truth: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts
    minos : bool
        set true to evaluate minos errors
    """

    if verbose > 0:
      logger.log("Initializing fit", "info")

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
      logger.log(f"components {list(mom_components.keys())}", "info")
      
    # --- Initialize PDF Builder ---
    mom_builder = MomPDFBuilder()
    
    # --- Loop over Components and Build Model ---
    for proc in mom_components:
        comp_config = mom_components[proc]
        
        # Build PDF using the builder with cleaner keyword arguments
        pdfs[proc], norms[proc] = mom_builder.build(
            obs=obs_mom,
            params_tot=pars,
            process=proc,
            model=comp_config['pdf'],
            pardict=comp_config['pars'],
            treat_params=comp_config['treat_params'],
            fit_range=fit_range,
            constraints=constraints,
            advanced_config=comp_config,
            use_advanced=bool(comp_config.get('advanced_pars'))
        )
        
        if bool(comp_config.get('advanced_pars')) and 'nll_sources' in comp_config['advanced_pars']:
            sources = comp_config['advanced_pars']['nll_sources']
            if not isinstance(sources, list): 
                sources = [sources]
            
            for nll_source in sources:
                # Use getattr to safely check for 'simul_source' without crashing
                sim_data = getattr(nll_source, 'simul_source', None)
                
                if sim_data is not None:
                    if verbose > 0:
                      logger.log(f"Activating simultaneous fit for: {proc}", "info")
                    # Only call get_nll if data is actually present to unpack
                    aux_nlls.extend(nll_source.get_nll(pars))
                else:
                    if verbose > 0:
                      logger.log(f"Using fixed resolution/loss for: {proc}", "info")

    # --- build combined PDF ---
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))


    # --- optional: load external constraints/templates (standalone uncertainty artifacts)
    if constraints_dir is not None:
      logger.log(f"Using constraints/templates from: {constraints_dir}", "info")
      try:
        specs = load_constraints_json(constraints_dir)
        extra_constraints = build_zfit_constraints_from_specs(pars, specs, logger=logger)
        # append to existing constraints list used by the loss builder
        constraints.extend(extra_constraints)
        # expose loaded templates (not automatically injected; available for later use)
        #templates = load_templates_npz(constraints_dir)
        #if templates:
        #  logger.log(f'Loaded templates: {list(templates.keys())}', 'info')
      except Exception as e:
        logger.log(f'Failed to load constraints from {constraints_dir}: {e}', 'error')
    
    # Convert data to zfit Data
    mom_zfit = DataPreparationManager.to_zfit_data(mom_mag, obs_mom, name='momentum')


    if verbose > 0:
      logger.log("Running minimizer", "info")


    # --- Loss function creation (MODIFIED) ---
    # Build the main loss from the Extended NLL and initial constraints
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=mom_zfit, constraints=constraints)
    
    # Add the auxiliary NLL terms 
    for nll in aux_nlls:
        loss = loss + nll # zfit overloads the '+' operator for loss addition

    minimizer = zfit.minimize.Minuit()
    # --- Attempt minimize with graceful diagnostic on FailMinimizeNaN ---
    try:
      result = minimizer.minimize(loss, params=pars)
    except FailMinimizeNaN as e:
      logger.log('Minimizer raised FailMinimizeNaN — dumping diagnostics', 'error')
      lv = loss.value()
      logger.log(f'loss.value() = {lv}', 'error')
      logger.log('Params:', 'error')
      for p in pars:
        name = getattr(p, 'name', str(p))
        val = float(p.value())
        fl = getattr(p, 'floating', None)
        logger.log(f'  {name} value={val} floating={fl} finite={np.isfinite(val)}', 'error')

      logger.log(f'Constraints (count): {len(constraints)}', 'error')
      for c in constraints:
        logger.log(f'   - {type(c)} {getattr(c, "param", getattr(c, "params", repr(c)))}', 'error')

      # Evaluate combined pdf and per-component pdfs on the data
      comb_vals = zfit.run(combine_pdf.pdf(mom_zfit))
      carr = np.asarray(comb_vals)
      logger.log(f'combine_pdf stats: min={np.nanmin(carr):.3e} max={np.nanmax(carr):.3e} n_nan={np.isnan(carr).sum()} n_zero={(carr==0).sum()}', 'error')
      for proc, pdf in pdfs.items():
        vals = zfit.run(pdf.pdf(mom_zfit))
        arr = np.asarray(vals)
        logger.log(f'PDF {proc} stats: min={np.nanmin(arr):.3e} max={np.nanmax(arr):.3e} n_nan={np.isnan(arr).sum()} n_zero={(arr==0).sum()}', 'error')
      # re-raise for upstream handling
      raise
    
    if verbose > 0:
      logger.log("Finished minimizing", "info")

      
    # --- Minos Error Calculation  ---
    if minos == True:
      try:
        param_errors, _ = result.errors(method='minuit_minos')
      except Exception as e:
        logger.log('Invalid fit, postfit parameters may not be optimal', 'error')
        logger.log(traceback.format_exc(), 'max')

    if result.valid == True:
      logger.log('fit is valid', 'success')
    else:
      logger.log('fit is not valid', 'warning')
      
    # Plot after fit (optional)
    if plot_results:
      if verbose > 0:
        logger.log('Plotting results', 'info')
      logger.log(str(pars), 'max')


      try:
        plotmom_fit(mom_mag, count_particle_types, fit_range, [(proc, pdfs[proc], norms[proc]) for proc in mom_components.keys()], plot_truth)
        ts = int(time.time())
        fname = f"fit_mom_{ts}.png"
        try:
          plt.savefig(fname)
          logger.log(f"Saved fit figure to {fname}", "info")
        except Exception as e:
          logger.log(f"Failed to save fit figure: {e}", "error")
        plt.close()
      except Exception as e:
        logger.log(f'plotmom_fit failed: {e}', 'error')
    
    # --- NLL Scan Plotting  ---
    if plot_NLL and plot_results:
      # performs optional scan to draw NLL plot:
      from plot_module import plot_nll_scan
      plot_nll_scan(pars, loss, minimizer, mom_mag, count_particle_types, result, fit_range, verbose=verbose)

    # produce bin-by-bin momentum confusion plot (true vs fitted fractions)
    if plot_results:
      try:
        bin_by_bin_mom_confusion(mom_mag, count_particle_types, [(proc, pdfs[proc], norms[proc]) for proc in mom_components.keys()], fit_range, bin_width=0.5, filename_prefix='mom_confusion_1d')
      except Exception as e:
        logger.log(f'Failed to produce bin-by-bin momentum confusion plot: {e}', 'error')
      
    # Signal yield is always N_CE (first parameter)
    poi = pars[0]
    logger.log(f'Selected POI for return: {getattr(poi, "name", repr(poi))}', 'info')
    return result, poi, loss, aux_nlls, combine_pdf, constraints

def Unbinned_fit_time(times, count_particle_types, fit_range_low, fit_range_hi, plot_truth=False, verbose=0, plot_NLL=False, plot_results=True):
    """
    Configures and calls the unbinned maximum likelihood fit for time using zfit

    Parameters
    ----------
    data : awkward array (with cuts applied)
        your data array post-processing
    fit_range_low, fit_range_hi : float, float
        min and max of fit range (args in the main function)
    plot_truth: bool
        show the MC truth processes on the histogram
    verbose : 1
        print progress statements and debug printouts

    """
    if verbose > 0:
      logger.log('Initializing time fit', 'info')
    fit_range = (fit_range_low, fit_range_hi)
    obs_time = zfit.Space('time', limits=fit_range)
    
    # Initialize time PDF builder
    time_builder = TimePDFBuilder()
    
    # Build PDF components
    pars = []
    pdfs = {}
    norms = {}
    for proc in time_components:
        comp_config = time_components[proc]
        pdfs[proc], norms[proc] = time_builder.build(
            obs=obs_time,
            params_tot=pars,
            process=proc,
            model=comp_config['pdf'],
            pardict=comp_config.get('pars'),
            fit_range=fit_range
        )

    # build combined PDF
    combine_pdf = zfit.pdf.SumPDF(list(pdfs.values()))

    # Convert data to zfit Data
    time_zfit = DataPreparationManager.to_zfit_data(times, obs_time, name='time')
    
    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=time_zfit)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    if verbose > 0:
      logger.log('Finished minimizing time fit', 'info')
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except Exception as e:
      logger.log('Invalid fit, postfit parameters may not be optimal', 'error')
      logger.log(traceback.format_exc(), 'max')
    if result.valid == True:
      logger.log('fit is valid', 'success')
    else:
      logger.log('fit is not valid', 'warning')
    
    # --- NLL Scan Plotting  ---
    if plot_NLL and plot_results:
      # performs optional scan to draw NLL plot:
      from plot_module import plot_nll_scan
      plot_nll_scan(pars, loss, minimizer, times, count_particle_types, result, fit_range, verbose=verbose)
    
    # Plot after fit (optional)
    if plot_results:
      if verbose > 0:
        logger.log('Plotting time fit', 'info')
      try:
        plottime_fit(times, count_particle_types, fit_range, [(proc, pdfs[proc], norms[proc]) for proc in time_components.keys()], plot_truth)
        ts = int(time.time())
        fname_t = f"fit_time_{ts}.png"
        try:
          plt.savefig(fname_t)
          logger.log(f"Saved time-fit figure to {fname_t}", "info")
        except Exception as e:
          logger.log(f"Failed to save time-fit figure: {e}", "error")
      except Exception as e:
        logger.log(f'plottime_fit failed: {e}', 'error')

      # produce bin-by-bin time confusion plot (true vs fitted fractions)
      try:
        bin_by_bin_time_confusion(times, count_particle_types, [(proc, timepdfs[proc], norms[proc]) for proc in time_components.keys()], fit_range, bin_width=50.0, filename_prefix='time_confusion_1d')
      except Exception as e:
        logger.log(f'Failed to produce bin-by-bin time confusion plot: {e}', 'error')

      plt.close()

    # Signal yield is always N_CE (first parameter)
    poi = pars[0]
    logger.log(f'Selected POI for return (time fit): {getattr(poi, "name", repr(poi))}', 'info')
    return result, poi, loss, combine_pdf

def Unbinned_2d_fit_mom_time(mom_mag, times, count_particle_types, fit_range_mom, fit_range_time, plot_truth=False, verbose=0, plot_NLL=False, plot_results=True, constraints_dir=None):
    """
    Configures and calls the unbinned maximum likelihood fit for momentum and time using zfit

    Parameters
    ----------
    data : awkward array (with cuts applied)
        your data array post-processing
    fit_range_mom, fit_range_time : [float, float] [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    plot_truth: bool
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
    
    # Initialize 2D PDF builder
    momtime_builder = MomTimePDFBuilder()
    
    # Loop over mom components
    for proc in mom_components:
      comp_config = mom_components[proc]
      time_model = time_components.get(proc, {}).get('pdf', 'uniform')

      # Build 2D PDF using the builder (cleaner, no try/except needed!)
      pdf2d, N, mom_subpdf, time_subpdf = momtime_builder.build(
          obs_mom=obs_mom,
          obs_time=obs_time,
          mom_params_tot=mompars,
          time_params_tot=timepars,
          process=proc,
          mom_model=comp_config['pdf'],
          time_model=time_model,
          pardict=comp_config['pars'],
          treat_params=comp_config['treat_params'],
          fit_range=fit_range_mom,
          constraints=constraints,
          advanced_config=comp_config,
          use_advanced=bool(comp_config.get('advanced_pars'))
      )

      pdfs[proc] = pdf2d
      norms[proc] = N
      mompdfs[proc] = mom_subpdf
      
      # Collect NLL sources from either top-level 'nll' or advanced_pars['nll_sources']
      if 'nll' in mom_components[proc].keys():
        nll_sources.append(mom_components[proc]['nll'])
      adv = comp_config.get('advanced_pars')
      if adv and 'nll_sources' in adv:
        sources = adv['nll_sources']
        if not isinstance(sources, list):
          sources = [sources]
        for s in sources:
          nll_sources.append(s)

      # Also build a time-only (non-extended) PDF for plotting
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

    # --- optional: load external constraints/templates (standalone uncertainty artifacts)
    if constraints_dir is not None:
      try:
        specs = load_constraints_json(constraints_dir)
        extra_constraints = build_zfit_constraints_from_specs(pars, specs, logger=logger)
        # append to existing constraints list used by the loss builder
        constraints.extend(extra_constraints)
        # expose loaded templates (not automatically injected; available for later use)
        templates = load_templates_npz(constraints_dir)
        if templates:
          logger.log(f'Loaded templates: {list(templates.keys())}', 'info')
      except Exception as e:
        logger.log(f'Failed to load constraints from {constraints_dir}: {e}', 'error')

    # Now collect auxiliary NLL terms from any sources that needed the full param list
    for src in nll_sources:
      try:
        nlls.extend(src.get_nll(pars))
      except Exception as e:
        logger.log(f'Failed to get NLL from source: {e}', 'warning')

    # Loss function and minimizer
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit, constraints=constraints)
    for nll in nlls:
      loss = loss + nll
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss, params=pars)
    
    if verbose > 0:
      logger.log("Finished minimizing 2D fit", 'info')
    try:
        param_errors, _ = result.errors(method='minuit_minos')
    except Exception as e:
        logger.log('Invalid fit, postfit parameters may not be optimal', 'error')
    if result.valid == True:
      logger.log("2D fit is valid", 'success')
    else:
      logger.log("2D fit is not valid", 'info')
    
    # --- NLL Scan Plotting  ---
    if plot_NLL and plot_results:
      # performs optional scan to draw NLL plot:
      from plot_module import plot_nll_scan
      plot_nll_scan(pars, loss, minimizer, mom_mag, count_particle_types, result, fit_range_mom, verbose=verbose)
    
    # Plot after fit (optional)
    if plot_results:
      if verbose > 0:
        logger.log("Plotting 2D fit results", 'info')

      # plot time fit — pass the time-only PDFs (projection of 2D) and yields
      try:
        plottime_fit(times, count_particle_types, fit_range_time, [(proc, timepdfs[proc], norms[proc]) for proc in mom_components.keys()], plot_truth)
        ts = int(time.time())
        fname_time_2d = f"fit_2d_time_{ts}.png"
        try:
          plt.savefig(fname_time_2d)
          logger.log(f"Saved 2D-fit figure to {fname_time_2d}", "info")
        except Exception as e:
          logger.log(f"Failed to save 2D-fit figure: {e}", "error")
        plt.close()
      except Exception as e:
        logger.log(f'plottime_fit failed: {e}', 'error')

      # plot mom fit using the 1D momentum sub-PDFs (projections)
      try:
        plotmom_fit(mom_mag, count_particle_types, fit_range_mom, [(proc, mompdfs[proc], norms[proc]) for proc in mom_components.keys()], plot_truth)
        fname_mom_2d = f"fit_2d_mom_{ts}.png"
        try:
          plt.savefig(fname_mom_2d)
          logger.log(f"Saved 2D-fit figure to {fname_mom_2d}", "info")
        except Exception as e:
          logger.log(f"Failed to save 2D-fit figure: {e}", "error")
        plt.close()
      except Exception as e:
        logger.log(f'plotmom_fit failed: {e}', 'error')

      # produce bin-by-bin momentum confusion plot (true vs fitted fractions)
      try:
        bin_by_bin_mom_confusion(mom_mag, count_particle_types, [(proc, mompdfs[proc], norms[proc]) for proc in mom_components.keys()], fit_range_mom, bin_width=0.5, filename_prefix='mom_confusion_2d')
      except Exception as e:
        logger.log(f'Failed to produce bin-by-bin momentum confusion plot: {e}', 'error')

      # produce bin-by-bin time confusion plot (true vs fitted fractions)
      try:
        bin_by_bin_time_confusion(times, count_particle_types, [(proc, timepdfs[proc], norms[proc]) for proc in mom_components.keys()], fit_range_time, bin_width=50.0, filename_prefix='time_confusion_2d')
      except Exception as e:
        logger.log(f'Failed to produce bin-by-bin time confusion plot: {e}', 'error')
    
    # Signal yield is always N_CE (first parameter in mompars)
    poi = pars[0] if pars else None
    logger.log(f'Selected POI for return (2D fit): {getattr(poi, "name", repr(poi))}', 'info')
    return result, poi, loss, combine_pdf, norms
