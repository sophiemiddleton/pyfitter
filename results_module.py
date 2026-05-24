import awkward as ak
import numpy as np
import csv
import pickle
import zfit
import matplotlib.pyplot as plt
import traceback
from pyutils.pylogger import Logger
from hepstats.hypotests.parameters import POI
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests.calculators import FrequentistCalculator
from hepstats.hypotests import Discovery
from hepstats.hypotests import UpperLimit
from hepstats.hypotests.parameters import POIarray
from hepstats.hypotests import ConfidenceInterval

class ResultsClass:
  """Class to interpret results: provide discovery tests and limits, print to BAT.jl readable file etc.
  """
  def __init__(self, data, result, verbose=0):
        """Initialise the results class
        
        Parameters:
          result : zfit fite result
          data : zfit array
          verbose: verbosity
          rmue : the derived rmue (need to understand how to include efficiencies)
          pvalue : pvalue
          sigma : number of sigma significance result
        """
        self.result = result
        self.data = data # flattened mom mag list with cuts applied
        self.verbose = verbose
        self.rmue = 0
        self.pvalue = 0
        self.sigma = 0
        self.logger = Logger(print_prefix="[Results] ", verbosity=self.verbose)

  def GetSignifcance(self, par, loss, opt='freq'): #FIXME - concept, not fully tested
    """ compute significance of signal result 

    Parameters
    ----------
      par : zfit parameters
      loss : zfit loss function
      opt : option for how to compute (either frequentist (freq) or asymptotic (asym)
    """
    
    # the null hypothesis
    sig_yield_poi = POI(par, 0)
    minimizer = zfit.minimize.Minuit()
    
    if opt == 'freq':
      # construction of the calculator instance
      calculator = FrequentistCalculator(input=loss, minimizer=minimizer)
      calculator.bestfit = self.result
      calculator = FrequentistCalculator(input=self.result, minimizer=minimizer)
    elif opt == 'asym':
      # construction of the calculator instance
      calculator = AsymptoticCalculator(input=loss, minimizer=minimizer) # asimov_bins=100
      calculator.bestfit = self.result
      # equivalent to above
      calculator = AsymptoticCalculator(input=self.result, minimizer=minimizer) # asimov_bins=100

    else:
      self.logger.log('Invalid calculator chosen', 'error')
      return
    
    #calculate significance
    if self.verbose > 0:
      self.logger.log('Calculating significance', 'info')
      self.logger.log('If significance is inf this means numerical precision or too few toys', 'info')
    discovery = Discovery(calculator=calculator, poinull=sig_yield_poi)
    significance = discovery.result()

    if self.verbose > 0:
      self.logger.log(f'Result signal significance: {significance}', 'info')
    
    self.pvalue = significance[0]
    self.sigma = significance[1]
    if self.verbose > 0:
      self.logger.log(f'p-value: {self.pvalue}', 'info')
      self.logger.log(f'{self.sigma} sigma', 'info')
    
    return significance

  def GetUL(self, par, loss, nlls, combine_pdf, constraints, fitlow, fithigh, sig_yield=0, CL= 0.90, opt='freq'): #FIXME - concept, not fully tested
    """ compute an upper limit in case where no significant signal yield note: use asym option for quick fit 

    Parameters
    ----------
      par : zfit parameters
      loss : zfit loss function
      combine_pdf: zfit combined pdf
      fitlow, fithigh : fit range
      sig_yield : observed CEs from fit
      CL : confidence level for limit default is 90%
      opt : option for how to compute (either frequentist (freq) or asymptotic (asym)
    """
    sig_yield_poi = POI(par, 0)
    minimizer = zfit.minimize.Minuit()
    # Sets the values of the parameters to the self.result of the simultaneous fit
    zfit.param.set_values(loss.get_params(), self.result)

    # Creates a sampler that will draw events from the model
    if self.verbose > 0:
      self.logger.log('Creating sampler', 'info')
    sampler = combine_pdf.create_sampler()

    # Creates new loss
    data_np = ak.to_numpy(ak.flatten(self.data, axis=None))
    fit_range = (fitlow, fithigh)
    obs_mom = zfit.Space('x', limits=fit_range)
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_mom)
    nll_simultaneous_low_sig = zfit.loss.ExtendedUnbinnedNLL(model=combine_pdf, data=data_zfit, constraints=constraints)
    for nll in nlls:
        nll_simultaneous_low_sig = loss+nll

    # Samples with sig_yield. Since the model is extended the number of signal generated is drawn from a poisson distribution with lambda = sig_yield.
    if self.verbose > 0:
      self.logger.log(f'Resampling with N_CE = {sig_yield} as mean', 'info')

    # Try the native sampler resample API, but fall back to temporarily setting
    # model parameter values with clipping if resample fails (zfit may try to
    # assign out-of-bounds values to constrained parameters during resample).
    temp_param_context = None
    try:
      sampler.resample({par: sig_yield})
    except Exception:
      # build a full parameter->value mapping based on the current fit result
      try:
        params_all = tuple(loss.get_params())
      except Exception:
        params_all = tuple()

      vals = []
      for p in params_all:
        try:
          vals.append(float(self.result.params[p.name]['value']))
        except Exception:
          try:
            vals.append(float(p.value()))
          except Exception:
            vals.append(0.0)

      # overwrite the POI value with the requested injected signal
      for i, p in enumerate(params_all):
        try:
          if p.name == getattr(par, 'name', None):
            vals[i] = float(sig_yield)
            break
        except Exception:
          continue

      # Use zfit.param.set_values with clip=True to avoid ValueError
      try:
        temp_param_context = zfit.param.set_values(params_all, tuple(vals), clip=True)
      except Exception:
        temp_param_context = None

    # Create the calculator inside the temporary parameter-setting context
    # if we had to fall back to clipped parameter assignment.
    if temp_param_context is not None:
      ctx = temp_param_context
      with ctx:
        if opt == 'asym':
          calculator_low_sig = AsymptoticCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer)
        elif opt == 'freq':
          calculator_low_sig = FrequentistCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer, ntoysnull=1000,ntoysalt=1000)
        else:
          self.logger.log('Invalid limit calculator chosen', 'error')
          return
    else:
      if opt == 'asym':
        calculator_low_sig = AsymptoticCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer)
      elif opt == 'freq':
        calculator_low_sig = FrequentistCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer, ntoysnull=1000,ntoysalt=1000)
      else:
        self.logger.log('Invalid limit calculator chosen', 'error')
        return
      
    if self.verbose > 0:
      self.logger.log('Calculating significance for UL', 'info')
      self.logger.log('If significance is inf this means numerical precision or too few toys', 'info')
    discovery_low_sig = Discovery(calculator=calculator_low_sig, poinull=sig_yield_poi)
    discovery_low_sig.result()
    if self.verbose > 0:
      self.logger.log(f'discovery result: {discovery_low_sig.result()}', 'info')
      try:
        self.logger.log(f'best fit params: {calculator_low_sig.bestfit.params}', 'max')
      except Exception:
        pass
    
    #Background only hypothesis.
    bkg_only = POI(par, 0)
    # Range of Nsig values to scan - make adaptive based on fitted POI value
    # Use 2-3x the fitted POI value to ensure UL is sufficiently constraining
    fitted_poi = None
    try:
        fitted_poi = float(self.result.params[par.name]['value'])
    except Exception:
        fitted_poi = None
    
    if fitted_poi is not None and fitted_poi > 0:
        # Adaptive range: scan from 0 to ~2x the fitted POI value
        scan_max = max(50.0, 2.5 * abs(fitted_poi))
        sig_yield_scan = POIarray(par, np.linspace(0, scan_max, 60))
    else:
        # Fallback: conservative range
        sig_yield_scan = POIarray(par, np.linspace(0, 100, 60))
    
    if self.verbose > 0:
      self.logger.log(f'UL scan range: fitted POI={fitted_poi}, scan max={sig_yield_scan.value[-1] if hasattr(sig_yield_scan, "value") else "unknown"}', 'info')

    ul = UpperLimit(calculator=calculator_low_sig, poinull=sig_yield_scan, poialt=bkg_only, qtilde=True)
    ul.limits_result = None  # will be populated below if successful

    try:
      ul_limits = ul.upperlimit(alpha=0.05, CLs=True)
      ul.limits_result = ul_limits  # attach for caller to retrieve
    except Exception as e:
      self.logger.log(f'upperlimit() failed: {e}', 'error')
      self.logger.log(traceback.format_exc(), 'max')

    # plotting of the UL scan
    # NOTE: plt.show() intentionally omitted — it blocks in subprocess workers.
    if self.verbose > 0:
      self.logger.log(str(ul), 'info')
      self.logger.log(f'Result upper limit at {CL} % CL {ul}', 'success')
    
    return ul
    
  def WriteFittedData(self, min_v, max_v):
    """ Write data used in fit to csv (i,mom,time) Note: should be in format useful to BAT"""
    flat_mom = ak.flatten(self.data, axis = None)
    flat_np = np.array(flat_mom)

    # Create a boolean mask where elements are greater than or equal to 85
    mask = (flat_np >= min_v) & (flat_np < max_v)

    # Use the mask to filter the array and keep only the elements where the mask is True
    filtered_array = flat_np[mask]
    file_path = 'output_data.csv'

    with open(file_path , 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in filtered_array:
            csv_writer.writerow([item])

    if self.verbose > 0:
      self.logger.log(f"Data written to {file_path}", 'success')
    
  def WriteResult(self):
    """ Write result to csv file for safe keeping """
    file_path = 'output_fitresult.csv'
    with open(file_path, 'w') as csvfile:
      csvfile.write('Param,Value\n')
      for i, par in enumerate(self.result.params):
        
        csvfile.write(f"{par.name},{par.value().numpy()}\n")

    
    if self.verbose > 0:
      self.logger.log(f"Result written to {file_path}", "success")

  def WritePkl(self):
    """Outputs zfit result to a pkl file
    """
    # Specify the filename for your pickle file
    filename = "output_fitresult.pkl"
    my_data = [{'Param': [], 'Value' :[]}]
    for i, par in enumerate(self.result.params):
      my_data[0]['Param'].append(par.name)
      my_data[0]['Value'].append(par.value().numpy())
    # Save the list to the .pkl file
    try:
      with open(filename, 'wb') as file:
        pickle.dump(my_data, file)
      self.logger.log(f"List successfully saved to {filename}", "success")
    except Exception as e:
      self.logger.log(f"Error saving list: {e}", "error")
      self.logger.log(traceback.format_exc(), "max")

  def SensitivityFromMocks(self, mock_samples, fit_runner, result_key='ul', alpha=0.05, CL=0.90, verbose=0):
    """Estimate expected sensitivity from an ensemble of mock datasets.

    This helper runs a user-supplied `fit_runner` on each mock dataset and
    collects a numeric summary (by default an upper limit) returned by the
    runner. It reports the median expected value and +/-1 and +/-2 sigma bands.

    Parameters
    ----------
    mock_samples : iterable
      Iterable of mock data arrays (e.g. 1D numpy arrays of momenta) to be
      passed to `fit_runner`.
    fit_runner : callable
      Function with signature `res = fit_runner(data)` where `data` is one
      mock sample. `res` may be:
        - a numeric value (interpreted as the desired metric), or
        - a dict-like object containing `result_key` with a numeric value, or
        - an object from which a float can be coerced.
    result_key : str
      If `fit_runner` returns a dict, use this key to extract the numeric
      metric (default: 'ul' for upper limit).
    alpha, CL : float
      Unused by the routine itself but available for the runner if needed.
    verbose : int
      Verbosity level.

    Returns
    -------
    dict containing:
      - 'median': median of collected metrics
      - 'p16','p84': 1 sigma lower/upper (16th/84th percentiles)
      - 'p025','p975': 2 sigma lower/upper (2.5th/97.5th percentiles)
      - 'values': raw list of values

    Notes
    -----
    This method intentionally delegates the fitting work to `fit_runner` so
    it remains decoupled from specific fitting workflows and can be used with
    both 1D and 2D fit runners. The runner should be responsible for any
    model construction, constraints loading, and returning a numeric metric
    for each mock dataset.
    """
    vals = []
    for i, samp in enumerate(mock_samples):
      try:
        res = fit_runner(samp)
        if isinstance(res, dict):
          if result_key in res:
            v = float(res[result_key])
          else:
            # try to coerce a single-entry dict
            try:
              v = float(list(res.values())[0])
            except Exception:
              raise ValueError(f"fit_runner returned dict without key {result_key}")
        elif isinstance(res, (int, float, np.floating, np.integer)):
          v = float(res)
        else:
          # try coercion
          v = float(res)
      except Exception as e:
        if verbose:
          self.logger.log(f"Mock {i} fit failed: {e}", 'error')
        continue
      vals.append(v)

    if len(vals) == 0:
      raise RuntimeError('No successful mock fits; cannot estimate sensitivity')

    arr = np.array(vals, dtype=float)
    out = {
      'median': float(np.median(arr)),
      'p16': float(np.percentile(arr, 16)),
      'p84': float(np.percentile(arr, 84)),
      'p025': float(np.percentile(arr, 2.5)),
      'p975': float(np.percentile(arr, 97.5)),
      'values': vals,
    }

    if verbose:
      self.logger.log(f"Sensitivity estimate: median={out['median']}, p16/p84={out['p16']}/{out['p84']}", 'info')

    return out

  def ReadPkl(self, filename):
    """test to read in a zfit result (e.g. to compare to a previous result)
    """
    # To confirm it worked, you can load the data back:
    loaded_data = None
    my_data = self.result
    try:
      with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
      self.logger.log(f"List successfully loaded from {filename}", "success")
      self.logger.log(str(loaded_data), "max")
    except Exception as e:
      self.logger.log(f"Error loading list: {e}", "error")
      self.logger.log(traceback.format_exc(), "max")
