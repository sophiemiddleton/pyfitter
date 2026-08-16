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

  def GetUL(self, par, loss, nlls, combine_pdf, constraints, fitlow, fithigh, sig_yield, CL=0.90, opt='asym', ntoysnull=1000, ntoysalt=1000):
    """ Compute upper limit of signal result using hepstats calculators.

    Parameters
    ----------
      par : zfit parameter object
        The parameter of interest (POI), e.g., N_CE.
      loss : zfit loss object
        The UnbinnedNLL loss function used for the fit.
      nlls : list
        List of auxiliary or constrained NLLs.
      combine_pdf : zfit PDF object
        The total combined probability density function model.
      constraints : list or None
        Any additional external structural constraints.
      fitlow : float
        Lower boundary of the main physical fit window.
      fithigh : float
        Upper boundary of the main physical fit window.
      sig_yield : float
        Observed signal parameter yield from the nominal fit.
      CL : float, optional
        Confidence level threshold (defaults to 0.90 for a 90% CL limit).
      opt : str, optional
        Inference strategy: 'asym' for Profile Likelihood, 'freq' for full Toys.
      ntoysnull : int, optional
        Number of null hypothesis background pseudo-experiments (for frequentist opt).
      ntoysalt : int, optional
        Number of alternative hypothesis signal pseudo-experiments (for frequentist opt).
    """
    self.logger.log(f"Configuring statistical inference analyzer for parameter: {par.name}", "info")
    
    # 1. Define the Parameter of Interest (POI) scan array grid
    # Create an adaptive bounding limit centered safely around your observed yield value
    if sig_yield is not None and sig_yield > 0:
        scan_max = max(50.0, 2.5 * abs(sig_yield))
        sig_yield_scan = POIarray(par, np.linspace(0, scan_max, 60))
    else:
        sig_yield_scan = POIarray(par, np.linspace(0, 100, 60))
        
    # 2. Build the appropriate hepstats Calculator context
    if opt == 'freq':
        self.logger.log("Option 'freq' detected. Initializing Frequentist Calculator...", "info")
        try:
            # Frequentist toys *require* a sampler to perform thousands of pseudo-data generation tasks
            sampler = combine_pdf.create_sampler()
            calculator = FrequentistCalculator(
                input=loss, 
                minimizer=self.result.minimizer, 
                ntoysnull=ntoysnull, 
                ntoysalt=ntoysalt, 
                sampler=sampler
            )
        except AssertionError as e:
            self.logger.log("Caught known zfit Extended Product PDF sampler bug! Attempting fallback constructor...", "warning")
            # Fallback strategy: allow FrequentistCalculator to build its internal sampling layers independently
            calculator = FrequentistCalculator(
                input=loss, 
                minimizer=self.result.minimizer, 
                ntoysnull=ntoysnull, 
                ntoysalt=ntoysalt
            )
    else:
        self.logger.log("Option 'asym' detected. Bypassing sampler to initialize Profile Likelihood Asymptotic Calculator...", "info")
        # For 'asym', we evaluate the likelihood profile directly from the data distribution.
        # This completely skips create_sampler(), bypassing the internal zfit AssertionError bug!
        calculator = AsymptoticCalculator(
            input=loss, 
            minimizer=self.result.minimizer
        )

    # 3. Construct the Upper Limit engine wrapper
    # Define the alternative hypothesis POI (fixed to 0.0 for a background-only reference)
    from hepstats.hypotests.parameters import POI
    poi_alt = POI(par, 0.0)
    
    try:
        # Standard signature for newer hepstats versions using explicit named arguments
        ul_analyzer = UpperLimit(calculator, poialt=poi_alt, poinull=sig_yield_scan)
    except TypeError:
        # Fallback signature for versions requiring strictly positional arguments
        self.logger.log("Adapting to positional signature layout for UpperLimit constructor...", "info")
        ul_analyzer = UpperLimit(calculator, sig_yield_scan, poi_alt)
    
    # Calculate the limit (alpha = 1 - Confidence Level; e.g. alpha=0.10 for 90% CL)
    alpha_significance = 1.0 - CL
    
    try:
        # Use CLs method to handle low-statistics background boundaries cleanly
        # Inside results_module.py (Line 176):

        # To this:
        ul_analyzer.upperlimit(alpha=alpha_significance, CLs=True, unidim_solver='brentq')
                
        if hasattr(ul_analyzer, 'limits_result') and ul_analyzer.limits_result is not None:
            obs_limit = ul_analyzer.limits_result.get('observed', float('nan'))
            exp_limit = ul_analyzer.limits_result.get('expected', float('nan'))
            self.logger.log(f"--- Limit Results Processed ---", "success")
            self.logger.log(f"  Observed Upper Limit: {obs_limit:.3f} events", "info")
            self.logger.log(f"  Expected Upper Limit: {exp_limit:.3f} events", "info")
        else:
            self.logger.log("Limit calculation completed, but results dictionary structure is empty.", "warning")
            
    except Exception as err:
        self.logger.log(f"Failed to extract limit boundary points from the calculator curve: {err}", "error")
        traceback.print_exc()

    return ul_analyzer
    
  
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
