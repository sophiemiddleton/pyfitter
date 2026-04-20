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
        try:
          self.logger = Logger(print_prefix="[Results] ", verbosity=self.verbose)
        except Exception:
          self.logger = None
        
  def CalculateRmue(self, n_ce, n_dio):#FIXME requires work
    """ we need to understand how to normalize our signal note: use asym option for quick fit"""
    # as an estimate, use true values for POT
    eff_DIO = 0.11
    frac_sampled = 3.6370937564509995e-11
    N_stopped_mu = n_dio/(frac_sampled*0.39)
    N_nodecay = N_stopped_mu*0.61
    Rmue = par/(N_nodecay)
    return Rmue

  def plotlimit(self, ul, alpha=0.05, CLs=True, ax=None):
      """
      plot pvalue scan for different values of a parameter of interest (observed, expected and +/- sigma bands)

      Args:
          ul: UpperLimit instance
          alpha (float, default=0.05): significance level
          CLs (bool, optional): if `True` uses pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
              else as $$p_{clsb} = p_{null}$
          ax (matplotlib axis, optionnal)

      """
      if ax is None:
          ax = plt.gca()

      poivalues = ul.poinull.values
      pvalues = ul.pvalues(CLs=CLs)

      if CLs:
          cls_clr = "r"
          clsb_clr = "b"
      else:
          cls_clr = "b"
          clsb_clr = "r"

      color_1sigma = "mediumseagreen"
      color_2sigma = "gold"

      ax.plot(
          poivalues,
          pvalues["cls"],
          label="Observed CL$_{s}$",
          marker=".",
          color="k",
          markerfacecolor=cls_clr,
          markeredgecolor=cls_clr,
          linewidth=2.0,
          ms=11,
      )

      ax.plot(
          poivalues,
          pvalues["clsb"],
          label="Observed CL$_{s+b}$",
          marker=".",
          color="k",
          markerfacecolor=clsb_clr,
          markeredgecolor=clsb_clr,
          linewidth=2.0,
          ms=11,
          linestyle=":",
      )

      ax.plot(
          poivalues,
          pvalues["clb"],
          label="Observed CL$_{b}$",
          marker=".",
          color="k",
          markerfacecolor="k",
          markeredgecolor="k",
          linewidth=2.0,
          ms=11,
      )

      ax.plot(
          poivalues,
          pvalues["expected"],
          label="Expected CL$_{s}-$Median",
          color="k",
          linestyle="--",
          linewidth=1.5,
          ms=10,
      )

      ax.plot(
          [poivalues[0], poivalues[-1]],
          [alpha, alpha],
          color="r",
          linestyle="-",
          linewidth=1.5,
      )

      ax.fill_between(
          poivalues,
          pvalues["expected"],
          pvalues["expected_p1"],
          facecolor=color_1sigma,
          label="Expected CL$_{s} \\pm 1 \\sigma$",
          alpha=0.8,
      )

      ax.fill_between(
          poivalues,
          pvalues["expected"],
          pvalues["expected_m1"],
          facecolor=color_1sigma,
          alpha=0.8,
      )

      ax.fill_between(
          poivalues,
          pvalues["expected_p1"],
          pvalues["expected_p2"],
          facecolor=color_2sigma,
          label="Expected CL$_{s} \\pm 2 \\sigma$",
          alpha=0.8,
      )

      ax.fill_between(
          poivalues,
          pvalues["expected_m1"],
          pvalues["expected_m2"],
          facecolor=color_2sigma,
          alpha=0.8,
      )

      ax.set_ylim(-0.01, 1.1)
      ax.set_ylabel("p-value")
      ax.set_xlabel("parameter of interest")
      ax.legend(loc="best", fontsize=14)

      return ax


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
      if self.logger:
        self.logger.log('Invalid calculator chosen', 'error')
      else:
        print('[py-fitter/results_module/GetSignificance] ❌ ERROR! Invalid calculator chosen')
      return
      return
    
    #calculate significance
    if self.verbose > 0:
      if self.logger:
        self.logger.log('Calculating significance', 'info')
        self.logger.log('If significance is inf this means numerical precision or too few toys', 'info')
      else:
        print('[py-fitter/results_module/GetSignificance] ✅  calculating significance')
        print('[py-fitter/results_module/GetSignificance] ❌ CHECK: if signficance is inf this means that the numerical precision is not high enough or that the number of toys is not large enough. For example if all toys are rejected, the result is (0.0, inf)')
    discovery = Discovery(calculator=calculator, poinull=sig_yield_poi)
    significance = discovery.result()

    if self.verbose > 0:
      if self.logger:
        self.logger.log(f'Result signal significance: {significance}', 'info')
      else:
        print('[py-fitter/results_module/GetSignificance] ✅  result signal significance', significance)
    
    self.pvalue = significance[0]
    self.sigma = significance[1]
    if self.verbose > 0:
      if self.logger:
        self.logger.log(f'p-value: {self.pvalue}', 'info')
        self.logger.log(f'{self.sigma} sigma', 'info')
      else:
        print("p-value", self.pvalue)
        print(self.sigma,"sigma")
    
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
    if self.verbose > 0 and self.logger:
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
      if self.logger:
        self.logger.log(f'Resampling with N_CE = {sig_yield} as mean', 'info')
      else:
        print(f'[py-fitter/results_module/GetUL] ✅ resampling with N_CE = {sig_yield} as mean')

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
          if self.logger:
            self.logger.log('Invalid limit calculator chosen', 'error')
          else:
            print('[py-fitter/results_module/GetUL] ❌ ERROR! Invalid limit calculator chosen')
          return
    else:
      if opt == 'asym':
        calculator_low_sig = AsymptoticCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer)
      elif opt == 'freq':
        calculator_low_sig = FrequentistCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer, ntoysnull=1000,ntoysalt=1000)
      else:
        if self.logger:
          self.logger.log('Invalid limit calculator chosen', 'error')
        else:
          print('[py-fitter/results_module/GetUL] ❌ ERROR! Invalid limit calculator chosen')
        return
      
    if self.verbose > 0:
      if self.logger:
        self.logger.log('Calculating significance for UL', 'info')
        self.logger.log('If significance is inf this means numerical precision or too few toys', 'info')
      else:
        print('[py-fitter/results_module/GetUL] ✅  calculating significance')
        print('[py-fitter/results_module/GetUL] ❌ CHECK: if signficance is inf this means that the numerical precision is not high enough or that the number of toys is not large enough. For example if all toys are rejected, the result is (0.0, inf)')
    discovery_low_sig = Discovery(calculator=calculator_low_sig, poinull=sig_yield_poi)
    discovery_low_sig.result()
    if self.verbose > 0:
      if self.logger:
        self.logger.log(f'discovery result: {discovery_low_sig.result()}', 'info')
        try:
          self.logger.log(f'best fit params: {calculator_low_sig.bestfit.params}', 'max')
        except Exception:
          pass
      else:
        print("[py-fitter/fit_module/GetUL] ✅ discovery result",discovery_low_sig.result())
        print(f'[py-fitter/results_module/GetUL] ✅ best fit params {calculator_low_sig.bestfit.params}')
    
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
        if self.logger:
            self.logger.log(f'UL scan range: fitted POI={fitted_poi}, scan max={sig_yield_scan.value[-1] if hasattr(sig_yield_scan, "value") else "unknown"}', 'info')
        else:
            print(f'[py-fitter/results_module/GetUL] UL scan range: fitted POI={fitted_poi}')

    ul = UpperLimit(calculator=calculator_low_sig, poinull=sig_yield_scan, poialt=bkg_only)


    try:
      ul.upperlimit(alpha=0.05, CLs=True)
    except Exception as e:
      if self.logger:
        self.logger.log(f'upperlimit() failed: {e}', 'error')
        self.logger.log(traceback.format_exc(), 'max')
      else:
        print(f'[py-fitter/results_module/GetUL] ❌ upperlimit() failed: {e}')
        import traceback as _tb
        print(_tb.format_exc())

    # plotting of the UL scan (may be absent if upperlimit() failed)
    f = plt.figure(figsize=(9, 8))
    try:
      plotlimit(ul, alpha=0.05, CLs=False)
    except Exception:
      pass
    plt.xlabel("Nsig");
    plt.show()
    if self.verbose > 0:
      if self.logger:
        self.logger.log(str(ul), 'info')
        self.logger.log(f'Result upper limit at {CL} % CL {ul}', 'success')
      else:
        print(ul)
        print(f'[py-fitter/results_module/GetUL] ✅  result upper limit at {CL} % CL {ul}')
    #plotlimit(ul, CLs=False)
    
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
      if self.logger:
        self.logger.log(f"Data written to {file_path}", 'success')
      else:
        print(f"[py-fitter/results_module/WriteFittedData] ✅ Data written to {file_path}")
    
  def WriteResult(self):
    """ Write result to csv file for safe keeping """
    file_path = 'output_fitresult.csv'
    with open(file_path, 'w') as csvfile:
      csvfile.write('Param,Value\n')
      for i, par in enumerate(self.result.params):
        
        csvfile.write(f"{par.name},{par.value().numpy()}\n")

    
    if self.verbose > 0:
      if self.logger:
        self.logger.log(f"Result written to {file_path}", "success")
      else:
        print(f"[py-fitter/results_module/WriteFittedData] ✅ Result written to {file_path}")

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
      if self.logger:
        self.logger.log(f"List successfully saved to {filename}", "success")
      else:
        print(f"List successfully saved to {filename}")
    except Exception as e:
      if self.logger:
        self.logger.log(f"Error saving list: {e}", "error")
        self.logger.log(traceback.format_exc(), "max")
      else:
        print(f"Error saving list: {e}")
        print(traceback.format_exc())

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
          if self.logger:
            self.logger.log(f"Mock {i} fit failed: {e}", 'error')
          else:
            print(f"Mock {i} fit failed: {e}")
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
      if self.logger:
        self.logger.log(f"Sensitivity estimate: median={out['median']}, p16/p84={out['p16']}/{out['p84']}", 'info')
      else:
        print(f"Sensitivity estimate: median={out['median']}, p16/p84={out['p16']}/{out['p84']}")

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
      if self.logger:
        self.logger.log(f"List successfully loaded from {filename}", "success")
        self.logger.log(str(loaded_data), "max")
      else:
        print(f"\nList successfully loaded from {filename}:")
        print(loaded_data)
        print(f"Type of loaded data: {type(loaded_data)}")
        print(f"Is loaded_data equal to my_data? {loaded_data == my_data}")
    except Exception as e:
      if self.logger:
        self.logger.log(f"Error loading list: {e}", "error")
        self.logger.log(traceback.format_exc(), "max")
      else:
        print(f"Error loading list: {e}")
        print(traceback.format_exc())
