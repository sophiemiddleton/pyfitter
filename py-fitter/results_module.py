import awkward as ak
import numpy as np
import csv
import pickle
import zfit

from hepstats.hypotests.parameters import POI
from hepstats.hypotests.calculators import AsymptoticCalculator
from hepstats.hypotests.calculators import FrequentistCalculator
from hepstats.hypotests import Discovery
from hepstats.hypotests import UpperLimit
from hepstats.hypotests.parameters import POIarray
#from utils import *
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
        
  def CalculateRmue(self):#FIXME requires work
    """ we need to understand how to normalize our signal note: use asym option for quick fit"""
    # as an estimate, use true values for POT
    return 1e-13
    
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
      print('[py-fitter/results_module/GetSignificance] ❌ ERROR! Invalid calculator chosen')
      return
    
    #calculate significance
    if self.verbose > 0:
      print('[py-fitter/results_module/GetSignificance] ✅  calculating significance')
      print('[py-fitter/results_module/GetSignificance] ❌ CHECK: if signficance is inf this means that the numerical precision is not high enough or that the number of toys is not large enough. For example if all toys are rejected, the result is (0.0, inf)')
    discovery = Discovery(calculator=calculator, poinull=sig_yield_poi)
    significance = discovery.result()

    if self.verbose > 0:
      print('[py-fitter/results_module/GetSignificance] ✅  result signal significance', significance)
      
    self.pvalue = significance[0]
    self.sigma = significance[1]
    if self.verbose > 0:
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
    if self.verbose > 0:
      print("[py-fitter/fit_module/GetUL] ✅ creating sampler")
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
      print(f'[py-fitter/results_module/GetUL] ✅ resampling with N_CE = {sig_yield} as mean')
    sampler.resample({par: sig_yield})

    if opt == 'asym':
      calculator_low_sig = AsymptoticCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer)
    elif opt == 'freq':
      calculator_low_sig = FrequentistCalculator(input=nll_simultaneous_low_sig, minimizer=minimizer, ntoysnull=1000,ntoysalt=1000)
      # see https://github.com/scikit-hep/hepstats/blob/main/src/hepstats/hypotests/calculators/frequentist_calculator.py for details
    else:
      print('[py-fitter/results_module/GetUL] ❌ ERROR! Invalid limit calculator chosen')
      return
      
    if self.verbose > 0:
      print('[py-fitter/results_module/GetUL] ✅  calculating significance')
      print('[py-fitter/results_module/GetUL] ❌ CHECK: if signficance is inf this means that the numerical precision is not high enough or that the number of toys is not large enough. For example if all toys are rejected, the result is (0.0, inf)')
    discovery_low_sig = Discovery(calculator=calculator_low_sig, poinull=sig_yield_poi)
    discovery_low_sig.result()
    if self.verbose > 0:
      print("[py-fitter/fit_module/GetUL] ✅ discovery result",discovery_low_sig.result())
      print(f'[py-fitter/results_module/GetUL] ✅ best fit params {calculator_low_sig.bestfit.params}')
    
    #Background only hypothesis.
    bkg_only = POI(par, 0)
    # Range of Nsig values to scan.
    sig_yield_scan = POIarray(par, np.linspace(0,570,550))#FIXME - hardcoded

    ul = UpperLimit(calculator=calculator_low_sig, poinull=sig_yield_scan, poialt=bkg_only)
    ul.upperlimit(alpha=1-CL);
    if self.verbose > 0:
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
      print(f"[py-fitter/results_module/WriteFittedData] ✅ Data written to {file_path}")
    
  def WriteResult(self):
    """ Write result to csv file for safe keeping """
    file_path = 'output_fitresult.csv'
    with open(file_path, 'w') as csvfile:
      csvfile.write('Param,Value\n')
      for i, par in enumerate(self.result.params):
        
        csvfile.write(f"{par.name},{par.value().numpy()}\n")

    
    if self.verbose > 0:
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
    print(my_data)
    # Save the list to the .pkl file
    try:
        with open(filename, 'wb') as file:
            pickle.dump(my_data, file)
        print(f"List successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving list: {e}")

  def ReadPkl(self, filename):
    """test to read in a zfit result (e.g. to compare to a previous result)
    """
    # To confirm it worked, you can load the data back:
    loaded_data = None
    my_data = self.result
    try:
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
        print(f"\nList successfully loaded from {filename}:")
        print(loaded_data)
        print(f"Type of loaded data: {type(loaded_data)}")
        print(f"Is loaded_data equal to my_data? {loaded_data == my_data}")
    except Exception as e:
        print(f"Error loading list: {e}")
