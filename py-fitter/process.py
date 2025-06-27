import hist
import gc
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse

from fit_module import *
from results_module import ResultsClass
from analyze import Analyze
from mom_components import mom_components
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyplot import Plot
from pyutils.pyprint import Print
from pyutils.pyselect import Select
from pyutils.pyvector import Vector

class AnaProcessor(Skeleton):
    """custom file processor 
    
    This class inherits from the Skeleton defined in pyutils/pyprocess base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self, file_list_path, jobs=1 ):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        # Call the parent class's __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = file_list_path#"/exp/mu2e/app/users/sophie/analysis/LikelihoodAnalysis/py-fitter/filelist.txt"

        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
            ],
            "crv" : [
                "crvcoincs.time",
            ],
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trk.status",
                "trkqual.valid",
                "trkqual.result"
            ],
            "trkfit" : [
                "trksegs",
                "trksegsmc",
                "trksegpars_lh"
            ],
            "trkmc" : [
                "trkmcsim",
                "trkmc.valid"
            ]
        }
        #self.filelist = "filelist.txt"          # text file containing list of files
        self.use_remote = False     # Use remote file via mdh
        # self.location = "tape"     # File location
        self.max_workers = jobs      # Limit the number of workers
        self.verbosity = 2         # Set verbosity 
        self.use_processes = True  # Use processes rather than threads
        
        # Now add your own analysis-specific parameters 

        # Init analysis methods
        # Would be good to load an analysis config here 
        self.analyse = Analyze(
            verbosity=0
        )
            
        # Custom prefix for log messages from this processor
        self.print_prefix = "[AnaProcessor] "
        print(f"{self.print_prefix}Initialised")
    
    # ==========================================
    # Define the core processing logic
    # ==========================================
    # This method overrides the parent class's process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """
        try:
            # Create a local pyprocess Processor to extract data from this file
            # This uses the configuration parameters from our class
            processor = Processor(
                use_remote=self.use_remote,     # Use remote file via mdh
                location=self.location,         # File location
                verbosity=0 # self.verbosity        # Reduce output in worker threads
            )
            
            # Process the files using multithreading
            data = processor.process_data(
                file_list_path = self.file_list_path, 
                # file_name = self.file_name,
                # defname = defname, # Alternatively, you can provide a SAM definition
                branches = self.branches
              
            )
            
            # ---- Analysis ----            
            results = self.analyse.execute(data, file_name)

            # Clean up
            gc.collect()

            return results 
        
        except Exception as e:
            # Handle any errors that occur during processing
            print(f"{self.print_prefix}Error processing {file_name}: {e}")
            return None

def combine_arrays(results):
    """Combine filtered arrays from multiple files
    """
    arrays_to_combine = []
    # Check if we have results
    if not results:
        return None
    # Loop through all files
    for result in results: #
        if len(result) == 0:
            continue
        # Concatenate arrays
        arrays_to_combine.append(result)
    return ak.concatenate(arrays_to_combine)

def categorize_tracks( data, mismatch=False):
    array_tmp = ak.copy(data['trkmc'])

    i_mask = (array_tmp['trkmcsim']['rank'] == 0) & (array_tmp['trkmcsim']['nhits'] > 0)
    for branch in ak.fields(array_tmp):
        for leaf in ak.fields(array_tmp[branch]):
            if array_tmp[branch].layout.minmax_depth[1] > 2:
                mask_vec = ak.broadcast_arrays(array_tmp[branch],i_mask,depth_limit=3)[1]
                array_tmp[branch,leaf] = array_tmp[branch,leaf].mask[mask_vec]
            else:
                array_tmp[branch,leaf] = array_tmp[branch,leaf].mask[i_mask]

    if mismatch:
        pStartCode = ak.max(ak.flatten(array_tmp['trkmcsim']['startCode'],axis=2),axis=1,mask_identity=True)
        pGenCode = ak.max(ak.flatten(array_tmp['trkmcsim']['gen'],axis=2),axis=1,mask_identity=True)

    else:
        pStartCode = ak.flatten(ak.drop_none(array_tmp['trkmcsim']['startCode']),axis=2,mask_identity=True)
        pGenCode = ak.flatten(ak.drop_none(array_tmp['trkmcsim']['gen']),axis=2,mask_identity=True)
    pStartCode = ak.fill_none(pStartCode,-1)
    pGenCode = ak.fill_none(pGenCode,-1)
  
    categories = ak.zeros_like(pStartCode)
    for icat, idict in enumerate(mom_components.values()):
        startCodes = idict['startCode']
        genCodes = idict['genCode']
        goodCode = ak.zeros_like(pStartCode,dtype=bool)
        for startCode in startCodes:
            for genCode in genCodes:
                goodStartCode = ak.ones_like(pStartCode,dtype=bool) if startCode is None else (pStartCode == startCode)
                goodGenCode = ak.ones_like(pGenCode,dtype=bool) if genCode is None else (pGenCode == genCode)
                goodCode = goodCode | (goodStartCode & goodGenCode)
        
        categories = categories + (icat+1) * (goodCode)
    return categories
    
    
# Create an instance of our custom processor
def  main(args):
  """ main driver function to run analysis
  """
  ana_processor = AnaProcessor(args.file, args.jobs)
  results = ana_processor.execute()

  # Create an instance of our custom processor
  pre_fit = combine_arrays(results)

  # run cat
  if int(args.cat) == 1:
    track_cat = categorize_tracks(pre_fit, args.mismatch) #just pre-fit here worked but misaaligned .mask[(trk_front)]
    track_cat = (ak.broadcast_arrays(pre_fit['trkfit']['trksegs','time'],track_cat)[1])

  # select only track front to fit to
  selector = Select()
  trk_front = selector.select_surface(pre_fit['trkfit'], sid=0)
   
  trkfit_ent = pre_fit['trkfit']["trksegs"].mask[(trk_front)]

  if int(args.cat) == 1:
    track_cat = track_cat.mask[(trk_front)]
    track_cat = ak.flatten(track_cat, axis=None)
  else:
    track_cat = []
    
  # make vector mag branch
  vector = Vector()
  mom_mag = vector.get_mag(trkfit_ent ,'mom')

  mom_mag = ak.nan_to_none(mom_mag)
  mom_mag = ak.drop_none(mom_mag)
  
  #call the fitter
  if(args.fittype == "mom1D"):
    fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(mom_mag, track_cat,  (args.fitrange_low[0]), (args.fitrange_hi[0]), bool(args.cat), args.verbose)
    print('[py-fitter/main] ✅  Fit result: ', fitresult,'\n', 'for  fit')
    if (int(args.interpret) == 1):
      result_output = ResultsClass(mom_mag, fitresult,  args.verbose)
      result_output.WriteFittedData()
      result_output.WriteResult()
      result_output.GetSignifcance(par, loss, 'asym')
      #result_output.GetUL(par, loss, nlls, combine_pdf, constraints,(args.fitrange_low[0]), (args.fitrange_hi[0]),result.params['N_CE']['value'],0.90,'freq')
  elif(args.fittype == "time1D"):
    print("working on it")
    #FIXME
    #result = Unbinned_fit_time(array_cut, (args.fitrange_low[0]), (args.fitrange_hi[0]),bool(args.cat), args.verbose)
    #print('[py-fitter/main] ✅ Fit result: ', result,'\n', 'for ',args.fittype,' fit')
  elif(args.fittype == "momtime2D"):
    print("working on it")
    #FIXME
    #result = Unbinned_2d_fit_mom_time(array_cut, [(args.fitrange_low[0]),(args.fitrange_hi[0])], [(args.fitrange_low[1]),(args.fitrange_hi[1])],bool(args.cat), args.verbose)
   #print('[py-fitter/main]✅  Fit result: ', result,'\n', 'for ',args.fittype,' fit')  
  else:
    raise Exception("[py-fitter/main] ❌ ERROR: choice of fit type does not exist, please choose: mom1D, time1D or momtime2D")
      
      
  
def PrintArgs(args):
  """
  prints users input parameters
  """
  print("========= [py-fitter/main]✅  Analyzing with user opts: ===========")
  print("file:", args.file)
  print("number of processes (njobs - optimal is 1 per file):", args.jobs)
  print("fittype: ", args.fittype)
  print("range: ", args.fitrange_low, args.fitrange_hi)
  print("categorize: ", args.cat)
  print("mismatch: ", args.mismatch)
  print("verbose: ", args.verbose)
  print("interpret: ", args.interpret)

if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--fittype", type=str, default="mom1D", help="fittype implemented opts: mom1D, time1D, momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=[95,640], nargs='+', help="minimum to fit ordered mom, time")
    parser.add_argument("--fitrange_hi", type=float, default=[110,1650], nargs='+',help="maximum to fit  ordered mom, time")
    parser.add_argument("--interpret", type=int, default=0, help="writes data and fit results to csv")
    parser.add_argument("--cat", type=int, default=0, help="Categorize tracks by MC matching")
    parser.add_argument("--mismatch", type=int, default=0, help="This is an old sample with MC - reco trk mismatch")
    parser.add_argument("--verbose", default=1, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)




