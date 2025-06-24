import hist
import gc
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse

#from cut_module import CutClass
#from fit_module import *
#from mc_module import *
#from results_module import ResultsClass
from analyze import Analyze

from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyplot import Plot
from pyutils.pyprint import Print
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
class AnaProcessor(Skeleton):
    """Your custom file processor 
    
    This class inherits from the Skeleton base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        # Call the parent class's __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = "/exp/mu2e/app/users/sophie/analysis/LikelihoodAnalysis/py-fitter/filelist.txt"
        
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
                "trksegpars_lh"
            ],
            "trkmc" : [
                "trkmcsim"
            ]
        }
        #self.filelist = "filelist.txt"          # text file containing list of files
        self.use_remote = False     # Use remote file via mdh
        # self.location = "tape"     # File location
        self.max_workers = 1      # Limit the number of workers
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
        # array = ak.Array(result["filtered_data"])
        if len(result) == 0:
            continue
        # Concatenate arrays
        arrays_to_combine.append(result)
    return ak.concatenate(arrays_to_combine)


# Create an instance of our custom processor
def  main(args):
  ana_processor = AnaProcessor()
  results = ana_processor.execute()
  # Create an instance of our custom processor
 
  ana_processor = AnaProcessor()
  results = ana_processor.execute()

  pre_fit = combine_arrays(results)
 
  """
  n,bins,patch = plt.hist(pre_fit, color='black', bins=50, range=((98.,113.)), histtype='step')
  plt.yscale('log')
  plt.show()
  """

  result, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(pre_fit, 98., 113., 1,1)
  print('[py-fitter/main] ✅  Fit result: ', result,'\n', 'for  fit')
  
def PrintArgs(args):
  """
  prints users input parameters
  """
  print("========= [py-fitter/main]✅  Analyzing with user opts: ===========")
  print("file:", args.file," with ", args.dirname, args.treename)
  print("fittype: ", args.fittype)
  print("range: ", args.fitrange_low, args.fitrange_hi)
  print("cut list: ", args.cuts)
  print("showMC: ", args.showMC)
  print("categorize: ", args.cat)
  print("mismatch: ", args.mismatch)
  print("verbose: ", args.verbose)

if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--singlefile", type=int, required=False, default=1,help="use if just one root file input")
    parser.add_argument("--dirname", type=str, default="EventNtuple", help="dirname e.g. EventNtuple")
    parser.add_argument("--treename", type=str, default="ntuple", help="treename e.g. ntuple")
    parser.add_argument("--fittype", type=str, default="mom1D", help="fittype implemented opts: mom1D, time1D, momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=[95,640], nargs='+', help="minimum to fit ordered mom, time")
    parser.add_argument("--fitrange_hi", type=float, default=[113,1650], nargs='+',help="maximum to fit  ordered mom, time")
    parser.add_argument("--cuts", type=str, default="SU2020", help="cut e.g. SU2020")
    parser.add_argument("--writeoutput", type=int, default=0, help="writes data and fit results to csv")
    parser.add_argument("--showMC", type=int, default=0, help="will use MC information")
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




