import sys
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse

#from import_module import ImportClass
from cut_module import CutClass
from fit_module import *
from mc_module import *
from results_module import ResultsClass

import sys 
sys.path.append("../../pyutils/pyutils") # import pyutils

# pyutils classes
from pyprocess import Processor 
from pyvector import Vector

# Define the path to our example file
def  main(args):
    """
    main driving function of the entire analysis
    """
    if args.verbose > 0:
      print("[py-fitter/main] ✅ beginning analysis of ",args.file," with ", args.dirname, args.treename, "verbosity set to", args.verbose)
    # Import the data from the ntuple convert it into an Awkward array
    mds = Processor(verbosity=args.verbose)

    list_branch_trk = ["trksegs","trksegpars_lh","trk.nactive","trk.status","trk.pdg","trkqual.result"]
    list_branch_crv = ["crvsummary.","crvcoincs.time"]
    if (int(args.singlefile) == 1):
      array_trk = mds.process_data(
          file_name=args.file,
          branches=list_branch_trk
      )
    else:
      array_trk = mds.process_data(
          file_list_path = args.file,
           branches=list_branch_trk
      )

    # use vector package to include magnitude:
    vector = Vector()
    mom_mag = vector.get_mag(array_trk["trksegs"],'mom')
    array_trk['trksegs', 'mom.mag'] = mom_mag
    
    # import crv branches
    if (int(args.singlefile) == 1):
      array_crv = mds.process_data(
          file_name=args.file,
          branches=list_branch_crv
      )
    else:
      array_crv = mds.process_data(
          file_list_path =args.file,
          branches = list_branch_crv
      )
    # use our custom cut class
    cuts = CutClass(str(args.cuts), True, args.verbose)
    
    # apply cuts:
    if args.verbose > 0:
      print("[py-fitter/main] ✅ applying cut list ",args.cuts)
    array_cut = cuts.ApplyCut(array_trk, array_crv)

    if int(args.cat) > 0:
        if args.verbose > 0:
          print("[py-fitter/main] ✅ cat option set, looking at MC info")
        list_branch_mc  = ["trkmcsim"]
        if (int(args.singlefile) == 1):
          array_mc = mds.import_file(
              file_name=args.file,
              branches=list_branch_mc
          )
        else:
          array_mc = mds.import_dataset(
              file_list_path =args.file,
               branches=list_branch_mc
          )
        track_cat = cuts.CategorizeTracks(array_mc,args.mismatch)
        array_cut['trksegs','cat'] = ak.broadcast_arrays(array_cut['trksegs','time'],track_cat)[1]

    if(args.fittype == "mom1D"):
      result, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(array_cut, (args.fitrange_low[0]), (args.fitrange_hi[0]), bool(args.cat),args.verbose, args.DIO_efficiency_file, args.DIO_resolution_file)
      print('[py-fitter/main] ✅  Fit result: ', result,'\n', 'for ',args.fittype,' fit')

      result_output = ResultsClass(array_cut, result,  args.verbose)
      #result_output.GetSignifcance(par, loss, 'freq')
      #result_output.GetUL(par, loss, nlls, combine_pdf, constraints,(args.fitrange_low[0]), (args.fitrange_hi[0]),result.params['N_CE']['value'],0.90,'freq')

      if (int(args.writeoutput) == 1):
        result_output.WriteFittedData()
        result_output.WriteResult()

    elif(args.fittype == "time1D"):
      result = Unbinned_fit_time(array_cut, (args.fitrange_low[0]), (args.fitrange_hi[0]),bool(args.cat), args.verbose)
      print('[py-fitter/main] ✅ Fit result: ', result,'\n', 'for ',args.fittype,' fit')
      
    elif(args.fittype == "momtime2D"):
       result = Unbinned_2d_fit_mom_time(array_cut, [(args.fitrange_low[0]),(args.fitrange_hi[0])], [(args.fitrange_low[1]),(args.fitrange_hi[1])],bool(args.cat), args.verbose)
       print('[py-fitter/main]✅  Fit result: ', result,'\n', 'for ',args.fittype,' fit')
       
    else:
      raise Exception("[py-fitter/main] ❌ ERROR: choice of fit type does not exist, please choose: mom1D, time1D or momtime2D")
      
    plt.show()

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
    parser.add_argument("--fitrange_hi", type=float, default=[115,1650], nargs='+',help="maximum to fit  ordered mom, time")
    parser.add_argument("--cuts", type=str, default="SU2020", help="cut e.g. SU2020")
    parser.add_argument("--writeoutput", type=int, default=1, help="writes data and fit results to csv")
    parser.add_argument("--showMC", type=int, default=0, help="will use MC information")
    parser.add_argument("--cat", type=int, default=0, help="Categorize tracks by MC matching")
    parser.add_argument("--mismatch", type=int, default=0, help="This is an old sample with MC - reco trk mismatch")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    parser.add_argument("--DIO_efficiency_file", type=str, default=None, help="DIO efficiency file path")
    parser.add_argument("--DIO_resolution_file", type=str, default=None, help="DIO resolution file path")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)
