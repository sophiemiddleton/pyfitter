import sys
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse

from import_module import ImportClass
from cut_module import CutClass
from fit_module import *
from mc_module import *
from recoplot_module import PlotRecoMomEnt

def  main(args):
    """
    main driving function of the entire analysis
    """
    if args.verbose == 1:
      print("importing file",args.file," with ", args.dirname, args.treename)
    # Import the data from the ntuple convert it into an Awkward array
    mds1 = ImportClass(args.file, args.dirname, args.treename)

    # find track fit branches for cuts:
    # FIXME temporary only import branches that behaves correctly with the cuts
    #array_trk = mds1.Import(filter_branch="trk[!h]*")
    list_branch_trk = ["trk","trksegs","trksegpars_lh","trkqual"]
    list_branch_crv = ["crvsummary.","crvcoincs"]
    array_trk = mds1.Import(list_branch=list_branch_trk)
    array_trk = mds1.AddVectorMag(array_trk,'trksegs', 'mom') #FIXME - assume a momentum fit, probably true
    array_crv = mds1.Import(list_branch=list_branch_crv)

    # apply cuts:
    if args.verbose == 1:
      print("applying cut list",args.cuts)
    cuts = CutClass(str(args.cuts), True)

    if int(args.cat) == 1:
        if args.verbose == 1:
          print("cat option set, looking at MC info")
        list_branch_mc  = ["trkmcsim"]
        array_mc  = mds1.Import(list_branch=list_branch_mc)
        track_cat = cuts.CategorizeTracks(array_mc,args.mismatch)
        array_trk['trksegs','cat'] = ak.broadcast_arrays(array_trk['trksegs','time'],track_cat)[1]
    
    array_cut = cuts.ApplyCut(array_trk, array_crv)

    # Use when you want to incorporate MC information FIXME - this just repeats the above
    if int(args.showMC) ==1:
        if args.verbose == 1:
          print("showMC opt set, looking at MC trksegmcs")
        list_branch_mc = ["trkmc","trkmcsim","trksegsmc"]
        list_branch_crv_mc = ["crvsummarymc.","crvcoincsmc","crvcoincsmcplane"]
        array_MC = mds1.Import(list_branch=list_branch_mc)
        #array_crv_MC = mds1.Import(list_branch=list_branch_crv_mc)
        cuts_MC = CutClass(args.cuts, False)
        array_cut_MC = cuts_MC.ApplyCutMC(array_MC, array_cut)
        #print('Before cut:\n', count_MC(array_MC))
        #print('After cut:\n', count_MC(data_cut_MC))

    if(args.fittype == "mom1D"):
      result = Unbinned_fit_mom(array_cut, (args.fitrange_low), (args.fitrange_hi), bool(args.cat))
      print('Fit result: ', result) # FIXME you should have these sent to a file too as an option, to allow compare to BAT
    elif(args.fittype == "time1D"):
      result = Unbinned_fit_time(array_cut, (args.fitrange_low), (args.fitrange_hi))
      print('Fit result: ', result) # FIXME you should have these sent to a file too as an option, to allow compare to BAT
    elif(args.fittype == "momtime2D"):
       result = Unbinned_2d_fit_mom_time(array_cut, [(args.fitrange_low[0]),(args.fitrange_hi[0])], [(args.fitrange_low[1]),(args.fitrange_hi[1])])
       print('Fit result: ', result) # FIXME you should have these sent to a file too as an option, to allow compare to BAT
    else:
      raise Exception("ERROR: choice of fit type does not exist, please choose: mom1D, time1D or momtime2D")
      
    plt.show()

def PrintArgs(args):
  """
  prints users input parameters
  """
  print("=========Analyzing with user opts: ===========")
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
    parser.add_argument("--file", type=str, required=True, help="file")
    parser.add_argument("--dirname", type=str, default="EventNtuple", help="dirname e.g. EventNtuple")
    parser.add_argument("--treename", type=str, default="ntuple", help="treename e.g. ntuple")
    parser.add_argument("--fittype", type=str, default="mom1D", help="fittype implemented opts: mom1D, time1D, momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=95, nargs='+', help="minimum to fit ordered mom, time")
    parser.add_argument("--fitrange_hi", type=float, default=115, nargs='+',help="maximum to fit  ordered mom, time")
    parser.add_argument("--cuts", type=str, default="SU2020", help="cut e.g. SU2020")
    parser.add_argument("--showMC", type=int, default=0, help="will use MC information")
    parser.add_argument("--cat", type=int, default=0, help="Categorize tracks by MC matching")
    parser.add_argument("--mismatch", type=int, default=0, help="This is an old sample with MC - reco trk mismatch")
    parser.add_argument("--verbose", default=1, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # if verbose print the user input
    if(args.verbose == 1):
      PrintArgs(args)
    
    # run main function
    main(args)
