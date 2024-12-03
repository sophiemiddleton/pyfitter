# Main program which calls all the modules in the correct order with the correct inputs

import sys
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse

from import_module import ImportClass
from cut_module import CutClass
from fit_module import Unbinned_fit_mom
from mc_module import *
from recoplot_module import PlotRecoMomEnt

def  main(args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    mds1 = ImportClass(args.filelist, args.treename, args.branchname)

    # find track fit branches for cuts:
    # FIXME temporary only import branches that behaves correctly with the cuts
    #array_trk = mds1.Import(filter_branch="trk[!h]*")
    list_branch_trk = ["trk","trksegs","trksegpars_lh","trkcalohit", "trkmats"]
    list_branch_crv = ["crvsummary.","crvcoincs"]
    array_trk = mds1.Import(list_branch=list_branch_trk)
    array_trk = mds1.AddMomentumBranch(array_trk)
    array_crv = mds1.Import(list_branch=list_branch_crv)

    # apply cuts:
    cuts = CutClass("SU2020", False)
    array_cut = cuts.ApplyCut(array_trk, array_crv)
    array_trk['trk','trk.nactive'].show()
    array_cut['trk','trk.nactive'].show()

    # Use when you want to incorporate MC information FIXME - this just repeats the above
    if int(args.showMC) ==1:
        list_branch_mc = ["trkmc","trkmcsim","trksegsmc"] #,"trkmcvd","trkcalohitmc","trksegsmc"]
        list_branch_crv_mc = ["crvsummarymc.","crvcoincsmc","crvcoincsmcplane"]
        array_MC = mds1.Import(list_branch=list_branch_mc)
        #array_crv_MC = mds1.Import(list_branch=list_branch_crv_mc)
        cuts_MC = CutClass("SU2020", False)
        array_cut_MC = cuts_MC.ApplyCutMC(array_MC, array_cut)
        #print('Before cut:\n', count_MC(array_MC))
        #print('After cut:\n', count_MC(data_cut_MC))

    result = Unbinned_fit_mom(array_cut, (args.fitrange_mom_low), (args.fitrange_mom_hi))
    print('Fit result: ', result) # TODO you should have these sent to a file too as an option....
    plt.show()

if __name__ == "__main__":
    # example use: python main.py --filelist "pass0a.tka" --treename "EventNtuple" --branchname "ntuple --fitrange_low 95 --fitrange_hi 115 --showMC 1
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--filelist", type=str, default="nts.lborrel.ensembleMDS1aOnSpillTriggered.MDC2020ai_perfect_v1_3.0.root", help="filename")
    parser.add_argument("--treename", type=str, default="EventNtuple", help="treename")
    parser.add_argument("--branchname", type=str, default="ntuple", help="branchname")
    parser.add_argument("--fitrange_mom_low", type=float, default=95, help="fitrange_mom_low")
    parser.add_argument("--fitrange_mom_hi", type=float, default=115, help="fitrange_mom_hi")
    parser.add_argument("--showMC", type=int, default=0, help="showMC")
    parser.add_argument("--verbose", default=0, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
