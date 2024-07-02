# Main program which calls all the modules in the correct order with the correct inputs

import sys
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from Import_module import ImportClass
from Cut_module import CutClass
from Fit_module import Unbinned_fit_mom
import argparse
from MC_module import *
from RecoPlot_module import PlotRecoMomEnt

def  main(args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    i_MC = ImportClass(args.filelist, args.treename, args.branchname)
    #array_all = i_MC.Import()

    # find track fit branches for cuts:
    array_TRK = i_MC.Import_branches(["demfit", "demlh"])
    array_TRKQUAL = i_MC.Import_branches([ "demtrkqual"])

    # apply cuts:
    cuts = CutClass("MDC2024", False)
    array_CRV = i_MC.Import_branches(["crvcoincs"])
    data_cut = cuts.ApplyCut_mom(array_TRK, array_TRKQUAL, array_CRV) # returns momentum only

    # MC information
    if args.hasMC == 1:
        array_MC = i_MC.Import_branches(["demfit","demlh","demmcsim"])
        cuts_MC = CutClass("MDC2024", False)
        data_cut_MC = cuts_MC.ApplyCut_mom(array_MC, array_TRKQUAL, array_CRV)
        print('Before cut:\n', count_MC(array_MC))
        print('After cut:\n', count_MC(data_cut_MC))


    result = Unbinned_fit_mom(data_cut, args.fitrange_mom_low, args.fitrange_mom_hi)
    print('Fit result: ', result) # TODO you should have these sent to a file too as an option....
    plt.show()



if __name__ == "__main__":
    # example use: python main.py --filelist "trkana7.root" --treename "TrkAnaNeg" --branchname "trkana" --fit_range_low 95 --fit_range_hi 115 --showplots True
    parser = argparse.ArgumentParser()
    #parser.add_argument("--filelist", default="nts.mu2e.ensemble-MixBB-CEDIO-1month-p95MeVc-Triggered.MDC2020ad_perfect_v1_2.0.tka", help="filename")
    parser.add_argument("--filelist", default="nts.mu2e.ensemble-1BB-CEDIOCRYCosmic-2400000s-p95MeVc-Triggered.MDC2020ae_best_v1_3.0.tka", help="filename")
    parser.add_argument("--treename", default="TrkAna", help="treename")
    parser.add_argument("--branchname", default="trkana", help="branchname")
    parser.add_argument("--fitrange_mom_low", default=95, help="fitrange_mom_low")
    parser.add_argument("--fitrange_mom_hi", default=115, help="fitrange_mom_hi")
    parser.add_argument("--showplots", default=0, help="showplots")
    parser.add_argument("--hasMC", default=1, help="hasMC")
    parser.add_argument("--use_CRV", default=1, help="use_CRV")
    parser.add_argument("--verbose", default=0, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
