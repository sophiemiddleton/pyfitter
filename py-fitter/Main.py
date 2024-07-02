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
from MCPlot_module import *
from RecoPlot_module import PlotRecoMomEnt

def  main(args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    i_MC = ImportClass(args.filelist, args.treename, args.branchname)

    # find track fit branches for cuts:
    array_TRK = i_MC.Import_branches(["demfit", "demlh"])
    array_TRKQUAL = i_MC.Import_branches([ "demtrkqual"])
    array_dem = i_MC.Import_branches(["dem.nactive"])

    # apply cuts:
    cuts = CutClass("MDC2024", False)
    array_CRV = i_MC.Import_branches(["crvcoincs"])
    data_np = cuts.ApplyCut_mom(array_TRK, array_TRKQUAL, array_dem, array_CRV) # returns momentum only


    result = Unbinned_fit_mom(data_np, args.fitrange_mom_low, args.fitrange_mom_hi)
    print('Fit result: ', result) # TODO you should have these sent to a file too as an option....
    plt.show()



if __name__ == "__main__":
    # example use: python main.py --filelist "trkana7.root" --treename "TrkAnaNeg" --branchname "trkana" --fit_range_low 95 --fit_range_hi 115 --showplots True
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", default="nts.mu2e.trkana-reco-CE-DIO-1month.MDC2020ad_perfect_v1_2.0.tka", help="filename")
    parser.add_argument("--treename", default="TrkAna", help="treename")
    parser.add_argument("--branchname", default="trkana", help="branchname")
    parser.add_argument("--fitrange_mom_low", default=95, help="fitrange_mom_low")
    parser.add_argument("--fitrange_mom_hi", default=115, help="fitrange_mom_hi")
    parser.add_argument("--showplots", default=0, help="showplots")
    parser.add_argument("--hasMC", default=0, help="hasMC")
    parser.add_argument("--use_CRV", default=1, help="use_CRV")
    parser.add_argument("--verbose", default=0, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
