# Main program which calls all the modules in the correct order with the correct inputs

import sys
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from import_module import ImportClass
from cut_module import CutClass
from fit_module import Unbinned_fit_mom
import argparse
from mc_module import *
from recoplot_module import PlotRecoMomEnt

def  main(args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    i_MC = ImportClass(args.filelist, args.treename, args.branchname)

    # find track fit branches for cuts:
    demfits = i_MC.Import_branches(["demfit", "demlh"])
    trkqual = i_MC.Import_branches(["demtrkqual.result"])
    demhits = i_MC.Import_branches(["dem.nactive"])

    # apply cuts:
    cuts = CutClass("MDC2024", False)
    crvcoin = i_MC.Import_branches(["crvcoincs"])
    data_cut = cuts.ApplyCut_mom(demfits, trkqual, demhits, crvcoin) # returns momentum only

    # Use when you want to incoperate MC information FIXME - this just repeats the above
    if int(args.showMC) ==1:
        array_MC = i_MC.Import_branches(["demfit","demlh","demmcsim"])
        cuts_MC = CutClass("MDC2024", False)
        data_cut_MC = cuts_MC.ApplyCut_mom(array_MC, trkqual, demhits, crvcoin)
        print('Before cut:\n', count_MC(array_MC))
        print('After cut:\n', count_MC(data_cut_MC))

    result = Unbinned_fit_mom(data_cut, (args.fitrange_mom_low), (args.fitrange_mom_hi))
    print('Fit result: ', result) # TODO you should have these sent to a file too as an option....
    plt.show()

if __name__ == "__main__":
    # example use: python main.py --filelist "pass0a.tka" --treename "TrkAna" --branchname "trkana" --fitrange_low 95 --fitrange_hi 115 --showMC 1
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--filelist", type=str, default="nts.mu2e.ensemble-1BB-CEDIOCRYCosmic-2400000s-p95MeVc-Triggered.MDC2020ae_best_v1_3.0.tka", help="filename")
    parser.add_argument("--treename", type=str, default="TrkAna", help="treename")
    parser.add_argument("--branchname", type=str, default="trkana", help="branchname")
    parser.add_argument("--fitrange_mom_low", type=float, default=95, help="fitrange_mom_low")
    parser.add_argument("--fitrange_mom_hi", type=float, default=115, help="fitrange_mom_hi")
    parser.add_argument("--showMC", type=int, default=0, help="showMC")
    parser.add_argument("--verbose", default=0, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
