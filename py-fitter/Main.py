# Main program which calls all the modules in the correct order with the correct inputs

import sys
import numpy as np
import matplotlib.pyplot as plt

from Import_module import ImportClass
from Cut_module import CutClass
from Fit_module import Unbinned_fit_mom
import argparse
from MCPlot_module import *
from RecoPlot_module import *

def  main(args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    i_MC = ImportClass(args.filelist, args.treename, args.branchname)

     # find track fit branches for downstream (d) electron (em) loop helix (lh):
    array_LHFit = i_MC.Import_branch("demfit", "demlh")
    data_np = i_MC.Import_mom(array_LHFit)
    # Apply the selection cuts
    #cuts = CutClass("su2020", False) # TODO - this should pass out the fits
    #array_LHFit_cuts = cuts.ApplyCut(array_LHFit) # TODO - this should be passed to the fitter

    if (args.verbose != 0):
        print('\nMC count (before cut):')
        gen_count = count_MC(array_LHFit)
        print(gen_count)
        print('\nMC count (after cut):')
        gen_count_cuts = count_MC(array_LHFit_cuts)
        print(gen_count_cuts)

    if (args.showplots == 1):
        PlotRecoMomEnt(array_LHFit, args.fitrange_mom_low, args.fitrange_mom_hi)

    result = Unbinned_fit_mom(data_np, args.fitrange_mom_low, args.fitrange_mom_hi)
    print('Fit result: ', result) # TODO you should have these sent to a file too as an option....
    plt.show()

    #if (args.hasMC is 1 and args.showplots is 1):
        #plot_MC(array_LHFit, ('deent','mom')) FIXME won't work in MDC2024
        #plot_MC(array_LHFit, ('de','t0'))
        #plot_MC_comparison(MC_count_cuts, result)


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
    parser.add_argument("--verbose", default=0, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()
    main(args)
