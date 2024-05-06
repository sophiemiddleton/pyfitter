# Main
# Main program which calls all the modules in the correct order with the correct inputs

import sys
import numpy as np
import matplotlib.pyplot as plt

from Import_module import ImportClass
from Cut_module import CutClass
from Fit_module import Unbinned_fit_mom
import argparse
from optparse import OptionParser

def  main(options, args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    i_MC = ImportClass(options.filelist, options.treename, options.branchname)
    array_MC = i_MC.Import()

    # Apply the selection cuts from cd3
    cuts = CutClass("su2020", False)
    array_MC_cuts = cuts.ApplyCut(array_MC)

    if options.verbose != 0:
        print('\nMC count (before cut):')
        gen_count = count_MC(array_MC)
        print(gen_count)
        print('\nMC count (after cut):')
        gen_count_cuts = count_MC(array_MC_cuts)
        print(gen_count_cuts)

    if options.showplots:
        plot_MC(array_MC, ('deent','mom'))
        plot_MC(array_MC, ('de','t0'))

    result = Unbinned_fit_mom(array_MC_cuts, options.fitrange_mom_low, options.fitrange_mom_hi)
    print('Fit result: ', result) # TODO you should have these sent to a file too as an option....
    plt.show()

if __name__ == "__main__":

    parser = OptionParser()

    # example use: python main.py --filelist "trkana7.root" --treename "TrkAnaNeg" --branchname "trkana"
    parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    parser.add_option('-f','--filelist', dest='filelist', default = 'trkana7.root',help='filelist', metavar='fdir')
    parser.add_option('-t','--treename', dest='treename', default = 'TrkAnaNeg',help='treename', metavar='tdir')
    parser.add_option('-n','--branchname', dest='branchname', default = 'trkana', help='branchname', metavar='bdir')
    parser.add_option('-l','--fitrange_mom_low', dest='fitrange_mom_low', default = 95, help='fitrange_mom_low', metavar='ldir')
    parser.add_option('-g','--fitrange_mom_hi', dest='fitrange_mom_hi', default = 115, help='fitrange_mom_hi', metavar='hdir')
    parser.add_option('-a','--showplots', dest='showplots', default =False, help='showplots', metavar='sdir')
    parser.add_option('-v','--verbose', dest='verbose', default =0, help='verbose', metavar='vdir')

    (options, args) = parser.parse_args()

    main(options, args)
