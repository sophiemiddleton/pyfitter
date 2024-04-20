# Main
# Main program which calls all the modules in the correct order with the correct inputs

# Original Author: Leo Borrel
# Edits: Sophie Middleton
# Date: 2024-04-19

import sys
import numpy as np
import matplotlib.pyplot as plt

from Import_module import ImportClass
from Cut_module import CutClass
from Fit_module import Unbinned_fit_mom
import argparse
from optparse import OptionParser
##from PDF_list import PDF

def  main(options, args):
    # Import the data from the Trkana root tree and convert it into an Awkward array
    i = ImportClass(options.filelist, options.treename, options.branchname)
    array = i.Import()

    # Apply the selection cuts from cd3
    cuts = CutClass("cd3", False)
    array_cuts = cuts.ApplyCut(array)

    Unbinned_fit_mom(array_cuts, 95,115) # TODO why does this not work? options.fitrange_mom_low, options.fitrange_mom_hi)
    plt.show()

if __name__ == "__main__":

    parser = OptionParser()

    # example use: python main.py --filelist "trkana7.root" --treename "TrkAnaNeg" --branchname "trkana"
    parser.add_option('-b', action='store_true', dest='noX', default=False, help='no X11 windows')
    parser.add_option('-f','--filelist', dest='filelist', default = 'trkana7.root',help='filelist', metavar='fdir')
    parser.add_option('-t','--treename', dest='treename', default = 'TrkAnaNeg',help='treename', metavar='tdir')
    parser.add_option('-n','--branchname', dest='branchname', default = 'trkana', help='branchname', metavar='bdir')
    parser.add_option('-l','--fitrange_mom_low', dest='fitrange_mom_low', default = '95', help='fitrange_mom_low', metavar='ldir')
    parser.add_option('-g','--fitrange_mom_hi', dest='fitrange_mom_hi', default = '115', help='fitrange_mom_hi', metavar='hdir')

    (options, args) = parser.parse_args()

    main(options, args)
