# Import class
# Import the data stored in a root tree by TrkAna and save it as an Awkward array
# Additional functions allows to dump the TrkAna tree as csv

import sys
import uproot
import awkward as ak
import pandas
import numpy as np


class ImportClass :

    def __init__(self, fileName, treeName, branchName):
        """Initialise the Class Object"""

        self.FileName= fileName
        self.TreeName = treeName
        self.BranchName = branchName
        self.Array = ak.Array

    def Import(self):
        """ Import root tree and save it as an Awkward array """

        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]

        # Uproot5 using awkward array
        self.Array = input_tree.arrays(library='ak')

        return self.Array

    def printAllField(self):
        """Print all the field variable in the array with their type"""
        return self.Array.type.show()
