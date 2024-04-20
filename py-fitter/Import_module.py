# Import class
# Import the data stored in a root tree by TrkAna and save it as an Awkward array
# Additional functions allows to dump the TrkAna tree as csv

# Original Author: Leo Borrel
# Edits: Sophie Middleton
# Date: 2024-04-19


import sys
import uproot
import awkward as ak
import pandas
import numpy as np

class ImportClass :

    def __init__(self, fileName, treeName, branchName, flatten = False):
        """Initialise the Class Object"""

        self.FileName= fileName
        self.TreeName = treeName
        self.BranchName = branchName
        self.Flatten = flatten
        self.Array = ak.Array

    def Import(self):
        """ Import root tree and save it as an Awkward array """

        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]

        # Uproot5 using awkward array
        self.Array = input_tree.arrays(library='ak')
        return self.Array

    def GetFlatname(self, branchname, featurename, index):
        """flatten the tree branch names for making querying later easier e.g. de.status --> de_status """

        flattened_branches = self.BranchName.replace(".", "_")
        if not isinstance(self.BranchName, str):
            flattened_branches = self.BranchName.decode("utf-8")
        if self.FeatureName is not None:
            self.FeatureName.replace(".", "_")
            flattened_branches += "_" + self.FeatureName
        if index != ():
            flattened_branches += "[" + "][".join(str(x) for x in index) + "]"

        return flattened_branches
