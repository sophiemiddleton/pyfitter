# Import the data stored in a root tree by TrkAna and save it as an Awkward array

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


    def Import(self, filter_branch="*"):
        """ Import root tree and save it as an Awkward array """
        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]
        self.Array = input_tree.arrays(filter_name = filter_branch, library='ak')

        return self.Array

    def Import_branch(self, branch_name):
        """ Import only one single branch in an awkward array """
        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]
        self.Array = input_tree[branch_name].array(library='ak')

        return self.Array


    def AddMomentumBranch(self, array_trk):
        """ Add momentum branch """
        array_trk['trksegs','mom.mag'] = np.sqrt((array_trk['trksegs','mom','fCoordinates','fX'])**2 + (array_trk['trksegs','mom','fCoordinates','fY'])**2 + (array_trk['trksegs','mom','fCoordinates','fZ'])**2)

        return array_trk


    def printAllField(self):
        """Print all the field variable in the array with their type"""
        return self.Array.type.show()
