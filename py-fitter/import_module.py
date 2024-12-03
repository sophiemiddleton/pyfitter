# Import the data stored in a root tree by TrkAna and save it as an Awkward array

import sys
import uproot
import awkward as ak
import pandas
import numpy as np

#FIXME - use pyutils
class ImportClass :

    def __init__(self, fileName, treeName, branchName):
        """Initialise the Class Object"""
        self.FileName= fileName
        self.TreeName = treeName
        self.BranchName = branchName
        self.Array = ak.Array

    def Import(self, list_branch = [], filter_name="*"):
        """ Import root tree and save it as an Awkward array """
        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]
        self.Array = input_tree.arrays(list_branch, library='ak')
        return self.Array

    def Import_branch(self, branch_name):
        """ Import only one single branch in an awkward array """
        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]
        self.Array = input_tree[branch_name].array(library='ak')
        return self.Array

    def AddMomentumBranch(self, array_trk):
        """ Add momentum branch """ 
        # FIXME - use utils
        # register the vector class
        vector.register_awkward()

        # make the Vector 3D
        trkvect3D = ak.zip({
            "x": branch[str(vectorname)]["fCoordinates"]["fX"],
            "y": branch[str(vectorname)]["fCoordinates"]["fY"],
            "z": branch[str(vectorname)]["fCoordinates"]["fZ"],
        }, with_name="Vector3D")
        
        mag = vector.mag
        array_trk['trksegs','mom.mag'] = mag

        return array_trk


    def printAllField(self):
        """Print all the field variable in the array with their type"""
        #FIXME use utils
        return self.Array.type.show()
