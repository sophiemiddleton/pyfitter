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

    def Import(self):
        """ Import root tree and save it as an Awkward array """
        input_file = uproot.open(self.FileName)
        input_tree = input_file[self.TreeName][self.BranchName]

        # Uproot5 using awkward array
        self.Array = input_tree.arrays(library='ak')
        return self.Array

    def Import_branches(self, leafnames):
        """ import list of branches from trk ana"""
        trkana = uproot.open(self.FileName+":"+str(self.TreeName)+"/"+str(self.BranchName))
        list_names = []
        for i, leafname  in enumerate(leafnames):
            #branches = trkana.arrays(filter_name=["/"+str(leafname)+"/", "/"+str(leafname_field)+"/"])
            list_names.append("/"+str(leafname)+"/")
        branches = trkana.arrays(filter_name=list_names)
        ak.flatten(branches, axis=None)
        return branches

    def Import_mom(self, array_MC):
        """ import array select reco mom """
        array_MC['demfit_mom'] = np.sqrt((array_MC['demfit']['mom']['fCoordinates']['fX'])**2 + (array_MC['demfit']['mom']['fCoordinates']['fY'])**2 + (array_MC['demfit']['mom']['fCoordinates']['fZ'])**2)
        trk_ent_mask = (array_MC['demfit']['sid']==0)
        data_np = np.array(ak.flatten(array_MC[(trk_ent_mask)]['demfit_mom']))
        return data_np

    def printAllField(self):
        """Print all the field variable in the array with their type"""
        return self.Array.type.show()
