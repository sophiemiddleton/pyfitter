# Cut class
# Apply the selection cuts of choice, no optimization is done in this fill

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import math

class CutClass:

    def __init__(self,  opt = 'SU2020', use_CRV = True):
        self.use_CRV = use_CRV
        self.Event_cut = {} # Cut applied at the event level
        self.SID_select = {} # Select the Surface ID
        self.Track_cut = {} # Cut applied at the track level
        self.Trksegs_cut = {}# Cut applied at the trksegs level
        self.CRV_cut = {}   # Cut applied based on CRV coincidence
        self.MC_cut = {}    # Cut applied on the MC data
        if opt == 'SU2020':
            self.Event_cut = {
            }
            self.SID_select = {
                "'trksegs','sid'": 0    # Look at the track at the entrance of the tracker
            }
            self.Track_cut = {
                "'trk','trk.pdg'": 11,   # trk uses e- hypothesis for Kalman fit
                "'trk','trk.nactive'": [20, float('inf')] # active hits in the tracker
            }
            self.Trksegs_cut = {
                #"'trksegs','sid'": 0,   # Look at the track at the entrance of the tracker
                "'trksegs','mom','fCoordinates','fZ'": [0, float('inf')],   # Look at downstream tracks
                "'trksegs','time'": [640., 1650], #inTimeWindow
                "'trksegpars_lh','t0err'": [0, 0.9], #intimeErr
                "'trksegpars_lh','maxr'": [450., 680.], #inMaxRCut
                #"'trkqual.result'": [0.2, float('inf')], # trk qual
            }
            self.CRV_cut = {
                "crv_coincidence" : 150. # cut anything with a track - crv time difference less than this
            }

    def ApplyCut(self, array_trk, array_crv):
        """ function applies cuts to MDS1 """
        print("\nApplying cuts\n")
        print("# of events before cut: ", ak.num(array_trk, axis=0))
        print("# of tracks before cut: ", ak.count(array_trk['trk','trk.status']))

        array_cut = ak.copy(array_trk)  # Use copy to keep the initial array untouched

        # Event level cut: TODO when there is one
        
        # Track level cut
        for key, value in self.Track_cut.items():
            print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            ApplyMaskTrk(array_cut, mask)
            #array_trk[eval(key)].show()
            #array_cut[eval(key)].show()

        # For SID selection, because of the different size between the MC and the trk branches we cut completely the SID we don't want to look at. TODO It sounds very complicated to make something more general at this point (L. Borrel, 11/2024)
        for key, value in self.SID_select.items():
            print(eval(key))

            mask_sid = array_trk[eval(key)] == value

            for branch in ak.fields(array_trk):
                for leaf in ak.fields(array_trk[branch]):
                    if (branch == 'trksegs'):# or (branch == 'trksegpars_lh') or (branch == 'trksegsmc'):
                        array_cut[branch,leaf] = array_trk[branch,leaf].mask[mask_sid]   # Here instead of applying a mask, we remove the trksegs with other SID to keep the MC and trk arrays at the same size

        # Track segments level cut
        for key, value in self.Trksegs_cut.items():
            print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            print("# of tracks passing this cut: ", ak.sum(mask))
            
            ApplyMaskTrksegs(array_cut, mask)
            #mask.show()
            #array_trk[eval(key)].show()
            #array_cut[eval(key)].show()

        # CRV cuts
        if self.use_CRV:
            print('crv cut')
            builder = ak.ArrayBuilder()
            for i_evt, evt in enumerate(array_trk['trksegs','time']):
                builder.begin_list()
                for i_trk, trk in enumerate(evt):
                    flag = True
                    if ak.num(ak.drop_none(trk), axis=0) > 0:
                        for i_crv, crv in enumerate(array_crv['crvcoincs','crvcoincs.time', i_evt]):
                            if np.abs(trk[0] - crv) < self.CRV_cut.get('crv_coincidence'):
                                flag = False
                    builder.boolean(flag)
                builder.end_list()

            coincidence = builder.snapshot()
            ApplyMaskTrk(array_cut, coincidence)

        #print("# of events after all the cuts: ", ak.num(array_trk, axis=0)
        print("# of tracks after all the cuts: ", ak.count(array_cut['trksegs','time']))

        return array_cut


    def ApplyCutMC(self, array_mc, array_trk_cut):
        """ Apply the trk cut on the MC array"""
        """ Reproduce the combination of all masks applied on the trk array and apply it on the MC array """
        print("\nApplying cuts on MC array\n")
        print("# of events before cut: ", ak.num(array_mc, axis=0))
        print("# of tracks before cut: ", ak.count(array_mc['trkmc','trkmc.valid']))
        array_mc_cut = ak.copy(array_mc)

        # Event level cut: TODO when there is one

        # Track level cut
        mask = ~(ak.is_none(array_trk_cut['trk','trk.nactive'], axis=1))
        ApplyMaskTrk(array_mc_cut, mask)
        mask.show()
        array_mc['trkmc','trkmc.nactive'].show()
        array_mc_cut['trkmc','trkmc.nactive'].show()

        # Track segments level cut
        mask = ~(ak.is_none(array_trk_cut['trksegs','time'], axis=2))
        ApplyMaskTrksegs(array_mc_cut, mask)
        mask.show()
        array_mc['trksegsmc','time'].show()
        array_mc_cut['trksegsmc','time'].show()

        #print("# of events after all the cuts: ", ak.num(array_trk, axis=0)
        print("# of tracks after all the cuts: ", ak.count(array_mc_cut['trksegsmc','time']))


def ApplyMaskTrk(array_cut, mask):
    """ Apply the mask onto each track-level branch of the array """
    for branch in ak.fields(array_cut):
        for leaf in ak.fields(array_cut[branch]):
            array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask]

    return array_cut


def ApplyMaskTrksegs(array_cut, mask):
    """ Apply the mask onto each track-segments-level branch of the array """
    for branch in ak.fields(array_cut):
        if (branch == 'trksegs') or (branch == 'trksegpars_lh') or (branch == 'trkmcsegsmc'):
            for leaf in ak.fields(array_cut[branch]):
                array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask]

    return array_cut


