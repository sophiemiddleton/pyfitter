# Cut class
# Apply the selection cuts of choice, no optimization is done in this fill

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import math

class CutClass:

    def __init__(self,  opt = 'MDC2024', use_CRV = True):
        self.use_CRV = use_CRV
        self.Event_cut = {} # Cut applied at the event level
        self.SID_select = {}# Cut to only select track values at the entrance of the tracker
        self.Track_cut = {} # Cut applied at the track level
        self.CRV_cut = {}   # Cut applied based on CRV coincidence
        self.MC_cut = {}    # Cut applied on the MC data
        if opt == 'MDC2024': #TODO these are SU2020 cuts, and some are missing....
            self.Event_cut = {
            }
            self.SID_select = {
                "'trksegs','sid'" : 0
            }
            self.Track_cut = {
                "'trk.nactive'" : [20, float('inf')], # active hits in the tracker
                "'trksegs','time'" : [640., 1650], #inTimeWindow
                "'trksegpars_lh','t0err'" : [0, 0.9], #intimeErr
                "'trksegpars_lh','maxr'" : [450., 680.], #inMaxRCut
                #"'trkqual.result'" : [0.2, float('inf')], # trk qual
            }
            self.CRV_cut = {
                "crv_coincidence" : [50.,150.] # cut anything with a track - crv time difference less than this
            }

    def ApplyCut(self, array_trk, array_crv):
        """ function applies cuts to MDC2024 trkana """
        print("Applying cuts\n")
        print("# of events before cut: ", ak.num(array_trk, axis=0))
        print("# of tracks before cut: ", ak.sum(ak.num(array_trk['trk.status'], axis=1)))

        array_cut = array_trk

        # Event level cut: TODO when there is one
        
        # Surface ID (SID) selection
        for key, value in self.SID_select.items():
            print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            array_trksegs_cut = array_cut['trksegs'].mask[mask]
            array_trksegpars_lh_cut = array_cut['trksegpars_lh'].mask[mask]
            array_cut['trksegs'] = array_trksegs_cut
            array_cut['trksegpars_lh'] = array_trksegpars_lh_cut
            array_cut[eval(key)].show()

        # Track level cut
        for key, value in self.Track_cut.items():
            print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            print("# of tracks passing this cut: ", ak.sum(mask))
            array_cut = array_cut.mask[mask]
            array_cut[eval(key)].show()

        return array_trk


    def ApplyCRVCut(self, demfits, crvcoin, cut_value):
        """ function applies time based cut on comparison of TRK and CRV times """
        # Remove events without any track
        trk_valid = ak.num(demfits['trksegs','time'], axis=1) > 0
        mask_trk = ak.mask(demfits, trk_valid)
        crv_coinc = np.ones(ak.num(demfits, axis=0), dtype='bool')    # Create a boolean array of True for the mask
        for evt_idx, evt in enumerate(demfits['trksegs','time']):   # loop through every event
            for trk_idx, trk in enumerate(evt):     # loop through every track within an event
                if ak.num(trk, axis=0) > 0:       # rare case when an event has 2 tracks, with one being empty (Ask why?)
                    for crv_idx, crv in enumerate(crvcoin['crvcoincs.time', evt_idx]):    # loop through CRV hits
                        if np.abs(trk[0] - crv) < cut_value:
                            crv_coinc[evt_idx] = False
        return crv_coinc
