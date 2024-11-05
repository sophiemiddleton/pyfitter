# Cut class
# Apply the selection cuts of choice, no optimization is done in this fill

import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import math

class CutClass:

    def __init__(self,  opt = 'MDC2024', use_CRV = True):
        self.use_CRV = use_CRV
        self.Cut_List = {}
        if opt == 'MDC2024': #TODO these are SU2020 cuts, and some are missing....
            self.Cut_List = {
                "trk.nactive" : [20, float('inf')], # active hits in the tracker
                #"trk.sid" : [0], # select values at the entrance of the tracker
                #"trksegs.time" : [640., 1650], #inTimeWindow
                #"trksegpars_lh.t0err" : [0, 0.9], #intimeErr
                #"trksegpars_lh.maxr" : [450., 680.], #inMaxRCut
                #"trkqual.result" : [0.2, float('inf')], # trk qual
                "crv_coincidence" : [50.,150.] # cut anything with a track - crv time difference less than this
            }

    def ApplyCut(self, array_trk, array_crv):
        """ function applies cuts to MDC2024 trkana """
        print("Applying cuts\n")
        print("array size before cut: ", ak.num(array_trk, axis=0))

        for key, value in self.Cut_List.items():
            print(key)

            if key == 'crv_coincidence':
                print("crv coincidence")

            else:
                if type(value) == int:
                    mask = array_trk[key] == value
                elif len(value) == 2:
                    mask = (array_trk[key] >= value[0]) & (array_trk[key] <= value[1])

            print("# of events passing this cut: ", ak.sum(mask))

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
