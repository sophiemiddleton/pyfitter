# Cut class
# Apply the selection cuts on the data listed as a pandas dataframe

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
                "demfit.time" : [640., 1650], #inTimeWindow
                "demlh.t0err.max" : [0.9], #intimeErr
                "demlh.maxr" : [450., 680.], #inMaxRCut
                "demtrkqual.result" : [0.2], # trk qual
                #dem', 'nactive') : [20],
                 "time_diff" : [150.] # cut anything with a track - crv time difference less than this
            }
    def ApplyCut_mom(self, array_TRK, array_TRKQUAL, array_CRV = None):
        """ function applies cuts to MDC2024 trkana """
        array_TRK['demfit_mom'] = np.sqrt((array_TRK['demfit']['mom']['fCoordinates']['fX'])**2 + (array_TRK['demfit']['mom']['fCoordinates']['fY'])**2 + (array_TRK['demfit']['mom']['fCoordinates']['fZ'])**2)
        trk_ent_mask = (array_TRK['demfit']['sid']==0)
        # build masks for cuts
        time_cut_mask_min = (array_TRK['demfit']['time']>=self.Cut_List["demfit.time"][0])
        time_cut_mask_max = (array_TRK['demfit']['time']<self.Cut_List["demfit.time"][1])
        timeerr_cut_mask = (array_TRK['demlh']['t0err']< self.Cut_List["demlh.t0err.max"][0])
        maxr_cut_mask = (array_TRK['demlh']['maxr']> self.Cut_List["demlh.maxr"][0])
        """
        trkqual_mask = (array_TRKQUAL['result'] > self.Cut_List["demtrkqual.result"][0]) # FIXME - here due to TrkAna v5 issue
        data_np = np.array(ak.flatten(array_TRK[ (trkqual_mask ) & (trk_ent_mask) & (time_cut_mask_max) & (time_cut_mask_min) & (timeerr_cut_mask) & (maxr_cut_mask)  ]['demfit_mom'], axis=None))
        """
        # look for CRV coincidences
        crv_cut_mask = self.ApplyCRVCutNew(array_TRK, array_CRV, self.Cut_List['time_diff'][0])

        # apply all cuts and convert to numpy array
        data_np = np.array(ak.flatten(array_TRK[(crv_cut_mask) & (trk_ent_mask) & (time_cut_mask_max) & (time_cut_mask_min) & (timeerr_cut_mask) & (maxr_cut_mask)  ]['demfit_mom'], axis=None))
        return data_np

    def ApplyCRVCut(self, array_TRK, array_CRV, cut_value):
        """ function applies time based cut on comparison of TRK and CRV times """ # FIXME this is a bit hacky ...
        # Remove events without any track
        trk_valid = ak.num(array_TRK['demfit','time'], axis=1) > 0
        mask_trk = ak.mask(array_TRK, trk_valid)
        crv_coinc = np.ones(ak.num(array_TRK, axis=0), dtype='bool')    # Create a boolean array of True for the mask
        for evt_idx, evt in enumerate(array_TRK['demfit','time']):   # loop through every event
            for trk_idx, trk in enumerate(evt):     # loop through every track within an event
                if ak.num(trk, axis=0) > 0:       # rare case when an event has 2 tracks, with one being empty (Ask why?)
                    for crv_idx, crv in enumerate(array_CRV['crvcoincs.time', evt_idx]):    # loop through CRV hits
                        if np.abs(trk[0] - crv) < cut_value:
                            crv_coinc[evt_idx] = False

        return crv_coinc

