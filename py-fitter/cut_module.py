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
                "demfit.time" : [640., 1650], #inTimeWindow
                "demlh.t0err.max" : [0.9], #intimeErr
                "demlh.maxr" : [450., 680.], #inMaxRCut
                "demtrkqual.result" : [0.2], # trk qual
                "dem.nactive" : [20], # active hits in fit
                "time_diff" : [50.,150.] # cut anything with a track - crv time difference less than this
            }

    def ApplyCut_mom(self, demfits, trkqual, demhits, crvcoin = None):
        """ function applies cuts to MDC2024 trkana """

        # find magnitude
        demfits['demfit_mom'] = np.sqrt((demfits['demfit']['mom']['fCoordinates']['fX'])**2 + (demfits['demfit']['mom']['fCoordinates']['fY'])**2 + (demfits['demfit']['mom']['fCoordinates']['fZ'])**2)

        # select trak fit at entrance
        trk_ent_mask = (demfits['demfit']['sid']==0)

        # build masks for cuts
        time_cut_mask_min = (demfits['demfit']['time'] >= self.Cut_List["demfit.time"][0])
        time_cut_mask_max = (demfits['demfit']['time'] < self.Cut_List["demfit.time"][1])
        timeerr_cut_mask = (demfits['demlh']['t0err'] < self.Cut_List["demlh.t0err.max"][0])
        maxr_cut_mask = (demfits['demlh']['maxr'] > self.Cut_List["demlh.maxr"][0])
        active_mask = (demhits['dem.nactive'] > self.Cut_List["dem.nactive"][0])

        # apply trkqual cut # FIXME - commented here due to TrkAna v5 issue
        # trkqual_mask = (trkqual['demtrkqual.result'] > self.Cut_List["demtrkqual.result"][0])

        # look for CRV coincidences
        crv_cut_mask = self.ApplyCRVCut(demfits, crvcoin, self.Cut_List['time_diff'][1])

        # apply all cuts and convert to numpy array
        data_cut = demfits[(active_mask) & (crv_cut_mask) & (trk_ent_mask) & (time_cut_mask_max) & (time_cut_mask_min) & (timeerr_cut_mask) & (maxr_cut_mask)]
        return data_cut

    def ApplyCRVCut(self, demfits, crvcoin, cut_value):
        """ function applies time based cut on comparison of TRK and CRV times """
        # Remove events without any track
        trk_valid = ak.num(demfits['demfit','time'], axis=1) > 0
        mask_trk = ak.mask(demfits, trk_valid)
        crv_coinc = np.ones(ak.num(demfits, axis=0), dtype='bool')    # Create a boolean array of True for the mask
        for evt_idx, evt in enumerate(demfits['demfit','time']):   # loop through every event
            for trk_idx, trk in enumerate(evt):     # loop through every track within an event
                if ak.num(trk, axis=0) > 0:       # rare case when an event has 2 tracks, with one being empty (Ask why?)
                    for crv_idx, crv in enumerate(crvcoin['crvcoincs.time', evt_idx]):    # loop through CRV hits
                        if np.abs(trk[0] - crv) < cut_value:
                            crv_coinc[evt_idx] = False
        return crv_coinc
