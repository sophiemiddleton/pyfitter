# Cut class
# Apply the selection cuts on the data listed as a pandas dataframe

import awkward as ak
import numpy as np

class CutClass:

    def __init__(self,  opt = 'MDC2024', use_CRV = True):
        self.use_CRV = use_CRV
        self.Cut_List = {}
        if opt == 'MDC2024': #TODO these are SU2020 cuts, and some are missing....
            self.Cut_List = {
                ('demfit','time') : [640., 1650],    #inTimeWindow
                ('demlh','t0err') : [0,0.9], #intimeErr
                ('demlh','maxr') : [450., 680.], #inMaxRCut
                ('demtrkqual','result') : [0.2,1],   # trk qual
                ( 'dem', 'nactive') : [20.,1000.],
                ( 'TRK_CRV','time') : [150.] # cut anything with a track - crv time difference less than this
            }
    def ApplyCut_MDC2024_mom(self, array_TRK, array_TRKQUAL, array_CRV = None):
        """ function applies cuts to MDC2024 trkana --> work in progress!!!!"""
        array_TRK['demfit_mom'] = np.sqrt((array_TRK['demfit']['mom']['fCoordinates']['fX'])**2 + (array_TRK['demfit']['mom']['fCoordinates']['fY'])**2 + (array_TRK['demfit']['mom']['fCoordinates']['fZ'])**2)
        trk_ent_mask = (array_TRK['demfit']['sid']==0)
        time_cut_mask_min = (array_TRK['demfit']['time']>=640)
        time_cut_mask_max = (array_TRK['demfit']['time']<1650)
        timeerr_cut_mask = (array_TRK['demlh']['t0err']<0.9)
        maxr_cut_mask = (array_TRK['demlh']['maxr']>450.)

        if (array_CRV == None):
            trkqual_mask = (array_TRKQUAL['result'] > 0.2) # FIXME - this should apply in both cases, here due to TrkAna v5 issue
            data_np = np.array(ak.flatten(array_TRK[ (trkqual_mask ) & (trk_ent_mask) & (time_cut_mask_max) & (time_cut_mask_min) & (timeerr_cut_mask) & (maxr_cut_mask)  ]['demfit_mom'], axis=None))
            return data_np
        else:
            array_CRVTRKTimes = self.ApplyCRVCut(array_TRK, array_CRV)
            crv_cut_mask = (array_CRVTRKTimes['time_diff'] > 150)
            data_np = np.array(ak.flatten(array_TRK[(crv_cut_mask) & (trk_ent_mask) & (time_cut_mask_max) & (time_cut_mask_min) & (timeerr_cut_mask) & (maxr_cut_mask)  ]['demfit_mom'], axis=None))
            return data_np

    def ApplyCRVCut(self, array_TRK, array_CRV):
        """ function applies time based cut on comparison of TRK and CRV times """ # FIXME this is a bit hacky ...
        TRK_CRV_Times_all = []
        for i, j in enumerate(array_TRK):
            trktime = ak.firsts(ak.firsts(array_TRK['demfit']['time'])) # for SID == 0
            maxCRVtime = ak.max(array_CRV["crvcoincs.time"][i])
            time_diff = 0
            if(trktime[i] != None and maxCRVtime!=None):
                time_diff = trktime[i] - maxCRVtime
            TRK_CRV_Times_all.append({"time_diff" :time_diff})
        TRK_CRV_Times_awk = ak.Array(TRK_CRV_Times_all)
        return TRK_CRV_Times_awk
