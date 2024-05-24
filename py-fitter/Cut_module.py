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
                ('demlh','t0') : [700., 1695],    #inTimeWindow
                ('demlh','t0err') : [0,0.9], #intimeErr
                ('demlh','maxr') : [450., 680.], #inMaxRCut
                ('demtrkqual','result') : [0.2,1],   # trk qual
                ( 'dem', 'nactive') : [20.,1000.]
                #('demfit','mom') : [95., 115.]   #recomom --> done in fitter
            }
    def ApplyCut_MDC2024_mom(self, array_MC):
        """ function applies cuts to MDC2024 trkana --> work in progress!!!!"""
        array_MC['demfit_mom'] = np.sqrt((array_MC['demfit']['mom']['fCoordinates']['fX'])**2 + (array_MC['demfit']['mom']['fCoordinates']['fY'])**2 + (array_MC['demfit']['mom']['fCoordinates']['fZ'])**2)
        trk_ent_mask = (array_MC['demfit']['sid']==0)
        time_cut_mask = (array_MC['demlh']['t0']>=650)
        timeerr_cut_mask = (array_MC['demlh']['t0err']<0.9)
        maxr_cut_mask = (array_MC['demlh']['maxr']>450.)
        #active_cut_mask = (array_MC['dem']['nactive']>20) TODO - different type of array
        #trkqual_cut_mask = (array_MC['demtrkqual']['result']>0.2) TODO - different type of array
        data_np = np.array(ak.flatten(array_MC[(trk_ent_mask) & (time_cut_mask) & (timeerr_cut_mask) & (maxr_cut_mask)  ]['demfit_mom'], axis=None))
        return data_np

    def ApplyCRVCut_MDC2024(self, array):
        ## TODO:
        print("TODO: cuts not currently being applied!!!!\n")

    def ApplyCut(self, array):
        """ function applies cuts to MDC2018 trkana """
        array_cut = array
        print("Applying cuts\n")
        for key, value in self.Cut_List.items():
            print(key)

            if self.use_CRV == True:
                n_event = ak.num(array_cut, axis=0)
                condition = np.full(n_event, True)
                for i in range(n_event):
                    bestcrv = array_cut['bestcrv',i]
                    if bestcrv >= 0:
                        if (array_cut['de','t0',i] - array_cut['crvinfo._timeWindowStart',i,bestcrv] > value[0]) & (array_cut['de','t0',i] - array_cut['crvinfo._timeWindowStart',i,bestcrv] < value[1]):
                            condition[i] = False
                array_cut = array_cut[condition]

            else:
                array_cut = array_cut[(array_cut[key] > value[0]) & (array_cut[key] <= value[1])]

        print('array size before cuts', ak.num(array, axis=0))
        print('array size after cuts', ak.num(array_cut, axis=0))

        return array_cut
