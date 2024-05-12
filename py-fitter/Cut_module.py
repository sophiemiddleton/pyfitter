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
    def ApplyCut_MDC2024(self, array):
        """ function applies cuts to MDC2024 trkana --> work in progress!!!!"""
        print("TODO: cuts not currently being applied!!!!\n")
        array_cut = array
        #array['demfit_mom'] = np.sqrt((array['demfit']['mom']['fCoordinates']['fX'])**2 + (array['demfit']['mom']['fCoordinates']['fY'])**2 + (array['demfit']['mom']['fCoordinates']['fZ'])**2)
        #trk_ent_mask = (array_cut['demfit']['sid']==0)
        #demlh_cut_mask = (array_cut['demlh']['t0'] >= 1000) #& (array_cut['demlh']['t0err'] < 0.9) & (array_cut['demlh']['maxr'] < 680 )#& array_cut['demtrkqual']['result'] > 0.2)
        #array_cut = array_cut[(trk_ent_mask) & (demlh_cut_mask)]
        #ak.flatten(array_cut[(trk_ent_mask) & (demlh_cut_mask)], axis=0)
        print('array size before cuts', ak.num(array, axis=0))
        print('array size after cuts', ak.num(array_cut, axis=0))

        return array_cut

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
