# Cut class
# Apply the selection cuts on the data listed as a pandas dataframe

import awkward as ak
import numpy as np

class CutClass:

    def __init__(self, opt = 'su2020', use_CRV = True):
        self.use_CRV = use_CRV
        self.Cut_List = {}

        if opt == 'su2020': # TODO make sure you are using SU2020 cuts
            self.Cut_List = {
                ('de','status') : [0., float('inf')] ,  #goodfit
                'trigbits' : [0., float('inf')], #triggeredbits0x208
                ('de','t0') : [700., 1695],    #inTimeWindow
                ('deent','td') : [0.577350, 1.000],    #inTanDipCut
                ('deent','d0') : [-80., 105.], #inD0Cut
                'inMaxRCut' : [450., 680.], #inMaxRCut
                'noCRVHit' : [-50.0 , 150.0],   # rejection time window around a crv hit
                ('dequal','TrkQual') : [0.8, float('inf')],  #TrkQual
                ('dequal','TrkPID') : [0.95, float('inf')],  #TrkPID
                ('ue','status') : [float('-inf'), 0.],   #noUpstream
                ('deent','mom') : [95., float('inf')]   #recomom
                #('deent','mom') : [95., 115.]   #FIXME changed to have fixed bin size in plot
            }

    def ApplyCut(self, array):
        array_cut = array
        print("Applying cuts\n")
        for key, value in self.Cut_List.items():
            print(key)

            if key == 'noCRVHit':
                if self.use_CRV == True:
                    n_event = ak.num(array_cut, axis=0)
                    condition = np.full(n_event, True)
                    for i in range(n_event):
                        bestcrv = array_cut['bestcrv',i]
                        if bestcrv >= 0:
                            if (array_cut['de','t0',i] - array_cut['crvinfo._timeWindowStart',i,bestcrv] > value[0]) & (array_cut['de','t0',i] - array_cut['crvinfo._timeWindowStart',i,bestcrv] < value[1]):
                                condition[i] = False
                    array_cut = array_cut[condition]

            elif key == "trigbits":
                array_cut = array_cut[(array_cut['trigbits']&0x208) > 0]

            elif key == "inMaxRCut":
                array_cut = array_cut[(array_cut['deent','d0'] + 2/array_cut['deent','om'] > value[0]) & (array_cut['deent','d0'] + 2/array_cut['deent','om'] < value[1])]

            else:
                array_cut = array_cut[(array_cut[key] > value[0]) & (array_cut[key] <= value[1])]

        print('array size before cut', ak.num(array, axis=0))
        print('array size after cut', ak.num(array_cut, axis=0))

        return array_cut
