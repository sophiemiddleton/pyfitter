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
        self.Track_cut = {} # Cut applied at the track level
        self.Trksegs_cut = {}# Cut applied at the trksegs level
        self.CRV_cut = {}   # Cut applied based on CRV coincidence
        self.MC_cut = {}    # Cut applied on the MC data
        if opt == 'MDC2024': #TODO these are SU2020 cuts, and some are missing....
            self.Event_cut = {
            }
            self.Track_cut = {
                "'trk','trk.nactive'" : [20, float('inf')] # active hits in the tracker
            }
            self.Trksegs_cut = {
                "'trksegs','sid'" : 0,
                "'trksegs','time'" : [640., 1650], #inTimeWindow
                "'trksegpars_lh','t0err'" : [0, 0.9], #intimeErr
                "'trksegpars_lh','maxr'" : [450., 680.], #inMaxRCut
                #"'trkqual.result'" : [0.2, float('inf')], # trk qual
            }
            self.CRV_cut = {
                "crv_coincidence" : 150. # cut anything with a track - crv time difference less than this
            }

    def ApplyCut(self, array_trk, array_crv):
        """ function applies cuts to MDC2024 trkana """
        print("Applying cuts\n")
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
            array_trk[eval(key)].show()
            array_cut[eval(key)].show()
            #array_trksegs_cut = array_cut['trksegs'].mask[mask]
            #array_trksegpars_lh_cut = array_cut['trksegpars_lh'].mask[mask]
            #array_cut['trksegs'] = array_trksegs_cut
            #array_cut['trksegpars_lh'] = array_trksegpars_lh_cut

        # Track segments level cut
        for key, value in self.Trksegs_cut.items():
            print(eval(key))

            if type(value) == int:
                mask = array_trk[eval(key)] == value
            elif len(value) == 2:
                mask = (array_trk[eval(key)] >= value[0]) & (array_trk[eval(key)] <= value[1])

            print("# of tracks passing this cut: ", ak.sum(mask))
            
            ApplyMaskTrksegs(array_cut, mask)
            mask.show()
            array_trk[eval(key)].show()
            array_cut[eval(key)].show()

        # CRV cuts
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

        #print("# ofevents after all the cuts: ", ak.num(array_trk, axis=0)
        print("# of tracks after all the cuts: ", ak.count(array_cut['trksegs','time']))

        return array_cut

def ApplyMaskTrk(array_cut, mask):
    """ Apply the mask onto each branch of the array """
    for branch in ak.fields(array_cut):
        for leaf in ak.fields(array_cut[branch]):
            array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask]

    return array_cut


def ApplyMaskTrksegs(array_cut, mask):
    """ Apply the mask onto each branch of the array """
    for branch in ak.fields(array_cut):
        if (branch == 'trksegs') or (branch == 'trksegpars_lh'):
            for leaf in ak.fields(array_cut[branch]):
                array_cut[branch,leaf] = array_cut[branch,leaf].mask[mask]

    return array_cut


