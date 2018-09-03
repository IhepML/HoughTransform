# coding: utf-8
'''
import sys
sys.path.insert(0, '.')
'''
from TrackCenters import TrackCenters
from HoughTransform import HoughTransform
import Plot
import random
import numpy as np
import matplotlib.pyplot as plt


class GetHfEachEvent(object):
    def __init__(self, sig_samp, bak_samp, y_pre_samp):
        evt_id = sig_samp['iev'].drop_duplicates()
        evt = random.sample(evt_id.tolist(), 4)

        y_pre_sig_samp = y_pre_samp[:len(sig_samp)]
        y_pre_bak_samp = y_pre_samp[len(sig_samp):]

        pos_sig = 0
        pos_bak = 0
        self.hf_scr_sig = []
        self.hf_scr_bak = []
        for evt_id in evt_id:
            data_sig = sig_samp[sig_samp.iev == evt_id]
            data_bak = bak_samp[bak_samp.iev == evt_id]
            y_pre_sig = y_pre_sig_samp[pos_sig:pos_sig + len(data_sig)]
            y_pre_bak = y_pre_bak_samp[pos_bak:pos_bak + len(data_bak)]
            pos_sig += len(data_sig)
            pos_bak += len(data_bak)
            wet = np.concatenate((y_pre_sig, y_pre_bak), axis=0)
            hf = HoughTransform(data_sig, data_bak, wet)
            if evt_id in evt:
                Plot.putout(hough=TrackCenters(), signals=data_sig, backgrounds=data_bak, trackcenter=False,
                            circlebysig=False, circlebytrackcenter=True, tkctrbywt=True,
                            backgrounds_=False, backgroundsbywt=True, signals_=False,
                            signalsbywt=True, vt_sig=hf.vt_sigs, vt_bak=hf.vt_baks,
                            vt_points=hf.vt_points, vt_points_sig=hf.vt_points_sig)
            self.hf_scr_sig = self.hf_scr_sig + hf.vt_sigs.tolist()
            self.hf_scr_bak = self.hf_scr_bak + hf.vt_baks.tolist()
        self.hf_scr = self.hf_scr_sig + self.hf_scr_bak
        # print(self.hf_scr[:10])
        plt.hist(self.hf_scr_sig, 100, density=True, alpha=0.5, label="signals")
        plt.hist(self.hf_scr_bak, 100, density=True, alpha=0.5, label="backgrounds")
        plt.legend(prop={'size': 10})
        plt.xlabel("votes")
        plt.ylabel("rate")
        plt.show()
