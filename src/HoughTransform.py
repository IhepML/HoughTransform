# coding: utf-8
'''
import sys
sys.path.insert(0, '.')
'''
from TrackCenters import *
import numpy as np
from scipy.spatial.distance import cdist

XY_NAME = ['xe0', 'ye0']


# perform Hough Transform
class HoughTransform(object):

    def __init__(self, data_sig, data_bak, y_pre, xy_name=XY_NAME):
        hf = TrackCenters()
        r_max = hf.r_max
        r_min = hf.r_min
        trk_rho_sgma = hf.trk_rho_sgma

        xy_sig = data_sig[xy_name]
        xy_bak = data_bak[xy_name]
        xy_hits = np.concatenate((xy_sig, xy_bak), axis=0)

        n_of_points = np.sum(hf.n_by_layer)

        dist = np.zeros([len(xy_hits), n_of_points])
        result = np.zeros([len(xy_hits), n_of_points])
        weight = np.transpose([y_pre])
        vt_points = np.zeros(n_of_points)

        # vote on track centers/points
        dist = cdist(xy_hits, hf.xy_points)
        result = np.where(dist <= r_max + trk_rho_sgma, dist, 0)
        result = np.where(result >= r_min - trk_rho_sgma, 1, 0)
        vote_table = result * weight

        self.vt_points_sig = np.sum(result[:len(data_sig)], axis=0)  # ##################
        max_vpon_sig = np.amax(self.vt_points_sig)
        min_vpon_sig = np.amin(self.vt_points_sig)
        self.vt_points_sig = (self.vt_points_sig - min_vpon_sig) / (max_vpon_sig - min_vpon_sig)  # normalized

        vt_points = vote_table.sum(axis=0)
        max_vpon = np.amax(vt_points)
        min_vpon = np.amin(vt_points)
        # print(max_vpon, min_vpon)
        vt_points = (vt_points - min_vpon) / (max_vpon - min_vpon) * 15
        self.vt_points = vt_points / 15  # normalized

        # vote on signals and backgrounds
        wet_points = np.exp(vt_points)
        self.vt_hits = np.sum(result * wet_points, axis=1)
        vt_max = np.amax(self.vt_hits)
        vt_min = np.amin(self.vt_hits)
        self.vt_hits = (self.vt_hits - vt_min) / (vt_max - vt_min)  # normalized
        self.vt_sigs = self.vt_hits[:len(data_sig)]
        self.vt_baks = self.vt_hits[len(data_sig):]
