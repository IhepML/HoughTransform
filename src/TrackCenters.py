# coding: utf-8

import numpy as np
from math import *

CDC_R_MAX = 80
CDC_R_MIN = 50
R_MAX = 39
R_MEAN = 33
R_MIN = 28
TRK_RHO_SGMA = 3
TRGT_RHO = 10
RHO_BINS = 20


# divide potential track center area into small bins
class TrackCenters(object):
    def __init__(self):
        # signals.r.max,signals.r.min = 80.199996,53.0
        self.r_max = R_MAX
        self.r_min = R_MIN
        self.trk_rho_sgma = TRK_RHO_SGMA
        self.r_mean = R_MEAN

        self.r_max_center = CDC_R_MAX - R_MIN
        self.r_min_center = CDC_R_MIN - R_MAX  # max(r_max - trgt_rho, cdc_rho_min - r_max)

        self.rho_bins = RHO_BINS
        self.n_layer = np.arange(RHO_BINS)
        self.r_by_layer = np.zeros(self.rho_bins, dtype="float")
        self.n_by_layer = np.zeros(self.rho_bins, dtype="int")
        self.rho_gap = (self.r_max_center - self.r_min_center) / (self.rho_bins - 1)
        self.arc_gap = self.rho_gap

        # assign values of r,n to each layer
        self.r_by_layer = [self.r_min_center + self.n_layer[n] * self.rho_gap for n in self.n_layer]
        self.n_by_layer = [int(2 * pi * self.r_by_layer[n] / self.arc_gap) for n in self.n_layer]

        # assign values of r,phi to each point
        self.n_points = np.arange(np.sum(self.n_by_layer))
        self.rho_points = np.zeros(np.sum(self.n_by_layer), dtype="float")
        self.phi_points = np.zeros(np.sum(self.n_by_layer), dtype="float")

        self.rho_points = np.repeat(self.r_by_layer, self.n_by_layer)

        self.n_first = np.zeros(self.rho_bins, dtype="int")
        for i in self.n_layer:
            self.n_first[i] = np.sum(self.n_by_layer[:i])
        self.n_points_on_layer = self.n_points - np.repeat(self.n_first, self.n_by_layer)
        self.phi_points = self.n_points_on_layer * self.arc_gap / self.rho_points

        # assign values of x,y to each point
        self.x_points = self.rho_points * np.cos(self.phi_points)
        self.y_points = self.rho_points * np.sin(self.phi_points)
        self.xy_points = np.column_stack((self.x_points, self.y_points))
        # print(self.n_points)
