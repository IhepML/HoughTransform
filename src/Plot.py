# coding: utf-8
"""将事例的横截面画出"""

import matplotlib.pyplot as plt
import numpy as np


class Cylinder(object):
    """
    define the cylindrical array of points read in from positional information.
    It returns a flat enumerator of the points in the array.

    param:
    r_max  the maximal radius of the geometry
    r_min  the minimal radius of the geometry
    """
    def __init__(self, r_max=None, r_min=None, rho_bins=None, point_x=None, point_y=None, layer_id=None, arc_res=0):
        self.r_max = r_max
        self.r_min = r_min
        self.rho_bins = rho_bins
        self.arc_res = arc_res
        self.point_x = point_x
        self.point_y = point_y
        self.point_layers = layer_id
        _, self.n_by_layer = np.unique(layer_id, return_counts=True)
        self.first_point = self._get_first_point(self.n_by_layer)
        self.n_points = sum(self.n_by_layer)

    def _get_first_point(self, n_by_layer):
        """
        Returns the point_id of the first point in each layer
        param:
        n_by_layer: the number of points in each layer

        return:
        numpy array of first point in each layer
        """
        first_point = np.zeros(len(n_by_layer), dtype=int)
        for i in range(len(n_by_layer)):
            first_point[i] = sum(n_by_layer[:i])
        return first_point

    def get_rhos_and_phis(self):
        """
        Returns the position of each point in radial system

        return:
        pair of numpy.arrays of shape [n_points],
         - first one contains rho's(radii)
         - second one contains phi's(angles)
        """
        # the distance between layers
        drho = (self.r_max - self.r_min) / (self.rho_bins - 1)
        # a array of radii on each layer
        r_track_cent = [self.r_min + drho * n for n in range(self.rho_bins)]
        if self.arc_res == 0:
            self.arc_res = drho
        n_by_layer = [int(2 * np.pi * r_track_cent[n] / self.arc_res) for n in range(self.rho_bins)]
        # the phi between two bins which are on the same layer
        dphi_by_layer = [self.arc_res / r_track_cent[n] for n in range(self.rho_bins)]
        rho_by_points = []
        phi_by_layer = []
        for i in range(self.rho_bins):
            point_phi_n = [dphi_by_layer[i] * n for n in range(n_by_layer[i])]
            a = np.ones((len(point_phi_n), 1), dtype='float')
            rho_by_points.extend(a * r_track_cent[i])
            phi_by_layer.extend(point_phi_n)
        return rho_by_points, phi_by_layer

    def get_points_rho_and_phi(self):
        rs = []
        phis = []
        for point_x, point_y in zip(self.point_x, self.point_y):
            if point_x == 0:
                if point_y > 0:
                    phi = np.pi / 2
                else:
                    phi = 3 * np.pi / 2
            elif point_x < 0:
                phi = np.arctan(point_y / point_x) + np.pi
            else:
                phi = np.arctan(point_y / point_x)
            r = np.sqrt(point_x ** 2 + point_y ** 2)
            rs.append(r)
            phis.append(phi)
        return rs, phis


def plot_add_circle(x, y, radius, color="#32CD32", lw=1, center_weight=None, spread=0, l_alpha=0.8,
                    s_alpha=0.025, fill=False, edgecolor='#32CD32', **kwargs):
    """
    Add a circle to our plot

    :param x:        x location of circle centre
    :param y:        y location of circle centre
    :param radius:   radius of circle
    :param color:    color of circle
    :param lw:       line width of circle
    :param spread:   spread of circle, symmetric
    :param l_alpha:  overall normalization on weight of line
    :param s_alpha:  overall normalization on weight of spread
    """
    # TODO check gca() usage here
    if center_weight is not None:
        lw = center_weight
    plot_circle = plt.Circle((x, y), radius, transform=plt.gca().transData._b,
                             color=color, fill=fill, alpha=l_alpha, lw=lw, edgecolor=edgecolor, **kwargs)
    plt.gca().add_artist(plot_circle)


def putout(hough=None, signals=None, backgrounds=None, trackcenter=False, circlebysig=True,
           circlebytrackcenter=False, tkctrbywt=True, backgrounds_=False, backgroundsbywt=True,
           signals_=False, signalsbywt=True, vt_sig=None, vt_bak=None, vt_points=None, vt_points_sig=None):
    # draw the cells
    r_max = max(max(backgrounds.r), max(signals.r))
    r_min = min(min(backgrounds.r), min(signals.r))
    r_sig = signals.r
    r_bkg = backgrounds.r
    r_c = list(r_sig)
    r_c.extend(list(r_bkg))
    rho_bins = len(np.unique(r_c))

    vt_sig_nom = vt_sig * 15
    vt_bak_nom = vt_bak * 15
    vt_bak_nom = np.exp(vt_bak_nom)
    vt_sig_nom = np.exp(vt_sig_nom)
    max_vt = max(np.amax(vt_sig_nom), np.amax(vt_bak_nom))
    min_vt = max(np.amin(vt_sig_nom), np.amin(vt_bak_nom))
    vt_sig_nom = (vt_sig_nom - min_vt) / (max_vt - min_vt) * 25
    vt_bkg_nom = (vt_bak_nom - min_vt) / (max_vt - min_vt) * 25

    for _ in range(2):
        # event_id = random.randint(0, max_event_id)

        # draw the wire cells
        track = Cylinder(r_max=r_max, r_min=r_min, rho_bins=rho_bins)
        rho_by_points, phi_by_points = track.get_rhos_and_phis()
        fig = plt.figure(figsize=(8, 8))
        axs = fig.add_subplot(111, projection='polar')
        axs.scatter(phi_by_points, rho_by_points, s=1, c='g', alpha=0.3, marker='.')

        # draw the track centers
        if trackcenter:
            axs.scatter(hough.phi_points, hough.rho_points, s=1, c='#FF8C00', alpha=1, marker='.', zorder=10)

        # draw the circles centered by the signals
        if circlebysig:
            x_s = list(signals.xe0)
            y_s = list(signals.ye0)
            for i in range(len(x_s)):
                plot_add_circle(x_s[i], y_s[i], hough.r_min)

        # draw the track centers by weight
        if tkctrbywt:
            axs.scatter(hough.phi_points, hough.rho_points, s=vt_points, c='#FF8C00', alpha=1, marker='o', zorder=10)

        # draw the circles centered by trackcenters which are  weighted
        if circlebytrackcenter:
            vt_points = np.where(vt_points >= 0.6, vt_points, 0)
            for i in range(len(hough.x_points)):
                plot_add_circle(hough.x_points[i], hough.y_points[i], hough.r_min,
                                center_weight=vt_points[i])

        # draw backgrounds first
        if backgrounds_ or backgroundsbywt:
            track = Cylinder(point_x=backgrounds.xe0, point_y=backgrounds.ye0)
            r_bkg0, phi_bkg0 = track.get_points_rho_and_phi()
            track = Cylinder(point_x=backgrounds.xe1, point_y=backgrounds.ye1)
            r_bkg1, phi_bkg1 = track.get_points_rho_and_phi()
            if backgrounds_:
                r_bkg = [r_bkg0, r_bkg1]
                phi_bkg = [phi_bkg0, phi_bkg1]
                axs.plot(phi_bkg, r_bkg, '-', c='r', lw=1, alpha=0.5)
            if backgroundsbywt:
                axs.scatter(phi_bkg0, r_bkg0, s=15, c='', alpha=0.3, edgecolors='r',
                            marker='o', zorder=10)
                axs.scatter(phi_bkg0, r_bkg0, s=vt_bkg_nom, c='r', alpha=1, marker='.', zorder=10)

        # draw the signals
        if signals_ or signalsbywt:
            track = Cylinder(point_x=signals.xe0, point_y=signals.ye0)
            r_sig0, phi_sig0 = track.get_points_rho_and_phi()
            track = Cylinder(point_x=signals.xe1, point_y=signals.ye1)
            r_sig1, phi_sig1 = track.get_points_rho_and_phi()
            if signals_:
                r_sig = [r_sig0, r_sig1]
                phi_sig = [phi_sig0, phi_sig1]
                axs.plot(phi_sig, r_sig, '-', c='b', lw=1, alpha=0.5)
            if signalsbywt:
                axs.scatter(phi_sig0, r_sig0, s=15, c='', alpha=0.3, edgecolors='b',
                            marker='o', zorder=10)
                axs.scatter(phi_sig0, r_sig0, s=vt_sig_nom, c='b', alpha=1, marker='.', zorder=10)

        axs.grid(True, linestyle="--", alpha=0.5)
        axs.set_rgrids([0, 50, 83])
        axs.set_rlim(0, 83)
        plt.show()
        plt.close()
        pass
