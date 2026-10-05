# forward.py
# 正向模型：TMM + 球面几何
# 支持s/p偏振平均

import numpy as np
import tmm
from config import (N_SUB, N_INC, R_APERTURE, N_R_SAMPLES,
                    R_CURV, THETA_DEP)


def incident_angle(r, R=R_CURV):
    """入射角 (rad)"""
    return np.arcsin(np.clip(r / R, 0, 0.999))


def thickness_deviation(r, R=R_CURV, theta_dep_deg=THETA_DEP):
    """膜厚偏差因子 δ(r)"""
    theta = incident_angle(r, R)
    theta_dep = np.radians(theta_dep_deg)
    delta = np.cos(theta_dep) / np.cos(theta + theta_dep)
    return np.clip(delta, 1.0, 1.1)


def reflectance_sphere(wls, layers, R, r, pol='avg',
                       n_sub=N_SUB, n_inc=N_INC,
                       use_deviation=True):
    """
    球面上半径r处的反射率
    pol: 's' | 'p' | 'avg'
    """
    theta = incident_angle(r, R)
    d_scale = thickness_deviation(r, R) if use_deviation else 1.0

    n_list = [n_inc] + [n for n, d in layers] + [n_sub]
    d_list = [np.inf] + [d * d_scale for n, d in layers] + [np.inf]

    R_arr = np.zeros(len(wls))
    for i, wl in enumerate(wls):
        if pol == 'avg':
            R_s = tmm.coh_tmm('s', n_list, d_list, theta, wl)['R']
            R_p = tmm.coh_tmm('p', n_list, d_list, theta, wl)['R']
            R_arr[i] = (R_s + R_p) / 2
        else:
            R_arr[i] = tmm.coh_tmm(pol, n_list, d_list, theta, wl)['R']
    return R_arr


def reflectance_planar(wls, layers, pol='avg',
                       n_sub=N_SUB, n_inc=N_INC):
    """平面反射率（法向入射，s=p）"""
    n_list = [n_inc] + [n for n, d in layers] + [n_sub]
    d_list = [np.inf] + [d for n, d in layers] + [np.inf]

    R_arr = np.zeros(len(wls))
    for i, wl in enumerate(wls):
        R_arr[i] = tmm.coh_tmm('s', n_list, d_list, 0, wl)['R']
    return R_arr


def sphere_average_reflectance(wls, layers, R,
                               r_max=R_APERTURE,
                               n_r=N_R_SAMPLES, pol='avg'):
    """球面平均反射率（用于优化目标）"""
    rs = np.linspace(0, r_max, n_r)
    spectra = np.array([
        reflectance_sphere(wls, layers, R, r, pol=pol)
        for r in rs
    ])
    return spectra.mean(axis=0), spectra


def layers_to_arrays(layers):
    n_arr = np.array([n for n, d in layers])
    d_arr = np.array([d for n, d in layers])
    return n_arr, d_arr