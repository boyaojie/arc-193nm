# identifiability.py
# 可辨识性分析：FIM + Sobol

import numpy as np
from scipy.optimize import approx_fprime
from config import (WLS, WL_TARGET, LAYERS_7, N_LAF3, N_MGF2,
                    R_CURV, R_APERTURE, N_R_SAMPLES)
from forward import sphere_average_reflectance


# ============================================================
# 参数向量
# p = [d1, d2, d3, d4, d5, d6, d7, n_LaF3, n_MgF2, R]
# ============================================================

PARAM_NAMES = [f"d{i+1}" for i in range(7)] + ["n_LaF3", "n_MgF2", "R"]


def params_to_layers(p):
    """p[0:7]=厚度, p[7]=n_LaF3, p[8]=n_MgF2"""
    d = p[0:7]
    n_la = p[7]
    n_mg = p[8]
    materials = [n_la, n_mg, n_la, n_mg, n_la, n_mg, n_la]
    return [(materials[i], d[i]) for i in range(7)]


def spectrum_from_params(p, wls=WLS, n_r=N_R_SAMPLES, pol='avg'):
    """p = [d1..d7, n_LaF3, n_MgF2, R]"""
    layers = params_to_layers(p)
    R = p[9]
    spec, _ = sphere_average_reflectance(wls, layers, R,
                                         r_max=R_APERTURE, n_r=n_r,
                                         pol=pol)
    return spec


def get_initial_params():
    """从[3]的Table 5取初始参数"""
    d0 = np.array([d for n, d in LAYERS_7])
    return np.concatenate([d0, [N_LAF3, N_MGF2, R_CURV]])


# ============================================================
# Jacobian
# ============================================================

def compute_jacobian(p0, wls=WLS, n_r=N_R_SAMPLES, eps=1e-4, pol='avg'):
    """计算 Jacobian J = dR/dp"""
    def f(p):
        return spectrum_from_params(p, wls, n_r, pol)
    J = approx_fprime(p0, f, epsilon=eps)
    return J


# ============================================================
# Fisher信息矩阵
# ============================================================

def compute_fim(J, sigma=0.01):
    """FIM = J^T Sigma^-1 J"""
    F = J.T @ J / sigma ** 2
    return F


def fim_metrics(F):
    """条件数、CRLB、特征值"""
    eigvals = np.linalg.eigvalsh(F)
    cond = eigvals.max() / (eigvals.min() + 1e-30)
    F_inv = np.linalg.pinv(F)
    CRLB = np.diag(F_inv)
    return eigvals, cond, CRLB


def unidentifiable_direction(F):
    """最不可辨识方向（最小特征值对应特征向量）"""
    eigvals_s, eigvecs = np.linalg.eigh(F)
    v_min = eigvecs[:, 0]
    return v_min, eigvals_s


# ============================================================
# Sobol（用SALib）
# ============================================================

def sobol_analysis(problem, param_values, Y, calc_second_order=False):
    """Sobol分析"""
    from SALib.analyze import sobol
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                       print_to_console=False)
    return Si