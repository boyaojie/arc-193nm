# curvature.py
# 实验3：曲率对可辨识性的影响
# 生成 Figure 3
# A组：固定边缘角（44.4°），R=20-150 mm，r_max=0.7R
# B组：固定孔径（105 mm），R=150-1000 mm，边缘角随R减小

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

from config import (WLS, WL_TARGET, LAYERS_7, N_LAF3, N_MGF2,
                    R_APERTURE, FIG_DIR)
from forward import sphere_average_reflectance
from identifiability import (compute_fim, unidentifiable_direction,
                             PARAM_NAMES)

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
})

# ============================================================
# 全局
# ============================================================
WLS_SUB = WLS[::2]  # 200波长
d0 = np.array([d for n, d in LAYERS_7])


def compute_metrics(R, r_max, wls=WLS_SUB, n_r=3, eps=1e-4):
    """在指定R和r_max下计算FIM指标"""
    p0 = np.concatenate([d0, [N_LAF3, N_MGF2, R]])

    def f(p):
        layers = [(p[7] if i % 2 == 0 else p[8], p[i])
                  for i in range(7)]
        spec, _ = sphere_average_reflectance(
            wls, layers, p[9], r_max=r_max, n_r=n_r, pol='avg')
        return spec

    J = approx_fprime(p0, f, epsilon=eps)
    F = compute_fim(J, sigma=0.01)

    # 正则化
    F_reg = F + 1e-12 * np.eye(F.shape[0]) * np.trace(F) / F.shape[0]

    eigvals = np.linalg.eigvalsh(F_reg)
    cond = eigvals.max() / eigvals.min()
    F_inv = np.linalg.pinv(F_reg)
    CRLB = np.diag(F_inv)

    eigvals_s, eigvecs = np.linalg.eigh(F_reg)
    v_min = eigvecs[:, 0]

    theta_edge = np.degrees(np.arcsin(np.clip(r_max / R, 0, 0.999)))

    return {
        'R': R,
        'r_max': r_max,
        'theta_edge': theta_edge,
        'cond': cond,
        'eig_min': eigvals.min(),
        'eig_max': eigvals.max(),
        'eigvals': eigvals,
        'v_min': v_min,
        'CRLB': CRLB,
    }


# ============================================================
# A组：固定边缘角44.4°（r_max = 0.7R）
# ============================================================
R_A = [20, 30, 50, 75, 100, 125, 150]
results_A = []

print("===== Group A: Fixed edge angle (44.4°) =====")
for R in R_A:
    r_max = 0.7 * R
    print(f"\nR = {R} mm, r_max = {r_max:.1f} mm ...")
    res = compute_metrics(R, r_max)
    results_A.append(res)
    print(f"  θ_edge = {res['theta_edge']:.2f}°")
    print(f"  cond = {res['cond']:.3e}")
    print(f"  λ_min = {res['eig_min']:.3e}")

# ============================================================
# B组：固定孔径105 mm（边缘角随R变化）
# ============================================================
R_B = [150, 200, 300, 500, 1000]
results_B = []

print("\n===== Group B: Fixed aperture (105 mm) =====")
for R in R_B:
    r_max = R_APERTURE
    print(f"\nR = {R} mm, r_max = {r_max:.1f} mm ...")
    res = compute_metrics(R, r_max)
    results_B.append(res)
    print(f"  θ_edge = {res['theta_edge']:.2f}°")
    print(f"  cond = {res['cond']:.3e}")
    print(f"  λ_min = {res['eig_min']:.3e}")

# ============================================================
# 合并
# ============================================================
all_results = results_A + results_B
Rs = [r['R'] for r in all_results]
conds = [r['cond'] for r in all_results]
eig_mins = [r['eig_min'] for r in all_results]
eig_maxs = [r['eig_max'] for r in all_results]
theta_edges = [r['theta_edge'] for r in all_results]

# ============================================================
# 图3：3子图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# ---- (a) 条件数 vs R ----
axes[0].loglog([r['R'] for r in results_A],
               [r['cond'] for r in results_A],
               'o-', linewidth=2, markersize=8,
               color='steelblue', label='Fixed θ_edge = 44.4°')
axes[0].loglog([r['R'] for r in results_B],
               [r['cond'] for r in results_B],
               's-', linewidth=2, markersize=8,
               color='coral', label='Fixed aperture = 105 mm')
axes[0].axvline(x=150, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Curvature radius R (mm)')
axes[0].set_ylabel('FIM condition number')
axes[0].set_title('(a) Condition number vs R')
axes[0].grid(True, alpha=0.3, which='both')
axes[0].legend(fontsize=8)

# ---- (b) 特征值 vs R ----
axes[1].loglog([r['R'] for r in results_A],
               [r['eig_max'] for r in results_A],
               'o-', linewidth=2, markersize=8, color='coral',
               label='A: λ_max')
axes[1].loglog([r['R'] for r in results_A],
               [r['eig_min'] for r in results_A],
               'o--', linewidth=2, markersize=8, color='steelblue',
               label='A: λ_min')
axes[1].loglog([r['R'] for r in results_B],
               [r['eig_max'] for r in results_B],
               's-', linewidth=2, markersize=8, color='coral',
               alpha=0.5)
axes[1].loglog([r['R'] for r in results_B],
               [r['eig_min'] for r in results_B],
               's--', linewidth=2, markersize=8, color='steelblue',
               alpha=0.5)
axes[1].axvline(x=150, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Curvature radius R (mm)')
axes[1].set_ylabel('FIM eigenvalue')
axes[1].set_title('(b) Eigenvalues vs R')
axes[1].grid(True, alpha=0.3, which='both')
axes[1].legend(fontsize=8)

# ---- (c) 条件数 vs 边缘入射角 ----
axes[2].semilogy(theta_edges, conds, 'o-', linewidth=2, markersize=10,
                 color='purple')
for r in all_results:
    axes[2].annotate(f"R={r['R']}",
                     xy=(r['theta_edge'], r['cond']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=7)
axes[2].set_xlabel('Edge incident angle θ_edge (°)')
axes[2].set_ylabel('FIM condition number')
axes[2].set_title('(c) Condition number vs edge angle')
axes[2].grid(True, alpha=0.3, which='both')

plt.tight_layout()
png_path = os.path.join(FIG_DIR, 'fig3_curvature.png')
pdf_path = os.path.join(FIG_DIR, 'fig3_curvature.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")

# ============================================================
# 保存数据
# ============================================================
np.savez('curvature_results.npz',
         R_A=np.array([r['R'] for r in results_A]),
         cond_A=np.array([r['cond'] for r in results_A]),
         eig_min_A=np.array([r['eig_min'] for r in results_A]),
         eig_max_A=np.array([r['eig_max'] for r in results_A]),
         theta_A=np.array([r['theta_edge'] for r in results_A]),
         R_B=np.array([r['R'] for r in results_B]),
         cond_B=np.array([r['cond'] for r in results_B]),
         eig_min_B=np.array([r['eig_min'] for r in results_B]),
         eig_max_B=np.array([r['eig_max'] for r in results_B]),
         theta_B=np.array([r['theta_edge'] for r in results_B]))
print("Saved: curvature_results.npz")

# ============================================================
# 打印关键数据
# ============================================================
print("\n===== Group A: Fixed edge angle =====")
print(f"{'R (mm)':>8} {'r_max':>8} {'θ_edge':>8} {'cond':>12} {'λ_min':>12}")
for r in results_A:
    print(f"{r['R']:>8} {r['r_max']:>8.1f} {r['theta_edge']:>8.2f} "
          f"{r['cond']:>12.3e} {r['eig_min']:>12.3e}")

print("\n===== Group B: Fixed aperture =====")
print(f"{'R (mm)':>8} {'r_max':>8} {'θ_edge':>8} {'cond':>12} {'λ_min':>12}")
for r in results_B:
    print(f"{r['R']:>8} {r['r_max']:>8.1f} {r['theta_edge']:>8.2f} "
          f"{r['cond']:>12.3e} {r['eig_min']:>12.3e}")

# ============================================================
# 不可辨识方向对比
# ============================================================
print("\n===== Unidentifiable direction: R=20 vs R=1000 =====")
for r in all_results:
    if r['R'] in [20, 1000]:
        print(f"\nR = {r['R']} mm (θ_edge={r['theta_edge']:.2f}°):")
        for name, v in zip(PARAM_NAMES, r['v_min']):
            marker = " <-- dominant" if abs(v) > 0.3 else ""
            print(f"  {name}: {v:+.4f}{marker}")