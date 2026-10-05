# fim.py
# 实验2a：Fisher 信息矩阵分析
# 生成 Figure 2：仅 (a) Jacobian 热图

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (WLS, WL_TARGET, FIG_DIR)
from identifiability import (get_initial_params, compute_jacobian,
                             compute_fim, fim_metrics,
                             unidentifiable_direction, PARAM_NAMES)

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 科研论文风格
# ============================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# 计算
# ============================================================
p0 = get_initial_params()

print("Computing Jacobian...")
J = compute_jacobian(p0, wls=WLS, n_r=4)
print("Jacobian shape:", J.shape)

F = compute_fim(J, sigma=0.01)
eigvals, cond, CRLB = fim_metrics(F)
v_min, eigvals_s = unidentifiable_direction(F)

# ============================================================
# 图：仅 (a) Jacobian 热图
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

J_plot = np.abs(J)
if J_plot.shape[0] != len(p0):
    J_plot = J_plot.T

im = ax.imshow(J_plot, aspect='auto', cmap='viridis',
               extent=[WLS[0], WLS[-1], len(p0) - 0.5, -0.5],
               origin='upper', interpolation='nearest')

ax.axvline(x=WL_TARGET, color='w', ls='--', lw=1.0, alpha=0.9)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Parameter')
ax.set_yticks(range(len(p0)))
ax.set_yticklabels(PARAM_NAMES, fontsize=8)
ax.set_title('(a) Jacobian magnitude $|\\partial R / \\partial p|$',
             fontweight='bold', pad=6)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.ax.tick_params(labelsize=7)
cbar.set_label('$|\\partial R / \\partial p|$', fontsize=8)

plt.tight_layout()

png_path = os.path.join(FIG_DIR, 'fig2_jacobian.png')
pdf_path = os.path.join(FIG_DIR, 'fig2_jacobian.pdf')

plt.savefig(png_path)
plt.savefig(pdf_path, format='pdf')
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# 保存数据
np.savez('fim_results.npz',
         J=J, F=F, eigvals=eigvals, CRLB=CRLB, v_min=v_min, p0=p0)
print("Saved: fim_results.npz")