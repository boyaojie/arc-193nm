# fig2_combined.py
# 读 fim_results.npz 和 sobol_results.npz
# 生成 Figure 3：(b) FIM 特征值、(c) CRLB、(d) 最不可辨识方向、(e) Sobol

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import WLS, WL_TARGET, FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 科研论文风格
# ============================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,
    'legend.frameon': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

C_BLUE   = '#0072BD'
C_ORANGE = '#D95319'
C_GREEN  = '#77AC30'
C_RED    = '#A2142F'

# ============================================================
# 读取数据
# ============================================================
fim_data = np.load('fim_results.npz')
sobol_data = np.load('sobol_results.npz')

J = fim_data['J']
eigvals = fim_data['eigvals']
CRLB = fim_data['CRLB']
v_min = fim_data['v_min']
p0 = fim_data['p0']

crlb_std = np.sqrt(CRLB)

S1 = sobol_data['S1']
ST = sobol_data['ST']
sobol_names = sobol_data['names']

PARAM_NAMES = ['$d_1$', '$d_2$', '$d_3$', '$d_4$', '$d_5$',
               '$d_6$', '$d_7$', '$n_{\\rm LaF_3}$',
               '$n_{\\rm MgF_2}$', '$R$']

cond = eigvals.max() / eigvals.min()

# ============================================================
# 图：(b)(c)(d)(e) 四子图
# ============================================================
fig = plt.figure(figsize=(13.5, 6.5))
gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=0.75)

# ------------------------------------------------------------
# (b) FIM 特征值谱
# ------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 0:3])
idx = np.arange(1, len(eigvals) + 1)
ax_b.semilogy(idx, eigvals, 'o-', color=C_BLUE, lw=1.5,
              markersize=5, markerfacecolor='white',
              markeredgewidth=1.2, markeredgecolor=C_BLUE)
ax_b.set_xlabel('Eigenvalue index')
ax_b.set_ylabel('Eigenvalue')
ax_b.set_title(f'(b) FIM eigenvalue spectrum (cond = {cond:.1e})',
               fontweight='bold', pad=6)
ax_b.grid(True, which='major', alpha=0.25, lw=0.5)
ax_b.set_xticks(idx)

# ------------------------------------------------------------
# (c) CRLB
# ------------------------------------------------------------
ax_c = fig.add_subplot(gs[0, 3:6])
y_pos = np.arange(len(p0))
ax_c.barh(y_pos, crlb_std, color=C_ORANGE, edgecolor='black',
          lw=0.4, height=0.7)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(PARAM_NAMES, fontsize=7)
ax_c.set_xlabel('CRLB')
ax_c.set_title('(c) CRLB', fontweight='bold', pad=6)
ax_c.invert_yaxis()
ax_c.set_xscale('log')
ax_c.grid(True, which='major', alpha=0.25, lw=0.5, axis='x')

# ------------------------------------------------------------
# (d) 最不可辨识方向
# ------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 0:3])
colors_d = [C_RED if abs(v) > 0.3 else C_BLUE for v in v_min]
ax_d.bar(range(len(p0)), v_min, color=colors_d,
         edgecolor='black', lw=0.4, width=0.65)
ax_d.axhline(0, color='black', lw=0.6)
ax_d.set_xticks(range(len(p0)))
ax_d.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=7)
ax_d.set_ylabel('Weight')
ax_d.set_title('(d) Most unidentifiable direction',
               fontweight='bold', pad=6)
ax_d.grid(True, alpha=0.25, lw=0.5, axis='y')

# ------------------------------------------------------------
# (e) Sobol
# ------------------------------------------------------------
ax_e = fig.add_subplot(gs[1, 3:6])
x = np.arange(len(sobol_names))
w = 0.38
ax_e.bar(x - w/2, S1, w, color=C_BLUE, edgecolor='black',
         lw=0.4, label='$S1$ (first-order)')
ax_e.bar(x + w/2, ST, w, color=C_GREEN, edgecolor='black',
         lw=0.4, label='$ST$ (total-order)')
ax_e.set_xticks(x)
ax_e.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=7)
ax_e.set_ylabel('Sobol index')
ax_e.set_title('(e) Sobol global sensitivity indices',
               fontweight='bold', pad=6)
ax_e.legend(loc='upper right', frameon=False, fontsize=7)
ax_e.grid(True, alpha=0.25, lw=0.5, axis='y')
ax_e.set_ylim(0, max(ST.max(), S1.max()) * 1.15)

# ============================================================
# 保存
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'fig')
os.makedirs(OUT_DIR, exist_ok=True)

png_path = os.path.join(OUT_DIR, 'fig3_identifiability.png')
pdf_path = os.path.join(OUT_DIR, 'fig3_identifiability.pdf')

plt.savefig(png_path)
plt.savefig(pdf_path, format='pdf')
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")