# validate.py
# 实验1：正向模型验证
# 生成 Figure 1（精美版）

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from config import (WLS, WL_TARGET, LAYERS_7, R_CURV, R_APERTURE,
                    R_SAMPLES, FIG_DIR, N_R_SAMPLES)
from forward import (incident_angle, thickness_deviation,
                     reflectance_sphere, reflectance_planar,
                     sphere_average_reflectance)

# ============================================================
# 确保 fig/ 存在
# ============================================================
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 全局样式（改进点 1：统一字体）
# ============================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# 统一配色（改进点 6）
# ============================================================
C_BLUE   = '#1f77b4'
C_RED    = '#d62728'
C_GREEN  = '#2ca02c'
C_ORANGE = '#ff7f0e'
C_PURPLE = '#9467bd'
C_GRAY   = '#7f7f7f'

# ============================================================
# 图1：4子图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
ax_a, ax_b, ax_c, ax_d = axes.flatten()

# ------------------------------------------------------------
# (a) 入射角分布
# ------------------------------------------------------------
r_arr = np.linspace(0, R_APERTURE, 200)
theta_arr = np.degrees(incident_angle(r_arr, R_CURV))
theta_err = 0.02 * theta_arr  # ±2%

ax_a.fill_between(r_arr, theta_arr - theta_err, theta_arr + theta_err,
                  color=C_BLUE, alpha=0.2, label=r'$\pm 2\%$ error')
ax_a.plot(r_arr, theta_arr, color=C_BLUE, lw=1.8, label=r'$\theta(r)$')
ax_a.axhline(44.4, color=C_RED, ls='--', lw=1.0,
             label=r'$44.4^{\circ}$ (edge)')
ax_a.axvline(84, color=C_GREEN, ls=':', lw=1.0,
             label=r'$r=84$ mm (edge start)')
ax_a.set_xlabel('Radial position $r$ (mm)')
ax_a.set_ylabel(r'Incident angle $\theta$ ($^{\circ}$)')
ax_a.set_title('(a) Incident angle distribution')
ax_a.set_xlim(0, R_APERTURE)
ax_a.set_ylim(0, 50)
ax_a.legend(loc='upper left', frameon=True, edgecolor='0.8')
ax_a.grid(True, which='major', alpha=0.3, lw=0.5)
ax_a.grid(False, which='minor')
ax_a.xaxis.set_minor_locator(MultipleLocator(10))
ax_a.yaxis.set_minor_locator(MultipleLocator(5))

# ------------------------------------------------------------
# (b) R→∞退化
# ------------------------------------------------------------
spec_planar = reflectance_planar(WLS, LAYERS_7)
spec_sphere_inf = reflectance_sphere(WLS, LAYERS_7, R=1e6, r=1.0,
                                     use_deviation=False)

ax_b.plot(WLS, spec_planar * 100, color=C_BLUE, lw=1.8,
          label=r'Planar ($R\to\infty$)')
ax_b.plot(WLS, spec_sphere_inf * 100, color=C_RED, ls='--', lw=1.2,
          label=r'Sphere $R=10^{6}$ mm')
ax_b.axvline(WL_TARGET, color=C_GRAY, ls=':', lw=1.0,
             label=f'{WL_TARGET:.1f} nm')
ax_b.set_xlabel('Wavelength (nm)')
ax_b.set_ylabel('Reflectance (%)')
ax_b.set_title('(b) Planar-to-spherical degeneration')
ax_b.set_xlim(WLS[0], WLS[-1])
ax_b.legend(loc='upper left', frameon=True, edgecolor='0.8')
ax_b.grid(True, which='major', alpha=0.3, lw=0.5)
ax_b.grid(False, which='minor')
ax_b.xaxis.set_minor_locator(MultipleLocator(10))
ax_b.yaxis.set_minor_locator(MultipleLocator(5))

err = np.max(np.abs(spec_sphere_inf - spec_planar))
print(f"Degeneration error: {err:.3e}")

# ------------------------------------------------------------
# (c) 球面反射谱
# ------------------------------------------------------------
colors_c = [C_PURPLE, C_BLUE, C_GREEN, C_ORANGE, '#bcbd22', '#17becf']

for i, r in enumerate(R_SAMPLES):
    spec = reflectance_sphere(WLS, LAYERS_7, R_CURV, r, pol='avg')
    ax_c.plot(WLS, spec * 100, linewidth=1.5,
              color=colors_c[i], label=f'r={r} mm')

ax_c.plot(WLS, spec_planar * 100, color=C_GRAY, ls='--', lw=1.2,
          label='planar', alpha=0.8)

ax_c.axvline(WL_TARGET, color=C_GRAY, ls=':', lw=1.0)
ax_c.set_xlabel('Wavelength (nm)')
ax_c.set_ylabel('Reflectance (%)')
ax_c.set_title('(c) Spherical reflectance spectra')
ax_c.set_xlim(WLS[0], WLS[-1])
ax_c.legend(fontsize=7, loc='upper left', ncol=2,
            frameon=True, edgecolor='0.8')
ax_c.grid(True, which='major', alpha=0.3, lw=0.5)
ax_c.grid(False, which='minor')
ax_c.xaxis.set_minor_locator(MultipleLocator(10))
ax_c.yaxis.set_minor_locator(MultipleLocator(5))

# ------------------------------------------------------------
# (d) 膜厚偏差
# ------------------------------------------------------------
delta_arr = thickness_deviation(r_arr, R_CURV)

ax_d.plot(r_arr, delta_arr, color=C_GREEN, lw=1.8,
          label=r'$\delta(r)$')
ax_d.axhline(1.0, color=C_GRAY, ls='--', lw=1.0,
             label=r'$\delta=1.0$ (nominal)')
ax_d.axhline(1.1, color=C_RED, ls='--', lw=1.0,
             label=r'$\delta=1.1$ (constraint)')
ax_d.set_xlabel('Radial position $r$ (mm)')
ax_d.set_ylabel(r'Thickness deviation $\delta(r)$')
ax_d.set_title('(d) Film thickness deviation')
ax_d.set_xlim(0, R_APERTURE)
ax_d.set_ylim(0.98, 1.12)
ax_d.legend(loc='lower right', frameon=True, edgecolor='0.8')
ax_d.grid(True, which='major', alpha=0.3, lw=0.5)
ax_d.grid(False, which='minor')
ax_d.xaxis.set_minor_locator(MultipleLocator(10))
ax_d.yaxis.set_minor_locator(MultipleLocator(0.02))

# ------------------------------------------------------------
# 布局与保存（改进点 7）
# ------------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(wspace=0.28, hspace=0.32)

png_path = os.path.join(FIG_DIR, 'fig1_forward.png')
pdf_path = os.path.join(FIG_DIR, 'fig1_forward.pdf')

plt.savefig(png_path)
plt.savefig(pdf_path, format='pdf')
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# ============================================================
# 验证数据
# ============================================================
i193 = np.argmin(np.abs(WLS - WL_TARGET))

print("\n===== Validation =====")
print(f"Edge incident angle:  {np.degrees(incident_angle(R_APERTURE, R_CURV)):.4f}°")
print(f"Degeneration error:   {err:.3e}")
print(f"R at 193 nm (planar): {spec_planar[i193] * 100:.4f}%")
print(f"Delta range:          [{delta_arr.min():.4f}, {delta_arr.max():.4f}]")

print("\n===== R at 193 nm per radius (s+p avg) =====")
for r in R_SAMPLES:
    spec_r = reflectance_sphere(WLS, LAYERS_7, R_CURV, r, pol='avg')
    theta_r = np.degrees(incident_angle(r, R_CURV))
    print(f"  r={r:3d} mm, θ={theta_r:5.2f}°, R(193nm)={spec_r[i193]*100:.4f}%")

spec_avg, _ = sphere_average_reflectance(WLS, LAYERS_7, R_CURV, pol='avg')
print(f"\nSphere avg R at 193 nm: {spec_avg[i193] * 100:.4f}%")

print("\n===== High-angle reflectance (θ>40°) =====")
r_high = np.linspace(84, R_APERTURE, 20)
spec_high = []
for r in r_high:
    theta_r = np.degrees(incident_angle(r, R_CURV))
    if theta_r > 40:
        spec_r = reflectance_sphere(WLS, LAYERS_7, R_CURV, r, pol='avg')
        spec_high.append(spec_r[i193])
if spec_high:
    print(f"  Max R at 193 nm (θ>40°): {max(spec_high)*100:.4f}%")