# validate.py
# 实验1：正向模型验证
# 生成 Figure 1

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# 全局样式
# ============================================================
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
# 图1：4子图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# ------------------------------------------------------------
# (a) 入射角分布
# ------------------------------------------------------------
r_arr = np.linspace(0, R_APERTURE, 200)
theta_arr = np.degrees(incident_angle(r_arr, R_CURV))

axes[0, 0].plot(r_arr, theta_arr, 'b-', linewidth=2, label='θ(r)')
axes[0, 0].axhline(y=44.4, color='r', linestyle='--', alpha=0.6,
                   label='44.4° (edge)')
axes[0, 0].axvline(x=84, color='g', linestyle=':', alpha=0.6,
                   label='r=84 mm (edge start)')
axes[0, 0].fill_between(r_arr, theta_arr - 2.0, theta_arr + 2.0,
                        color='b', alpha=0.15, label='±2° error')
axes[0, 0].set_xlabel('Radial position r (mm)')
axes[0, 0].set_ylabel('Incident angle θ (°)')
axes[0, 0].set_title('(a) Incident angle distribution')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=8, loc='upper left')
axes[0, 0].set_xlim(0, R_APERTURE)
axes[0, 0].set_ylim(0, 50)

# ------------------------------------------------------------
# (b) R→∞退化
# ------------------------------------------------------------
spec_planar = reflectance_planar(WLS, LAYERS_7)
spec_sphere_inf = reflectance_sphere(WLS, LAYERS_7, R=1e6, r=1.0,
                                     use_deviation=False)

axes[0, 1].plot(WLS, spec_planar * 100, 'b-', linewidth=2,
                label='Planar (R→∞)')
axes[0, 1].plot(WLS, spec_sphere_inf * 100, 'r--', linewidth=1.5,
                label='Sphere R=1e6 mm')
axes[0, 1].axvline(x=WL_TARGET, color='k', linestyle=':', alpha=0.5,
                   label=f'{WL_TARGET} nm')
axes[0, 1].set_xlabel('Wavelength (nm)')
axes[0, 1].set_ylabel('Reflectance (%)')
axes[0, 1].set_title('(b) R→∞ degenerates to planar')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_xlim(WLS[0], WLS[-1])

err = np.max(np.abs(spec_sphere_inf - spec_planar))
print(f"Degeneration error: {err:.3e}")

# ------------------------------------------------------------
# (c) 球面反射谱
# ------------------------------------------------------------
colors = plt.cm.viridis(np.linspace(0, 1, len(R_SAMPLES)))

for i, r in enumerate(R_SAMPLES):
    spec = reflectance_sphere(WLS, LAYERS_7, R_CURV, r, pol='avg')
    axes[1, 0].plot(WLS, spec * 100, linewidth=1.8,
                    color=colors[i], label=f'r={r} mm')

axes[1, 0].plot(WLS, spec_planar * 100, 'k--', linewidth=1.5,
                label='planar', alpha=0.7)

axes[1, 0].axvline(x=WL_TARGET, color='k', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Wavelength (nm)')
axes[1, 0].set_ylabel('Reflectance (%)')
axes[1, 0].set_title(f'(c) Spherical spectra (R={R_CURV} mm, s+p avg)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=7, loc='upper right', ncol=2)
axes[1, 0].set_xlim(WLS[0], WLS[-1])

# ------------------------------------------------------------
# (d) 膜厚偏差
# ------------------------------------------------------------
delta_arr = thickness_deviation(r_arr, R_CURV)

axes[1, 1].plot(r_arr, delta_arr, 'g-', linewidth=2, label='δ(r)')
axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5,
                   label='δ=1.0 (nominal)')
axes[1, 1].axhline(y=1.1, color='r', linestyle='--', alpha=0.5,
                   label='δ=1.1 (constraint)')
axes[1, 1].set_xlabel('Radial position r (mm)')
axes[1, 1].set_ylabel('Thickness deviation δ(r)')
axes[1, 1].set_title('(d) Film thickness deviation')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=8, loc='lower right')
axes[1, 1].set_xlim(0, R_APERTURE)
axes[1, 1].set_ylim(0.98, 1.12)

# ------------------------------------------------------------
# 保存
# ------------------------------------------------------------
plt.tight_layout()

png_path = os.path.join(FIG_DIR, 'fig1_forward.png')
pdf_path = os.path.join(FIG_DIR, 'fig1_forward.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

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

# 每个r的193 nm反射率
print("\n===== R at 193 nm per radius (s+p avg) =====")
for r in R_SAMPLES:
    spec_r = reflectance_sphere(WLS, LAYERS_7, R_CURV, r, pol='avg')
    theta_r = np.degrees(incident_angle(r, R_CURV))
    print(f"  r={r:3d} mm, θ={theta_r:5.2f}°, R(193nm)={spec_r[i193]*100:.4f}%")

# 球面平均反射率
spec_avg, _ = sphere_average_reflectance(WLS, LAYERS_7, R_CURV, pol='avg')
print(f"\nSphere avg R at 193 nm: {spec_avg[i193] * 100:.4f}%")

# 高角反射率（θ>40°）
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