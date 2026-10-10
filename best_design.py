# best_design.py
# 从 benchmark_results.npz 提取最优 PINN 设计
# 生成：Table 11、Figure 12（截面图）、Figure 13a/13b（径向反射率）

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import WLS, WL_TARGET, R_APERTURE, FIG_DIR, N_SUB
from forward import reflectance_sphere, sphere_average_reflectance

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs('paper', exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

# ============================================================
# 1. 读取最优设计
# ============================================================
res = np.load('exp10_results.npz')

d_opt  = res['d_pinn']
n_la   = float(res['n_la_pinn'])
n_mg   = float(res['n_mg_pinn'])
R_curv = float(res['R_pinn'])

r_max_eval = min(R_APERTURE, 0.7 * R_curv)

print("===== Optimal PINN design =====")
print(f"  n_LaF3 = {n_la:.4f}, n_MgF2 = {n_mg:.4f}")
print(f"  R      = {R_curv:.2f} mm")
print(f"  r_max  = {r_max_eval:.2f} mm  (evaluation aperture)")
for i, d in enumerate(d_opt):
    mat = 'LaF3' if i % 2 == 0 else 'MgF2'
    print(f"  d{i+1} ({mat}) = {d:.2f} nm")

layers_opt = [
    (n_la if i % 2 == 0 else n_mg, d_opt[i])
    for i in range(7)
]

# ============================================================
# 2. Table 11
# ============================================================
with open('paper/table11.md', 'w', encoding='utf-8') as f:
    f.write("# Table 11. Optimal layer thicknesses of the "
            "PINN-designed 7-layer LaF3/MgF2 AR coating.\n\n")
    f.write("| Layer | Material | Thickness (nm) |\n")
    f.write("|---|---|---|\n")
    for i, d in enumerate(d_opt):
        mat = 'LaF$_3$' if i % 2 == 0 else 'MgF$_2$'
        f.write(f"| d$_{i+1}$ | {mat} | {d:.2f} |\n")
    f.write(f"\n*Note:* $n_{{LaF_3}}$ = {n_la:.4f}, "
            f"$n_{{MgF_2}}$ = {n_mg:.4f} at 193 nm; "
            f"substrate fused silica $n_{{sub}}$ = {N_SUB}; "
            f"$R_{{curv}}$ = {R_curv:.1f} mm; "
            f"evaluation aperture radius = {r_max_eval:.1f} mm "
            f"(= min(105, 0.7R)).\n")
print("Saved: paper/table11.md")

# ============================================================
# 3. Figure 12：膜系层结构（水平条形图，配色优化）
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

# ---- 配色（改后）----
C_LA  = '#1f4e79'   # LaF3 深蓝（更深，对比更强）
C_MG  = '#ffe08a'   # MgF2 浅黄（蓝黄对比，黑白打印也清晰）

materials = ['LaF$_3$' if i % 2 == 0 else 'MgF$_2$' for i in range(7)]
labels = [f'$d_{i+1}$ ({materials[i]})' for i in range(7)]
colors = [C_LA if i % 2 == 0 else C_MG for i in range(7)]

y_pos = np.arange(7)[::-1]   # d1 在顶部
bars = ax.barh(y_pos, d_opt, height=0.65,
               color=colors, edgecolor='k', linewidth=1.2)

# 每根条右边标注厚度
for i, (bar, d) in enumerate(zip(bars, d_opt)):
    ax.text(d + 3, bar.get_y() + bar.get_height()/2,
            f'{d:.2f} nm',
            va='center', ha='left', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel('Layer thickness (nm)', fontsize=12)
ax.set_xlim(0, max(d_opt) * 1.25)
ax.grid(True, alpha=0.3, axis='x')
ax.set_title('Layer structure of the optimal PINN design',
             fontweight='bold', fontsize=13, pad=10)

# 图例（配色同步更新）
handles = [Patch(facecolor=C_LA, edgecolor='k', label='LaF$_3$'),
           Patch(facecolor=C_MG, edgecolor='k', label='MgF$_2$')]
ax.legend(handles=handles, loc='upper right',
          bbox_to_anchor=(1.0, 0.85), fontsize=11, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig12_schematic.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig12_schematic.pdf'),
            format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}fig12_schematic.png / .pdf")

# ============================================================
# 4. Figure 13a：径向反射率（0 – 0.7R，有效区域）
# ============================================================
r_a = np.linspace(0, r_max_eval, 200)
R_a = np.array([
    reflectance_sphere(np.array([WL_TARGET]), layers_opt,
                       R_curv, r, pol='avg')[0]
    for r in r_a
]) * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_a, R_a, color='#0072BD', lw=2.2, label='Radial reflectance')
ax.axhline(1.0, color='gray', ls='--', lw=1.2,
           label='1.0% reference')
ax.axhline(1.6, color='gray', ls=':', lw=1.2,
           label='1.6% reference')
ax.axhline(R_a.mean(), color='red', ls='-.', lw=1.2,
           label=f'Mean = {R_a.mean():.3f}%')

ax.set_xlabel('Radial position $r$ (mm)', fontsize=13)
ax.set_ylabel('Reflectance at 193 nm (%)', fontsize=13)
ax.set_title(f'Radial reflectance (valid aperture $r\\leq{r_max_eval:.1f}$ mm)',
             fontweight='bold', fontsize=15, pad=12)
ax.set_xlim(0, r_max_eval)
ax.set_ylim(0, max(2.0, R_a.max() * 1.15))
ax.legend(fontsize=10, frameon=False, loc='upper right')
ax.grid(True, alpha=0.25, lw=0.5)
ax.tick_params(labelsize=11)

# ← 加"uniformity"标注
unif_a = (1 - R_a.std() / R_a.mean()) * 100
ax.text(0.05, 0.05,
        f'Uniformity = {unif_a:.1f}%',
        transform=ax.transAxes, fontsize=11,
        fontweight='bold',
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white',
                  edgecolor='gray',
                  alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig13a_radial_valid.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig13a_radial_valid.pdf'),
            format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}fig13a_radial_valid.png / .pdf")

# ============================================================
# 5. Figure 13b：径向反射率（0 – 105 mm，全孔径，log 纵轴）
# ============================================================
r_b = np.linspace(0, R_APERTURE, 300)
R_b = np.array([
    reflectance_sphere(np.array([WL_TARGET]), layers_opt,
                       R_curv, r, pol='avg')[0]
    for r in r_b
]) * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(r_b, R_b, color='#0072BD', lw=2.2,
            label='Radial reflectance')
ax.axhline(1.0, color='gray', ls='--', lw=1.2,
           label='1.0% reference')
ax.axhline(1.6, color='gray', ls=':', lw=1.2,
           label='1.6% reference')
ax.axvline(r_max_eval, color='red', ls='-.', lw=1.2,
           label=f'Optimization boundary $r={r_max_eval:.1f}$ mm')

ax.set_xlabel('Radial position $r$ (mm)', fontsize=13)
ax.set_ylabel('Reflectance at 193 nm (%)', fontsize=13)
ax.set_title('Radial reflectance across the full 105 mm aperture',
             fontweight='bold', fontsize=15, pad=12)
ax.set_xlim(0, R_APERTURE)
ax.set_ylim(0.5, 100)
ax.legend(fontsize=10, frameon=False, loc='upper left')
ax.grid(True, which='both', alpha=0.25, lw=0.5)
ax.tick_params(labelsize=11)

# ← 加"edge effect"标注
ax.text(0.95, 0.55,
        f'Edge effect:\nR rises to {R_b.max():.0f}%',
        transform=ax.transAxes, fontsize=11,
        fontweight='bold', color='red',
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='white',
                  edgecolor='red',
                  alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig13b_radial_full.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig13b_radial_full.pdf'),
            format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}fig13b_radial_full.png / .pdf")

# ============================================================
# 6. 用全光谱复现 benchmark_results.npz 的 avg/max/unif
# ============================================================
spec_full, _ = sphere_average_reflectance(
    WLS, layers_opt, R_curv,
    r_max=r_max_eval, n_r=10, pol='avg')

avg_full  = float(np.mean(spec_full) * 100)
max_full  = float(np.max(spec_full)  * 100)
unif_full = float((1 - spec_full.std() / (spec_full.mean() + 1e-30)) * 100)

print("\n===== Verify vs benchmark_results.npz =====")
print(f"  avg_R  : {avg_full:.4f}%   (npz: "
      f"{float(res['avg_R_pinn'])*100:.4f}%)")
print(f"  max_R  : {max_full:.4f}%   (npz: "
      f"{float(res['max_R_pinn'])*100:.4f}%)")
print(f"  unif   : {unif_full:.4f}%   (npz: "
      f"{float(res['unif_pinn'])*100:.4f}%)")

print("\n===== Figure 13a summary (193 nm, r <= 0.7R) =====")
print(f"  Mean : {R_a.mean():.4f}%")
print(f"  Min  : {R_a.min():.4f}%")
print(f"  Max  : {R_a.max():.4f}%")
print(f"  Unif : {(1 - R_a.std()/R_a.mean())*100:.4f}%")

print("\n===== Figure 13b summary (193 nm, r <= 105 mm) =====")
print(f"  Mean : {R_b.mean():.4f}%")
print(f"  Min  : {R_b.min():.4f}%")
print(f"  Max  : {R_b.max():.4f}%")
print(f"  Unif : {(1 - R_b.std()/R_b.mean())*100:.4f}%")
print(f"\n  d1..d7 (nm) : {np.round(d_opt, 2).tolist()}")