# sobol.py
# 实验2b：Sobol全局敏感度分析
# 生成 Figure 2 (e)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 用新版sobol采样（消除DeprecationWarning）
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol

from config import (WLS, WL_TARGET, N_LAF3, N_MGF2, R_CURV,
                    FIG_DIR, SEED)
from identifiability import spectrum_from_params

os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

# ============================================================
# 降采样
# ============================================================
WLS_SUB = WLS[::4]
print(f"Wavelength samples: {len(WLS_SUB)}")

# ============================================================
# 参数范围
# ============================================================
d0 = np.array([53.72, 30.15, 49.98, 23.23, 47.87, 58.44, 124.48])

bounds = []
for d in d0:
    bounds.append([d * 0.9, d * 1.1])   # 缩小到 ±10%
bounds.append([N_LAF3 * 0.99, N_LAF3 * 1.01])
bounds.append([N_MGF2 * 0.99, N_MGF2 * 1.01])
bounds.append([R_CURV * 0.9, R_CURV * 1.1])

names = [f"d{i+1}" for i in range(7)] + ["n_LaF3", "n_MgF2", "R"]
problem = {'num_vars': len(names), 'names': names, 'bounds': bounds}

# ============================================================
# 采样（用新版API，样本数256）
# ============================================================
print("Sampling...")
param_values = sobol_sample.sample(problem, 256,
                                   calc_second_order=False)
print("Samples:", param_values.shape)

# ============================================================
# 计算输出
# ============================================================
i193 = np.argmin(np.abs(WLS_SUB - WL_TARGET))
print("Computing spectra...")
Y = np.zeros(len(param_values))
for i, p in enumerate(param_values):
    spec = spectrum_from_params(p, wls=WLS_SUB, n_r=2, pol='avg')
    Y[i] = spec[i193]
    if i % 500 == 0:
        print(f"  {i}/{len(param_values)}")

# ============================================================
# Sobol
# ============================================================
Si = sobol.analyze(problem, Y, calc_second_order=False,
                   print_to_console=False)

# 截断负值到0
S1 = np.clip(Si['S1'], 0, None)
ST = np.clip(Si['ST'], 0, None)

# 确保 ST >= S1
ST = np.maximum(ST, S1)

print("\n===== Sobol indices (corrected) =====")
for name, s1, st in zip(names, S1, ST):
    print(f"  {name}: S1={s1:.4f}, ST={st:.4f}")

print(f"\nSum S1 = {S1.sum():.4f}")
print(f"Sum ST = {ST.sum():.4f}")

# ============================================================
# 图：Sobol条形图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, S1, w, label='First-order (S1)',
       color='steelblue', edgecolor='k')
ax.bar(x + w/2, ST, w, label='Total (ST)',
       color='coral', edgecolor='k')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('Sobol index')
ax.set_title('Global sensitivity analysis (Sobol indices)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(ST.max(), S1.max()) * 1.15)

# 标注最大值
max_idx = np.argmax(S1)
ax.annotate(f'max S1={S1[max_idx]:.3f}',
            xy=(max_idx - w/2, S1[max_idx]),
            xytext=(max_idx + 0.5, S1[max_idx] + 0.05),
            arrowprops=dict(arrowstyle='->', color='k'),
            fontsize=9)

plt.tight_layout()
png_path = os.path.join(FIG_DIR, 'fig2_sobol.png')
pdf_path = os.path.join(FIG_DIR, 'fig2_sobol.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")

# ============================================================
# 保存
# ============================================================
np.savez('sobol_results.npz',
         S1=S1, ST=ST, names=names)
print("Saved: sobol_results.npz")