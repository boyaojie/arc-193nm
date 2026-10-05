# robust.py
# 实验7：鲁棒性分析（最终版 v6，精进4 + 精进5）
# 输出：fig/fig7_robust.png、fig/fig7_robust.pdf、robust_results.npz

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import WLS, WL_TARGET, FIG_DIR, SEED
from forward import sphere_average_reflectance

os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

# ============================================================
# 1. 加载数据 + 训练时的归一化参数
# ============================================================
print("===== Robustness Analysis (with PINN, v6) =====")

data = np.load('dataset.npz')
spectra = data['spectra'][:50000]
params = data['params'][:50000]

norm = np.load('norm_params_pinn.npz')
Y_min = norm['Y_min']
Y_max = norm['Y_max']

X = np.concatenate([
    spectra,
    params[:, 9:10] / 200.0
], axis=1).astype(np.float32)

# ============================================================
# 2. 网络结构（和 pinn.py 完全一致）
# ============================================================
class PINN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


device = 'cpu'
model = PINN(X.shape[1], 9).to(device)

model_path = 'pinn.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✓ Loaded: {model_path}")
print(f"  First layer weight mean: {model.net[0].weight.data.mean().item():.6f}")

# ============================================================
# 3. 选基准参数（两阶段筛选，A+C 方案）
# ============================================================
avg_R_all = spectra.mean(axis=1)
max_R_all = spectra.max(axis=1)
unif_all = 1.0 - spectra.std(axis=1) / (avg_R_all + 1e-30)

MAX_R_THRESHOLD = 0.10
mask = max_R_all < MAX_R_THRESHOLD
if mask.sum() < 100:
    threshold = np.percentile(max_R_all, 10)
    mask = max_R_all <= threshold
    print(f"⚠️  Max R < {MAX_R_THRESHOLD*100:.0f}% 样本不足 100，"
          f"放宽到 Max R <= {threshold*100:.2f}%")

print(f"  Candidates (Max R < {MAX_R_THRESHOLD*100:.0f}%): {mask.sum()}")

idx_candidates = np.where(mask)[0]
TOP_K = min(200, len(idx_candidates))
idx_top = idx_candidates[np.argsort(avg_R_all[idx_candidates])[:TOP_K]]

idx_best = int(idx_top[np.argmax(unif_all[idx_top])])

d_true = params[idx_best, :7].astype(np.float64)
n_la_true = float(params[idx_best, 7])
n_mg_true = float(params[idx_best, 8])
R_sample = float(params[idx_best, 9])

with torch.no_grad():
    x_best = torch.tensor(X[idx_best:idx_best+1])
    pred_norm = model(x_best).numpy()[0]
pred_denorm = pred_norm * (Y_max - Y_min) + Y_min

d_pred = pred_denorm[:7]
n_la_pred = float(pred_denorm[7])
n_mg_pred = float(pred_denorm[8])

print(f"\n===== Base design (idx={idx_best}) =====")
print(f"  True  d1-d7 : {np.round(d_true, 2)}")
print(f"  True  n_LaF3: {n_la_true:.4f}, n_MgF2: {n_mg_true:.4f}")
print(f"  True  R     : {R_sample:.1f} mm")
print(f"  True  avg R : {avg_R_all[idx_best]*100:.4f}%")
print(f"  True  max R : {max_R_all[idx_best]*100:.4f}%")
print(f"  True  unif. : {unif_all[idx_best]*100:.4f}%")
print(f"  Pred  d1-d7 : {np.round(d_pred, 2)}")
print(f"  Pred  n_LaF3: {n_la_pred:.4f}, n_MgF2: {n_mg_pred:.4f}")

# ============================================================
# 4. 基准光谱
# ============================================================
layers_true = [
    (n_la_true if i % 2 == 0 else n_mg_true, d_true[i])
    for i in range(7)
]
spec_target, _ = sphere_average_reflectance(
    WLS, layers_true, R_sample,
    r_max=min(105.0, 0.7 * R_sample), n_r=10, pol='avg')

target_avg_R = float(np.mean(spec_target) * 100)
target_max_R = float(np.max(spec_target) * 100)
target_unif = float(
    (1.0 - spec_target.std() / (spec_target.mean() + 1e-30)) * 100)
target_cv = float(
    spec_target.std() / (spec_target.mean() + 1e-30) * 100)

print(f"\nNominal avg R     : {target_avg_R:.4f}%")
print(f"Nominal max R     : {target_max_R:.4f}%")
print(f"Nominal uniformity: {target_unif:.4f}%  (definition: 1 - std/mean)")
print(f"Nominal CV        : {target_cv:.4f}%  (definition: std/mean)")

# ============================================================
# 5. 蒙特卡洛鲁棒性（两组：标准 / 严苛）
# ============================================================
print("\n===== Monte Carlo Robustness =====")

N_MC = 500

def run_mc(d_noise, n_noise, label):
    mc_avg_R = []
    mc_max_R = []
    mc_unif = []
    mc_cv = []
    for _ in range(N_MC):
        d_pert = d_true * (1 + np.random.normal(0, d_noise, 7))
        n_la_pert = n_la_true * (1 + np.random.normal(0, n_noise))
        n_mg_pert = n_mg_true * (1 + np.random.normal(0, n_noise))

        layers_pert = [
            (n_la_pert if j % 2 == 0 else n_mg_pert, d_pert[j])
            for j in range(7)
        ]
        spec, _ = sphere_average_reflectance(
            WLS, layers_pert, R_sample,
            r_max=min(105.0, 0.7 * R_sample), n_r=10, pol='avg')

        mean_R = np.mean(spec)
        std_R = np.std(spec)

        mc_avg_R.append(mean_R * 100)
        mc_max_R.append(np.max(spec) * 100)
        mc_unif.append((1.0 - std_R / (mean_R + 1e-30)) * 100)
        mc_cv.append(std_R / (mean_R + 1e-30) * 100)

    mc_avg_R = np.array(mc_avg_R)
    mc_max_R = np.array(mc_max_R)
    mc_unif = np.array(mc_unif)
    mc_cv = np.array(mc_cv)

    print(f"\n  --- {label} ---")
    print(f"  Avg R : mean={mc_avg_R.mean():.4f}%, "
          f"std={mc_avg_R.std():.4f}%, "
          f"95%CI=[{np.percentile(mc_avg_R, 2.5):.4f}, "
          f"{np.percentile(mc_avg_R, 97.5):.4f}]")
    print(f"  Max R : mean={mc_max_R.mean():.4f}%, "
          f"std={mc_max_R.std():.4f}%, "
          f"95%CI=[{np.percentile(mc_max_R, 2.5):.4f}, "
          f"{np.percentile(mc_max_R, 97.5):.4f}]")
    print(f"  Unif  : mean={mc_unif.mean():.4f}%, "
          f"std={mc_unif.std():.4f}%, "
          f"95%CI=[{np.percentile(mc_unif, 2.5):.4f}, "
          f"{np.percentile(mc_unif, 97.5):.4f}]")
    print(f"  CV    : mean={mc_cv.mean():.4f}%, "
          f"std={mc_cv.std():.4f}%, "
          f"95%CI=[{np.percentile(mc_cv, 2.5):.4f}, "
          f"{np.percentile(mc_cv, 97.5):.4f}]")

    return mc_avg_R, mc_max_R, mc_unif, mc_cv

# 标准：±3% / ±0.5%
mc_avg_R, mc_max_R, mc_unif, mc_cv = run_mc(
    0.03, 0.005, "Standard: ±3% thickness / ±0.5% index")

# 严苛：±5% / ±1%
mc_avg_R_h, mc_max_R_h, mc_unif_h, mc_cv_h = run_mc(
    0.05, 0.01, "Harsher: ±5% thickness / ±1% index")

# ============================================================
# 6. 温度鲁棒性：20°C vs 40°C
# ============================================================
print("\n===== Temperature Robustness =====")

dn_dT_la = 8.2e-6
dn_dT_mg = 7.5e-6
alpha_la = 9.0e-6
alpha_mg = 8.0e-6
dT = 20.0

layers_20 = [
    (n_la_true if i % 2 == 0 else n_mg_true, d_true[i])
    for i in range(7)
]
spec_20, _ = sphere_average_reflectance(
    WLS, layers_20, R_sample,
    r_max=min(105.0, 0.7 * R_sample), n_r=10, pol='avg')

layers_40 = []
for i in range(7):
    if i % 2 == 0:
        n_i = n_la_true + dn_dT_la * dT
        d_i = d_true[i] * (1 + alpha_la * dT)
    else:
        n_i = n_mg_true + dn_dT_mg * dT
        d_i = d_true[i] * (1 + alpha_mg * dT)
    layers_40.append((n_i, d_i))

spec_40, _ = sphere_average_reflectance(
    WLS, layers_40, R_sample,
    r_max=min(105.0, 0.7 * R_sample), n_r=10, pol='avg')

thermal_drift = float(np.mean(np.abs(spec_40 - spec_20)) * 100)
print(f"  Thermal drift: {thermal_drift:.4f}%")

# ============================================================
# 7. 图7：4 子图（精进5：图里写清均匀性定义）
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(mc_avg_R, bins=30, color='steelblue',
                edgecolor='k', alpha=0.7)
axes[0, 0].axvline(target_avg_R, color='r', linestyle='--',
                   linewidth=2, label=f'Nominal = {target_avg_R:.4f}%')
axes[0, 0].axvline(np.percentile(mc_avg_R, 2.5), color='gray',
                   linestyle=':', linewidth=1.5)
axes[0, 0].axvline(np.percentile(mc_avg_R, 97.5), color='gray',
                   linestyle=':', linewidth=1.5, label='95% CI')
axes[0, 0].set_xlabel('Average Reflectance (%)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('(a) Average reflectance distribution')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].hist(mc_unif, bins=30, color='coral',
                edgecolor='k', alpha=0.7)
axes[0, 1].axvline(target_unif, color='r', linestyle='--',
                   linewidth=2, label=f'Nominal = {target_unif:.4f}%')
axes[0, 1].axvline(np.percentile(mc_unif, 2.5), color='gray',
                   linestyle=':', linewidth=1.5, label='95% CI')
axes[0, 1].axvline(np.percentile(mc_unif, 97.5), color='gray',
                   linestyle=':', linewidth=1.5)
axes[0, 1].set_xlabel('Uniformity (%)  [definition: 1 - std/mean]')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('(b) Uniformity distribution')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[1, 0].hist(mc_max_R, bins=30, color='green',
                edgecolor='k', alpha=0.7)
axes[1, 0].axvline(target_max_R, color='r', linestyle='--',
                   linewidth=2, label=f'Nominal = {target_max_R:.4f}%')
axes[1, 0].axvline(np.percentile(mc_max_R, 2.5), color='gray',
                   linestyle=':', linewidth=1.5, label='95% CI')
axes[1, 0].axvline(np.percentile(mc_max_R, 97.5), color='gray',
                   linestyle=':', linewidth=1.5)
axes[1, 0].set_xlabel('Max Reflectance (%)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('(c) Max reflectance distribution')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].plot(WLS, spec_20 * 100, 'b-', linewidth=2, label='20°C')
axes[1, 1].plot(WLS, spec_40 * 100, 'r--', linewidth=2, label='40°C')
axes[1, 1].axvline(x=WL_TARGET, color='k', linestyle=':', alpha=0.5)
axes[1, 1].set_xlabel('Wavelength (nm)')
axes[1, 1].set_ylabel('Reflectance (%)')
axes[1, 1].set_title(f'(d) Thermal stability (drift={thermal_drift:.4f}%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
png_path = os.path.join(FIG_DIR, 'fig7_robust.png')
pdf_path = os.path.join(FIG_DIR, 'fig7_robust.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")

# ============================================================
# 8. 保存数据
# ============================================================
np.savez('robust_results.npz',
         mc_avg_R=mc_avg_R,
         mc_max_R=mc_max_R,
         mc_unif=mc_unif,
         mc_cv=mc_cv,
         mc_avg_R_h=mc_avg_R_h,
         mc_max_R_h=mc_max_R_h,
         mc_unif_h=mc_unif_h,
         mc_cv_h=mc_cv_h,
         target_avg_R=target_avg_R,
         target_max_R=target_max_R,
         target_unif=target_unif,
         target_cv=target_cv,
         thermal_drift=thermal_drift,
         d_true=d_true,
         n_la_true=n_la_true,
         n_mg_true=n_mg_true,
         R_sample=R_sample,
         d_pred=d_pred,
         n_la_pred=n_la_pred,
         n_mg_pred=n_mg_pred,
         idx_best=idx_best)
print("Saved: robust_results.npz")
print("\n===== ALL DONE =====")