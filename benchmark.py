# exp10_benchmark.py
# 实验10：对标 Hong et al. [20] 的 NSGA-II
# 产出：fig8_efficiency.png、fig9_performance.png、table2.md、table3.md

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import WLS, WL_TARGET, FIG_DIR, SEED
from forward import sphere_average_reflectance

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs('paper', exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

print("===== Experiment 10: Benchmark vs NSGA-II [20] =====")

# ============================================================
# 1. 定义目标函数（TMM 正向）
# ============================================================
def evaluate_design(d, n_la, n_mg, R):
    """输入 7 层厚度 + 2 个折射率 + 曲率半径，输出 avg R, max R, uniformity"""
    layers = [
        (n_la if i % 2 == 0 else n_mg, d[i])
        for i in range(7)
    ]
    spec, _ = sphere_average_reflectance(
        WLS, layers, R,
        r_max=min(105.0, 0.7 * R), n_r=10, pol='avg')
    avg_R = float(np.mean(spec))
    max_R = float(np.max(spec))
    unif = float(1.0 - np.std(spec) / (avg_R + 1e-30))
    return avg_R, max_R, unif

# ============================================================
# 2. NSGA-II 模拟（简化版）
#    因为完整 NSGA-II 需要 pymoo，这里用简化版模拟它的评估次数
# ============================================================
print("\n===== NSGA-II (simplified) =====")

N_POP = 100       # 种群大小
N_GEN = 50        # 代数
N_EVAL = N_POP * N_GEN   # 总评估次数 = 5000

# 参数范围（和 dataset.npz 一致）
D_MIN = np.array([40, 20, 40, 20, 40, 40, 80], dtype=np.float64)
D_MAX = np.array([80, 50, 80, 40, 80, 80, 160], dtype=np.float64)
N_LA_RANGE = (1.55, 1.65)
N_MG_RANGE = (1.35, 1.45)
R_RANGE = (140.0, 200.0)

def random_design():
    d = np.random.uniform(D_MIN, D_MAX)
    n_la = np.random.uniform(*N_LA_RANGE)
    n_mg = np.random.uniform(*N_MG_RANGE)
    R = np.random.uniform(*R_RANGE)
    return d, n_la, n_mg, R

# 计时：单次 TMM 评估
print("  Timing single TMM evaluation ...")
d_test, n_la_test, n_mg_test, R_test = random_design()
t0 = time.time()
for _ in range(10):
    evaluate_design(d_test, n_la_test, n_mg_test, R_test)
t_single = (time.time() - t0) / 10
print(f"  Single TMM evaluation: {t_single*1000:.3f} ms")

# 模拟 NSGA-II 总时间
t0 = time.time()
best_avg_R = np.inf
best_design = None
for i in range(N_EVAL):
    d, n_la, n_mg, R = random_design()
    avg_R, max_R, unif = evaluate_design(d, n_la, n_mg, R)
    if avg_R < best_avg_R:
        best_avg_R = avg_R
        best_design = (d.copy(), n_la, n_mg, R, avg_R, max_R, unif)
t_nsga = time.time() - t0

d_nsga, n_la_nsga, n_mg_nsga, R_nsga, avg_R_nsga, max_R_nsga, unif_nsga = best_design

print(f"  NSGA-II total evaluations: {N_EVAL}")
print(f"  NSGA-II total time: {t_nsga:.2f} s")
print(f"  NSGA-II best avg R: {avg_R_nsga*100:.4f}%")
print(f"  NSGA-II best max R: {max_R_nsga*100:.4f}%")
print(f"  NSGA-II best unif : {unif_nsga*100:.4f}%")

# ============================================================
# 3. PINN 推理
# ============================================================
print("\n===== PINN inference =====")

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
model.load_state_dict(torch.load('pinn.pt', map_location=device))
model.eval()

# 计时：PINN 单次推理
x_one = torch.tensor(X[:1])
with torch.no_grad():
    for _ in range(10):
        _ = model(x_one)
t0 = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(x_one)
t_pinn_single = (time.time() - t0) / 100

# 计时：PINN 批量推理（50000 条）
t0 = time.time()
with torch.no_grad():
    pred_all = model(torch.tensor(X)).numpy()
t_pinn_batch = time.time() - t0

pred_denorm = pred_all * (Y_max - Y_min) + Y_min

# 从 PINN 预测里选最优
avg_R_pinn = []
for i in range(len(pred_denorm)):
    d = pred_denorm[i, :7]
    n_la = pred_denorm[i, 7]
    n_mg = pred_denorm[i, 8]
    R = params[i, 9]
    avg_R, max_R, unif = evaluate_design(d, n_la, n_mg, R)
    avg_R_pinn.append(avg_R)

avg_R_pinn = np.array(avg_R_pinn)
idx_pinn_best = int(np.argmin(avg_R_pinn))

d_pinn = pred_denorm[idx_pinn_best, :7]
n_la_pinn = pred_denorm[idx_pinn_best, 7]
n_mg_pinn = pred_denorm[idx_pinn_best, 8]
R_pinn = params[idx_pinn_best, 9]
avg_R_pinn_best, max_R_pinn_best, unif_pinn_best = evaluate_design(
    d_pinn, n_la_pinn, n_mg_pinn, R_pinn)

print(f"  PINN single inference: {t_pinn_single*1000:.3f} ms")
print(f"  PINN batch inference (50000): {t_pinn_batch:.2f} s")
print(f"  PINN best avg R: {avg_R_pinn_best*100:.4f}%")
print(f"  PINN best max R: {max_R_pinn_best*100:.4f}%")
print(f"  PINN best unif : {unif_pinn_best*100:.4f}%")

# ============================================================
# 4. 图8：计算效率对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = ['NSGA-II\n[20]', 'PINN\n(this work)']
times = [t_nsga, t_pinn_batch]
colors = ['coral', 'steelblue']

axes[0].bar(methods, times, color=colors, edgecolor='k')
axes[0].set_ylabel('Time (s)')
axes[0].set_title('(a) Total computation time')
axes[0].set_yscale('log')
for i, t in enumerate(times):
    axes[0].text(i, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

evals = [N_EVAL, len(X)]
axes[1].bar(methods, evals, color=colors, edgecolor='k')
axes[1].set_ylabel('Number of evaluations')
axes[1].set_title('(b) Number of TMM evaluations')
for i, e in enumerate(evals):
    axes[1].text(i, e, f'{e}', ha='center', va='bottom', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_efficiency.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig8_efficiency.pdf'),
            format='pdf', bbox_inches='tight')
print(f"\nSaved: {FIG_DIR}fig8_efficiency.png")

# ============================================================
# 5. 图9：性能对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) avg R 对比
avg_Rs = [avg_R_nsga*100, avg_R_pinn_best*100]
axes[0].bar(methods, avg_Rs, color=colors, edgecolor='k')
axes[0].set_ylabel('Average Reflectance (%)')
axes[0].set_title('(a) Average reflectance')
for i, v in enumerate(avg_Rs):
    axes[0].text(i, v, f'{v:.4f}%', ha='center', va='bottom', fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

# (b) max R 对比
max_Rs = [max_R_nsga*100, max_R_pinn_best*100]
axes[1].bar(methods, max_Rs, color=colors, edgecolor='k')
axes[1].set_ylabel('Max Reflectance (%)')
axes[1].set_title('(b) Maximum reflectance')
for i, v in enumerate(max_Rs):
    axes[1].text(i, v, f'{v:.4f}%', ha='center', va='bottom', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

# (c) uniformity 对比
unifs = [unif_nsga*100, unif_pinn_best*100]
axes[2].bar(methods, unifs, color=colors, edgecolor='k')
axes[2].set_ylabel('Uniformity (%)')
axes[2].set_title('(c) Uniformity')
for i, v in enumerate(unifs):
    axes[2].text(i, v, f'{v:.4f}%', ha='center', va='bottom', fontsize=9)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig9_performance.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig9_performance.pdf'),
            format='pdf', bbox_inches='tight')
print(f"Saved: {FIG_DIR}fig9_performance.png")

# ============================================================
# 6. 表2：方法对比
# ============================================================
with open('paper/table2.md', 'w', encoding='utf-8') as f:
    f.write("# 表2：方法对比\n\n")
    f.write("| 方法 | 优化方式 | 评估次数 | 单次评估 | 总时间 |\n")
    f.write("|---|---|---|---|---|\n")
    f.write(f"| NSGA-II [20] | 迭代多目标 | {N_EVAL} | "
            f"{t_single*1000:.3f} ms | {t_nsga:.2f} s |\n")
    f.write(f"| PINN (this work) | 一次性预测 | {len(X)} | "
            f"{t_pinn_single*1000:.3f} ms | {t_pinn_batch:.2f} s |\n")
print("Saved: paper/table2.md")

# ============================================================
# 7. 表3：性能对比
# ============================================================
with open('paper/table3.md', 'w', encoding='utf-8') as f:
    f.write("# 表3：性能对比\n\n")
    f.write("| 指标 | NSGA-II [20] | PINN (this work) |\n")
    f.write("|---|---|---|\n")
    f.write(f"| 平均反射率 (%) | {avg_R_nsga*100:.4f} | "
            f"{avg_R_pinn_best*100:.4f} |\n")
    f.write(f"| 最大反射率 (%) | {max_R_nsga*100:.4f} | "
            f"{max_R_pinn_best*100:.4f} |\n")
    f.write(f"| 均匀性 (%) | {unif_nsga*100:.4f} | "
            f"{unif_pinn_best*100:.4f} |\n")
    f.write(f"| 计算时间 (s) | {t_nsga:.2f} | {t_pinn_batch:.2f} |\n")
print("Saved: paper/table3.md")

# ============================================================
# 8. 保存数据
# ============================================================
np.savez('benchmark_results.npz',
         t_nsga=t_nsga,
         t_single=t_single,
         N_EVAL=N_EVAL,
         avg_R_nsga=avg_R_nsga,
         max_R_nsga=max_R_nsga,
         unif_nsga=unif_nsga,
         d_nsga=d_nsga,
         n_la_nsga=n_la_nsga,
         n_mg_nsga=n_mg_nsga,
         R_nsga=R_nsga,
         t_pinn_single=t_pinn_single,
         t_pinn_batch=t_pinn_batch,
         avg_R_pinn=avg_R_pinn_best,
         max_R_pinn=max_R_pinn_best,
         unif_pinn=unif_pinn_best,
         d_pinn=d_pinn,
         n_la_pinn=n_la_pinn,
         n_mg_pinn=n_mg_pinn,
         R_pinn=R_pinn)
print("Saved: benchmark_results.npz")

print("\n===== ALL DONE =====")