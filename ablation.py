# ablation.py
# 实验6：曲率条件化消融实验（原版，50000 样本，300 epochs）
# 条件化 vs 非条件化 PINN

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import WLS, WL_TARGET, FIG_DIR, SEED

os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})

# ============================================================
# 参数（和实验5 一致：50000 样本，300 epochs）
# ============================================================
N_USE = 50000
EPOCHS = 300
BATCH_SIZE = 256

print(f"===== Ablation (50000 samples, 300 epochs) =====")
print(f"N_USE = {N_USE}, EPOCHS = {EPOCHS}")

# ============================================================
# 加载数据
# ============================================================
data = np.load('dataset.npz')
spectra = data['spectra'][:N_USE]
params = data['params'][:N_USE]

print("Data shape:", spectra.shape, params.shape)

# 输出归一化
Y_raw = params[:, :9].astype(np.float32)
Y_min = Y_raw.min(axis=0)
Y_max = Y_raw.max(axis=0)
Y = (Y_raw - Y_min) / (Y_max - Y_min + 1e-8)

# 两种输入
R_norm = params[:, 9:10] / 200.0
X_cond = np.concatenate([spectra, R_norm], axis=1).astype(np.float32)  # 含 R
X_nocond = spectra.astype(np.float32)                                   # 不含 R

print("X_cond:", X_cond.shape)
print("X_nocond:", X_nocond.shape)

# ============================================================
# 划分（两个模型用同一划分，保证公平）
# ============================================================
N = len(X_cond)
idx = np.random.permutation(N)
n_train = int(0.8 * N)
train_idx, test_idx = idx[:n_train], idx[n_train:]

# ============================================================
# 网络（和实验5 一致：更深、更宽）
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


# ============================================================
# 训练函数（纯数据驱动，无物理损失）
# ============================================================
def train_model(X, Y, name, epochs=EPOCHS, lr=5e-4):
    print(f"\n===== Training {name} =====")

    X_train = torch.tensor(X[train_idx])
    Y_train = torch.tensor(Y[train_idx])
    X_test = torch.tensor(X[test_idx])
    Y_test_t = torch.tensor(Y[test_idx])

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = PINN(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    losses = []
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            d_pred = model(xb)
            loss = loss_fn(d_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        avg = total / len(train_idx)
        losses.append(avg)
        scheduler.step()

        if epoch % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1) / 60
            print(f"  Epoch {epoch}, Loss={avg:.6f}, "
                  f"Elapsed={elapsed/60:.1f}min, ETA={eta:.1f}min")

    print(f"  Training {name} done in {(time.time()-t_start)/60:.1f} min")

    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(X_test).cpu().numpy()

    pred_denorm = pred * (Y_max - Y_min) + Y_min
    Y_test_denorm = Y_test_t.numpy() * (Y_max - Y_min) + Y_min

    r2 = [r2_score(Y_test_denorm[:, i], pred_denorm[:, i])
          for i in range(Y.shape[1])]

    print(f"\n{name} R²:")
    param_names = [f"d{i+1}" for i in range(7)] + ["n_LaF3", "n_MgF2"]
    for nm, r in zip(param_names, r2):
        print(f"  {nm}: {r:.4f}")

    return model, r2, losses


# ============================================================
# 训练两个模型
# ============================================================
model_cond, r2_cond, losses_cond = train_model(
    X_cond, Y, "Conditioned PINN (with R)")

model_nocond, r2_nocond, losses_nocond = train_model(
    X_nocond, Y, "Non-conditioned PINN (without R)")

# ============================================================
# 保存结果
# ============================================================
np.savez('ablation_results.npz',
         r2_cond=np.array(r2_cond),
         r2_nocond=np.array(r2_nocond),
         losses_cond=np.array(losses_cond),
         losses_nocond=np.array(losses_nocond))
print("Saved: ablation_results.npz")

# ============================================================
# 图6：消融实验
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

param_names = [f"d{i+1}" for i in range(7)] + ["n_LaF3", "n_MgF2"]
x = np.arange(len(param_names))
w = 0.35

# (a) R² 对比
axes[0].bar(x - w/2, r2_nocond, w,
            label='Non-conditioned (without R)',
            color='steelblue', edgecolor='k')
axes[0].bar(x + w/2, r2_cond, w,
            label='Conditioned (with R)',
            color='coral', edgecolor='k')
axes[0].set_xticks(x)
axes[0].set_xticklabels(param_names, rotation=45, ha='right')
axes[0].set_ylabel('R²')
axes[0].set_title('(a) R² comparison: conditioned vs non-conditioned')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, 1.05)

# (b) 训练曲线
axes[1].plot(losses_nocond, 'b-', linewidth=1.5, label='Non-conditioned')
axes[1].plot(losses_cond, 'r-', linewidth=1.5, label='Conditioned')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Total Loss')
axes[1].set_title('(b) Training curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
png_path = os.path.join(FIG_DIR, 'fig6_ablation.png')
pdf_path = os.path.join(FIG_DIR, 'fig6_ablation.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")
print("\n===== ALL DONE =====")