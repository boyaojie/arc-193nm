# baseline.py
# 实验4：基线MLP
# 输出归一化 + 更大网络 + scheduler

import os
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

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

# ============================================================
# 加载数据
# ============================================================
data = np.load('dataset.npz')
spectra = data['spectra']
params = data['params']

print("Data shape:", spectra.shape, params.shape)

# 输入：[spectra, R_norm]
X = np.concatenate([
    spectra,
    params[:, 9:10] / 200.0
], axis=1).astype(np.float32)

# 输出：归一化到[0,1]
Y_raw = params[:, :9].astype(np.float32)
Y_min = Y_raw.min(axis=0)
Y_max = Y_raw.max(axis=0)
Y = (Y_raw - Y_min) / (Y_max - Y_min + 1e-8)

np.savez('norm_params.npz', Y_min=Y_min, Y_max=Y_max)

print("X shape:", X.shape, "Y shape:", Y.shape)

# ============================================================
# 划分
# ============================================================
N = len(X)
idx = np.random.permutation(N)
n_train = int(0.8 * N)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_train = torch.tensor(X[train_idx])
Y_train = torch.tensor(Y[train_idx])
X_test = torch.tensor(X[test_idx])
Y_test = torch.tensor(Y[test_idx])

train_ds = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# ============================================================
# MLP（更大 + BatchNorm）
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1),
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

model = MLP(X.shape[1], Y.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=30)
loss_fn = nn.MSELoss()

# ============================================================
# 训练
# ============================================================
epochs = 500
train_losses = []

for epoch in range(epochs):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    avg = total / len(train_idx)
    train_losses.append(avg)
    scheduler.step(avg)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss={avg:.6f}")

# ============================================================
# 评估（反归一化）
# ============================================================
model.eval()
with torch.no_grad():
    pred = model(X_test.to(device)).cpu().numpy()

pred_denorm = pred * (Y_max - Y_min) + Y_min
Y_test_denorm = Y_test.numpy() * (Y_max - Y_min) + Y_min

param_names = [f"d{i+1}" for i in range(7)] + ["n_LaF3", "n_MgF2"]
r2 = [r2_score(Y_test_denorm[:, i], pred_denorm[:, i])
      for i in range(Y.shape[1])]

print("\n===== Baseline R² =====")
for name, r in zip(param_names, r2):
    print(f"  {name}: {r:.4f}")

torch.save(model.state_dict(), 'baseline_mlp.pt')
print("Saved: baseline_mlp.pt")

# ============================================================
# 图4：4子图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) 训练曲线
axes[0, 0].plot(train_losses, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE loss')
axes[0, 0].set_title('(a) Training curve')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# (b) R²
colors_bar = ['steelblue' if r > 0.8 else 'coral' for r in r2]
axes[0, 1].bar(np.arange(len(param_names)), r2,
               color=colors_bar, edgecolor='k')
axes[0, 1].set_xticks(np.arange(len(param_names)))
axes[0, 1].set_xticklabels(param_names, rotation=45, ha='right')
axes[0, 1].set_ylabel('R²')
axes[0, 1].set_title('(b) Baseline R² per parameter')
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5,
                   label='R²=0.8')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# (c) 预测 vs 真实（d1）
axes[1, 0].scatter(Y_test_denorm[:, 0], pred_denorm[:, 0],
                   s=3, alpha=0.3, color='steelblue')
lim = [Y_test_denorm[:, 0].min(), Y_test_denorm[:, 0].max()]
axes[1, 0].plot(lim, lim, 'r--', linewidth=1)
axes[1, 0].set_xlabel('True d1 (nm)')
axes[1, 0].set_ylabel('Predicted d1 (nm)')
axes[1, 0].set_title('(c) Prediction vs truth (d1)')
axes[1, 0].grid(True, alpha=0.3)

# (d) R分布
axes[1, 1].hist(params[:, 9], bins=50, color='coral', edgecolor='k')
axes[1, 1].set_xlabel('R (mm)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('(d) Curvature distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
png_path = os.path.join(FIG_DIR, 'fig4_baseline.png')
pdf_path = os.path.join(FIG_DIR, 'fig4_baseline.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")