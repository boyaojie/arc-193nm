# pinn.py
# 实验5：PINN（PyTorch 版，50000 样本，300 epochs）
# 输出：fig/fig5_pinn.png

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
# 参数
# ============================================================
N_USE = 50000
EPOCHS = 300
BATCH_SIZE = 256
LR = 5e-4

print(f"===== PINN (PyTorch, 50000 samples) =====")
print(f"N_USE = {N_USE}, EPOCHS = {EPOCHS}, BATCH = {BATCH_SIZE}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================
# 加载数据
# ============================================================
t0 = time.time()
data = np.load('dataset.npz')
spectra = data['spectra'][:N_USE]
params = data['params'][:N_USE]
print(f"Data loaded in {time.time()-t0:.2f}s")
print("Data shape:", spectra.shape, params.shape)

X = np.concatenate([
    spectra,
    params[:, 9:10] / 200.0
], axis=1).astype(np.float32)

Y_raw = params[:, :9].astype(np.float32)
Y_min = Y_raw.min(axis=0)
Y_max = Y_raw.max(axis=0)
Y = (Y_raw - Y_min) / (Y_max - Y_min + 1e-8)

np.savez('norm_params_pinn.npz', Y_min=Y_min, Y_max=Y_max)
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
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 网络
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

model = PINN(X.shape[1], Y.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=EPOCHS, eta_min=1e-5)
loss_fn = nn.MSELoss()

# ============================================================
# 训练
# ============================================================
print("\n===== Training =====")
train_losses = []
t_start = time.time()

for epoch in range(EPOCHS):
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
    train_losses.append(avg)
    scheduler.step()

    if epoch % 20 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1) / 60
        print(f"Epoch {epoch}, Loss={avg:.6f}, "
              f"Elapsed={elapsed/60:.1f}min, ETA={eta:.1f}min")

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), f'pinn_epoch{epoch+1}.pt')
        print(f"  Checkpoint saved at epoch {epoch+1}")

print(f"\nTraining done in {(time.time()-t_start)/60:.1f} min")

# ============================================================
# 评估
# ============================================================
model.eval()
with torch.no_grad():
    pred = model(X_test).cpu().numpy()

pred_denorm = pred * (Y_max - Y_min) + Y_min
Y_test_denorm = Y_test.numpy() * (Y_max - Y_min) + Y_min

param_names = [f"d{i+1}" for i in range(7)] + ["n_LaF3", "n_MgF2"]
r2 = [r2_score(Y_test_denorm[:, i], pred_denorm[:, i])
      for i in range(Y.shape[1])]

print("\n===== PINN R² =====")
for name, r in zip(param_names, r2):
    print(f"  {name}: {r:.4f}")

torch.save(model.state_dict(), 'pinn.pt')
np.savez('pinn_results.npz', r2=r2, param_names=param_names)
print("Saved: pinn.pt, pinn_results.npz")

# ============================================================
# 图5
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].plot(train_losses, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('(a) Training curve')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

colors_bar = ['steelblue' if r > 0.7 else 'coral' for r in r2]
axes[0, 1].bar(np.arange(len(param_names)), r2,
               color=colors_bar, edgecolor='k')
axes[0, 1].set_xticks(np.arange(len(param_names)))
axes[0, 1].set_xticklabels(param_names, rotation=45, ha='right')
axes[0, 1].set_ylabel('R²')
axes[0, 1].set_title('(b) R² per parameter')
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[1, 0].plot(WLS, spectra[test_idx[0]] * 100, 'b-',
                linewidth=1.5, label='Target')
d_pred_real = pred_denorm[0, :7]
R_sample = params[test_idx[0], 9]
layers_pred = [
    (1.590, d_pred_real[0]), (1.390, d_pred_real[1]), (1.590, d_pred_real[2]),
    (1.390, d_pred_real[3]), (1.590, d_pred_real[4]), (1.390, d_pred_real[5]),
    (1.590, d_pred_real[6])
]
spec_recon, _ = sphere_average_reflectance(
    WLS, layers_pred, R_sample,
    r_max=min(105.0, 0.7 * R_sample), n_r=2, pol='avg')
axes[1, 0].plot(WLS, spec_recon * 100, 'r--',
                linewidth=1.5, label='Reconstructed')
axes[1, 0].axvline(x=WL_TARGET, color='k', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Wavelength (nm)')
axes[1, 0].set_ylabel('Reflectance (%)')
axes[1, 0].set_title('(c) Spectrum reconstruction')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(Y_test_denorm[:, 0], pred_denorm[:, 0],
                   s=3, alpha=0.3, color='coral')
lim = [Y_test_denorm[:, 0].min(), Y_test_denorm[:, 0].max()]
axes[1, 1].plot(lim, lim, 'k--', linewidth=1)
axes[1, 1].set_xlabel('True d1 (nm)')
axes[1, 1].set_ylabel('Predicted d1 (nm)')
axes[1, 1].set_title('(d) Prediction vs truth (d1)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_pinn.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig5_pinn.pdf'), format='pdf', bbox_inches='tight')
print(f"\nSaved: {FIG_DIR}fig5_pinn.png")
print(f"Saved: {FIG_DIR}fig5_pinn.pdf")
print("\n===== ALL DONE =====")