# 实验4：数据集与基线性能

## Figure 4

**Figure 4.** Dataset distribution and baseline MLP performance. 
(a) Training curve. (b) R² per parameter. (c) Prediction vs truth (d1). 
(d) Curvature distribution.

## 3.4 Dataset and Baseline Performance

We generated a dataset of 50,000 samples, each consisting of a 
reflectance spectrum (400 wavelengths from 150 to 250 nm) and the 
corresponding parameter vector p = [d₁, ..., d₇, n_LaF₃, n_MgF₂, R]. 
The thicknesses were sampled from ±20% of the reference values 
from Hong et al. [20], the refractive indices from ±1%, and the 
curvature radius from R ∈ [20, 200] mm. The spherical average 
reflectance was computed at n_r = 4 radial positions to ensure 
accuracy, and Gaussian noise (σ = 0.005) was added to simulate 
measurement uncertainty.

A baseline MLP with 4 hidden layers (512-512-256-128) and BatchNorm 
was trained for 500 epochs using the Adam optimizer with a learning 
rate of 10⁻³ and a reduce-on-plateau scheduler. The dataset was 
split into 80% training and 20% testing.

Figure 4(a) shows the training curve, which converges to a final 
MSE of 0.0312. Figure 4(b) shows the R² for each parameter. The 
thickness parameters d₁, d₃, d₅, d₆ achieve R² of 0.879, 0.812, 
0.854, and 0.877, respectively, while d₂ and d₄ (both MgF₂ layers) 
achieve R² of only 0.163 and 0.191. The refractive indices n_LaF₃ 
and n_MgF₂ achieve R² of 0.188 and 0.099. The thickness d₇ achieves 
R² of 0.442.

These results are consistent with the identifiability analysis in 
Section 3.2, which identified the MgF₂ layers (d₂, d₄) as the 
most poorly identifiable parameters (CRLB std = 668.8 and 360.8, 
respectively). The low R² for these layers reflects their low 
refractive index contrast with LaF₃ (1.39 vs. 1.59), which 
reduces their influence on the reflectance spectrum. This directly 
motivates the physics-informed neural network in Section 3.5, 
which embeds the TMM forward model as a physical constraint to 
stabilize the estimation of these poorly identifiable parameters.

Figure 4(c) shows the prediction vs truth scatter plot for d₁, 
confirming the good agreement along the diagonal. Figure 4(d) 
shows the curvature distribution, which is uniform over [20, 200] mm.

The baseline performance establishes a reference for the 
physics-informed neural network in Section 3.5.

## 关键数据

### 表7：基线 R²

| 参数 | R² | 材料 | 评价 |
|---|---|---|---|
| d1 | 0.879 | LaF₃ | 好 |
| d2 | 0.163 | MgF₂ | 差（不可辨识） |
| d3 | 0.812 | LaF₃ | 好 |
| d4 | 0.191 | MgF₂ | 差（不可辨识） |
| d5 | 0.854 | LaF₃ | 好 |
| d6 | 0.877 | MgF₂ | 好 |
| d7 | 0.442 | LaF₃ | 中 |
| n_LaF₃ | 0.188 | 折射率 | 差（范围太小） |
| n_MgF₂ | 0.099 | 折射率 | 差（范围太小） |

## 方法要点

- **数据集规模：** 50,000 样本
- **采样范围：** 厚度 ±20%，折射率 ±1%，R ∈ [20, 200] mm
- **球面平均：** n_r = 4 径向采样点
- **噪声：** σ = 0.005
- **网络：** MLP，512-512-256-128，BatchNorm，Dropout 0.1
- **训练：** Adam，lr=10⁻³，500 epochs，ReduceLROnPlateau
- **划分：** 80% 训练 / 20% 测试
- **最终loss：** 0.0312

## 核心结论

1. **d1、d3、d5、d6 的 R² > 0.8**，可辨识性好
2. **d2、d4 的 R² < 0.2**，MgF₂层不可辨识（与实验2一致）
3. **n_LaF₃、n_MgF₂ 的 R² < 0.2**，因折射率范围太小（±1%）
4. **低 R² 参数正是实验2中的不可辨识参数**，为PINN提供动机

## 对应图片

- `fig/fig4_baseline.png`
- `fig/fig4_baseline.pdf`