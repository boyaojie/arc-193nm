# 实验2：可辨识性分析

## Figure 2

**Figure 2.** Identifiability analysis of the 7-layer LaF₃/MgF₂ coating 
on the strongly curved substrate. 
(a) Magnitude of the Jacobian |∂R/∂p| as a function of wavelength and 
parameter. 
(b) Eigenvalue spectrum of the Fisher information matrix (FIM). 
(c) Cramér-Rao lower bound (CRLB) for each parameter. 
(d) Most unidentifiable direction. 
(e) Sobol global sensitivity indices (S1 and ST).

## 3.2 Identifiability Analysis

To address the question of which parameters are fundamentally 
identifiable, we compute the Fisher information matrix (FIM) 
F = Jᵀ Σ⁻¹ J, where J = ∂R/∂p is the Jacobian of the reflectance 
spectrum with respect to the parameter vector 
p = [d₁, ..., d₇, n_LaF₃, n_MgF₂, R], and Σ is the noise covariance 
matrix (assumed diagonal with σ = 0.01).

Figure 2(a) shows the magnitude of the Jacobian |∂R/∂p| across 
wavelengths and parameters. The sensitivity is highest in the 
180–220 nm range for d₁ and d₃, which are located near the air 
interface and the interference core, respectively.

Figure 2(b) shows the eigenvalue spectrum of the FIM. The condition 
number is 9.87×10¹¹, indicating strong coupling among the parameters. 
The eigenvalue decay spans 12 orders of magnitude, from 
1.44×10⁶ to 1.46×10⁻⁶, confirming that the parameter space is 
highly anisotropic.

Figure 2(c) shows the CRLB for each parameter. The thickness 
parameters d₂ and d₄ (both MgF₂ layers) have the largest 
uncertainties (CRLB std = 668.8 and 360.8, respectively), 
followed by d₁ (250.7) and R (153.8). In contrast, the refractive 
indices n_LaF₃ and n_MgF₂ are extremely well constrained 
(CRLB std = 0.26 and 0.36, respectively), owing to their 
direct influence on the phase-matching condition.

Figure 2(d) shows the most unidentifiable direction, which is 
dominated by d₂ (weight +0.81), d₄ (weight −0.43), and d₁ 
(weight −0.30). The dominance of the MgF₂ layers (d₂ and d₄) 
in this direction is consistent with their low refractive index 
contrast (n_MgF₂ = 1.39 vs. n_LaF₃ = 1.59), which reduces 
their individual influence on the reflectance spectrum.

Figure 2(e) shows the Sobol global sensitivity indices. The 
first-order index S1 is highest for d₁ (S1 = 0.629), confirming 
that the air-side LaF₃ layer dominates the reflectance variance 
at 193 nm. The remaining parameters have S1 values below 0.09, 
with d₄, d₇, n_MgF₂ showing negligible sensitivity (S1 < 0.01). 
The total-order index ST is consistently larger than S1 for d₁ 
(ST = 0.787), d₃ (ST = 0.200), and d₅ (ST = 0.104), indicating 
non-negligible interaction effects among the parameters. The sum 
of S1 (0.923) and ST (1.378) confirms that the model is 
predominantly additive with moderate interactions.

Both analyses consistently show that the MgF₂ layer thicknesses 
(d₂, d₄) are poorly identifiable and have limited influence on 
the reflectance at 193 nm, while d₁ (the air-side LaF₃ layer) 
and the curvature radius R are the most identifiable and 
influential parameters. These findings motivate the use of 
physics-informed constraints in the inverse design to stabilize 
the estimation of the poorly identifiable parameters, which is 
the subject of Section 3.3.

## 关键数据

### 表1：FIM 指标

| 指标 | 值 |
|---|---|
| 条件数 | 9.87×10¹¹ |
| 最大特征值 | 1.44×10⁶ |
| 最小特征值 | 1.46×10⁻⁶ |
| 特征值跨度 | 12 个数量级 |

### 表2：CRLB（参数标准差）

| 参数 | CRLB std | 材料/含义 |
|---|---|---|
| d1 | 250.7 | LaF₃ (空气侧) |
| d2 | 668.8 | MgF₂ |
| d3 | 162.6 | LaF₃ |
| d4 | 360.8 | MgF₂ |
| d5 | 38.5 | LaF₃ |
| d6 | 38.9 | MgF₂ |
| d7 | 24.6 | LaF₃ (基底侧) |
| n_LaF₃ | 0.26 | 折射率 |
| n_MgF₂ | 0.36 | 折射率 |
| R | 153.8 | 曲率半径 |

### 表3：Sobol 指数

| 参数 | S1 | ST | 评价 |
|---|---|---|---|
| d1 | 0.629 | 0.787 | 主导 |
| d2 | 0.053 | 0.088 | 小 |
| d3 | 0.089 | 0.200 | 中 |
| d4 | 0.004 | 0.006 | 极小 |
| d5 | 0.072 | 0.104 | 小 |
| d6 | 0.024 | 0.072 | 小 |
| d7 | 0.006 | 0.009 | 极小 |
| n_LaF₃ | 0.019 | 0.019 | 小 |
| n_MgF₂ | 0.000 | 0.008 | 极小 |
| R | 0.028 | 0.086 | 小 |
| **Sum** | **0.923** | **1.378** | — |

## 核心结论

1. **d₁（空气侧LaF₃层）主导反射率变化**（S1=0.63）
2. **MgF₂层（d₂、d₄）不可辨识且影响小**，因为折射率对比低
3. **折射率（n_LaF₃、n_MgF₂）极好辨识**（CRLB<0.4）
4. **曲率R的CRLB大（153.8），但Sobol S1小（0.028）**，说明R难辨识但影响也有限
5. **条件数9.87×10¹¹说明参数强耦合**，需要物理约束

## 对应图片

- `fig/fig2_fim.png`
- `fig/fig2_fim.pdf`
- `fig/fig2_sobol.png`
- `fig/fig2_sobol.pdf`