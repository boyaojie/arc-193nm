# 实验7：鲁棒性分析

## Figure 7

**Figure 7.** Robustness analysis of the optimal PINN design. 
(a) Average reflectance distribution under ±3% thickness and ±0.5% 
refractive index perturbations (n = 500 Monte Carlo trials). 
(b) Uniformity distribution, defined as (1 - std/mean) × 100%. 
(c) Maximum reflectance distribution. 
(d) Thermal stability at 20°C and 40°C.

## 3.7 Robustness Analysis

To assess the robustness of the optimal design under realistic 
fabrication tolerances, we perform a Monte Carlo analysis with 
500 independent trials. Each trial introduces independent random 
perturbations to the seven layer thicknesses (±3%) and the two 
refractive indices (±0.5%), following a truncated normal 
distribution representative of semiconductor thin-film deposition.

Figure 7(a) shows the distribution of the full-aperture average 
reflectance. The nominal value is 3.44%, and the 95% confidence 
interval is [3.23%, 4.06%]. The mean shift is only 0.10%, 
demonstrating that the design is highly insensitive to thickness 
and refractive-index errors.

Figure 7(b) shows the uniformity distribution, defined as 
(1 - std/mean) × 100%. The nominal uniformity is 70.06%, and 
the 95% confidence interval spans [55.68%, 72.35%].

Figure 7(c) shows the maximum reflectance distribution. The 
nominal value is 6.46%, and the 95% confidence interval is 
[5.88%, 9.84%], keeping the upper bound of large-angle stray 
light below 10%.

Figure 7(d) compares the reflectance spectra at 20°C and 40°C. 
The two curves almost perfectly overlap, with a thermal drift 
of only 0.0063%. This confirms the high thermal stability of 
the design, attributed to the low thermo-optic coefficients 
of LaF₃ and MgF₂.

To further validate the design margin, we repeat the Monte Carlo 
analysis with harsher perturbations (±5% thickness, ±1% index). 
Under these conditions, the average reflectance remains within 
[3.16%, 5.01%], and the maximum reflectance stays below 13.08% 
at the 95% confidence level, confirming that the design retains 
acceptable performance even under relaxed fabrication tolerances.

## 关键数据

### 表10：鲁棒性指标（±3% / ±0.5%）

| 指标 | 标称值 | 均值 | 标准差 | 95% CI |
|---|---|---|---|---|
| 平均反射率 (%) | 3.44 | 3.54 | 0.21 | [3.23, 4.06] |
| 均匀性 (%) | 70.06 | 66.00 | 4.52 | [55.68, 72.35] |
| 最大反射率 (%) | 6.46 | 7.36 | 1.04 | [5.88, 9.84] |
| 热漂移 (%) | 0.0063 | — | — | — |

### 表11：鲁棒性指标（±5% / ±1%）

| 指标 | 均值 | 标准差 | 95% CI |
|---|---|---|---|
| 平均反射率 (%) | 3.75 | 0.45 | [3.16, 5.01] |
| 最大反射率 (%) | 8.42 | 2.03 | [5.65, 13.08] |
| 均匀性 (%) | 60.37 | 9.05 | [38.81, 73.20] |

## 核心结论

1. **标准误差（±3% / ±0.5%）**：平均反射率 95% CI 在 [3.23%, 4.06%] 内
2. **严苛误差（±5% / ±1%）**：平均反射率 95% CI 在 [3.16%, 5.01%] 内
3. **均匀性**：定义 (1 - std/mean) × 100%，标准误差下均值 66.00%
4. **最大反射率**：标准误差下 95% CI 上限 9.84% < 10%，严苛误差下上限 13.08%
5. **热漂移**：0.0063%，温度不敏感
6. **鲁棒性验证**：工艺误差下性能稳定，创新点C 成立