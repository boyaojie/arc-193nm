# 实验1：正向模型验证

## Figure 1

**Figure 1.** Forward model validation for the 7-layer LaF₃/MgF₂ coating 
on a strongly curved substrate (R = 150 mm, aperture = 105 mm). 
(a) Radial distribution of the incident angle. 
(b) R → ∞ degeneration validation. 
(c) Spherical reflectance spectra at six radial positions (s+p averaged). 
(d) Film thickness deviation along the radial direction.

## 3.1 Forward Model Validation

We validate the forward model using the 7-layer LaF₃/MgF₂ structure 
from Hong et al. [3]. Figure 1(a) shows the incident angle distribution, 
which rises nonlinearly from 0° at the paraxial region to 44.43° at the 
aperture edge (r = 105 mm), consistent with the geometry in [3]. 
The edge region (r > 84 mm) occupies only 20% of the aperture but 
accounts for nearly 30% of the total angular variation.

Figure 1(b) compares the planar reflectance with the spherical 
reflectance at R = 1×10⁶ mm. The two spectra overlap almost perfectly, 
with a maximum deviation of 3.39×10⁻¹³, confirming that the spherical 
model correctly degenerates to the planar case.

Figure 1(c) presents the spherical reflectance spectra at six radial 
positions, computed as the average of s- and p-polarized light. 
As r increases, the spectra shift systematically due to the 
angle-thickness coupling. Figure 1(d) shows the film thickness 
deviation δ(r), which increases from 1.0 at the center to 1.1 at the 
edge, consistent with the cosine law for planetary deposition.

The computed full-aperture average reflectance is 1.3612% at 193 nm, 
and the maximum high-angle reflectance (θ > 40°) is 1.5735%, both 
in excellent agreement with the values reported by Hong et al. [3] 
(1.3633% and 1.5749%, respectively). The reflectance at different 
radial positions ranges from 1.0556% at the center to 1.5735% at the 
edge, exhibiting a non-monotonic trend due to the combined effect of 
the incident-angle variation and the film thickness deviation. These 
results establish the correctness of the forward model and provide 
the basis for the identifiability analysis in Section 3.2.

## 关键数据

### 表1：正向模型验证结果

| 项 | 本文结果 | Hong et al. [3] | 差异 |
|---|---|---|---|
| 边缘入射角 | 44.4270° | 44.4° | 0.027° |
| R→∞退化误差 | 3.39×10⁻¹³ | — | — |
| 平面反射率（193 nm） | 1.0556% | — | — |
| 球面平均反射率 | 1.3612% | 1.3633% | 0.0021% |
| 高角反射率（θ>40°） | 1.5735% | 1.5749% | 0.0014% |
| 膜厚偏差范围 | 1.0–1.1 | 1.0–1.1 | 一致 |

### 表2：各径向位置的193 nm反射率

| r (mm) | θ (°) | R(193nm) |
|---|---|---|
| 0 | 0.00 | 1.0556% |
| 20 | 7.66 | 1.4585% |
| 40 | 15.47 | 1.3742% |
| 60 | 23.58 | 1.3077% |
| 80 | 32.23 | 1.4067% |
| 105 | 44.43 | 1.5735% |

## 对应图片

- `fig/fig1_forward.png`
- `fig/fig1_forward.pdf`