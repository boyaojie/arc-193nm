# 实验3：曲率对可辨识性的影响

## Figure 3

**Figure 3.** Effect of curvature and edge incident angle on 
identifiability. 
(a) FIM condition number vs curvature radius R for two groups: 
fixed edge angle (44.4°, blue circles) and fixed aperture (105 mm, 
orange squares). 
(b) Maximum and minimum FIM eigenvalues vs R. 
(c) FIM condition number vs edge incident angle θ_edge.

## 3.2 (Continued) Curvature Dependence

To disentangle the effects of curvature radius and edge incident 
angle on identifiability, we compute the FIM condition number for 
two groups of geometries:

**Group A (fixed edge angle):** R ∈ [20, 150] mm with 
r_max = 0.7R, so that the edge angle is fixed at 44.4°.

**Group B (fixed aperture):** R ∈ [150, 1000] mm with 
r_max = 105 mm, so that the edge angle decreases from 44.4° 
to 6.0°.

Figure 3(a) shows the condition number as a function of R. In 
Group A, the condition number remains nearly constant, varying 
by less than 0.4% from 2.199×10¹² at R = 20 mm to 2.207×10¹² 
at R = 150 mm, despite a 7.5× variation in curvature radius. This 
demonstrates that, **for a fixed edge incident angle, the 
identifiability is essentially independent of the curvature 
radius itself**.

In contrast, Group B shows a non-monotonic dependence. The 
condition number drops sharply from 2.207×10¹² at R = 150 mm 
(θ_edge = 44.4°) to 1.189×10¹⁰ at R = 200 mm (θ_edge = 31.7°), 
then rises to 3.417×10¹¹ at R = 300 mm (θ_edge = 20.5°), remains 
at 3.119×10¹¹ at R = 500 mm (θ_edge = 12.1°), and finally rises 
to 3.051×10¹² at R = 1000 mm (θ_edge = 6.0°). The minimum 
condition number occurs at R = 200 mm (θ_edge ≈ 32°), where 
both angle-thickness coupling and thickness-index coupling are 
moderately weak. For R > 500 mm, the condition number increases 
again as the system approaches normal incidence, where the 
thickness and refractive index parameters become strongly 
coupled and the curvature radius itself becomes unidentifiable.

Figure 3(b) shows the maximum and minimum FIM eigenvalues. In 
Group A, both eigenvalues are nearly constant, with λ_min ranging 
from 3.834×10⁻⁷ to 3.848×10⁻⁷ and λ_max around 8.5×10⁵. In 
Group B, λ_min increases by two orders of magnitude from 
3.834×10⁻⁷ at R = 150 mm to 5.440×10⁻⁵ at R = 200 mm, then 
decreases again for larger R.

Figure 3(c) plots the condition number against the edge incident 
angle for both groups. The A-group data collapse onto a nearly 
constant value at θ_edge = 44.4°, while the B-group data exhibit 
a non-monotonic U-shaped trend with a minimum around 30°. This 
reveals a fundamental trade-off: **identifiability is governed 
primarily by the edge incident angle, but the optimum occurs at 
an intermediate angle rather than at normal incidence**.

The most unidentifiable direction at R = 20 mm 
(θ_edge = 44.4°) is dominated by d₂ (weight −0.86), a MgF₂ layer, 
followed by d₁ (weight +0.31) and d₄ (weight +0.30). At 
R = 1000 mm (θ_edge = 6.0°), the dominant component shifts to 
R (weight +0.90), indicating that under near-normal incidence, 
the curvature radius itself becomes the least identifiable 
parameter, while the thickness parameters become relatively 
better constrained.

These results reveal a fundamental trade-off in AR coating 
design for strongly curved substrates. For the target 
application at R = 150 mm with a 105 mm aperture (θ_edge = 44.4°), 
the condition number is 2.207×10¹², and the MgF₂ layer 
thicknesses (d₂ and d₄) are the most poorly identifiable 
parameters. The identifiability can be improved by reducing 
the edge angle to ~30°, corresponding to a curvature radius of 
R ≈ 200 mm for the same aperture. However, this would reduce 
the optical power of the lens element, illustrating the 
design trade-off.

## 关键数据

### 表1：A组（固定边缘角44.4°）

| R (mm) | r_max (mm) | θ_edge (°) | 条件数 | λ_min |
|---|---|---|---|---|
| 20 | 14.0 | 44.43 | 2.199×10¹² | 3.848×10⁻⁷ |
| 30 | 21.0 | 44.43 | 2.199×10¹² | 3.848×10⁻⁷ |
| 50 | 35.0 | 44.43 | 2.201×10¹² | 3.845×10⁻⁷ |
| 75 | 52.5 | 44.43 | 2.202×10¹² | 3.843×10⁻⁷ |
| 100 | 70.0 | 44.43 | 2.202×10¹² | 3.842×10⁻⁷ |
| 125 | 87.5 | 44.43 | 2.205×10¹² | 3.837×10⁻⁷ |
| 150 | 105.0 | 44.43 | 2.207×10¹² | 3.834×10⁻⁷ |

### 表2：B组（固定孔径105 mm）

| R (mm) | θ_edge (°) | 条件数 | λ_min |
|---|---|---|---|
| 150 | 44.43 | 2.207×10¹² | 3.834×10⁻⁷ |
| 200 | 31.67 | 1.189×10¹⁰ | 5.440×10⁻⁵ |
| 300 | 20.49 | 3.417×10¹¹ | 1.388×10⁻⁶ |
| 500 | 12.12 | 3.119×10¹¹ | 1.565×10⁻⁶ |
| 1000 | 6.03 | 3.051×10¹² | 2.273×10⁻⁷ |

### 表3：不可辨识方向对比

| 参数 | R=20 mm (θ=44.4°) | R=1000 mm (θ=6.0°) |
|---|---|---|
| d1 | +0.3143 | -0.1350 |
| d2 | **-0.8639** | +0.3741 |
| d3 | +0.2500 | -0.0921 |
| d4 | +0.3000 | -0.1559 |
| d5 | -0.0453 | +0.0339 |
| d6 | -0.0113 | +0.0194 |
| d7 | +0.0114 | +0.0148 |
| n_LaF3 | +0.0001 | -0.0001 |
| n_MgF2 | -0.0004 | +0.0001 |
| R | -0.0099 | **+0.8985** |

## 核心结论

1. **固定边缘角下，条件数与曲率半径R无关**（A组变化<0.4%）
2. **固定孔径下，条件数呈U型：在θ_edge≈32°处最小**（B组）
3. **可辨识性主要由边缘入射角决定，而非R本身**
4. **强曲率（R=20 mm）下，MgF₂层（d₂）最不可辨识**
5. **近法向入射（R=1000 mm）下，曲率半径R本身最不可辨识**
6. **目标应用（R=150 mm，θ=44.4°）下，条件数2.207×10¹²，MgF₂层为最弱参数**
7. **存在设计权衡：减小边缘角到30°可改善可辨识性，但会降低光学元件的屈光力**

## 对应图片

- `fig/fig3_curvature.png`
- `fig/fig3_curvature.pdf`