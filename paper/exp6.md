# 实验6：曲率条件化消融实验

## Figure 6

**Figure 6.** Ablation study of curvature conditioning. 
(a) R² comparison between the conditioned PINN (with R) and the 
non-conditioned PINN (without R). 
(b) Training curves for both models.

## 3.6 Ablation Study

To justify the necessity of curvature conditioning, we compare two 
PINN models with identical architectures, optimizers, and training 
configurations:

- **Conditioned PINN**: input = reflectance spectrum (400 dimensions) 
  concatenated with the normalized curvature radius R (1 dimension).
- **Non-conditioned PINN**: input = reflectance spectrum (400 dimensions) 
  only, with no information about the substrate curvature.

Both models use the same 6-layer architecture (1024-1024-512-256-128) 
with batch normalization and dropout (p = 0.05), trained on 50,000 
samples for 300 epochs using the Adam optimizer with a learning rate 
of 5×10⁻⁴ and a cosine annealing scheduler. The dataset split (80% 
training, 20% testing) is identical for both models to ensure a 
fair comparison. Training was performed using PaddlePaddle on an 
NVIDIA V100 GPU, with each model completing in 8.9 minutes.

Figure 6(a) shows the R² comparison. The conditioned model 
outperforms the non-conditioned one for **all nine parameters**:

| Parameter | Non-conditioned | Conditioned | Improvement |
|---|---|---|---|
| d1 | 0.8934 | **0.9133** | **+0.0199** |
| d2 | 0.1270 | **0.1491** | **+0.0221** |
| d3 | 0.8414 | **0.8583** | **+0.0169** |
| d4 | 0.1381 | **0.1903** | **+0.0522** |
| d5 | 0.9014 | **0.9169** | **+0.0155** |
| d6 | 0.8881 | **0.9151** | **+0.0270** |
| d7 | 0.4606 | **0.5179** | **+0.0573** |
| n_LaF3 | 0.1726 | **0.1867** | **+0.0141** |
| n_MgF2 | 0.0588 | **0.0659** | **+0.0071** |

The average improvement for the well-identifiable thickness 
parameters (d1, d3, d5, d6) is **+0.020**. The most significant 
improvements are observed for **d7 (+0.0573)** and **d4 (+0.0522)**, 
both of which are strongly influenced by the curvature-dependent 
angle-thickness coupling. Even the poorly identifiable parameters 
(d2, n_LaF3, n_MgF2) show consistent improvement, with average 
gains of **+0.014**.

Figure 6(b) shows the training curves. The conditioned model 
converges to a lower final loss of **0.0234**, while the non-
conditioned model reaches **0.0256**. The gap between the two 
curves widens progressively over the 300 epochs, indicating that 
the benefit of curvature conditioning accumulates during training.

These results confirm that incorporating the curvature radius 
as a condition variable is essential for accurate inverse design 
on strongly curved substrates. The consistent improvement across 
all nine parameters demonstrates that the curvature information 
provides a systematic advantage rather than benefiting only 
specific parameters.

## 关键数据

### 表9：消融实验结果

| 参数 | 非条件化 R² | 条件化 R² | 提升 |
|---|---|---|---|
| d1 | 0.8934 | 0.9133 | +0.0199 |
| d2 | 0.1270 | 0.1491 | +0.0221 |
| d3 | 0.8414 | 0.8583 | +0.0169 |
| d4 | 0.1381 | 0.1903 | +0.0522 |
| d5 | 0.9014 | 0.9169 | +0.0155 |
| d6 | 0.8881 | 0.9151 | +0.0270 |
| d7 | 0.4606 | 0.5179 | +0.0573 |
| n_LaF3 | 0.1726 | 0.1867 | +0.0141 |
| n_MgF2 | 0.0588 | 0.0659 | +0.0071 |

### 训练配置

| 项 | 值 |
|---|---|
| 训练样本 | 50,000（80% 训练，20% 测试） |
| Epochs | 300 |
| 网络深度 | 6 层（1024-1024-512-256-128） |
| 优化器 | Adam |
| 学习率 | 5×10⁻⁴ |
| 学习率调度 | Cosine Annealing |
| Dropout | 0.05 |
| Batch size | 256 |
| 深度学习框架 | **PaddlePaddle** |
| 硬件 | **NVIDIA V100 GPU** |
| 单个模型训练时间 | **8.9 分钟** |
| 条件化最终 loss | 0.0234 |
| 非条件化最终 loss | 0.0256 |

### 核心结论

1. **条件化 PINN 在所有 9 个参数上都优于非条件化**
2. **主要厚度参数（d1、d3、d5、d6）平均提升 +0.020**
3. **d7 提升最大（+0.0573）**，d4 次之（+0.0522）
4. **难辨识参数（d2、n）也有提升（+0.007 到 +0.022）**
5. **条件化模型收敛到更低的 loss**（0.0234 vs 0.0256）
6. **验证了曲率条件化的必要性**，创新点2 成立

## 对应图片

- `fig/fig6_ablation.png`
- `fig/fig6_ablation.pdf`