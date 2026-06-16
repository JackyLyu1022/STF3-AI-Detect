# Random Split 实验结果汇总
> 自动生成时间：2026-05-28 21:35:36  
> Notebook：`notebooks/Random_Split_R1_R7_Experiments.ipynb`

## 统一参数

| 参数 | 值 |
|---|---:|
| `epochs` | `5` |
| `batch-size` | `1` |
| `num-frames` | `8` |
| `image-size` | `112` |
| `foundation-backbone` | `dinov2_vits14` |
| `wavelet-aug-prob` | `0.1` |
| `branch-dropout` | `0.1` |
| `aux-loss-weight` | `0.2` |
| `amp` | `True` |
| `train-csv` | `data\GenVideo-Val\splits\random_train.csv` |
| `val-csv` | `data\GenVideo-Val\splits\random_val.csv` |
| `test-csv` | `data\GenVideo-Val\splits\random_test.csv` |

## Random 消融汇总表（已完成实验）

| 编号 | Method | Spatial | Temporal | Frequency | Fusion | Test ACC | Test AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | frequency_wave |  |  | ✓ | - | 0.8657 | 0.9309 | 0.8440 | 0.8962 | 0.7976 |
| R5 | spatial_temporal_new | ✓ | ✓ |  | Transformer | 0.9735 | 0.9967 | 0.9705 | 0.9836 | 0.9578 |
| R7 | stf3_new | ✓ | ✓ | ✓ | Branch-token Transformer | 0.9750 | 0.9963 | 0.9722 | 0.9821 | 0.9625 |

## 单实验详情

### R1. `frequency_wave`

- 模型：`frequency_wave`
- 训练目录：`runs\random_frequency_wave`
- 测试输出目录：`outputs\random_frequency_wave`
- 状态：已完成 `5` epoch；Best Val Epoch = `4`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.5073 |
| Train ACC | 0.8154 |
| Train AUC | 0.8871 |
| Val Loss | 0.3834 |
| Val ACC | 0.8451 |
| Val AUC | 0.9170 |
| Val F1 | 0.8086 |
| Test ACC | 0.8657 |
| Test AUC | 0.9309 |
| Test F1 | 0.8440 |
| Test Precision | 0.8962 |
| Test Recall | 0.7976 |
| TN | 1384 |
| FP | 116 |
| FN | 254 |
| TP | 1001 |
| Num Test Samples | 2755 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_frequency_wave/frequency_examples.png)

### R2. `spatial_dino`

- 模型：`spatial_dino`
- 训练目录：`runs\random_spatial_dino`
- 测试输出目录：`outputs\random_spatial_dino`
- 状态：未完成或缺少 `history.json / metrics.json / predictions.csv`。

### R3. `temporal_d3_restrav`

- 模型：`temporal_d3_restrav`
- 训练目录：`runs\random_temporal_d3_restrav`
- 测试输出目录：`outputs\random_temporal_d3_restrav`
- 状态：未完成或缺少 `history.json / metrics.json / predictions.csv`。

### R4. `spatial_frequency_new`

- 模型：`spatial_frequency_new`
- 训练目录：`runs\random_spatial_frequency_new`
- 测试输出目录：`outputs\random_spatial_frequency_new`
- 状态：未完成或缺少 `history.json / metrics.json / predictions.csv`。

### R5. `spatial_temporal_new`

- 模型：`spatial_temporal_new`
- 训练目录：`runs\random_spatial_temporal_new`
- 测试输出目录：`outputs\random_spatial_temporal_new`
- 状态：已完成 `5` epoch；Best Val Epoch = `3`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.1152 |
| Train ACC | 0.9738 |
| Train AUC | 0.9964 |
| Val Loss | 0.1176 |
| Val ACC | 0.9679 |
| Val AUC | 0.9934 |
| Val F1 | 0.9638 |
| Test ACC | 0.9735 |
| Test AUC | 0.9967 |
| Test F1 | 0.9705 |
| Test Precision | 0.9836 |
| Test Recall | 0.9578 |
| TN | 1480 |
| FP | 20 |
| FN | 53 |
| TP | 1202 |
| Num Test Samples | 2755 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_spatial_temporal_new/frequency_examples.png)

### R6. `stf3_new_concat`

- 模型：`stf3_new_concat`
- 训练目录：`runs\random_stf3_new_concat`
- 测试输出目录：`outputs\random_stf3_new_concat`
- 状态：未完成或缺少 `history.json / metrics.json / predictions.csv`。

### R7. `stf3_new`

- 模型：`stf3_new`
- 训练目录：`runs\random_stf3_new`
- 测试输出目录：`outputs\random_stf3_new`
- 状态：已完成 `5` epoch；Best Val Epoch = `4`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.1330 |
| Train ACC | 0.9735 |
| Train AUC | 0.9959 |
| Val Loss | 0.1011 |
| Val ACC | 0.9712 |
| Val AUC | 0.9957 |
| Val F1 | 0.9677 |
| Test ACC | 0.9750 |
| Test AUC | 0.9963 |
| Test F1 | 0.9722 |
| Test Precision | 0.9821 |
| Test Recall | 0.9625 |
| TN | 1478 |
| FP | 22 |
| FN | 47 |
| TP | 1208 |
| Num Test Samples | 2755 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/random_stf3_new/frequency_examples.png)
