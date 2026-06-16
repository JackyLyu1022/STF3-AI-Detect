# OOD Split 实验结果汇总
> 自动生成时间：2026-06-04 13:45:02  
> Notebook：`notebooks/OOD_Split_R1_R7_Experiments.ipynb`

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
| `train-csv` | `data\GenVideo-Val\splits\ood_train.csv` |
| `val-csv` | `data\GenVideo-Val\splits\ood_val.csv` |
| `test-csv` | `data\GenVideo-Val\splits\ood_test.csv` |

## OOD 消融汇总表（已完成实验）

| 编号 | Method | Spatial | Temporal | Frequency | Fusion | Test ACC | Test AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | frequency_wave |  |  | ✓ | - | 0.8060 | 0.9354 | 0.8075 | 0.9581 | 0.6978 |
| R2 | spatial_dino | ✓ |  |  | - | 0.8858 | 0.9657 | 0.8923 | 0.9913 | 0.8112 |
| R3 | temporal_d3_restrav |  | ✓ |  | - | 0.8130 | 0.9550 | 0.8134 | 0.9722 | 0.6992 |
| R4 | spatial_frequency_new | ✓ |  | ✓ | Transformer | 0.8749 | 0.9636 | 0.8805 | 0.9940 | 0.7903 |
| R5 | spatial_temporal_new | ✓ | ✓ |  | Transformer | 0.8877 | 0.9795 | 0.8939 | 0.9959 | 0.8108 |
| R6 | stf3_new_concat | ✓ | ✓ | ✓ | Gated Concat | 0.8819 | 0.9744 | 0.8879 | 0.9941 | 0.8022 |
| R7 | stf3_new | ✓ | ✓ | ✓ | Branch-token Transformer | 0.9161 | 0.9651 | 0.9235 | 0.9854 | 0.8689 |

## 单实验详情

### R1. `frequency_wave`

- 模型：`frequency_wave`
- 训练目录：`runs\ood_frequency_wave`
- 测试输出目录：`outputs\ood_frequency_wave`
- 状态：已完成 `5` epoch；Best Val Epoch = `5`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.4926 |
| Train ACC | 0.8228 |
| Train AUC | 0.8924 |
| Val Loss | 0.3351 |
| Val ACC | 0.8716 |
| Val AUC | 0.9238 |
| Val F1 | 0.8149 |
| Test ACC | 0.8060 |
| Test AUC | 0.9354 |
| Test F1 | 0.8075 |
| Test Precision | 0.9581 |
| Test Recall | 0.6978 |
| TN | 1436 |
| FP | 64 |
| FN | 634 |
| TP | 1464 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_frequency_wave/frequency_examples.png)

### R2. `spatial_dino`

- 模型：`spatial_dino`
- 训练目录：`runs\ood_spatial_dino`
- 测试输出目录：`outputs\ood_spatial_dino`
- 状态：已完成 `5` epoch；Best Val Epoch = `4`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.0397 |
| Train ACC | 0.9912 |
| Train AUC | 0.9983 |
| Val Loss | 0.1300 |
| Val ACC | 0.9704 |
| Val AUC | 0.9919 |
| Val F1 | 0.9605 |
| Test ACC | 0.8858 |
| Test AUC | 0.9657 |
| Test F1 | 0.8923 |
| Test Precision | 0.9913 |
| Test Recall | 0.8112 |
| TN | 1485 |
| FP | 15 |
| FN | 396 |
| TP | 1702 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_dino/frequency_examples.png)

### R3. `temporal_d3_restrav`

- 模型：`temporal_d3_restrav`
- 训练目录：`runs\ood_temporal_d3_restrav`
- 测试输出目录：`outputs\ood_temporal_d3_restrav`
- 状态：已完成 `5` epoch；Best Val Epoch = `5`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.4161 |
| Train ACC | 0.8528 |
| Train AUC | 0.9223 |
| Val Loss | 0.2365 |
| Val ACC | 0.9144 |
| Val AUC | 0.9738 |
| Val F1 | 0.8796 |
| Test ACC | 0.8130 |
| Test AUC | 0.9550 |
| Test F1 | 0.8134 |
| Test Precision | 0.9722 |
| Test Recall | 0.6992 |
| TN | 1458 |
| FP | 42 |
| FN | 631 |
| TP | 1467 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_temporal_d3_restrav/frequency_examples.png)

### R4. `spatial_frequency_new`

- 模型：`spatial_frequency_new`
- 训练目录：`runs\ood_spatial_frequency_new`
- 测试输出目录：`outputs\ood_spatial_frequency_new`
- 状态：已完成 `5` epoch；Best Val Epoch = `3`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.1156 |
| Train ACC | 0.9734 |
| Train AUC | 0.9965 |
| Val Loss | 0.0964 |
| Val ACC | 0.9761 |
| Val AUC | 0.9963 |
| Val F1 | 0.9683 |
| Test ACC | 0.8749 |
| Test AUC | 0.9636 |
| Test F1 | 0.8805 |
| Test Precision | 0.9940 |
| Test Recall | 0.7903 |
| TN | 1490 |
| FP | 10 |
| FN | 440 |
| TP | 1658 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_frequency_new/frequency_examples.png)

### R5. `spatial_temporal_new`

- 模型：`spatial_temporal_new`
- 训练目录：`runs\ood_spatial_temporal_new`
- 测试输出目录：`outputs\ood_spatial_temporal_new`
- 状态：已完成 `5` epoch；Best Val Epoch = `3`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.1000 |
| Train ACC | 0.9775 |
| Train AUC | 0.9972 |
| Val Loss | 0.0772 |
| Val ACC | 0.9782 |
| Val AUC | 0.9975 |
| Val F1 | 0.9710 |
| Test ACC | 0.8877 |
| Test AUC | 0.9795 |
| Test F1 | 0.8939 |
| Test Precision | 0.9959 |
| Test Recall | 0.8108 |
| TN | 1493 |
| FP | 7 |
| FN | 397 |
| TP | 1701 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_spatial_temporal_new/frequency_examples.png)

### R6. `stf3_new_concat`

- 模型：`stf3_new_concat`
- 训练目录：`runs\ood_stf3_new_concat`
- 测试输出目录：`outputs\ood_stf3_new_concat`
- 状态：已完成 `5` epoch；Best Val Epoch = `5`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.0726 |
| Train ACC | 0.9945 |
| Train AUC | 0.9995 |
| Val Loss | 0.0904 |
| Val ACC | 0.9769 |
| Val AUC | 0.9983 |
| Val F1 | 0.9692 |
| Test ACC | 0.8819 |
| Test AUC | 0.9744 |
| Test F1 | 0.8879 |
| Test Precision | 0.9941 |
| Test Recall | 0.8022 |
| TN | 1490 |
| FP | 10 |
| FN | 415 |
| TP | 1683 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new_concat/frequency_examples.png)

### R7. `stf3_new`

- 模型：`stf3_new`
- 训练目录：`runs\ood_stf3_new`
- 测试输出目录：`outputs\ood_stf3_new`
- 状态：已完成 `5` epoch；Best Val Epoch = `3`。

| 指标 | 数值 |
|---|---:|
| Train Loss | 0.1201 |
| Train ACC | 0.9756 |
| Train AUC | 0.9971 |
| Val Loss | 0.0984 |
| Val ACC | 0.9745 |
| Val AUC | 0.9949 |
| Val F1 | 0.9661 |
| Test ACC | 0.9161 |
| Test AUC | 0.9651 |
| Test F1 | 0.9235 |
| Test Precision | 0.9854 |
| Test Recall | 0.8689 |
| TN | 1473 |
| FP | 27 |
| FN | 275 |
| TP | 1823 |
| Num Test Samples | 3598 |

#### 可视化

**混淆矩阵**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/confusion_matrix.png)

**ROC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/roc_curve.png)

**Loss 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/history_loss.png)

**ACC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/history_acc.png)

**AUC 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/history_auc.png)

**F1 曲线**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/history_f1.png)

**Generator 准确率**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/accuracy_by_generator.png)

**频域示例**

![](D:/VsCode Program/Python/content_security/final_project/outputs/figures/ood_stf3_new/frequency_examples.png)
