# AEGIS 外部测试集实验汇总

## 1. 实验目的

本实验用于验证 STF3 在 **外部数据集 AEGIS hard test set** 上的跨数据集泛化能力。

与 GenVideo 内部 Random / OOD 划分不同，AEGIS 不参与模型训练、验证或调参，仅作为外部测试集使用。因此，该实验主要用于回答：

> 已在 GenVideo OOD split 上训练完成的 STF3，能否泛化到新的真实视频来源和新的 AI 视频生成器？

## 1.1 AEGIS 数据集与论文信息

AEGIS 是 2025 年 ACM Multimedia（ACM MM 2025）数据集方向收录的 AI 生成视频真实性评估 benchmark，论文题目为：

```text
AEGIS: Authenticity Evaluation Benchmark for AI-Generated Video Sequences
```

论文与数据集链接：

| 类型 | 链接 |
|---|---|
| ACM Digital Library | https://dl.acm.org/doi/10.1145/3746027.3758295 |
| arXiv | https://arxiv.org/abs/2508.10771 |
| Hugging Face Dataset | https://huggingface.co/datasets/Clarifiedfish/AEGIS |
| ACM MM 2025 Accepted Papers - Datasets | https://acmmm2025.org/accepted-papers-datasets/ |

基本信息：

| 项目 | 内容 |
|---|---|
| 会议 | ACM Multimedia 2025 / ACM MM 2025 |
| 论文类型 | Dataset / Benchmark |
| 题目 | AEGIS: Authenticity Evaluation Benchmark for AI-Generated Video Sequences |
| 作者 | Jieyu Li, Xin Zhang, Joey Tianyi Zhou |
| DOI | 10.1145/3746027.3758295 |
| 本项目使用部分 | AEGIS hard test set |

在本文实验中，AEGIS 不参与 STF3、TALL 或其他 baseline 的训练过程，仅作为外部测试集，用于评估模型在跨数据集、跨真实视频来源和跨生成器场景下的泛化能力。

## 2. 数据集设置

AEGIS hard test set 本地路径：

```text
data/AEGIS
```

已生成适配本项目格式的测试 CSV：

```text
data/AEGIS/splits/aegis_hard_test.csv
```

数据规模：

| 类别 | 数量 |
|---|---:|
| Fake / AI-generated | 218 |
| Real | 218 |
| Total | 436 |

数据来源 / 生成器分布：

| 类型 | 来源 / 生成器 | 数量 |
|---|---|---:|
| Fake | kling | 111 |
| Fake | sora | 107 |
| Real | dvf | 109 |
| Real | youtube | 109 |

## 3. STF3 外部测试结果

测试模型：

```text
STF3-New R7_224_FAKEW12
```

模型 checkpoint：

```text
runs/ood_stf3_new_224_fakew12/best.pt
```

测试命令：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\ood_stf3_new_224_fakew12\best.pt `
  --data-root data\AEGIS `
  --csv data\AEGIS\splits\aegis_hard_test.csv `
  --batch-size 2 `
  --num-workers 0 `
  --amp `
  --out-dir outputs\aegis_stf3_new_224_fakew12
```

### 3.1 整体指标

| 指标 | 数值 |
|---|---:|
| ACC | 0.6858 |
| AUC | 0.7184 |
| F1 | 0.7523 |
| Precision | 0.6209 |
| Recall | 0.9541 |
| Samples | 436 |
| Time | 148.62 s |

### 3.2 混淆矩阵

原始结果：

```python
confusion_matrix = [[91, 127], [10, 208]]
```

对应含义：

| 真实类别 | 预测 Real | 预测 Fake |
|---|---:|---:|
| Real | TN = 91 | FP = 127 |
| Fake | FN = 10 | TP = 208 |

## 4. TALL 外部测试结果

测试模型：

```text
TALL-SWIN
```

模型 checkpoint：

```text
comparison_experiment/results/TALL/ood_tall_uniform8_e5/best.pt
```

输出目录：

```text
comparison_experiment/results/TALL/aegis_tall_uniform8_e3_best
```

运行设置：

| 设置项 | 数值 |
|---|---|
| eval_only | true |
| num_frames | 8 |
| frame_size | 112 |
| thumbnail_rows | 2 |
| batch_size | 2 |
| AMP | true |
| test_valid | 436 |
| test_errors | 0 |
| seconds | 319.87 s |

需要注意：本次 TALL AEGIS 实验中，脚本的 `val_csv` 和 `test_csv` 都指向了 `data/AEGIS/splits/aegis_hard_test.csv`。因此，下面的阈值型结果属于 **AEGIS 诊断性测试结果**，不是最严格意义上的“GenVideo validation threshold 迁移到 AEGIS test”的结果。AUC / AP 仍然可以作为阈值无关指标参考。

### 4.1 TALL 主要结果：max_f1 / max_balanced_acc / max_acc

TALL 在 `max_f1`、`max_balanced_acc` 和 `max_acc` 三个目标下选出了相同阈值，因此三者结果一致：

| 指标 | 数值 |
|---|---:|
| Threshold | 0.9931 |
| ACC | 0.8165 |
| AUC | 0.8846 |
| AP | 0.8631 |
| F1 | 0.8238 |
| Precision | 0.7924 |
| Recall | 0.8578 |
| Balanced ACC | 0.8165 |

混淆矩阵：

| 真实类别 | 预测 Real | 预测 Fake |
|---|---:|---:|
| Real | TN = 169 | FP = 49 |
| Fake | FN = 31 | TP = 187 |

### 4.2 TALL 高精度目标：precision_0.95_recall_max

该目标强制追求 Precision ≥ 0.95，因此阈值被提高，误报大幅减少，但漏检明显增加。

| 指标 | 数值 |
|---|---:|
| Threshold | 1.0000 |
| ACC | 0.6353 |
| AUC | 0.8846 |
| AP | 0.8631 |
| F1 | 0.4382 |
| Precision | 0.9538 |
| Recall | 0.2844 |
| Balanced ACC | 0.6353 |

混淆矩阵：

| 真实类别 | 预测 Real | 预测 Fake |
|---|---:|---:|
| Real | TN = 215 | FP = 3 |
| Fake | FN = 156 | TP = 62 |

### 4.3 TALL 结果解释

TALL 在 AEGIS 上的整体区分能力明显强于当前 STF3 的默认 0.5 阈值结果：

- TALL 的 AUC 为 **0.8846**，高于 STF3 的 **0.7184**，说明 TALL 在 AEGIS 上的 fake / real 排序能力更强。
- 在 `max_f1` 工作点下，TALL 的 ACC 为 **0.8165**，F1 为 **0.8238**，Precision 为 **0.7924**，整体更加均衡。
- 但 TALL 的 Recall 为 **0.8578**，低于 STF3 的 **0.9541**，说明 STF3 更倾向于高召回检出 AI 视频，而 TALL 更均衡、误报更少。

因此，TALL 可以作为 AEGIS 上较强的外部时空 baseline；而 STF3 当前结果更适合表述为“高召回 AI 视频初筛能力较强，但跨数据集校准仍需优化”。

## 5. WaveRep 外部测试结果

测试模型：

```text
WaveRep_DINOv2_G4
```

模型权重：

```text
comparison_experiment/WaveRep/demo/weights/weights_dinov2_G4.ckpt
```

输出目录：

```text
comparison_experiment/results/WaveRep/aegis_g4_uniform8
```

运行设置：

| 设置项 | 数值 |
|---|---|
| architecture | vit_base_patch14_reg4_dinov2.lvd142m |
| sampling | uniform |
| num_frames | 8 |
| crop_size | 504 |
| batch_size | 8 |
| val_valid | 436 |
| test_valid | 436 |
| val_errors | 0 |
| test_errors | 0 |
| seconds | 653.74 s |

需要注意：本次 WaveRep AEGIS 实验中，脚本的 `val_csv` 和 `test_csv` 都指向了 `data/AEGIS/splits/aegis_hard_test.csv`。因此，阈值型结果同样属于 **AEGIS 诊断性测试结果**。AUC / AP 是阈值无关指标，可以作为更稳定的跨模型参考。

### 5.1 WaveRep 各目标结果

| Objective | Threshold | ACC | AUC | AP | F1 | Precision | Recall | Balanced ACC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| precision_0.95_recall_max | 0.9632 | 0.7821 | 0.8452 | 0.8822 | 0.7278 | 0.9695 | 0.5826 | 0.7821 | 214 | 4 | 91 | 127 |

### 5.2 WaveRep 结果解释

WaveRep 在 AEGIS 上呈现出比较明显的“保守检测”特征：

- 在 `max_f1` 工作点下，WaveRep 的 F1 为 **0.7570**，ACC 为 **0.7615**，整体略高于 STF3 的默认 0.5 阈值 ACC，但 F1 与 STF3 接近。
- WaveRep 的 AUC 为 **0.8452**，AP 为 **0.8822**，说明其排序能力明显强于 STF3 当前默认输出结果，但低于 TALL。
- 当使用 `precision_0.95_recall_max` 目标时，WaveRep 的 Precision 达到 **0.9695**，FP 仅为 **4**，但 Recall 降至 **0.5826**，FN 增至 **91**。这说明 WaveRep 可以通过提高阈值显著减少误报，但代价是漏检更多 AI 视频。

因此，WaveRep 更适合作为一个高精度、低误报的外部预训练 baseline；而 STF3 更偏向高召回，TALL 在本次 AEGIS 诊断测试中整体指标更均衡。

## 6. D3 外部测试结果

测试模型：

```text
D3 XCLIP-16 L2
```

输出目录：

```text
comparison_experiment/results/D3/aegis_xclip16_l2
```

运行设置：

| 设置项 | 数值 |
|---|---|
| encoder | XCLIP-16 |
| loss | l2 |
| num_frames | 16 |
| image_size | 224 |
| channel_mode | BGR_official_cv2 |
| orientation | real_high |
| val_valid | 436 |
| test_valid | 436 |
| val_errors | 0 |
| test_errors | 0 |
| seconds | 570.92 s |

需要注意：本次 D3 AEGIS 实验中，脚本的 `val_csv` 和 `test_csv` 同样都指向了 `data/AEGIS/splits/aegis_hard_test.csv`，因此阈值型结果属于 **AEGIS 诊断性测试结果**。D3 是 training-free 方法，没有在 GenVideo 或 AEGIS 上重新训练。

### 6.1 D3 各目标结果

| Objective | Threshold | ACC | AUC | AP | F1 | Precision | Recall | Balanced ACC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| precision_0.95_recall_max | -3.4377 | 0.6284 | 0.6630 | 0.6168 | 0.7188 | 0.5782 | 0.9495 | 0.6284 | 67 | 151 | 11 | 207 |


## 7. AEGIS 上 STF3、TALL、WaveRep 与 D3 对比

为了避免混淆，下面的主对比采用：

- STF3：`src.evaluate` 默认 0.5 阈值结果；
- TALL：`max_f1` / `max_balanced_acc` / `max_acc` 共同对应的最优诊断阈值结果。
- WaveRep：`max_f1` 诊断阈值结果。
- D3：`max_f1` 诊断阈值结果。

| 模型 | ACC | AUC | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STF3-New R7_224_FAKEW12 | 0.6858 | 0.7184 | 0.7523 | 0.6209 | **0.9541** | 91 | 127 | **10** | **208** |
| TALL-SWIN | **0.8165** | **0.8846** | **0.8238** | **0.7924** | 0.8578 | 169 | 49 | 31 | 187 |
| WaveRep DINOv2 G4 | 0.7615 | 0.8452 | 0.7570 | 0.7714 | 0.7431 | **170** | **48** | 56 | 162 |
| D3 XCLIP-16 L2 | 0.6284 | 0.6630 | 0.7188 | 0.5782 | **0.9495** | 67 | 151 | 11 | 207 |

从该表可以看出：

- **STF3 的优势**：相对 TALL 和 WaveRep，Recall 更高、FN 更少，对 AI 生成视频更敏感；同时整体 F1 明显高于 D3；
- **TALL 的优势**：ACC、AUC、F1、Precision 更高，整体二分类性能更均衡；
- **WaveRep 的优势**：在 `max_f1` 工作点下 FP 略低于 TALL；在高精度工作点下可进一步将 FP 降至 4，但会牺牲大量 Recall；
- **D3 的特点**：Recall 高、FN 少，但 Precision、AUC 和 FP 表现较弱，说明其在 AEGIS 上更像高敏感度但高误报的 training-free baseline；
- **关键解释**：STF3 在 AEGIS 上主要问题不是漏检 fake，而是将较多 real 视频误判为 fake，说明跨数据集真实视频分布适应和阈值校准仍需改进。

## 8. STF3 结果解释

AEGIS 外部测试结果显示，STF3 在跨数据集场景下表现出明显的两面性：

1. **AI 视频检出能力较强**  
   Recall 达到 **0.9541**，说明 218 个 AI 生成视频中有 208 个被正确检出，仅漏检 10 个。  
   这表明 STF3 对 AI 生成痕迹较敏感，具备较强的高召回筛查能力。

2. **真实视频误报较多**  
   Precision 仅为 **0.6209**，Real 视频中有 127 个被误判为 Fake。  
   这说明模型在 AEGIS 的真实视频分布上校准不足，容易将部分真实视频中的压缩、画质、运动或来源差异识别为生成痕迹。

3. **整体跨数据集泛化仍有挑战**  
   ACC 为 **0.6858**，AUC 为 **0.7184**，相比 GenVideo OOD 测试结果有明显下降。  
   这说明 AEGIS 与 GenVideo 之间存在较强 domain shift，模型并不能直接在外部数据集上保持同等综合性能。

## 9. 答辩表述建议

推荐表述：

> 在 AEGIS 外部 hard test set 上，STF3 对 AI 生成视频保持了较高召回率，Recall 达到 95.41%，218 个 AI 视频中成功检出 208 个，仅漏检 10 个。这说明模型对 AI 生成痕迹具有较强敏感性，适合作为高召回的 AI 视频初筛模型。但同时，模型在真实视频上误报较多，Precision 和 ACC 下降明显，说明跨数据集场景下仍存在 domain shift 和阈值校准问题。

不建议表述：

> STF3 在 AEGIS 上整体表现非常好。

更严谨的结论是：

> STF3 在外部数据集上具有较强的 AI 视频检出能力，但完整二分类性能仍受到真实视频误报和跨数据集分布偏移的限制。

如果需要同时解释 TALL 对比结果，可以补充：

> 在 AEGIS 外部测试中，TALL 的整体二分类指标更均衡，说明强时空建模方法在该外部数据集上具有更好的整体泛化表现。但 STF3 的 AI 视频召回率更高，说明三分支证据融合对生成痕迹具有较强敏感性。后续优化重点应放在真实视频误报控制、阈值迁移校准和跨数据集 domain shift 适应上。

如果需要同时解释 WaveRep 对比结果，可以补充：

> WaveRep 作为官方预训练的外部 baseline，在 AEGIS 上具有较好的排序能力和较低误报潜力。特别是在高精度工作点下，WaveRep 可以将 FP 降至 4，但 Recall 明显下降。这说明不同模型存在不同工作点取向：STF3 更偏向高召回筛查，WaveRep 更适合保守判定，TALL 在本次测试中综合指标更均衡。

如果需要同时解释 D3 对比结果，可以补充：

> D3 作为 training-free baseline 在 AEGIS 上具有很高的 AI 视频召回率，但 AUC 和 Precision 较低，误报真实视频较多。这说明单纯依赖动态一致性差异在跨数据集 hard set 上不足以稳定区分真实视频和 AI 生成视频。相比之下，STF3 虽然仍存在真实视频误报，但综合 F1 和 AUC 高于 D3，说明三分支证据融合相较单一 training-free 动态指标具有一定提升。

## 10. RealRobust-UCF101 后续优化实验：负结果

为了解决 STF3 在 AEGIS 上真实视频误报较多的问题，进一步尝试了 RealRobust-UCF101 实验：在 GenVideo OOD train 基础上加入 2000 个 UCF-101 真实视频重新训练 STF3-New R7_224_FAKEW12。

该实验不使用 AEGIS 参与训练，AEGIS 仍作为外部测试集。

### 10.1 AEGIS 测试结果

| 模型 | ACC | AUC | F1 | Precision | Recall | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original STF3-New R7_224_FAKEW12 | 0.6858 | 0.7184 | 0.7523 | 0.6209 | 0.9541 | 0.4174 | 91 | 127 | 10 | 208 |
| RealRobust-UCF101 | 0.6514 | 0.7095 | 0.7185 | 0.6025 | 0.8899 | 0.4128 | 90 | 128 | 24 | 194 |

### 10.2 结论

RealRobust-UCF101 没有改善 AEGIS 外部测试表现：

- FP 从 127 增加到 128，真实视频误报没有下降；
- Specificity 从 0.4174 降至 0.4128，真实视频识别能力没有提升；
- FN 从 10 增加到 24，AI 视频漏检增加；
- Recall 从 0.9541 降至 0.8899，AI 视频检出能力下降；
- ACC、AUC、F1、Precision 均低于原始 STF3。

因此，RealRobust-UCF101 应作为 **负结果** 记录，不替代主模型。

### 10.3 解释

该负结果说明，**简单加入普通真实视频并不一定能降低外部数据集误报**。UCF-101 是动作识别数据集，虽然都是真实视频，但其视频来源、压缩方式、场景分布和 AEGIS 的 `real/youtube`、`real/dvf` 并不完全一致，因此不一定能作为有效 hard negatives。

更合理的后续方向是：

- 寻找更接近 AEGIS real 分布的真实视频，例如 YouTube/WebVid/Pexels/Pixabay；
- 对 AEGIS 中被误报的 real 视频做错误分析，再寻找相似真实样本；
- 使用阈值迁移校准或 domain calibration；
- 如果使用 AEGIS 自身做适应，必须划分 `AEGIS-adapt` 和 `AEGIS-heldout-test`，避免外部测试集泄漏。

答辩时可以表述为：

> 我尝试通过加入 UCF-101 真实视频降低 AEGIS 真实视频误报，但结果没有改善，反而整体指标下降。这说明问题不只是训练集中 real 数量不足，而是外部真实视频的分布差异与 hard negative 类型不匹配。该负结果帮助明确了后续优化方向：需要更接近目标域的真实视频 hard negatives，而不是简单堆叠普通真实视频。

## 11. 后续建议

为了进一步完善 AEGIS 外部测试实验，建议继续进行：

1. **按来源 / 生成器统计错误**
   - 分析 FP 主要来自 `real/youtube` 还是 `real/dvf`。
   - 分析 FN 主要来自 `fake/sora` 还是 `fake/kling`。

2. **绘制 fake probability 分布图**
   - 对比 Real 与 Fake 的预测概率分布。
   - 判断错误主要来自阈值偏移，还是特征区分能力不足。

3. **加入外部 baseline 对比**
   - 在 AEGIS 上继续测试 D3、WaveRep、TALL。
   - 构建跨数据集泛化对比表，观察 STF3 是否在高召回或综合指标上仍有优势。

4. **固定 GenVideo validation threshold 重新计算**
   - 当前 `src.evaluate` 默认使用 0.5 阈值。
   - 更严谨的做法是使用 GenVideo OOD validation 上确定的阈值，直接迁移到 AEGIS 测试集，避免在 AEGIS 上调参。
