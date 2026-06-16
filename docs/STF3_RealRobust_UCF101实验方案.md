# STF3-RealRobust UCF101 真实视频增强实验方案

## 1. 实验目的

当前 STF3-New R7_224_FAKEW12 在 AEGIS 外部测试集上的主要问题是：

```text
真实视频被误判为 AI 视频较多，即 FP 偏高。
```

因此，本实验不直接使用 AEGIS 参与训练，而是在 GenVideo OOD 训练集基础上加入额外真实视频 UCF-101，用于增强模型对真实视频分布的覆盖能力，观察是否能够降低 AEGIS 上的真实视频误报。

## 2. 数据重构原则

### 保持不变

- AEGIS 仍然只作为外部测试集；
- GenVideo OOD val/test 不加入训练；
- 原始 `data/GenVideo-Val/splits/*.csv` 不被覆盖。

### 新增内容

从 `data/UCF-101` 中按类别均匀抽样 2000 个真实视频，加入训练集。

## 3. 新数据集文件

生成脚本：

```text
scripts/build_realrobust_ucf101_splits.py
```

生成目录：

```text
data/STF3-RealRobust/splits
```

生成文件：

| 文件 | 作用 |
|---|---|
| `ood_train_genvideo_ucf101_real2000.csv` | GenVideo OOD train + 2000 UCF-101 real |
| `ood_val_genvideo_prefixed.csv` | GenVideo OOD val，路径改为相对 `data/` |
| `ood_test_genvideo_prefixed.csv` | GenVideo OOD test，路径改为相对 `data/` |
| `ucf101_real2000_manifest.csv` | 本次抽样的 UCF-101 real 清单 |
| `summary_real2000.json` | 数据统计摘要 |

数据统计：

| Split | Total | Real | Fake |
|---|---:|---:|---:|
| Train | 14273 | 9000 | 5273 |
| Val | 2429 | 1500 | 929 |
| Test | 3598 | 1500 | 2098 |

UCF-101 抽样情况：

```text
UCF-101 classes: 101
Added real videos: 2000
Per-class sampled videos: 19-20
```

## 4. 训练设置

本实验尽量复现原 STF3-New R7_224_FAKEW12 的训练设置：

| 参数 | 数值 |
|---|---|
| model | stf3_new |
| epochs | 5 |
| batch_size | 1 |
| num_frames | 8 |
| image_size | 224 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| foundation_backbone | dinov2_vits14 |
| fake_loss_weight | 1.2 |
| aux_loss_weight | 0.2 |
| branch_dropout | 0.1 |
| wavelet_aug_prob | 0.1 |

注意：虽然本实验加入了更多真实视频，但仍保留 `fake_loss_weight=1.2`，用于尽量保持与原 R7_224_FAKEW12 的可比性。

## 5. 运行命令

### 5.1 训练

```powershell
.\scripts\run_stf3_realrobust_ucf101_r7_train.ps1
```

输出：

```text
runs/ood_stf3_new_224_fakew12_ucf101real2000
```

### 5.2 GenVideo OOD 测试

```powershell
.\scripts\run_stf3_realrobust_ucf101_r7_eval_ood.ps1
```

输出：

```text
outputs/ood_stf3_new_224_fakew12_ucf101real2000
```

### 5.3 AEGIS 外部测试

```powershell
.\scripts\run_stf3_realrobust_ucf101_r7_eval_aegis.ps1
```

输出：

```text
outputs/aegis_stf3_new_224_fakew12_ucf101real2000
```

## 6. 重点观察指标

本实验的目标不是单纯提高 Recall，而是降低真实视频误报。因此重点观察：

| 指标 | 关注原因 |
|---|---|
| FP | 真实视频误报数量，核心优化目标 |
| Precision | FP 降低后应上升 |
| Specificity = TN / (TN + FP) | 真实视频识别能力 |
| Recall | 需要避免为了降低 FP 而过度牺牲 AI 检出率 |
| AUC | 阈值无关排序能力 |
| F1 | 综合参考 |

## 7. 预期答辩解释

如果 AEGIS FP 明显下降：

> 加入额外真实视频后，STF3 在 AEGIS 上的真实视频误报下降，说明原始模型的主要问题来自真实视频分布覆盖不足，而不是三分支结构完全失效。

如果 AEGIS FP 没有明显下降：

> UCF-101 虽然增加了真实视频数量，但其分布仍然与 AEGIS real/youtube、real/dvf 存在差异，说明后续需要更接近目标场景的真实视频 hard negatives，或进行更严格的跨数据集校准。

## 8. 实验结果：RealRobust-UCF101 为负结果

本实验完成后，在 AEGIS hard test set 上得到如下结果：

| 指标 | 数值 |
|---|---:|
| ACC | 0.6514 |
| AUC | 0.7095 |
| F1 | 0.7185 |
| Precision | 0.6025 |
| Recall | 0.8899 |
| Specificity | 0.4128 |
| TN | 90 |
| FP | 128 |
| FN | 24 |
| TP | 194 |

与原始 STF3-New R7_224_FAKEW12 在 AEGIS 上的结果对比：

| 模型 | ACC | AUC | F1 | Precision | Recall | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original STF3-New R7_224_FAKEW12 | 0.6858 | 0.7184 | 0.7523 | 0.6209 | 0.9541 | 0.4174 | 91 | 127 | 10 | 208 |
| RealRobust-UCF101 | 0.6514 | 0.7095 | 0.7185 | 0.6025 | 0.8899 | 0.4128 | 90 | 128 | 24 | 194 |

### 8.1 结论

RealRobust-UCF101 没有达到预期目标：

- FP 从 127 增加到 128，没有降低真实视频误报；
- Specificity 从 0.4174 降至 0.4128，真实视频识别能力没有改善；
- FN 从 10 增加到 24，AI 视频漏检变多；
- Recall 从 0.9541 降至 0.8899，模型对 AI 视频的敏感性下降；
- ACC、AUC、F1、Precision 均低于原始 STF3。

因此，本实验应作为 **负结果** 记录，不应替代主模型 `R7_224_FAKEW12`。

### 8.2 原因分析

虽然 UCF-101 增加了真实视频数量，但它没有降低 AEGIS 上的真实视频误报，可能原因包括：

1. **UCF-101 与 AEGIS real 分布不一致**  
   UCF-101 主要是动作识别视频，场景、压缩方式、拍摄风格与 AEGIS 的 `real/youtube`、`real/dvf` 并不完全一致。  
   因此，它并不是针对 AEGIS 误报模式的有效 hard negative。

2. **额外 real 数量增加不等于有效负样本增加**  
   本实验加入的是普通真实视频，而不是当前 STF3 容易误判的真实视频类型。  
   如果额外 real 与误报样本差异较大，模型无法学到降低 FP 所需的边界。

3. **训练分布被改变，fake 敏感性下降**  
   加入 2000 个 UCF real 后，训练集 real/fake 比例进一步偏向 real。  
   这可能改变了模型的决策边界，使其对 fake 的召回下降，表现为 FN 从 10 增加到 24。

### 8.3 后续建议

后续不建议继续简单堆叠 UCF-101 real。更合理的方向是：

- 使用更接近 AEGIS 的真实视频来源，例如 YouTube/WebVid/Pexels/Pixabay real clips；
- 做 hard negative mining：先分析 AEGIS 中被误报的 real 视频特征，再寻找相似真实视频加入训练；
- 尝试后处理校准或外部阈值迁移，而不是只靠加入普通 real 数据；
- 如果使用 AEGIS 自身进行适应，必须重新划分 `AEGIS-adapt` 与 `AEGIS-heldout-test`，不能再把完整 AEGIS 当作纯外部测试集。

### 8.4 答辩表述

推荐表述：

> 为了降低 AEGIS 上真实视频被误判为 AI 的问题，我尝试在 GenVideo 训练集上额外加入 2000 个 UCF-101 真实视频进行 RealRobust 训练。但结果显示，FP 没有下降，反而 Recall、F1 和 ACC 均下降。这说明简单增加普通真实视频并不能解决跨数据集误报问题，关键在于加入与目标误报模式更接近的 hard negative，而不是盲目扩大 real 类数量。
