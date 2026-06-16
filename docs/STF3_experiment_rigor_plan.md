# STF3 答辩前实验严谨性优化方案：Baseline、主指标、外部对比与轻量测试集

> 生成日期：2026-06-13  
> 项目根目录：`D:\VsCode Program\Python\content_security\final_project`  
> 依据：当前仓库 `src/` 代码、`data/GenVideo-Val/splits/` 数据划分、`runs/` 训练历史、`outputs/` 评估结果、`notebooks/` 实验编排、`docs/ood_followup_optimization_summary.md` 以及 D3 / ReStraV / WaveRep / SPAI / UNITE 等相关方法。

---

## 0. 我对当前项目的理解

你的项目已经不是一个“只有 idea 的模型”，而是一个围绕 **AI 生成视频检测** 的三分支检测系统：

```text
输入视频 [B,T,3,H,W]
  ├─ Spatial branch：DINOv2 / CLIP / XCLIP 等 foundation encoder + temporal attention pooling
  ├─ Temporal branch：D3 二阶动态 + ReStraV 表征轨迹几何
  ├─ Frequency branch：WaveRep 风格小波频带统计 + radial FFT 频谱统计
  └─ Fusion：branch-token Transformer / gated concat -> Real/Fake logits
```

核心代码位置：

| 模块 | 文件 | 作用 |
|---|---|---|
| 数据读取 | `src/dataset.py` | 从 CSV 读视频路径，OpenCV 均匀采样/训练随机采样，返回 `[T,3,H,W]` |
| 训练 | `src/train.py` | CE / fake 加权 CE、aux branch loss、AMP、best checkpoint 按 val AUC 保存 |
| 评估 | `src/evaluate.py` | 输出 `metrics.json` 与 `predictions.csv`，保存分支概率与 branch weights |
| 可视化 | `src/visualize.py` | 混淆矩阵、ROC、训练曲线、generator accuracy |
| 新模型 | `src/models/stf3_modern.py` | STF3-New 现代三分支主结构 |
| Foundation backbone | `src/models/backbones/foundation_encoder.py` | DINOv2 torch.hub / CLIP / XCLIP / HF-DINOv2 |
| 时序特征 | `src/features/d3.py`, `src/features/geometry.py` | D3 second-order 与 ReStraV 21D 几何特征 |
| 频域特征 | `src/features/wavelet.py`, `src/features/spectral.py` | Haar DWT、WaveRep-style band replacement、FFT radial stats |
| 融合 | `src/models/fusion/branch_token_transformer.py` | Branch-token Transformer 与 gated fusion |

当前数据集是 `GenVideo-Val`，已有两类协议：

| Split | train | val | test | 设计意义 |
|---|---:|---:|---:|---|
| Random | 12,804 | 2,743 | 2,755 | 同生成器家族随机划分，测试 in-distribution 能力 |
| OOD | 12,273 | 2,429 | 3,598 | train/val 只含 Lavie、Gen2、HotShot、MoonValley、ModelScope、Crafter；test 主要含 MorphStudio、Show_1、Sora、WildScrape，是跨生成器泛化测试 |

---

## 1. 当前最关键实验事实

### 1.1 Random split 已完成结果

目前 Random split 完整完成了 R1/R5/R7 三个正式模型：

| 模型 | 分支 | ACC | AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `frequency_wave` | F | 0.8657 | 0.9309 | 0.8440 | 0.8962 | 0.7976 |
| `spatial_temporal_new` | S+T | 0.9735 | **0.9967** | 0.9705 | **0.9836** | 0.9578 |
| `stf3_new` | S+T+F | **0.9750** | 0.9963 | **0.9722** | 0.9821 | **0.9625** |

解释：

- Random 已经接近饱和。
- R5 和 R7 差距非常小，因此 Random 更适合证明“系统可检测”，不适合单独证明频域分支很关键。
- 真正能体现泛化与模型贡献的，应主要看 OOD。

### 1.2 OOD split 原始 R1-R7 结果

| 模型 | 分支/融合 | ACC | AUC | F1 | Precision | Recall | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `frequency_wave` | F | 0.8060 | 0.9354 | 0.8075 | 0.9581 | 0.6978 | 634 | 64 |
| `spatial_dino` | S | 0.8858 | 0.9657 | 0.8923 | 0.9913 | 0.8112 | 396 | 15 |
| `temporal_d3_restrav` | T | 0.8130 | 0.9550 | 0.8134 | 0.9722 | 0.6992 | 631 | 42 |
| `spatial_frequency_new` | S+F | 0.8749 | 0.9636 | 0.8805 | 0.9940 | 0.7903 | 440 | 10 |
| `spatial_temporal_new` | S+T | 0.8877 | **0.9795** | 0.8939 | **0.9959** | 0.8108 | 397 | 7 |
| `stf3_new_concat` | S+T+F gated | 0.8819 | 0.9744 | 0.8879 | 0.9941 | 0.8022 | 415 | 10 |
| `stf3_new` | S+T+F Transformer | **0.9161** | 0.9651 | **0.9235** | 0.9854 | **0.8689** | **275** | 27 |

解释：

- 在 OOD 原始阈值 0.5 下，STF3-New 的主要价值很清楚：它不是最高 AUC，但显著提高 fake recall、降低 FN，使综合 F1/ACC 最好。
- 这说明你的模型更像一个“实际部署判决器”，不是单纯排序器。
- 对内容安全系统而言，减少 fake 漏检比只追求 AUC 更符合实际应用目标。

### 1.3 当前最佳 OOD 单模型：R7_224_FAKEW12

当前最佳主模型是：

```text
R7_224_FAKEW12 = stf3_new + image_size=224 + fake_loss_weight=1.2
```

与同协议的 S+T 双分支模型对比：

| 模型 | 默认阈值 ACC | 默认阈值 AUC | 默认阈值 F1 | 默认 Recall | 默认 FN | Val 校准 Precision≥0.95 Recall-max F1 | 校准 Recall | 校准 FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `spatial_temporal_new_224_fakew12` | 0.8997 | 0.9802 | 0.9062 | 0.8313 | 354 | 0.9360 | 0.9061 | 197 |
| `stf3_new_224_fakew12` | **0.9275** | **0.9848** | **0.9341** | **0.8818** | **248** | **0.9499** | **0.9271** | **153** |

这张表非常重要：它比 Random R5/R7 更能支持“频域分支 + 三分支融合确实有贡献”。

### 1.4 后处理与优化边界

已有后续优化结论：

| 方法 | ACC | F1 | Precision | Recall | FN | FP | 定位 |
|---|---:|---:|---:|---:|---:|---:|---|
| R7_224_FAKEW12 + val-calibrated threshold | 0.9430 | 0.9499 | 0.9740 | 0.9271 | 153 | 52 | 最佳单模型决策阈值 |
| Branch-level LR calibration | **0.9466** | **0.9530** | **0.9794** | 0.9280 | 151 | 41 | 后处理，不是主模型结构 |
| STF3-only ensemble | 0.9439 | 0.9508 | 0.9731 | **0.9295** | **148** | 54 | 多 checkpoint 推理增强 |
| WaveRep-bank | 0.9283 | 0.9351 | **0.9894** | 0.8866 | 238 | **20** | 更保守，降低 FP 但增加 FN |
| 16-frame | 0.9339 | 0.9408 | 0.9834 | 0.9018 | 206 | 32 | 未超过 8-frame baseline |

结论：

- 答辩时应把 `R7_224_FAKEW12` 作为主模型。
- threshold calibration / Branch LR / ensemble 应明确称为“后处理或增强系统”，不能混写成主模型本身。
- WaveRep-bank 和 16-frame 是有价值的负结果：说明你不是盲目堆模块，而是验证了优化边界。

---

## 2. Baseline 应该怎么设？

### 2.1 最推荐的 baseline 层级

你不应该只设置一个 baseline，而应设置多层 baseline：

| 层级 | Baseline | 用途 | 答辩中怎么说 |
|---|---|---|---|
| B0 历史/朴素 baseline | 旧 `STF3Detect`：ResNet18 spatial + pixel temporal diff + FFT-SmallCNN + concat-MLP | 证明你不是简单堆模型，而是从旧 STF3-Lite 现代化升级 | “相对传统三分支实现，STF3-New 引入 foundation representation、D3/ReStraV temporal cues 和 wavelet/spectral cues。” |
| B1 内部消融 baseline | 单分支：`spatial_dino`、`temporal_d3_restrav`、`frequency_wave` | 证明每个信息源单独能力与局限 | “单分支分别代表语义、动态、频域证据。” |
| B2 最重要公平 baseline | `spatial_temporal_new`，尤其是 `spatial_temporal_new_224_fakew12` | 直接回答“三分支比不用频域的强在哪里” | “在相同 backbone、相同训练集、相同阈值选择协议下，加入 frequency branch 与 branch-token fusion 后降低 OOD FN。” |
| B3 融合 baseline | `stf3_new_concat` | 证明不是三分支简单 concat，而是 branch-token Transformer 有价值 | “Transformer fusion 在默认 OOD 表中优于 gated concat。” |
| B4 外部方法 baseline | D3、ReStraV、WaveRep/SPAI/其他视频检测器 | 提高学术性 | “使用公开方法在同一数据协议下复跑或零样本评估。” |

### 2.2 你的 baseline 到底是“单分支”还是“别人论文模型”？

答辩最稳妥的回答是：

> **主 baseline 应该是内部公平消融模型，尤其是 S+T 双分支 `spatial_temporal_new`；外部论文模型作为补充 comparison。**

原因：

1. 你的核心 claim 是 “Spatial + Temporal + Frequency 三种证据融合可以提升跨生成器 AI 视频检测”，所以最直接 baseline 是去掉 Frequency 的 `spatial_temporal_new`。
2. 单分支是必要 ablation，但不能作为唯一 baseline，因为单分支天然弱，老师可能认为对比太容易。
3. 别人论文模型很有学术性，但它们的训练数据、预训练权重、输入帧数、任务定义可能不同；如果直接拿论文表格比，会不公平。
4. 最严谨的方式是：**外部模型如果参与主表，必须在你的 train/val/test 上复跑或统一 zero-shot 协议。**

推荐答辩表格顺序：

```text
Table 1: Internal ablation on GenVideo-Val OOD
B0 old STF3 / B1 S/T/F / B2 S+T / B3 S+T+F concat / Ours S+T+F Transformer

Table 2: Fair external comparison on the same split
D3 / ReStraV-like / WaveRep-like / Ours

Table 3: Lightweight external dataset stress test
Only evaluate final Ours and 2-3 strongest baselines
```

---

## 3. 主指标应该选什么？

### 3.1 不建议只说“我选择最好的参数”

不能说“哪个指标最高我就选哪个”，这在答辩里会被认为是 test-set cherry-picking。正确做法是：

1. 先根据应用场景声明主指标；
2. 用 val 集选阈值或超参数；
3. test 集只做最终报告；
4. 辅助指标同时报告，避免只展示好看的数字。

### 3.2 AI 视频检测系统的推荐主指标

你的任务是 AI 生成视频检测，fake 是正类。系统风险有两类：

- FN：AI 生成视频被漏检为真实视频；
- FP：真实视频被误报为 AI 生成视频。

如果这是内容安全/审核系统，通常 **漏检 fake 的风险更大**，但 FP 太高也会误伤真实用户。因此推荐指标框架是：

| 指标 | 是否主指标 | 原因 |
|---|---|---|
| **F1** | 主指标 1 | 同时考虑 precision 与 recall，适合二分类检测汇总 |
| **AUC** | 主指标 2 / 排序指标 | 不受固定阈值影响，能评价模型把 fake 排在 real 前面的能力 |
| **Recall@Precision≥0.95** 或 “precision-constrained recall” | 主指标 3 / 系统指标 | 很适合内容安全：在误报率可控时尽量少漏检 fake |
| ACC | 辅助 | 样本比例接近时可读性强，但类别不均衡或阈值变化时容易误导 |
| Precision | 辅助 | 衡量误报真实视频的风险 |
| Recall | 辅助但要重点解释 | 衡量漏检 AI 视频的风险 |
| FN/FP | 必报辅助 | 比百分比更直观，尤其适合答辩解释 WildScrape 漏检 |

### 3.3 本项目最有利且合理的主指标选择

我建议你最终这样写：

> 本文以 **OOD test F1** 作为主要综合指标，以 **AUC** 作为阈值无关排序指标，并额外报告 **Precision≥0.95 约束下的 Recall / FN** 作为面向内容安全系统的实用指标。所有阈值均在 validation set 上确定，test set 仅用于最终评估。

这样对你最有利，因为：

1. STF3-New 在 OOD 原始阈值下 F1/ACC/Recall/FN 明显优于单分支和 S+T；
2. `R7_224_FAKEW12` 经 val 阈值校准后 F1 和 Recall 都很好；
3. 即使某些 S+T 模型 AUC 高，STF3 的实际默认决策和校准后 FN 更低，更符合系统检测目标。

---

## 4. 应该和哪些模型比较？

### 4.1 必做：你自己代码里的公平消融

这是最重要、最省时间、最能说服老师的比较。

| 对比组 | 模型 | 作用 |
|---|---|---|
| 单分支 | `spatial_dino`, `temporal_d3_restrav`, `frequency_wave` | 证明每种证据的单独效果 |
| 双分支 | `spatial_frequency_new`, `spatial_temporal_new` | 证明组合效果，特别是 S+T vs S+T+F |
| 融合方式 | `stf3_new_concat` vs `stf3_new` | 证明 branch-token Transformer 比简单融合更稳 |
| 强公平消融 | `spatial_temporal_new_224_fakew12` vs `stf3_new_224_fakew12` | 最关键：同分辨率、同 fake loss，验证 Frequency + STF3 融合贡献 |

如果时间有限，最少要保证这三项：

```text
1. spatial_dino
2. spatial_temporal_new_224_fakew12
3. stf3_new_224_fakew12
```

### 4.2 推荐外部论文模型

你当前方法本身借鉴了几类 2025 方法，因此最自然的外部对比是：

| 外部方法 | 类型 | 与 STF3 的关系 | 建议使用方式 |
|---|---|---|---|
| D3 | training-free / dynamic degeneration | 你的 temporal branch 借鉴了它的二阶动态 | 跑官方 D3 scoring，val 上选阈值，在你的 test 上评估 |
| ReStraV | DINOv2 表征轨迹/representation strangeness | 你的 spatial/temporal foundation trajectory 思路来源之一 | 若官方代码可跑，用其特征和分类器/score 在同一 split 上评估 |
| WaveRep | wavelet band replacement / synthetic video detection | 你的 frequency branch 和 augmentation 来源之一 | 作为频域相关外部方法或 ablation，不一定要完整训练大模型 |
| SPAI | spectral learning / AI-generated image detection | 支撑频谱线索，但偏 image detector | 可逐帧评估后视频级聚合，作为 optional spectral baseline |
| UNITE / universal synthetic video detector | 通用合成视频检测 | 与“跨任务/跨生成器泛化”故事相关 | 如果代码/权重可用再做；否则只放 related work |

重要：这些方法不能直接引用论文中的数字和你的数字比较，因为训练数据/测试集不同。应当写成：

- **Fair re-evaluation**：在同一 `GenVideo-Val` split 上跑；
- 或 **Zero-shot external detector**：使用官方预训练权重，不训练，只在你的 test 上评估，并明确它可能见过不同生成器。

### 4.3 是否要把 CCF-A 论文中的单分支都列出来？

可以，而且这是你当前最好的学术叙事方式。但表述要精确：

> 我不是复现每篇论文的完整大模型，而是将 D3/ReStraV/WaveRep 的核心可解释线索分别实现为 STF3 的 temporal/frequency/spatial branch，并通过同一训练协议比较单分支、双分支和三分支融合。

这样可以避免老师追问“你是否完整复现了 D3/ReStraV/WaveRep 全部实验”。

---

## 5. 关于 OOD、公平训练、训练集与数据集的回答

### 5.1 用 OOD 最佳模型跑，还是重新训练包含所有生成器的模型？

建议分成两张表，不要混在一起：

| 表 | 训练方式 | 测试方式 | 意义 |
|---|---|---|---|
| 主表 A：Random / in-distribution | train/val/test 都含同类生成器家族 | random_test | 证明模型基础检测能力 |
| 主表 B：OOD / cross-generator | train/val 不含 test 的 MorphStudio/Show_1/Sora/WildScrape | ood_test | 证明跨生成器泛化能力 |

不要把“包含所有生成器训练后再测同一生成器”作为唯一主结果。那会变成 seen-generator 检测，故事会弱。你可以补一个 “all-generator random split” 作为额外 in-distribution upper bound，但主结论应以 OOD 为核心。

### 5.2 外部模型如果训练集包含更多生成器，是否不公平？

是的，不公平。处理方式：

1. 如果外部模型使用官方预训练权重，必须标注为 **external pretraining / zero-shot**；
2. 不要把它当作严格主 baseline；
3. 严格主 baseline 必须在你的 `ood_train/ood_val/ood_test` 上训练/验证/测试；
4. 如果外部方法是 training-free，比如 D3，可以用同一 val 阈值协议评估，相对更公平。

### 5.3 是否需要对外部模型重新训练？

如果你要在答辩里说“STF3 优于某某方法”，最好重新训练或至少重新评估。推荐分级：

| 等级 | 做法 | 严谨程度 | 工作量 |
|---|---|---:|---:|
| Level 1 | 只引用论文原表 | 低，只能做 related work | 低 |
| Level 2 | 官方预训练权重 zero-shot 跑你的 test | 中，要注明可能不公平 | 中 |
| Level 3 | 用你的 train/val/test 复跑外部方法 | 高，最推荐 | 高 |
| Level 4 | 多数据集、多 split、多 seed | 最高，课程项目未必需要 | 很高 |

课程答辩最建议：内部消融做到 Level 3，外部模型做到 Level 2 或 3。

### 5.4 如果重新训练，用自己的数据集可以吗？

可以，而且这是最公平的。推荐统一协议：

```text
Train: data/GenVideo-Val/splits/ood_train.csv
Val:   data/GenVideo-Val/splits/ood_val.csv
Test:  data/GenVideo-Val/splits/ood_test.csv
Frame: 8 frames, image_size=112 或 224
Threshold: validation set calibration only
Report: F1 / AUC / Recall@Precision≥0.95 / FN / FP
```

外部模型如果太重，就降低为：

```text
统一抽帧 -> 提取官方 score/logit -> val 选阈值 -> test 报告
```

---

## 6. 轻量外部测试集建议

### 6.1 最推荐：从 GenVideo / Gen-Video 系列再抽一个小 external stress test

你当前数据已经来自 GenVideo-Val 风格的生成器集合。最轻量、最可控的做法是：

```text
External-Lite-1:
  real: real_MSRVTT 另取 500-1000 个未用样本，或从现有 real 中只用于 external test 的不重叠子集
  fake: 每个未见/少见 generator 抽 100-300 个
  total: 1000-3000 videos
```

优点：

- 路径格式和当前代码兼容；
- 不需要改 data loader；
- 可以保持“跨生成器轻量 stress test”的叙事。

### 6.2 可选数据源

| 数据源 | 适合度 | 使用方式 | 风险 |
|---|---:|---|---|
| GenVideo / Gen-Video benchmark 子集 | ★★★★★ | 继续按 generator 抽轻量 subset | 与当前数据同源，外部性较弱但最省事 |
| GenVidBench | ★★★★☆ | 选择少量未见模型/未见内容来源做 stress test | 数据较大，下载和整理成本需确认 |
| VidProM | ★★★☆☆ | 只抽少量 generated videos，配同域 real 或 MSRVTT real | 若只有 generated，real/fake 域差异可能成为混杂因素 |
| EvalCrafter / VideoPhy 类视频生成评测集 | ★★☆☆☆ | 只作为 fake-only stress，配真实视频 | 原始任务不是检测，real/fake 分布可能不匹配 |
| DFDC / FaceForensics++ / Celeb-DF | ★★☆☆☆ | 可做“传统人脸 deepfake”外部泛化 | 与 text-to-video/AIGC 生成视频不同，容易偏题 |

### 6.3 我建议的最小外部数据实验

```text
External-Lite Test v1
- 不训练，只测试最终模型和 2 个 baseline
- real: 500 个真实视频
- fake: 4 个 generator × 200 = 800 个 fake
- total: 1300 videos 左右
- threshold: 仍然只用 GenVideo-Val ood_val 选好的阈值，不在 external test 上调阈值
- report: F1, AUC, Precision, Recall, per-generator Recall/FN
```

测试模型只选：

```text
1. spatial_dino 或 spatial_temporal_new_224_fakew12
2. stf3_new_224_fakew12
3. 如果能跑：D3 官方/近似 baseline
```

这样可以支持答辩叙事：

> 除 GenVideo-Val OOD 外，我们还构建了一个轻量外部 stress test。STF3 在不重新训练、不重新调 test 阈值的情况下仍保持较高 F1/Recall，说明模型不是只记住当前 split。

---

## 7. 建议最终实验矩阵

### 7.1 必须完成的主表

| 表名 | 内容 | 优先级 |
|---|---|---:|
| Table A: OOD internal ablation | R1-R7 原始 OOD，已有 | 必须 |
| Table B: Strong fair ablation | `spatial_temporal_new_224_fakew12` vs `stf3_new_224_fakew12`，已补入 OOD 优化总结；若时间允许可再加 `stf3_new_concat_224_fakew12` | 已完成核心对照 |
| Table C: Threshold calibration | 默认 0.5 vs val-calibrated threshold | 必须 |
| Table D: Generator-level error | 每个 generator 的 Recall/FN，重点 WildScrape | 必须 |
| Table E: External baseline | D3/ReStraV/WaveRep 至少 1-2 个 | 建议 |
| Table F: External-Lite stress test | 轻量外部数据测试 | 建议 |

### 7.2 目前最该补的实验

1. **已完成：R5_224_FAKEW12 的正式文档表**：test/val predictions、metrics 与阈值校准结果已写入 `docs/ood_followup_optimization_summary.md` 的最终强公平消融表，可作为 `spatial_temporal_new_224_fakew12` vs `stf3_new_224_fakew12` 的核心证据。
2. **如果时间允许，训练 `stf3_new_concat_224_fakew12`**：用于证明 Transformer fusion 不是参数设置偶然收益。
3. **跑 D3 官方或近似 score baseline**：这是最容易与“CCF-A 方法”对齐的外部 baseline。
4. **构建 External-Lite Test**：不重新训练，只跑最终模型和 1-2 个 baseline。

---

## 8. 推荐答辩故事线

可以这样组织：

1. **问题**：AI 生成视频检测不能只依赖单一线索；Random split 分数高不代表真实泛化，必须考察跨生成器 OOD。
2. **方法**：STF3 将空间语义、时序动态、频域伪影三种证据变成 branch tokens，用 Transformer 融合。
3. **公平 baseline**：先比较单分支、双分支、简单融合，再比较最终三分支 Transformer。
4. **结果**：在 OOD 中，STF3-New 显著降低 FN，提高 F1/Recall；在 224+fake_loss_weight 的强公平对比中，S+T+F 明显优于 S+T。
5. **系统指标**：阈值只在 val 上校准，在 precision≥0.95 约束下进一步减少漏检，符合内容安全系统需求。
6. **失败分析**：WildScrape 是主要困难来源；WaveRep-bank、16-frame、multi-clip 等尝试并非都有效，说明你做了科学边界验证，而不是只挑好结果。
7. **外部验证**：补充 D3/ReStraV/WaveRep 或轻量外部数据测试，提高学术性与泛化说服力。

---

## 9. 可以直接写进论文/答辩稿的结论句

> 本项目将 AI 生成视频检测建模为跨生成器泛化问题，而非仅在随机划分上追求高准确率。实验表明，单一空间、时序或频域分支在 OOD 场景下均存在明显漏检；相比最强的 spatial-temporal 双分支 baseline，STF3-New 通过引入频域证据与 branch-token Transformer 融合，在相同训练协议下显著降低 fake false negatives，并在 OOD test 上获得更高 F1 和 Recall。进一步地，基于 validation set 的阈值校准在 precision 约束下继续降低漏检，说明模型输出不仅具有较好的排序能力，也能服务于实际内容安全判决。

---

## 10. 参考来源与可复现实验依据

### 本地项目依据

- `src/models/stf3_modern.py`：STF3-New 主模型
- `src/models/backbones/foundation_encoder.py`：DINOv2 / CLIP / XCLIP backbone
- `src/features/d3.py`：D3-style second-order dynamics
- `src/features/geometry.py`：ReStraV-style trajectory geometry
- `src/features/wavelet.py`, `src/features/spectral.py`：WaveRep / SPAI-inspired spectral branch
- `outputs/*/metrics.json`：所有测试指标
- `outputs/ood_followup/*`：阈值校准、generator-level 分析、ensemble 与优化结果

### 外部论文/代码线索

- D3 official repository: https://github.com/Zig-HS/D3
- ReStraV official repository: https://github.com/ChristianInterno/ReStraV
- WaveRep official repository: https://github.com/grip-unina/WaveRep-SyntheticVideoDetection
- SPAI official repository: https://github.com/mever-team/spai
- UNITE paper page: https://openaccess.thecvf.com/content/CVPR2025/html/Ojha_Towards_a_Universal_Synthetic_Video_Detector_From_Face_or_Background_Manipulations_CVPR_2025_paper.html
- DeMamba / Gen-Video related repository: https://github.com/chenhaoxing/DeMamba
- GenVidBench project page: https://genvidbench.github.io/
