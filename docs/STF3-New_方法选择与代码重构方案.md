# STF3-New 方法选择与代码重构方案（第 1-3 步）

> 生成日期：2026-05-27  
> 目标：保留 GenVideo-Val、Random/OOD 两类实验和 STF3（Spatial/Temporal/Frequency）总体思想；推翻旧版 ResNet18 + 一阶帧差 + FFT-SmallCNN + concat-MLP 的实现，改为可借鉴 2025/2026 高质量论文代码的现代化 STF3-New。

---

## 1. 最终论文方法选择

### 1.1 选择原则

本项目不是换题，也不是换数据集，而是将 STF3 的每个模块替换为近两年论文中真实使用的检测思想和模型实现。因此筛选时我采用以下优先级：

1. **年份优先**：优先 2025/2026。
2. **会议优先**：优先 CCF-A，底线 CCF-B。
3. **代码优先**：优先官方代码能直接阅读、复用、改写。
4. **适配项目优先**：必须能落到你现有三分支结构中，而不是只作为相关工作。
5. **硬件可行**：优先 RTX 4060 8GB 可通过 frozen encoder、feature cache、small adapter 跑通的方法。

---

### 1.2 最优采用组合

我建议最终主线采用以下组合：

| 分支/模块 | 最终采用方法 | 来源论文 | 年份/会议 | CCF 参考 | 官方代码 | 采用原因 |
|---|---|---|---|---|---|---|
| Spatial | **DINOv2 / XCLIP frozen foundation encoder + adapter** | ReStraV / D3 | NeurIPS 2025 / ICCV 2025 | A / A | https://github.com/ChristianInterno/ReStraV / https://github.com/Zig-HS/D3 | 替代 ResNet18。ReStraV 使用 DINOv2 ViT-S/14，D3 使用 CLIP/XCLIP/DINO 系列编码器，都是现代视觉基础模型。 |
| Temporal | **D3 二阶动态特征 + ReStraV 表征轨迹几何特征** | D3 / ReStraV | ICCV 2025 / NeurIPS 2025 | A / A | 同上 | 替代一阶帧差。D3 提取一阶相邻 embedding 距离的二阶变化；ReStraV 提取 DINOv2 表征轨迹的距离、转角和曲率统计。 |
| Frequency | **WaveRep 小波频带增强 + Wavelet/Spectral feature adapter，SPAI 作为可选 score 分支** | WaveRep / SPAI | NeurIPS 2025 / CVPR 2025 | A / A | https://github.com/grip-unina/WaveRep-SyntheticVideoDetection / https://github.com/mever-team/spai | 替代 FFT + SmallCNN。WaveRep 有官方 PyTorch DWT 增强代码；SPAI 是 2025 CVPR 谱学习检测，可作为逐帧频谱 score 或理论支撑。 |
| Fusion | **Branch-token Transformer / gated attention fusion** | 借鉴 UNITE 类全帧 transformer 观点，自行轻量实现 | CVPR 2025 思路 | A | UNITE 未见稳定官方代码 | 替代 concat + MLP。把 S/T/F 三个分支变成 token，经 Transformer 或 gated attention 自适应融合。 |

最终建议方法名：

```text
STF3-New / STF3-Modern
= DINO/XCLIP Spatial Encoder
+ D3-ReStraV Temporal Dynamics
+ WaveRep-SPAI Frequency Modeling
+ Branch-Token Attention Fusion
```

---

### 1.3 为什么不把所有 2026 方法都放进主线

| 方法 | 年份/会议 | 是否进主线 | 原因 |
|---|---|---:|---|
| STALL | CVPR 2026 | 否，作为 OOD/zero-shot baseline 或后续扩展 | 仓库目前主要是项目页，缺少完整可运行代码；方法很新，但不适合作为主线重构依赖。 |
| NSG-VD | NeurIPS 2025 Spotlight | 否，作为高质量相关工作/可选高级实验 | 代码完整，但 README 要求 RTX 3090 24GB、约 2.4TB 磁盘和扩散模型特征，课程项目成本过高。 |
| VideoVeritas / Skyra | ICML/CVPR 2026 | 否，作为展望 | 偏 MLLM/RL/解释推理，工程成本大，和轻量 STF3 检测主线不完全匹配。 |
| UNITE | CVPR 2025 | 只借鉴融合思想 | 论文质量高，但可复用代码不如 D3/ReStraV/WaveRep/SPAI 明确。 |

---

## 2. STF3-New 总体架构设计

### 2.1 旧版 STF3-Lite 的问题

当前代码的核心结构是：

```text
frames [B,T,3,H,W]
  ├─ SpatialBranch: ResNet18(frame) -> mean over T -> 256
  ├─ TemporalBranch: abs(frame[t]-frame[t-1]) -> SmallCNN -> 128
  ├─ FrequencyBranch: FFT log-mag -> SmallCNN -> 128
  └─ concat -> MLPHead -> logits
```

主要老旧点：

1. ResNet18 是传统 CNN，空间表征能力和新生成模型泛化能力不足。
2. TemporalBranch 只看像素级一阶帧差，无法捕捉高阶动态异常。
3. FrequencyBranch 只做 FFT log magnitude，频域建模太浅。
4. SmallCNN 过于简单，且不是近年论文主流。
5. concat + MLP 无法学习不同分支在不同视频上的可信度。
6. 训练脚本只支持朴素 CE loss，没有分支级辅助监督、branch dropout、feature cache 等现代实验组织。

---

### 2.2 新架构总览

```text
Input video frames [B,T,3,H,W]
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
  Modern frame preprocessing                    WaveRep augmentation
  - resize/crop 224 or 336                      - DWT band replacement
  - model-specific normalize                    - train only
        │                                              │
        ├──────────────────────┬───────────────────────┤
        │                      │                       │
        ▼                      ▼                       ▼
Spatial Foundation Branch  Temporal Dynamics Branch  Frequency-Spectral Branch
(DINOv2 / XCLIP)           (D3 + ReStraV)             (Wavelet + SPAI-lite)
        │                      │                       │
   s_token [B,D]          t_token [B,D]           f_token [B,D]
        │                      │                       │
        └──────────────┬───────┴───────────────┬───────┘
                       ▼                       
            Branch Token Fusion Transformer
            - [CLS], [S], [T], [F] tokens
            - branch type embedding
            - attention/gated weighting
            - optional branch dropout
                       │
                       ▼
              classifier head -> logits
```

---

### 2.3 Spatial Branch：从 ResNet18 改成视觉基础模型

#### 采用方法

- 主选：**DINOv2 ViT-S/14**，来自 ReStraV 官方实现。
- 可选：**XCLIP-base-patch16 / CLIP ViT-B/16**，来自 D3 官方实现。

#### 设计

```text
frames [B,T,3,H,W]
  -> model-specific preprocess
  -> frozen DINOv2/XCLIP encoder
  -> CLS / pooler output [B,T,C]
  -> temporal attention pooling or mean pooling
  -> projection adapter -> s_token [B,D]
```

#### 代码借鉴点

- ReStraV `dinov2_features.py`：
  - `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
  - `forward_features()` 提取 `x_norm_clstoken` 和 patch tokens。
- D3 `models/D3_model.py`：
  - `CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")`
  - `XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")`
  - `AutoModel.from_pretrained("facebook/dinov2-base")`

#### 替代关系

```text
旧：torchvision.models.resnet18 + mean pooling
新：DINOv2/XCLIP frozen encoder + adapter + temporal attention pooling
```

---

### 2.4 Temporal Branch：从一阶帧差改成 D3 + ReStraV

#### 采用方法 A：D3 二阶动态

D3 的代码逻辑是：

```text
frame embeddings z_t
  -> first-order distance: dist(z_t, z_{t+1})
  -> second-order difference: dist_{t+1} - dist_t
  -> mean/std as detection statistics
```

这比当前 `abs(frame[t]-frame[t-1])` 更现代，因为它不是在像素空间做差，而是在 CLIP/XCLIP/DINO embedding 空间计算动态变化。

#### 采用方法 B：ReStraV 表征轨迹几何

ReStraV 的 21 维特征包括：

```text
7 early stepwise distances
6 early turning angles
8 global statistics: mean/min/max/var for distances and angles
```

#### 新 Temporal Branch 设计

```text
frame embeddings Z [B,T,C]
  ├─ D3SecondOrder:
  │    dist_t = L2/Cosine(z_t, z_{t+1})
  │    d2_t = dist_{t+1} - dist_t
  │    stats = mean/std/max/top-k
  │
  ├─ ReStraVGeometry:
  │    delta_t = z_{t+1} - z_t
  │    distance, angle, curvature stats -> 21-D
  │
  └─ temporal adapter MLP -> t_token [B,D]
```

#### 替代关系

```text
旧：pixel abs difference + SmallCNN
新：foundation embedding trajectory + D3 second-order statistics + ReStraV curvature features
```

---

### 2.5 Frequency Branch：从 FFT-SmallCNN 改成 WaveRep + SPAI-lite

#### 采用方法 A：WaveRep 小波频带增强

WaveRep 官方代码提供 `augmentation/pytorch_wt.py`，核心是：

- `Troch_DWT`：可微/torch 形式的 DWT 分解；
- `Troch_iDWT`：逆小波重建；
- `waverep_d2`：在训练时随机替换某些低/高频带，引导模型关注 forensic traces。

#### 采用方法 B：SPAI 频谱学习

SPAI 是 CVPR 2025，核心是 masked spectral learning 和 spectral reconstruction similarity。它是图像检测方法，但视频可逐帧处理后做视频级聚合。

#### 新 Frequency Branch 设计

```text
frames [B,T,3,H,W]
  ├─ Wavelet decomposition: LL/LH/HL/HH at 1-3 levels
  ├─ Radial FFT statistics: low/mid/high frequency energy
  ├─ Optional SPAI frame score: mean/max/top-k over T
  └─ spectral adapter MLP/Transformer -> f_token [B,D]
```

重点：不再使用旧的 `SmallCNN`。频域分支以“小波多频带统计 + 谱学习 score/adapter”为主。

#### 替代关系

```text
旧：FFT log magnitude -> SmallCNN
新：WaveRep DWT bands + radial spectral statistics + optional SPAI score adapter
```

---

### 2.6 Fusion：从 concat + MLP 改成 Branch-Token Transformer

#### 旧方式问题

当前融合方式：

```python
feat = torch.cat([s, t, f], dim=1)
logits = MLP(feat)
```

它无法表达：

- 当前视频更应该相信哪个分支；
- OOD 生成器下频域/时域特征权重变化；
- 分支之间的相互校验。

#### 新方式

```text
s_token, t_token, f_token -> project to same dim D
add branch type embedding
prepend learnable cls token
TransformerEncoder / gated attention
cls output -> classifier logits
```

可进一步加入：

- branch dropout：训练时随机丢掉某个分支，避免过度依赖单一特征；
- aux heads：每个分支单独输出一个 logit，便于做消融和辅助 loss；
- confidence-aware late fusion：评估时保存各分支概率，便于分析。

#### 替代关系

```text
旧：concat + MLPHead
新：[CLS,S,T,F] branch-token transformer + gated branch weights + aux heads
```

---

## 3. 旧代码到新代码的完整替换清单

### 3.1 模型层替换

| 当前文件/模块 | 当前实现 | 问题 | 新实现 | 参考论文/代码 |
|---|---|---|---|---|
| `src/models/spatial_branch.py` | ResNet18 + mean pooling | 传统 CNN，泛化弱 | `DINOv2SpatialBranch` / `XCLIPSpatialBranch` | ReStraV / D3 |
| `src/models/temporal_branch.py` | 像素一阶帧差 + SmallCNN | 只看局部低级变化 | `D3SecondOrderBranch` + `ReStraVGeometryBranch` | D3 / ReStraV |
| `src/models/frequency_branch.py` | FFT log-mag + SmallCNN | 频域建模浅 | `WaveletSpectralBranch` + optional `SPAIScoreBranch` | WaveRep / SPAI |
| `SmallCNN` | 手写浅层 CNN | 老旧且不是论文核心 | 删除 | - |
| `MLPHead` | concat 后两层 MLP | 融合方式弱 | `BranchTokenFusionTransformer` | 借鉴 transformer 融合思想 |
| `STF3Detect` | 单文件硬编码三分支 | 扩展困难 | `STF3Modern`，模块化分支+融合 | 本项目新框架 |

---

### 3.2 数据与特征层替换

| 当前位置 | 当前做法 | 新做法 |
|---|---|---|
| `dataset.py/read_video_frames` | OpenCV 均匀采样 + resize 到同一尺寸 | 保留采样逻辑，但输出原始 RGB tensor；新增 model-specific preprocess，支持 DINOv2/XCLIP/SPAI 的 normalize/resize。 |
| `random_sample` | 简单随机 temporal crop | 增加 ReStraV 风格 center-window clip、D3 风格固定 T 帧、OOD 评估 deterministic sampling。 |
| 无 feature cache | 每次训练重复抽基础模型特征 | 新增 `scripts/extract_foundation_features.py`，缓存 DINO/XCLIP embeddings，降低 RTX 4060 压力。 |
| 无频域增强 | 只训练原帧 | 新增 WaveRep DWT band replacement augmentation，仅训练时开启。 |
| 无 paired augmentation | 只单样本 transform | WaveRep 需要 batch 内/数据集内帧对，可实现 batch-pair band replacement。 |

---

### 3.3 训练层替换

| 当前文件/逻辑 | 当前实现 | 新实现 |
|---|---|---|
| `train.py --model choices` | `spatial/frequency/temporal/stf3` | 新增 `spatial_dino`, `temporal_d3`, `temporal_restrav`, `frequency_wave`, `stf3_new`。 |
| Loss | 单一 CrossEntropy | `CE(main) + lambda_aux * CE(aux branches)`，可选 consistency loss。 |
| Optimizer | AdamW 全参数 | foundation encoder 默认 freeze，只训练 adapters/fusion；可选最后 N 层微调。 |
| Scheduler | CosineAnnealingLR | 保留，但增加 warmup 更稳。 |
| AMP | 可选 | 保留；foundation encoder + batch size 1/2 时必须建议开启。 |
| 实验记录 | history.json | 保留，同时保存 branch logits、branch weights、config。 |

---

### 3.4 评估与可视化替换

| 当前输出 | 新增输出 |
|---|---|
| ACC / AUC / F1 | 保留 |
| confusion matrix / ROC | 保留 |
| accuracy_by_generator | 保留，Random/OOD 都要有 |
| 无分支权重分析 | 新增 `branch_attention_by_generator.png` |
| 无分支单独预测 | 新增 `predictions.csv` 中保存 `prob_s/prob_t/prob_f/prob_fused` |
| 无新版消融 | 新增 `ablation_summary.csv` |

---

### 3.5 推荐新代码结构

```text
src/
  models/
    backbones/
      dinov2_encoder.py
      xclip_encoder.py
      clip_encoder.py
    branches/
      spatial_dino_branch.py
      temporal_d3_branch.py
      temporal_restrav_branch.py
      frequency_wavelet_branch.py
      frequency_spai_branch.py
    fusion/
      branch_token_transformer.py
      gated_fusion.py
    stf3_modern.py
  features/
    geometry.py              # ReStraV 21-D features
    d3.py                    # first-order distance + second-order difference
    wavelet.py               # WaveRep DWT utilities
    spectral.py              # FFT radial stats / SPAI score aggregation
  transforms/
    video_transforms.py
    waverep_augment.py
scripts/
  extract_foundation_features.py
  run_stf3_new_random.ps1
  run_stf3_new_ood.ps1
  summarize_ablation.py
configs/
  stf3_new.yaml
  stf3_new_ood.yaml
```

---

## 4. 最终推荐实验矩阵

保留你的两大实验：

1. **Random Split**：同分布综合性能。
2. **OOD Split**：跨生成器泛化，重点指标。

新增消融如下：

| 实验名 | Spatial | Temporal | Frequency | Fusion | 目的 |
|---|---|---|---|---|---|
| old_stf3 | ResNet18 | 一阶帧差 | FFT-SmallCNN | concat-MLP | 旧版对照 |
| spatial_dino | DINOv2 | - | - | linear | 验证新空间分支 |
| temporal_d3 | - | D3 | - | linear | 验证二阶动态 |
| temporal_restrav | - | ReStraV | - | linear | 验证轨迹几何 |
| frequency_wave | - | - | Wavelet/Spectral | linear | 验证新频域分支 |
| stf3_new_concat | DINOv2 | D3+ReStraV | Wavelet | concat-MLP | 验证仅替换分支的收益 |
| stf3_new | DINOv2/XCLIP | D3+ReStraV | WaveRep/SPAI-lite | branch-token transformer | 最终方法 |
| stf3_new_no_waveaug | DINOv2/XCLIP | D3+ReStraV | Wavelet no aug | branch-token transformer | 验证 WaveRep 增强 |
| stf3_new_no_d3 | DINOv2/XCLIP | ReStraV only | Wavelet | branch-token transformer | 验证 D3 |
| stf3_new_no_restrav | DINOv2/XCLIP | D3 only | Wavelet | branch-token transformer | 验证 ReStraV |

---

## 5. 最终建议：最优解版本

如果只选一个最合理、最能落地、最容易写进论文的方法，我建议：

```text
STF3-New = DINOv2 Spatial + D3-ReStraV Temporal + WaveRep Wavelet Frequency + Branch-Token Transformer Fusion
```

SPAI 不作为第一阶段必须复现的核心分支，而作为：

1. 频域分支的理论参考；
2. 如果预训练权重下载成功，则作为 optional `SPAIScoreBranch`；
3. 论文中替代 FreqNet 的 2025 CVPR 频域相关工作。

STALL、NSG-VD、VideoVeritas、Skyra 不进入主线，但可以放入相关工作或后续展望。

---

## 6. 第 4 步代码实现前的明确任务边界

下一步真正动代码时，不应只是改 3 个分支文件，而应完成：

1. 新建现代化模型目录结构；
2. 实现 DINOv2/XCLIP backbone wrapper；
3. 实现 D3 second-order feature；
4. 实现 ReStraV 21-D geometry feature；
5. 实现 WaveRep DWT augmentation 和 wavelet spectral features；
6. 实现 branch-token fusion transformer；
7. 修改 `train.py/evaluate.py` 支持 `stf3_new` 和分支消融；
8. 更新实验脚本，继续跑 Random/OOD 两类实验。

这才符合“推翻旧代码，但保留任务、数据集和实验协议”的老师反馈。
