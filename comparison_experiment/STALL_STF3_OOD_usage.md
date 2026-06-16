# STALL 对比实验环境与运行说明

## 1. 仓库与环境

STALL 已下载到：

```text
comparison_experiment\STALL
```

当前复用 ReStraV 的独立环境：

```text
comparison_experiment\.venv_restrav
```

原因：

- STALL 和 ReStraV 都需要 PyTorch / torchvision / numpy / pandas / sklearn / tqdm；
- 已补充 STALL 额外依赖：`opencv-python`, `datasets`, `pyarrow`, `tabulate`；
- STALL 官方 `environment.yml` 建议 Python 3.10，但当前 Python 3.11 环境已通过核心导入检查。

## 2. 当前状态

已经完成：

- 克隆 STALL 官方仓库；
- 克隆 DINOv3 仓库到 `comparison_experiment\STALL\dinov3`；
- STALL 官方 VATEX 标定参数已随仓库存在：

```text
comparison_experiment\STALL\precomputed\stall_params_vatex_dino_v3.npz
```

尚需手动完成：

```text
comparison_experiment\STALL\dinov3\weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

该文件需要从 Meta DINOv3 官方页面下载：

```text
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
```

选择：

```text
ViT-L/16 distilled
pretraining dataset: LVD-1689M
filename: dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## 3. 为什么需要 DINOv3 权重

STALL 的本地视频模式不是直接读取 RGB 训练分类器，而是：

```text
video -> DINOv3 frame embeddings -> spatial likelihood + temporal likelihood -> final realness score
```

官方已经提供 VATEX real-video 标定参数，但你的视频仍然需要 DINOv3 提取 embedding。

## 4. 与 STF3 的公平性设置

当前一键脚本使用：

```text
duration = 1 second
target_fps = 8
effective_frames = 8
```

这和 STF3 的 `num_frames=8` 尽量接近。

注意：

- STALL 是 training-free，不使用你的 fake train set；
- 当前脚本只使用 `ood_val.csv` 选阈值，`ood_test.csv` 测试；
- 使用相同正类定义：`fake = 1`；
- 指标与 D3/STF3 一致：ACC/AUC/AP/F1/Precision/Recall/Balanced ACC/FN/FP。

## 5. 运行命令

下载并放好 DINOv3 权重后，在项目根目录运行：

```powershell
.\comparison_experiment\run_stall_ood_t8.ps1
```

输出目录：

```text
comparison_experiment\results\STALL\ood_vatex_dinov3_t8
```

主要输出文件：

```text
results.json
results.md
results.csv
val_scores.csv
test_scores.csv
max_f1\metrics.json
precision_0.95_recall_max\metrics.json
```

## 6. 答辩定位

推荐定位：

```text
External spatial-temporal dual-branch training-free baseline
```

推荐表述：

> STALL is evaluated as a recent CVPR 2026 training-free spatial-temporal dual-branch baseline. It uses real-video calibrated spatial and temporal likelihoods, without training on our fake generators. We evaluate it under the same OOD validation/test protocol as STF3.

