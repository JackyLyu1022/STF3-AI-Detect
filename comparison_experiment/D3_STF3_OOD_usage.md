# D3 vs STF3 OOD 对比运行说明

## 1. D3 是否需要训练？

不需要。D3 是 **training-free** 外部检测方法。

在本项目中的对比流程是：

```text
ood_val.csv 视频 -> D3 二阶动态分数 -> 在 val 上判断分数方向并选择阈值
ood_test.csv 视频 -> D3 二阶动态分数 -> 使用 val 上固定好的方向与阈值评估 test
```

因此 D3 只做推理和阈值校准，不训练检测器。

## 2. 环境位置

D3 使用独立虚拟环境，避免污染 STF3 主环境：

```text
comparison_experiment/.venv_d3
```

已安装的核心版本：

```text
torch==2.8.0+cu128
torchvision==0.23.0+cu128
transformers==4.57.0
timm==1.0.20
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
opencv-python==4.12.0.88
```

说明：D3 官方 `requirements.txt` 中 `numpy==2.3.3` 与 `opencv-python==4.12.0.88` 存在依赖冲突，因为该 OpenCV 版本要求 `numpy<2.3.0`。因此本环境使用 `numpy==2.2.6`，其余保持接近 D3 官方配置。

## 3. 适配脚本

主脚本：

```text
comparison_experiment/run_d3_ood_compare.py
```

它直接读取你的原始视频 CSV：

```text
data/GenVideo-Val/splits/ood_val.csv
data/GenVideo-Val/splits/ood_test.csv
```

输出：

```text
comparison_experiment/results/D3/ood_xclip16_l2/
  results.md
  results.csv
  results.json
  val_scores.csv
  test_scores.csv
  max_f1/metrics.json
  max_f1/predictions.csv
  max_balanced_acc/metrics.json
  precision_0.95_recall_max/metrics.json
  max_acc/metrics.json
```

## 4. 完整 OOD 运行命令

在项目根目录运行：

```powershell
.\comparison_experiment\run_d3_ood_xclip16_l2.ps1
```

或者直接运行：

```powershell
.\comparison_experiment\.venv_d3\Scripts\python.exe comparison_experiment\run_d3_ood_compare.py `
  --encoder XCLIP-16 `
  --loss l2 `
  --device cuda:0 `
  --num-frames 16 `
  --image-size 224 `
  --out-dir comparison_experiment\results\D3\ood_xclip16_l2
```

## 5. Smoke test 状态

已完成极小样本测试：

```text
val: 2 samples
test: 2 samples
encoder: XCLIP-16
loss: l2
device: cuda:0
result dir: comparison_experiment/results/D3/smoke_xclip16_l2
```

结论：环境、模型导入、视频读取、D3 分数计算、val 阈值选择和 test 指标输出均已跑通。

## 6. 论文/答辩中如何描述

推荐写法：

> D3 is evaluated as a training-free external baseline. We do not directly quote the numbers from the original paper. Instead, D3 scores are computed on the same GenVideo-Val OOD validation and test splits. Score orientation and thresholds are selected only on the validation split and then fixed for the test split.

中文：

> 本文将 D3 作为 training-free 外部对比方法。为避免不同训练集造成的不公平比较，我们不直接引用 D3 原论文结果，而是在相同的 GenVideo-Val OOD validation/test split 上重新计算 D3 分数，并仅使用 validation set 确定分数方向和阈值，再固定到 test set 评估。

