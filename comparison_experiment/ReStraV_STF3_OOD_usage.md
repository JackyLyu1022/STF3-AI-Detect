# ReStraV 对比实验环境说明

## 1. 环境位置

ReStraV 使用独立虚拟环境，避免污染主项目环境和 D3 环境：

```powershell
comparison_experiment\.venv_restrav
```

Python 版本：

```powershell
.\comparison_experiment\.venv_restrav\Scripts\python.exe --version
```

当前配置为 Python 3.11。

## 2. 为什么没有直接使用官方 requirements.txt

`comparison_experiment\ReStraV\requirements.txt` 中的版本对 Windows/当前 Python 环境不完全友好：

- 官方写了 `torch==2.5.1` 和 `torchcodec==0.9.1`，但 TorchCodec 0.9.1 更适合与 PyTorch 2.9 系列搭配。
- 官方写了较老的 `numpy/scipy/h5py/pandas`，如果用 Python 3.12 会出现 wheel/兼容性问题。
- TorchCodec 在 Windows 上还需要 FFmpeg shared DLL，而不是只有 `ffmpeg.exe` 的 essentials build。

因此本环境采用“独立环境 + 兼容版本”的方式：

```text
torch        2.9.1+cu128
torchvision  0.24.1+cu128
torchcodec   0.9.1
Python       3.11
```

这不改变 ReStraV 方法本身，只是为了让其在本机可运行。

## 3. 本地 FFmpeg shared 配置

TorchCodec 需要 FFmpeg DLL。已下载到：

```powershell
comparison_experiment\tools\ffmpeg-shared
```

并在 `.venv_restrav` 的 `sitecustomize.py` 中自动添加 DLL 搜索路径，因此之后直接运行该环境的 Python 即可。

## 4. 环境验证命令

推荐直接运行：

```powershell
.\comparison_experiment\run_restrav_env_check.ps1
```

或手动验证：

```powershell
.\comparison_experiment\.venv_restrav\Scripts\python.exe -c "import torch, torchcodec; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); from torchcodec.decoders import VideoDecoder; print('ok')"
```

如果输出 `ok` 且 CUDA 为 `True`，说明 ReStraV 的核心解码和 GPU 环境正常。

## 5. 下一步实验建议

ReStraV 和 D3 不同，它不是完全零训练的最终分类器：

1. ReStraV 先用 DINOv2 从视频中提取 21 维 temporal geometry 特征。
2. 然后用这些 21 维特征训练一个轻量 MLP。
3. 最后在 test set 上评估。

因此公平对比 STF3 时，建议不要直接运行官方 `train.py` 的 50/50 随机划分，而应改成：

- 使用 STF3 的 `ood_train.csv` 训练 ReStraV MLP；
- 使用 STF3 的 `ood_val.csv` 选择阈值；
- 使用 STF3 的 `ood_test.csv` 做最终测试；
- 指标与 D3/STF3 保持一致：ACC、AUC、AP、F1、Precision、Recall、Balanced ACC、TN/FP/FN/TP。

推荐下一步新增脚本：

```text
comparison_experiment/run_restrav_ood_compare.py
```

已新增该脚本，并提供一键运行脚本：

```powershell
.\comparison_experiment\run_restrav_ood_dinov2_t24_w2.ps1
```

该脚本会完成：

- 按 CSV 读取 train/val/test；
- 提取并缓存 ReStraV 21-D 特征；
- 在 train 上训练 MLP；
- 在 val 上选择阈值；
- 在 test 上输出统一格式结果。

输出目录：

```text
comparison_experiment\results\ReStraV\ood_dinov2_t24_w2_e5
```

主要结果文件：

```text
results.json
results.md
results.csv
train_scores.csv
val_scores.csv
test_scores.csv
max_f1\metrics.json
precision_0.95_recall_max\metrics.json
```

如果中断后重跑，已经提取好的特征会从以下目录自动读取，不会重复提取：

```text
comparison_experiment\results\ReStraV\ood_dinov2_t24_w2_e5\features
```

如果确实需要重新提取特征，在手动命令中加：

```powershell
--force-features
```

## 6. 使用官方预训练权重直接测试

如果要避免在本项目数据上重新训练 ReStraV MLP，可以使用 eval-only 脚本：

```powershell
.\comparison_experiment\run_restrav_official_pretrained_ood.ps1
```

运行前需要先把官方 ReStraV 推理文件放到：

```text
comparison_experiment\ReStraV\pretrained
```

需要的文件：

```text
model.pt
mean.npy
std.npy
best_tau.npy   # 可选
```

当前本地克隆的官方 GitHub 仓库没有自带这些权重文件，GitHub Releases 也没有发现发布包。因此如果作者之后提供权重，需要手动放入上述目录。

该 eval-only 脚本会：

- 不使用 `ood_train.csv` 训练；
- 只对 `ood_val.csv` 和 `ood_test.csv` 提取/读取 ReStraV 特征；
- 加载官方 `model.pt / mean.npy / std.npy`；
- 按官方标签约定 `1=real, 0=fake` 将 `prob_real` 转成项目正类分数 `fake_score = 1 - prob_real`；
- 仍然在本项目的 OOD validation split 上选择阈值，并在 OOD test split 上报告 ACC/AUC/AP/F1/Precision/Recall/Balanced ACC/FN/FP；
- 如果存在 `best_tau.npy`，额外输出 `official_best_tau` 这一行。

为了减少重复耗时，脚本会复用之前 24-frame/2-second 实验的特征缓存：

```text
comparison_experiment\results\ReStraV\ood_dinov2_t24_w2_e5\features
```

如果没有该缓存，也会自动重新提取 val/test 特征。
