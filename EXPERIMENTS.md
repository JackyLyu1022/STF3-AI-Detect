# STF³-Detect 实验运行指南

本文件对应开题报告中的第二、三、四、五阶段代码。

## 1. 激活环境

```powershell
cd "D:\VsCode Program\Python\content_security\final_project"
.\.venv\Scripts\Activate.ps1
```

## 2. 数据检查

```powershell
python scripts\prepare_genvideo_val.py
```

输出应包含：

- `data/GenVideo-Val/metadata.csv`
- `data/GenVideo-Val/splits/random_train.csv`
- `data/GenVideo-Val/splits/random_val.csv`
- `data/GenVideo-Val/splits/random_test.csv`
- `data/GenVideo-Val/splits/ood_train.csv`
- `data/GenVideo-Val/splits/ood_val.csv`
- `data/GenVideo-Val/splits/ood_test.csv`

## 3. 快速冒烟测试

```powershell
python -m src.train --model spatial --epochs 1 --batch-size 2 --num-frames 4 --image-size 112 --max-train-samples 12 --max-val-samples 8 --out-dir runs\smoke_spatial
python -m src.evaluate --checkpoint runs\smoke_spatial\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 2 --max-samples 8 --out-dir outputs\smoke_eval
python -m src.visualize --predictions outputs\smoke_eval\predictions.csv --history runs\smoke_spatial\history.json --out-dir outputs\smoke_figures
```

## 4. 正式随机划分实验

RTX 4060 8GB 建议先用：`batch-size=2`，`num-frames=8`，`image-size=112`。
稳定后再尝试 `num-frames=16` 或 `image-size=224`。

### 空间分支 baseline

```powershell
python -m src.train --model spatial --epochs 5 --batch-size 2 --num-frames 8 --image-size 112 --out-dir runs\random_spatial
python -m src.evaluate --checkpoint runs\random_spatial\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 2 --out-dir outputs\random_spatial
python -m src.visualize --predictions outputs\random_spatial\predictions.csv --history runs\random_spatial\history.json --out-dir outputs\random_spatial_figures
```

### 频域分支 baseline

```powershell
python -m src.train --model frequency --epochs 5 --batch-size 2 --num-frames 8 --image-size 112 --out-dir runs\random_frequency
python -m src.evaluate --checkpoint runs\random_frequency\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 2 --out-dir outputs\random_frequency
```

### 时序差分 baseline

```powershell
python -m src.train --model temporal --epochs 5 --batch-size 2 --num-frames 8 --image-size 112 --out-dir runs\random_temporal
python -m src.evaluate --checkpoint runs\random_temporal\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 2 --out-dir outputs\random_temporal
```

### 空间 + 频域双分支

```powershell
python -m src.train --model spatial_frequency --epochs 5 --batch-size 1 --num-frames 8 --image-size 112 --out-dir runs\random_spatial_frequency
python -m src.evaluate --checkpoint runs\random_spatial_frequency\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 1 --out-dir outputs\random_spatial_frequency
```

### STF³-Detect-Lite 三分支

```powershell
python -m src.train --model stf3 --epochs 5 --batch-size 1 --num-frames 8 --image-size 112 --out-dir runs\random_stf3
python -m src.evaluate --checkpoint runs\random_stf3\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 1 --out-dir outputs\random_stf3
python -m src.visualize --predictions outputs\random_stf3\predictions.csv --history runs\random_stf3\history.json --out-dir outputs\random_stf3_figures
```

## 5. 跨生成器 OOD 实验

训练集：`ood_train.csv`  
验证集：`ood_val.csv`  
测试集：`ood_test.csv`

OOD 测试生成器：`MorphStudio / Show_1 / Sora / WildScrape`

```powershell
python -m src.train --model stf3 --train-csv data\GenVideo-Val\splits\ood_train.csv --val-csv data\GenVideo-Val\splits\ood_val.csv --epochs 5 --batch-size 1 --num-frames 8 --image-size 112 --out-dir runs\ood_stf3
python -m src.evaluate --checkpoint runs\ood_stf3\best.pt --csv data\GenVideo-Val\splits\ood_test.csv --batch-size 1 --out-dir outputs\ood_stf3
python -m src.visualize --predictions outputs\ood_stf3\predictions.csv --history runs\ood_stf3\history.json --out-dir outputs\ood_stf3_figures
```

## 6. 结果文件说明

每个 `runs/<实验名>/` 下：

- `best.pt`：验证集最优模型
- `last.pt`：最后一轮模型
- `history.json`：训练过程指标

每个 `outputs/<实验名>/` 下：

- `metrics.json`：ACC / AUC / F1 / Precision / Recall / Confusion Matrix
- `predictions.csv`：每个视频的预测结果

每个可视化目录下：

- `confusion_matrix.png`
- `roc_curve.png`
- `accuracy_by_generator.png`
- `history_loss.png`
- `history_acc.png`
- `history_auc.png`
- `frequency_examples.png`

## 7. 建议最终报告表格

| 方法 | Random ACC | Random AUC | OOD ACC | OOD AUC |
|---|---:|---:|---:|---:|
| Spatial | - | - | - | - |
| Frequency | - | - | - | - |
| Temporal | - | - | - | - |
| Spatial + Frequency | - | - | - | - |
| STF³-Detect-Lite | - | - | - | - |

---

## 8. STF3-New 现代化重构实验

STF3-New 保留 Random / OOD 两类实验，但推翻旧版 ResNet18 + 一阶帧差 + FFT-SmallCNN + concat-MLP 实现，改为：

- Spatial：ReStraV/D3 风格 DINOv2 / XCLIP foundation encoder；
- Temporal：D3 二阶动态 + ReStraV 表征轨迹几何；
- Frequency：WaveRep 小波频带建模 + spectral statistics；
- Fusion：branch-token Transformer / gated fusion。

### 不下载基础模型的 smoke

```powershell
.\scripts\run_stf3_new_smoke.ps1
```

### STF3-New Random Split

```powershell
.\scripts\run_stf3_new_random.ps1
```

等价核心命令：

```powershell
python -m src.train --model stf3_new --train-csv data\GenVideo-Val\splits\random_train.csv --val-csv data\GenVideo-Val\splits\random_val.csv --epochs 5 --batch-size 1 --num-frames 8 --image-size 224 --foundation-backbone dinov2_vits14 --wavelet-aug-prob 0.1 --branch-dropout 0.1 --aux-loss-weight 0.2 --amp --out-dir runs\random_stf3_new
python -m src.evaluate --checkpoint runs\random_stf3_new\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 1 --amp --out-dir outputs\random_stf3_new
```

### STF3-New OOD Split

```powershell
.\scripts\run_stf3_new_ood.ps1
```

### 新版消融模式

| 模式 | 含义 |
|---|---|
| `spatial_dino` | DINOv2/XCLIP 空间分支 |
| `temporal_d3` | D3 二阶动态分支 |
| `temporal_restrav` | ReStraV 21-D 表征轨迹分支 |
| `temporal_d3_restrav` | D3 + ReStraV 联合时序分支 |
| `frequency_wave` | WaveRep/Spectral 频域分支 |
| `spatial_frequency_new` | 新空间分支 + 新频域分支，branch-token Transformer 融合 |
| `spatial_temporal_new` | 新空间分支 + 新时序分支，branch-token Transformer 融合 |
| `stf3_new_concat` | 新三分支 + gated concat 融合 |
| `stf3_new` | 新三分支 + branch-token Transformer 融合 |

注意：首次运行 `stf3_new` 默认会通过 `torch.hub` 下载 DINOv2 权重。
