# STF3-New Random / OOD 完整实验运行方案

> 目标：统一组内实验参数，先完成 Random Split 全部消融；后续按同一规范补 OOD。  
> 当前日期：2026-05-27  
> 项目路径：`D:\VsCode Program\Python\content_security\final_project`

---

## 0. 轮数与参数选择结论

### 推荐先跑几轮？

**建议第一轮完整实验统一跑 `epochs=5`，不建议一开始直接全部跑 `epochs=10`。**

原因：

1. 之前中等规模验证中，`stf3_new` 在前 1-3 个 epoch 已经达到很高验证 AUC，说明当前冻结 DINOv2 主干后，训练收敛速度较快。
2. Random 全量有约 12,804 个训练样本，10 epoch 会比较耗时；如果 7 个 Random 实验全部 10 epoch，可能需要 30+ 小时。
3. 消融实验最重要的是公平比较趋势。只要所有模型统一 epoch，`epochs=5` 可以作为第一版正式结果。
4. 如果后面发现某些模型的 `history.json` 里 val AUC 到第 5 轮仍在明显上升，再补跑 `epochs=10`。

### 最终论文建议

推荐采用两阶段策略：

| 阶段 | 用途 | epoch | 说明 |
|---|---|---:|---|
| 第一阶段 | 完整 Random / OOD 消融 | 5 | 用于快速得到主表和趋势 |
| 第二阶段 | 关键模型精跑 | 10 | 只对 `stf3_new`、`stf3_new_concat`、表现最强的双分支模型补跑 |

**注意：同一张表里的模型必须使用同样 epoch。** 例如 Random 消融表如果用 5 epoch，就所有 Random 模型都用 5 epoch。

---

## 1. 统一基础参数

所有实验默认使用以下设置：

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `epochs` | `5` | 第一轮完整实验推荐值 |
| `batch-size` | `1` | 当前 RTX 4060 Laptop GPU 稳定设置 |
| `num-frames` | `8` | 与老师计划一致 |
| `image-size` | `112` | 与老师计划一致，速度明显快于 224 |
| `foundation-backbone` | `dinov2_vits14` | 当前已下载权重，速度和显存较合适 |
| `wavelet-aug-prob` | `0.1` | 含频域分支的模型使用；无频域分支也保留不影响 |
| `branch-dropout` | `0.1` | 融合模型正则化 |
| `aux-loss-weight` | `0.2` | 分支辅助监督 |
| `lr` | `1e-4` | 代码默认值，不额外写也可以 |
| `weight-decay` | `1e-4` | 代码默认值，不额外写也可以 |
| `seed` | `42` | 代码默认值，统一随机性 |
| `amp` | 开启 | 混合精度，加速并省显存 |

不建议第一轮改动：

```text
--finetune-foundation
--foundation-backbone dinov2_vitb14
--image-size 224
```

这些会显著增加时间和显存压力，适合后期单独做提升实验。

---

## 2. 模型分类说明

### 单分支实验

| 模型 | 分支 | 作用 |
|---|---|---|
| `spatial_dino` | Spatial | DINOv2 空间语义分支 |
| `frequency_wave` | Frequency | Wavelet / Spectral 频域分支 |
| `temporal_d3_restrav` | Temporal | D3 + ReStraV 时序分支 |

### 双分支组合实验

| 模型 | 分支 | 作用 |
|---|---|---|
| `spatial_frequency_new` | Spatial + Frequency | 老师要求的 `spatial_frequency` 现代版 |
| `spatial_temporal_new` | Spatial + Temporal | 老师要求的 `spatial_temporal` 现代版 |

注意：不要用旧模型名：

```text
spatial_frequency
spatial_temporal
```

旧模型名对应旧版 ResNet/CNN 体系；正式新方法实验请使用：

```text
spatial_frequency_new
spatial_temporal_new
```

### 三分支融合实验

| 模型 | 分支 | 融合方式 | 作用 |
|---|---|---|---|
| `stf3_new_concat` | Spatial + Temporal + Frequency | Gated Concat | 融合方式对比 |
| `stf3_new` | Spatial + Temporal + Frequency | Branch-token Transformer | 最终方法 |

---

## 3. Random Split 全部实验

Random 是当前第一优先级。建议先完整跑下面 7 个。

### Random 数据

```text
train: data\GenVideo-Val\splits\random_train.csv
val:   data\GenVideo-Val\splits\random_val.csv
test:  data\GenVideo-Val\splits\random_test.csv
```

### Random 实验清单

| 编号 | 模型 | 输出目录 | 预计 5 epoch 时长 | 预计 10 epoch 时长 |
|---|---|---|---:|---:|
| R1 | `frequency_wave` | `runs/random_frequency_wave` | 0.8 - 1.3 h | 1.5 - 2.5 h |
| R2 | `spatial_dino` | `runs/random_spatial_dino` | 2.0 - 2.7 h | 4.0 - 5.0 h |
| R3 | `temporal_d3_restrav` | `runs/random_temporal_d3_restrav` | 2.0 - 2.8 h | 4.0 - 5.0 h |
| R4 | `spatial_frequency_new` | `runs/random_spatial_frequency_new` | 2.3 - 3.0 h | 4.5 - 5.5 h |
| R5 | `spatial_temporal_new` | `runs/random_spatial_temporal_new` | 2.5 - 3.2 h | 5.0 - 6.0 h |
| R6 | `stf3_new_concat` | `runs/random_stf3_new_concat` | 2.5 - 3.3 h | 5.0 - 6.0 h |
| R7 | `stf3_new` | `runs/random_stf3_new` | 2.8 - 3.6 h | 5.0 - 6.5 h |

Random 7 个实验总时长估计：

```text
5 epoch: 约 15 - 20 小时
10 epoch: 约 30 - 37 小时
```

---

## 4. Random 单个实验运行模板

把 `<MODEL>` 替换为对应模型名。

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model <MODEL> `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_<MODEL>

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_<MODEL>\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_<MODEL>
```

---

## 5. Random 每个实验的具体命令

### R1. Frequency only: `frequency_wave`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model frequency_wave `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_frequency_wave

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_frequency_wave\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_frequency_wave
```

### R2. Spatial only: `spatial_dino`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model spatial_dino `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_spatial_dino

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_spatial_dino\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_spatial_dino
```

### R3. Temporal only: `temporal_d3_restrav`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model temporal_d3_restrav `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_temporal_d3_restrav

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_temporal_d3_restrav\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_temporal_d3_restrav
```

### R4. Spatial + Frequency: `spatial_frequency_new`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model spatial_frequency_new `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_spatial_frequency_new

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_spatial_frequency_new\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_spatial_frequency_new
```

### R5. Spatial + Temporal: `spatial_temporal_new`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model spatial_temporal_new `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_spatial_temporal_new

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_spatial_temporal_new\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_spatial_temporal_new
```

### R6. STF3-New Concat Fusion: `stf3_new_concat`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model stf3_new_concat `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_stf3_new_concat

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_stf3_new_concat\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_stf3_new_concat
```

### R7. STF3-New Final: `stf3_new`

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model stf3_new `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_stf3_new

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\random_stf3_new\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_stf3_new
```

---

## 6. OOD Split 实验规划

OOD 后续建议也按同样 7 个模型跑，方便和 Random 表对齐。

### OOD 数据

```text
train: data\GenVideo-Val\splits\ood_train.csv
val:   data\GenVideo-Val\splits\ood_val.csv
test:  data\GenVideo-Val\splits\ood_test.csv
```

### OOD 实验清单

| 编号 | 模型 | 输出目录 | 预计 5 epoch 时长 | 预计 10 epoch 时长 |
|---|---|---|---:|---:|
| O1 | `frequency_wave` | `runs/ood_frequency_wave` | 0.8 - 1.3 h | 1.5 - 2.5 h |
| O2 | `spatial_dino` | `runs/ood_spatial_dino` | 2.0 - 2.7 h | 4.0 - 5.0 h |
| O3 | `temporal_d3_restrav` | `runs/ood_temporal_d3_restrav` | 2.0 - 2.8 h | 4.0 - 5.0 h |
| O4 | `spatial_frequency_new` | `runs/ood_spatial_frequency_new` | 2.3 - 3.0 h | 4.5 - 5.5 h |
| O5 | `spatial_temporal_new` | `runs/ood_spatial_temporal_new` | 2.5 - 3.2 h | 5.0 - 6.0 h |
| O6 | `stf3_new_concat` | `runs/ood_stf3_new_concat` | 2.5 - 3.3 h | 5.0 - 6.0 h |
| O7 | `stf3_new` | `runs/ood_stf3_new` | 2.8 - 3.6 h | 5.0 - 6.5 h |

如果组员时间有限，OOD 最低要求先跑：

```text
O2: ood_spatial_dino
O7: ood_stf3_new
```

---

## 7. OOD 单个实验运行模板

把 `<MODEL>` 替换为对应模型名。

```powershell
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --model <MODEL> `
  --train-csv data\GenVideo-Val\splits\ood_train.csv `
  --val-csv data\GenVideo-Val\splits\ood_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 112 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\ood_<MODEL>

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\ood_<MODEL>\best.pt `
  --csv data\GenVideo-Val\splits\ood_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\ood_<MODEL>
```

---

## 8. 推荐组内分工

### 方案 A：3 人分工，先跑 Random

| 人员 | 负责实验 |
|---|---|
| A | R1 `frequency_wave`、R2 `spatial_dino` |
| B | R3 `temporal_d3_restrav`、R4 `spatial_frequency_new` |
| C | R5 `spatial_temporal_new`、R6 `stf3_new_concat`、R7 `stf3_new` |

### 方案 B：4 人分工，Random + OOD 最低要求

| 人员 | 负责实验 |
|---|---|
| A | R1、R2 |
| B | R3、R4 |
| C | R5、R6 |
| D | R7、O2、O7 |

### 方案 C：多人完整 OOD

如果要完整跑 14 个实验，直接把 R1-R7 / O1-O7 分配给不同机器。每个人运行完后交付：

```text
runs/<split>_<model>/history.json
runs/<split>_<model>/best.pt
outputs/<split>_<model>/metrics.json
outputs/<split>_<model>/predictions.csv
```

---

## 9. 每个实验完成后检查什么？

每个实验至少检查这 3 个文件：

```text
runs/<split>_<model>/history.json
outputs/<split>_<model>/metrics.json
outputs/<split>_<model>/predictions.csv
```

重点记录：

| 指标 | 来源文件 |
|---|---|
| train loss / val loss | `history.json` |
| val ACC / AUC / F1 | `history.json` |
| test ACC / AUC / F1 / Precision / Recall | `metrics.json` |
| confusion matrix | `metrics.json` |
| FP / FN 样本 | `predictions.csv` |
| branch weight | `predictions.csv`，仅融合模型有 |

---

## 10. 阈值调优建议

默认 `evaluate.py` 使用 `fake_prob >= 0.5` 判断 fake。

正式表格建议先报告默认阈值 0.5 的结果。  
额外可以在验证集上找最佳阈值，再在 test 上重算 F1，作为补充表。

不要直接用 test 集调阈值，否则论文中不严谨。

---

## 11. 结果表建议

### Random 消融表

| Method | Spatial | Temporal | Frequency | Fusion | ACC | AUC | F1 | Precision | Recall |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `frequency_wave` |  |  | ✓ | - |  |  |  |  |  |
| `spatial_dino` | ✓ |  |  | - |  |  |  |  |  |
| `temporal_d3_restrav` |  | ✓ |  | - |  |  |  |  |  |
| `spatial_frequency_new` | ✓ |  | ✓ | Transformer |  |  |  |  |  |
| `spatial_temporal_new` | ✓ | ✓ |  | Transformer |  |  |  |  |  |
| `stf3_new_concat` | ✓ | ✓ | ✓ | Gated Concat |  |  |  |  |  |
| `stf3_new` | ✓ | ✓ | ✓ | Branch-token Transformer |  |  |  |  |  |

### OOD 泛化表

同 Random 表结构一致。若时间不够，先放：

```text
spatial_dino
stf3_new
```

但完整论文更建议放 7 个模型。

---

## 12. 后续可选优化参数

第一轮不要乱改参数。等完成完整 Random 后，再考虑以下优化：

| 优化项 | 建议 | 风险 |
|---|---|---|
| `epochs=10` | 只给关键模型补跑 | 时间翻倍 |
| `image-size=224` | 只给 `stf3_new` 精跑 | 时间和显存显著增加 |
| `foundation-backbone=dinov2_vitb14` | 后期尝试更大 DINOv2 | 可能显存不够，速度慢 |
| `branch-dropout=0.0/0.2` | 可做融合正则敏感性 | 增加实验数量 |
| `aux-loss-weight=0.1/0.3` | 可做辅助损失敏感性 | 增加实验数量 |
| `d3-loss=cos` | 可比较 D3 动态距离形式 | 增加实验数量 |

建议优先级：

```text
完整 Random 5 epoch
→ OOD 最低要求 5 epoch
→ OOD 完整 5 epoch
→ stf3_new 10 epoch
→ stf3_new image_size=224 精跑
```

---

## 13. 最终建议

当前最合理的执行路线：

```text
第一步：Random 7 个实验，epochs=5
第二步：整理 Random 消融表
第三步：OOD 至少跑 spatial_dino 和 stf3_new，epochs=5
第四步：如果时间允许，补 OOD 其余 5 个消融
第五步：只对 stf3_new / stf3_new_concat / 最强双分支补 10 epoch 或 224 分辨率
```

这样既能满足老师提出的 `spatial_frequency / spatial_temporal` 组合实验，又不会一开始把时间全部耗在 10 epoch 上。
