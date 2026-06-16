# OOD 后续优化研究过程与实验结果汇总

> 本文档用于汇总 STF3 后续优化实验。  
> 主模型关注对象：R7 `stf3_new` 及其后续变体。  
> 数据划分：`data/GenVideo-Val/splits/ood_train.csv`、`ood_val.csv`、`ood_test.csv`。
> 最后更新：2026-06-13。当前状态：`R5_224_FAKEW12` 已完成并写入最终公平消融表；在相同 224 分辨率、fake loss weight 1.2 与验证集阈值校准协议下，`R7_224_FAKEW12` 仍是当前最佳 STF3 单模型。

## 研究故事总览：从“模型能否检测”到“模型为什么能够泛化”

本项目的优化过程并不是围绕单一最高分进行无方向调参，而是沿着一个逐步收敛的问题链展开：

```text
Random split 接近饱和
    ↓
转向更困难的跨生成器 OOD split
    ↓
发现模型 Precision 很高但 Recall 偏低，主要错误为 FN
    ↓
先判断问题来自决策阈值，还是来自模型表示能力
    ↓
通过验证集阈值校准确认：阈值能够改善结果，但不能完全解决漏检
    ↓
通过 generator 级错误分析确认：绝大多数剩余 FN 集中在 WildScrape
    ↓
围绕“保留更多伪造痕迹”和“提高 fake 类敏感性”设计 224 + fake loss weight
    ↓
得到当前最强 STF3 单模型 R7_224_FAKEW12
    ↓
继续验证推理融合和 multi-clip，区分有效改进与无效直觉
    ↓
发现简单时间裁剪会破坏完整视频证据，multi-clip 失败
    ↓
进一步保持 STF3 独立性，验证有效 WaveRep 增强和完整视频 16 帧采样
    ↓
两个实验均未超过 R7_224_FAKEW12，说明当前最佳单模型仍应保留
```

因此，本文档不仅记录“跑了哪些实验”，也记录每个实验背后的问题、假设、证据和下一步决策。整个优化过程遵循以下原则：

1. **先诊断，再重训**：优先使用阈值校准和错误分析判断瓶颈，避免无目标增加 epoch。
2. **一次只改变少量变量**：分别验证 epoch、分辨率、fake loss、推理采样和后处理的作用。
3. **关注 OOD，而不是只看 Random split**：Random split 适合验证模型能否学习，但不足以证明跨生成器泛化。
4. **同时观察排序能力和决策结果**：AUC 高不等于默认阈值下 Recall 高；必须结合 ACC、F1、Precision、Recall、FN 和 FP。
5. **保留 STF3 的方法独立性**：外部模型或跨架构融合只作为分析参考；论文主结果应来自 STF3 本体或明确标注的 STF3 后处理。
6. **失败实验同样提供证据**：multi-clip 的失败说明当前模型依赖完整视频范围内的稳定时空证据，而不是任意局部片段。

本研究围绕三个核心问题展开：

| 核心问题 | 对应证据 |
|---|---|
| 三分支 STF3 是否比单分支或双分支更加均衡？ | R1-R7、R5/R7、后续 R5_224/R7_224 公平对照 |
| STF3 的 OOD 瓶颈来自阈值、训练目标还是输入证据不足？ | 阈值校准、fake loss、224、epoch 与 generator 错误分析 |
| 如何在不依赖外部模型的情况下继续提升 STF3？ | 有效 WaveRep donor-bank 与完整视频 16 帧采样，结果均未超过基准 |

当前可以形成的方法贡献不是简单堆叠已有模块，而是：

```text
以 Spatial / Temporal / Frequency 三类互补证据构建 STF3，
通过 branch-token Transformer 进行样本级融合，
并利用 OOD 驱动的错误分析，对输入分辨率、类别损失与训练增强进行有目标的优化。
```

## 一、研究起点：为什么必须从 Random split 转向 OOD split

STF3 的目标不是只识别训练中已经出现过的生成器，而是判断模型能否利用空间、时序和频域伪造痕迹，对未见生成器保持稳定检测能力。

Random split 中，同一生成器家族可能同时存在于 train、val 和 test。该设置可以验证训练流程和模型学习能力，但当指标接近饱和时，不同分支和融合方式之间的差异会被压缩。因此，Random split 上 R5 与 R7 接近，并不能直接说明频域分支无效，也不能充分证明三分支融合具有更强泛化性。

OOD split 将不同生成器分配到训练、验证和测试阶段，更接近真实部署情景。它使实验问题从：

```text
模型能否区分已知分布中的 real/fake？
```

转变为：

```text
模型能否识别来自未知生成器、未知压缩方式和未知内容分布的 fake 视频？
```

这也是后续所有优化都以 OOD test 为主要判断依据的原因。

## 二、从原始 R1-R7 得出的第一个诊断

原始 OOD 实验并没有呈现“某个模型全面碾压”的简单结果，而是揭示了不同模型能力之间的差异：

- R5 `spatial_temporal_new` 的 AUC 最高，说明它对 real/fake 的概率排序能力很强。
- R7 `stf3_new` 的 ACC、F1 和 Recall 最好，说明三分支 Transformer 融合在默认决策阈值下更加均衡。
- 多数模型 Precision 接近或超过 0.99，但 Recall 明显更低，说明模型通常不会把 real 错判为 fake，却容易把 fake 错判为 real。

由此形成了第一个核心判断：

> 当前主要瓶颈不是误报过多，而是模型决策过于保守，导致大量 fake 视频漏检。

这意味着后续优化不能只追求更高 Precision，也不能只看 AUC。实验重点应转向：

- 能否减少 FN；
- Recall 提升是否以不可接受的 FP 为代价；
- 漏检是否集中在特定生成器；
- 排序能力较强但阈值偏移的模型，能否通过验证集校准释放潜力。

## 三、为什么第一步是阈值校准，而不是立即重新训练

R5 的原始 AUC 为 0.9795，但默认阈值 0.5 下 Recall 只有 0.8108。高 AUC 与低 Recall 同时出现，通常说明模型能够较好排序样本，但概率刻度或默认决策边界不适合当前 OOD 分布。

因此，最先进行验证集阈值校准有三个原因：

1. **成本低**：只需要保存 val/test predictions，不需要重新训练。
2. **能够区分问题来源**：如果调整阈值即可大幅改善 Recall，则主要问题是决策边界；如果改善有限，则需要修改训练或表示。
3. **避免测试集泄漏**：所有阈值只在 `ood_val.csv` 上选择，再固定到 `ood_test.csv`。

本项目最终主要采用 `precision_0.95_recall_max`，原因不是它在所有数据上必然最优，而是它与任务目标一致：

```text
在 Precision 至少约为 0.95 的约束下，尽可能提高 Recall、减少 fake 漏检。
```

阈值实验表明，校准确实能够减少 FN，但无法彻底解决 WildScrape 漏检。这说明模型训练和输入表示仍有改进空间。

## 四、为什么进行 generator 级错误分析

总体 ACC 或 F1 只能说明模型整体表现，不能回答“模型究竟在哪里失败”。为了让后续优化具有方向性，本项目将 OOD test 按 generator 拆分，重点统计 Recall 和 FN。

分析后发现：

- MorphStudio、Show_1 和 Sora 已经能够取得很高 Recall；
- WildScrape 的表现显著更低；
- 在当前最佳单模型中，WildScrape 产生 141 个 FN，占总 FN 153 个中的约 92%。

这改变了后续实验的目标。模型已经不需要平均地提升所有 generator，而需要提升对困难、真实网络来源、可能包含压缩和复杂内容变化的视频的鲁棒性。

因此，后续每项改进都应回答两个问题：

1. 总体 ACC/F1/Recall 是否提高？
2. WildScrape FN 是否下降，且 real_MSRVTT FP 是否仍可接受？

## 五、从诊断到训练优化：为什么选择 epoch、fake loss weight 和 224

### 5.1 为什么补跑更多 epoch

补跑 R5/R7 的 6 epoch，最初用于检验一个简单假设：

> 原始训练轮数是否不足，模型是否仍处于欠拟合状态？

结果显示，仅增加 epoch 并没有稳定改善 OOD 结果。R7_224_FAKEW12 的最佳 checkpoint 甚至来自 epoch 2，而后续训练集指标继续提高、验证集 AUC 却没有继续提高。

这说明当前瓶颈不是单纯“训练时间不够”。继续盲目增加 epoch 可能只会让模型更适应训练生成器，而不会改善未知生成器泛化。

### 5.2 为什么提高 fake 类损失权重

原始 OOD 结果的主要错误是 FN，而非 FP。基于这一错误结构，训练时将 fake 类损失权重从 1.0 轻微提高到 1.2，目的是让模型对漏检 fake 样本承担更高代价：

```text
普通交叉熵：real 与 fake 错误代价相同
fake_loss_weight=1.2：fake 被判为 real 时，训练损失更高
```

这里没有一开始使用过大的权重，因为过度强调 fake 类可能导致大量 real 视频被误报为 fake。`1.2` 是一种温和、可解释的调整；`1.5` 用于观察更强权重是否继续有效。

结果证明，适度 fake 权重确实有价值，但其效果不是孤立的。它与 224 分辨率结合后，才获得当前最明显的 OOD 提升。

### 5.3 为什么从 112 提高到 224

提高输入分辨率的思路来自 STF3 三个分支共同依赖的伪造证据：

- 空间分支需要观察局部纹理、边缘和生成痕迹；
- 频域分支需要保留压缩、重采样和高频伪影；
- 时序分支依赖逐帧 DINOv2 表征，较高分辨率可能提供更稳定的轨迹特征。

112 输入能够降低计算成本，但下采样可能抹去细粒度证据。224 与 DINOv2 ViT-S/14 的 patch 结构兼容，并在计算成本和细节保留之间形成可接受折中。

单独运行 R7_224 并未全面改善结果，说明“更多像素”本身不是充分条件。其重要价值在于与 fake loss weight 结合后，使模型既看到更充分的伪造细节，又在训练目标中更加重视这些 fake 证据。

### 5.4 为什么组合 224 与 fake loss weight

该组合不是无依据的参数叠加，而是两个互补假设的联合验证：

```text
224：提高可观测证据质量
fake_loss_weight=1.2：提高模型利用 fake 证据的动力
```

如果只有 224，但模型仍保持保守决策，更多细节不一定转化为更高 Recall；如果只有 fake loss weight，但输入中细粒度伪造信息不足，模型也可能无法真正学到更可靠的 fake 特征。

R7_224_FAKEW12 的结果支持这种互补关系。相比 R7_224，验证集阈值校准后：

- ACC 从 0.9311 提升至 0.9430；
- F1 从 0.9383 提升至 0.9499；
- Recall 从 0.8990 提升至 0.9271；
- FN 从 212 降至 153；
- WildScrape FN 从 181 降至 141。

因此，当前最重要的训练优化结论不是“224 一定更好”或“提高 fake loss 一定更好”，而是：

> 对当前 STF3，细节保留与召回导向训练目标具有组合收益。

## 六、为什么探索概率融合与分支级校准

在得到较强单模型后，本项目进一步研究：不同 STF3 变体或内部三个分支是否具有互补性。

### 6.1 R5 + R7 融合的作用与限制

R5 具有较高 AUC，R7 具有更均衡的默认阈值表现，因此最初尝试二者概率融合。这一实验用于验证空间-时序模型的排序能力能否补充三分支模型。

融合后结果较强，但最优权重主要依赖 R5。这会削弱论文中“STF3 三分支主模型”的独立叙事。因此，R5 + R7 更适合作为补充分析，而不作为最终主方法。

### 6.2 为什么改为 STF3-only ensemble

为了保留 STF3 独立性，后续将融合限制为两个 R7/STF3 checkpoint：

```text
R7_224_FAKEW12 + R7_224
```

该方法取得最高 Recall 0.9295 和最低 FN 148，说明不同 STF3 训练目标之间确实存在互补性。但它需要运行两个 checkpoint，因此属于推理增强方案，而不是单个 STF3 模型。

### 6.3 为什么进行 branch-level Logistic Regression 校准

STF3 在评估时能够输出 spatial、temporal、frequency 三个分支的独立概率。分支级 LR 校准并未引入外部检测器，而是使用验证集学习不同分支在 OOD 决策中的组合方式。

该方法取得最高 ACC 0.9466、F1 0.9530，并将 real FP 从 52 降至 41。它证明三个内部证据具有互补性，也支持 branch-token STF3 的设计动机。

但该方法仍是验证集拟合的后处理器，因此论文中必须明确区分：

- `R7_224_FAKEW12`：最佳 STF3 单模型；
- `STF3 + branch-level calibration`：最佳综合后处理系统。

## 七、为什么尝试 multi-clip，以及为什么失败仍然重要

在发现 WildScrape 漏检集中后，一个自然假设是：单次均匀采样可能错过局部伪造片段，因此可以从视频多个时间区域取 clip，再聚合预测。

本项目尝试了：

```text
5 个 temporal clips
每个 clip 覆盖约 60% 视频
mean / top-2 mean 概率聚合
```

结果并未改善性能，反而显著增加 WildScrape FN。进一步观察 AUC/AP 后发现，multi-clip 不只是阈值不合适，而是排序能力本身下降。

这一失败带来了重要认识：

> 当前 STF3 依赖完整视频范围内稳定的空间、时序和频域联合证据。将视频裁剪成局部片段，可能破坏 D3/ReStraV 轨迹与全局频率统计。

因此，后续不再继续增加 clip 数量，也不再沿用 60% temporal crop。随后改为验证完整视频范围内均匀采样更多帧，即 16 帧实验。

## 八、实现复核带来的新发现：此前 WaveRep 增强实际未生效

在设计下一阶段实验时，对训练代码进行了重新审查。现有 WaveRep 增强原本依赖同一个 batch 中的其他视频作为频带 donor，并包含以下保护条件：

```text
batch size < 2 时直接返回原始输入
```

而本文 follow-up notebook 中的主要 R7 实验均使用 `batch_size=1`。因此，虽然训练配置中写有：

```text
wavelet_aug_prob = 0.1
```

此前实验实际上没有执行 WaveRep 频带替换。这个发现要求对已有结果作出准确解释：

- R7_224_FAKEW12 的频域分支仍然使用 Wavelet/FFT 统计；
- 但不能把它的提升归因于 WaveRep 训练增强；
- 真正有效的 WaveRep 增强需要单独验证；第 11 模块已经完成该验证，结果未超过基准。

为解决这一问题，新增 donor-bank 模式：保存上一条训练视频作为下一条视频的小波频带 donor，使 `batch_size=1` 时增强能够真正执行。该修改只发生在训练输入层：

```text
训练阶段：输入视频 + donor-bank WaveRep → 原有 STF3
推理阶段：原始输入视频 → 原有 STF3
```

它不增加模型分支、不修改融合结构、不增加推理参数，因此仍属于独立 STF3 单模型优化。

## 九、为什么选择 WaveRep-bank 与完整视频 16 帧，以及最终结果如何

在完成大量后处理和错误分析后，后续只保留两个能够增强 STF3 本体、且不会破坏模型独立性的实验。现在这两个实验已经完成。

### 9.1 `R7_224_FAKEW12_WAVEREP_BANK`

该实验保持当前最佳单模型的全部配置，仅让 WaveRep 增强真正生效：

```text
stf3_new
image_size = 224
num_frames = 8
fake_loss_weight = 1.2
wavelet_aug_mode = bank
wavelet_aug_prob = 0.1
```

研究假设是：通过交换不同训练视频的小波频带，减少模型对特定生成器频谱模式的依赖，促使频域分支学习更稳定的 forensic traces，从而改善 WildScrape 等困难 OOD 视频。

该实验风险在于：频带替换可能破坏当前 DINOv2 空间表征、D3/ReStraV 时序轨迹和频域统计之间已经形成的协同关系。最终结果证明该风险确实存在：WaveRep-bank 虽然降低了 FP，但显著增加 FN，未能改善 WildScrape。

### 9.2 `R7_224_F16_FAKEW12`

该实验将完整视频均匀采样帧数从 8 增加到 16，并关闭 WaveRep，以单独验证时序长度作用：

```text
stf3_new
image_size = 224
num_frames = 16
fake_loss_weight = 1.2
wavelet_aug_prob = 0.0
```

它与失败的 multi-clip 有本质区别：

- multi-clip 将视频裁剪成多个局部时间窗口；
- 16 帧实验仍覆盖完整视频，只提高轨迹采样密度。

更多帧原本可能为 D3 二阶动态和 ReStraV 轨迹几何提供更稳定的时间序列，也可能减少 8 帧采样遗漏关键异常的情况。最终结果显示，16 帧确实比 WaveRep-bank 更接近基准，但仍低于 8 帧 R7_224_FAKEW12，尤其没有改善 WildScrape。

### 9.3 两个实验完成后的判断

两个实验均未超过基准，因此不建议继续运行 `WaveRep-bank + 16F` 联合实验。原因是：

1. WaveRep-bank 单独实验已经降低总体 ACC/F1/Recall，并增加 WildScrape FN；
2. 16F 单独实验虽然优于 WaveRep-bank，但仍低于 8 帧基准；
3. 二者联合很可能同时继承频带扰动和长时序训练噪声，缺少继续投入长时间训练的证据。

这说明当前 STF3 的最强设置仍是 `R7_224_FAKEW12`，即 8 帧、224 分辨率、fake loss weight 1.2。

## 十、当前完整决策链

| 阶段 | 观察或问题 | 提出的假设 | 实验 | 结果与决策 |
|---|---|---|---|---|
| Random baseline | 指标接近饱和 | Random split 无法充分区分泛化能力 | 转向 OOD split | 后续以 OOD 为主 |
| OOD R1-R7 | Precision 高、Recall 低、FN 多 | 默认阈值可能过于保守 | val 阈值校准 | 有改善，但不能完全解决 |
| Generator 分析 | FN 高度集中在 WildScrape | 问题来自特定困难 OOD 分布 | generator-level Recall/FN | 后续重点观察 WildScrape |
| 增加 epoch | 可能训练不足 | 更多 epoch 可改善泛化 | R7_E6 | 无稳定收益，不继续盲目加 epoch |
| fake loss | fake 漏检代价不足 | 提高 fake 权重可减少 FN | R7_FAKEW12 | 有一定收益 |
| 224 输入 | 112 可能丢失细粒度痕迹 | 更高分辨率保留空间/频域证据 | R7_224 | 单独收益有限 |
| 224 + fake loss | 更多证据与召回导向目标互补 | 联合设置优于单独设置 | R7_224_FAKEW12 | 当前最佳 STF3 单模型 |
| 概率融合 | 不同模型可能互补 | R5/R7 概率可进一步提升 | R5 + R7 | 强，但削弱 STF3 独立叙事 |
| STF3 内部增强 | STF3 变体/分支可能互补 | 只用 STF3 证据进行后处理 | STF3-only ensemble / branch LR | 后处理提升，但不是单模型 |
| Multi-clip | 单次采样可能遗漏局部异常 | 多时间窗口可改善 Recall | 5-clip mean/top-2 | 失败，破坏完整视频证据 |
| 代码复核 | WaveRep 配置存在但 batch=1 | 增强可能实际未运行 | 检查实现 | 确认此前 WaveRep 未生效 |
| STF3 独立性优化 | 需要优化 STF3 本体 | 有效频域增强与更密时序可能改善 OOD | WaveRep-bank / 16F | 均未超过 R7_224_FAKEW12 |

---

以下章节保留完整实验数据、指标表和逐项结论，作为上述研究故事的证据档案。

## 1. 当前进度

已经完成的实验：

- 原始 OOD R1-R7 baseline 评估。
- 原始 OOD R1-R7 的验证集阈值校准分析。
- R5 + R7 概率融合实验。
- R7_E6：`stf3_new`，6 epoch，输入尺寸 112。
- R7_FAKEW12：`stf3_new`，6 epoch，输入尺寸 112，`fake_loss_weight=1.2`。
- R7_224：`stf3_new`，5 epoch，输入尺寸 224。
- R7_224_FAKEW12：`stf3_new`，5 epoch，输入尺寸 224，`fake_loss_weight=1.2`。
- R7_224_FAKEW12 的 `max_acc` 阈值搜索。
- R7_224_FAKEW12 + R7_224 的 STF3-only 概率融合。
- R7_224_FAKEW12 的分支级 Logistic Regression 后处理校准。
- R7_224_FAKEW12 的 5-clip mean 与 5-clip top-2 mean 推理实验。

已经完成代码实现并完成长时间训练的实验：

- R7_224_FAKEW12_WAVEREP_BANK：保持 STF3 结构不变，使 batch size 1 下 WaveRep 增强真正生效；结果未超过基准。
- R7_224_F16_FAKEW12：保持 STF3 结构不变，在完整视频范围均匀采样 16 帧；结果未超过基准。

当前实验边界：

- 两个第 11 模块实验均属于 STF3 单模型内部优化。
- 暂不加入外部 D3/ReStraV 独立分数或其他检测器融合，以保证 STF3 主模型独立性。
- 因两个实验单独运行后均低于基准，暂不继续组合 WaveRep-bank 与 16F。

## 2. 原始 OOD Baseline，默认阈值 0.5

| Experiment | ACC | AUC | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 frequency_wave | 0.8060 | 0.9354 | 0.8075 | 0.9581 | 0.6978 | 1436 | 64 | 634 | 1464 |
| R2 spatial_dino | 0.8858 | 0.9657 | 0.8923 | 0.9913 | 0.8112 | 1485 | 15 | 396 | 1702 |
| R3 temporal_d3_restrav | 0.8130 | 0.9550 | 0.8134 | 0.9722 | 0.6992 | 1458 | 42 | 631 | 1467 |
| R4 spatial_frequency_new | 0.8749 | 0.9636 | 0.8805 | 0.9940 | 0.7903 | 1490 | 10 | 440 | 1658 |
| R5 spatial_temporal_new | 0.8877 | 0.9795 | 0.8939 | 0.9959 | 0.8108 | 1493 | 7 | 397 | 1701 |
| R6 stf3_new_concat | 0.8819 | 0.9744 | 0.8879 | 0.9941 | 0.8022 | 1490 | 10 | 415 | 1683 |
| R7 stf3_new | 0.9161 | 0.9651 | 0.9235 | 0.9854 | 0.8689 | 1473 | 27 | 275 | 1823 |

阶段性结论：

- 在默认阈值 0.5 下，R7 是原始 OOD baseline 中最好的主模型结果。
- R5 在原始 baseline 中 AUC 最高，但默认阈值下 Recall 明显偏低。
- 当前 OOD 的主要错误类型是 FN，也就是 fake 视频被判成 real。

## 3. 原始 R7 的阈值校准结果

| ID | Method | Objective | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| R7 | stf3_new | max_f1 | 0.6001 | 0.9099 | 0.9171 | 0.9895 | 0.8546 | 305 | 19 |
| R7 | stf3_new | max_balanced_acc | 0.3960 | 0.9200 | 0.9276 | 0.9819 | 0.8789 | 254 | 34 |
| R7 | stf3_new | precision_0.95_recall_max | 0.3726 | 0.9202 | 0.9279 | 0.9804 | 0.8808 | 250 | 37 |

阶段性结论：

- 原始 R7 经过验证集阈值校准后有小幅提升。
- 对原始 R7 来说，最合适的校准目标是 `precision_0.95_recall_max`。
- 该设置将 FN 从 275 降到 250，同时 Precision 仍保持在 0.9804。

## 4. R5 + R7 概率融合

| Objective | Weight R5 | Weight R7 | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_f1 | 0.70 | 0.30 | 0.2827 | 0.9183 | 0.9255 | 0.9881 | 0.8704 | 272 | 22 |
| max_balanced_acc | 0.70 | 0.30 | 0.2827 | 0.9183 | 0.9255 | 0.9881 | 0.8704 | 272 | 22 |
| precision_0.95_recall_max | 0.85 | 0.15 | 0.0908 | 0.9336 | 0.9409 | 0.9779 | 0.9066 | 196 | 43 |

阶段性结论：

- R5 + R7 概率融合的后处理效果较强。
- 该方法不是主 STF3 架构本身，更适合作为增强推理方案或补充实验。
- 最好的融合设置主要依赖 R5 概率，同时加入少量 R7 概率。

## 5. R7 后续实验，默认阈值 0.5

| Experiment | ACC | AUC | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R7_E6 default | 0.8966 | 0.9678 | 0.9032 | 0.9943 | 0.8275 | 1490 | 10 | 362 | 1736 |
| R7_FAKEW12 default | 0.9163 | 0.9744 | 0.9235 | 0.9896 | 0.8656 | 1481 | 19 | 282 | 1816 |
| R7_224 default | 0.9044 | 0.9633 | 0.9110 | 0.9960 | 0.8394 | 1493 | 7 | 337 | 1761 |
| R7_224_FAKEW12 default | 0.9275 | 0.9848 | 0.9341 | 0.9930 | 0.8818 | 1487 | 13 | 248 | 1850 |

阶段性结论：

- R7_E6 和 R7_224 在默认阈值 0.5 下都偏保守，FN 偏多。
- 在这些后续实验里，R7_224_FAKEW12 默认阈值下 AUC、ACC、F1、Recall 都是最高，FN 也是最少。
- R7_224_FAKEW12 说明 `image_size=224` 和 `fake_loss_weight=1.2` 有组合收益，而不是两个单独设置简单互斥。
- 仅用默认阈值不足以公平比较这些模型，必须结合验证集阈值校准。

## 6. R7 后续实验，验证集校准阈值

阈值选择目标：`precision_0.95_recall_max`。

| ID | Method | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R7_E6 | stf3_new_e6 | 0.0173 | 0.9283 | 0.9357 | 0.9802 | 0.8951 | 220 | 38 |
| R7_FAKEW12 | stf3_new_fakew12 | 0.2457 | 0.9291 | 0.9364 | 0.9822 | 0.8947 | 221 | 34 |
| R7_224 | stf3_new_224 | 0.0044 | 0.9311 | 0.9383 | 0.9813 | 0.8990 | 212 | 36 |
| R7_224_FAKEW12 | stf3_new_224_fakew12 | 0.0545 | 0.9430 | 0.9499 | 0.9740 | 0.9271 | 153 | 52 |

阶段性结论：

- R7_224_FAKEW12 是目前 R7 系列中最好的单模型结果。
- 相比 R7_224，R7_224_FAKEW12 的 ACC 从 0.9311 提升到 0.9430，F1 从 0.9383 提升到 0.9499，Recall 从 0.8990 提升到 0.9271，FN 从 212 降到 153。
- 代价是 FP 从 36 增加到 52，Precision 从 0.9813 降到 0.9740，但仍高于 `precision_0.95_recall_max` 设定的约束目标。
- 如果论文主目标是减少 fake 漏检，当前最合理的单模型主结果是 R7_224_FAKEW12 + 验证集校准阈值。

当前最佳 R7 单模型候选：

```text
R7_224_FAKEW12 + validation-calibrated threshold
objective = precision_0.95_recall_max
threshold = 0.054459
ACC       = 0.9430
F1        = 0.9499
Precision = 0.9740
Recall    = 0.9271
FN        = 153
FP        = 52
```

## 7. Generator 级错误分析

阈值选择目标：`precision_0.95_recall_max`。

| ID | Method | Threshold | Generator | N | ACC | Recall | FN | FP | TN | TP |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| R7_E6 | stf3_new_e6 | 0.0173 | MorphStudio | 700 | 0.9857 | 0.9857 | 10 | 0 | 0 | 690 |
| R7_E6 | stf3_new_e6 | 0.0173 | Show_1 | 700 | 0.9629 | 0.9629 | 26 | 0 | 0 | 674 |
| R7_E6 | stf3_new_e6 | 0.0173 | Sora | 56 | 0.8571 | 0.8571 | 8 | 0 | 0 | 48 |
| R7_E6 | stf3_new_e6 | 0.0173 | WildScrape | 642 | 0.7259 | 0.7259 | 176 | 0 | 0 | 466 |
| R7_E6 | stf3_new_e6 | 0.0173 | real_MSRVTT | 1500 | 0.9747 |  | 0 | 38 | 1462 | 0 |
| R7_FAKEW12 | stf3_new_fakew12 | 0.2457 | MorphStudio | 700 | 0.9714 | 0.9714 | 20 | 0 | 0 | 680 |
| R7_FAKEW12 | stf3_new_fakew12 | 0.2457 | Show_1 | 700 | 0.9671 | 0.9671 | 23 | 0 | 0 | 677 |
| R7_FAKEW12 | stf3_new_fakew12 | 0.2457 | Sora | 56 | 0.8214 | 0.8214 | 10 | 0 | 0 | 46 |
| R7_FAKEW12 | stf3_new_fakew12 | 0.2457 | WildScrape | 642 | 0.7383 | 0.7383 | 168 | 0 | 0 | 474 |
| R7_FAKEW12 | stf3_new_fakew12 | 0.2457 | real_MSRVTT | 1500 | 0.9773 |  | 0 | 34 | 1466 | 0 |
| R7_224 | stf3_new_224 | 0.0044 | MorphStudio | 700 | 0.9886 | 0.9886 | 8 | 0 | 0 | 692 |
| R7_224 | stf3_new_224 | 0.0044 | Show_1 | 700 | 0.9729 | 0.9729 | 19 | 0 | 0 | 681 |
| R7_224 | stf3_new_224 | 0.0044 | Sora | 56 | 0.9286 | 0.9286 | 4 | 0 | 0 | 52 |
| R7_224 | stf3_new_224 | 0.0044 | WildScrape | 642 | 0.7181 | 0.7181 | 181 | 0 | 0 | 461 |
| R7_224 | stf3_new_224 | 0.0044 | real_MSRVTT | 1500 | 0.9760 |  | 0 | 36 | 1464 | 0 |
| R7_224_FAKEW12 | stf3_new_224_fakew12 | 0.0545 | MorphStudio | 700 | 0.9914 | 0.9914 | 6 | 0 | 0 | 694 |
| R7_224_FAKEW12 | stf3_new_224_fakew12 | 0.0545 | Show_1 | 700 | 0.9943 | 0.9943 | 4 | 0 | 0 | 696 |
| R7_224_FAKEW12 | stf3_new_224_fakew12 | 0.0545 | Sora | 56 | 0.9643 | 0.9643 | 2 | 0 | 0 | 54 |
| R7_224_FAKEW12 | stf3_new_224_fakew12 | 0.0545 | WildScrape | 642 | 0.7804 | 0.7804 | 141 | 0 | 0 | 501 |
| R7_224_FAKEW12 | stf3_new_224_fakew12 | 0.0545 | real_MSRVTT | 1500 | 0.9653 |  | 0 | 52 | 1448 | 0 |

Generator 级结论：

- R7_224_FAKEW12 同时改善了 MorphStudio、Show_1、Sora 和 WildScrape。
- 相比 R7_224，R7_224_FAKEW12 的 WildScrape FN 从 181 降到 141，这是本轮 8.7 最重要的变化。
- 相比 R7_FAKEW12，R7_224_FAKEW12 的 WildScrape FN 从 168 降到 141，说明 224 分辨率和 fake loss weight 的组合确实带来了额外收益。
- 但 WildScrape 仍然贡献了 141 个 FN，占 R7_224_FAKEW12 总 FN 153 个中的约 92%，仍是主要失败来源。
- real_MSRVTT 的 FP 从 R7_224 的 36 增加到 52，这是 Recall 提升的主要代价。

## 8. 当前解释

当前结果支持以下判断：

- Random split 很可能过于容易，不足以充分区分 STF3 设计细节。
- OOD split 更能反映模型泛化能力。
- R7 是本文主线中的 STF3 架构。
- 验证集阈值校准是必要步骤，因为多个后续模型在默认阈值 0.5 下偏保守。
- R7_224_FAKEW12 是目前已经完成的最强 R7 单模型变体。
- R7_224_FAKEW12 的提升不是只来自调低阈值；它在默认阈值下也已经优于原始 R7 和其他 R7 后续变体。
- WildScrape 仍然是最主要、尚未解决的 OOD 失败来源，但 8.7 已经明显降低了 WildScrape 漏检。

建议论文表述：

```text
最终的 branch-token Transformer STF3 模型在 OOD 评估中经过验证集阈值校准后取得最佳单模型表现。将输入分辨率提高到 224 并加入适度的 fake 类损失权重后，模型在 OOD test 上进一步提高 ACC、F1 和 Recall，并显著减少 FN。Generator 级分析显示，改进主要来自 Show_1、Sora 和 WildScrape 的漏检减少；其中 WildScrape 仍然是剩余错误的主要来源，说明当前 OOD 泛化瓶颈集中在特定困难生成器上。
```

## 9. R7_224_FAKEW12 训练情况

实验配置：

```text
model = stf3_new
epochs = 5
image_size = 224
fake_loss_weight = 1.2
pos_weight = none
```

训练观察：

- 最佳 checkpoint 来自 epoch 2，因为训练脚本按验证集 AUC 保存 best.pt。
- epoch 2 的验证集指标为 ACC 0.9897、AUC 0.9993、F1 0.9866、Precision 0.9839、Recall 0.9892。
- epoch 3 到 epoch 5 的训练集继续变好，但验证集 AUC 没有继续提高，说明继续训练不一定带来 OOD 泛化收益。
- 目前没有必要立刻把 epoch 数继续加大；如果要继续做训练实验，更应该先围绕 WildScrape 或 loss/采样策略做有目标的设计。

## 10. 第 10 模块后处理与推理优化结果

以下结果均使用验证集选择参数或阈值，再固定到 OOD test。为了突出 fake 漏检问题，主要比较 `precision_0.95_recall_max`。

| Method | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 baseline | 0.0545 | 0.9430 | 0.9499 | 0.9740 | 0.9271 | 153 | 52 |
| R7_224_FAKEW12 + R7_224 STF3-only ensemble | 0.0313 | 0.9439 | 0.9508 | 0.9731 | **0.9295** | **148** | 54 |
| R7_224_FAKEW12 + branch LR calibration | 0.0993 | **0.9466** | **0.9530** | **0.9794** | 0.9280 | 151 | **41** |
| R7_224_FAKEW12 5-clip mean | 0.4226 | 0.9258 | 0.9337 | 0.9741 | 0.8966 | 217 | 50 |
| R7_224_FAKEW12 5-clip top-2 mean | 0.5620 | 0.9227 | 0.9313 | 0.9676 | 0.8975 | 215 | 63 |

结论：

- 综合最佳后处理方案是 branch LR calibration，取得最高 ACC、F1 和 Precision。
- STF3-only ensemble 取得最高 Recall 和最低 FN，适合作为 Recall 导向增强方法。
- 两种 multi-clip 推理均明显低于单次 uniform sampling，不应作为当前最终方案。
- Branch LR 与 STF3-only ensemble 都只使用 STF3/R7 变体或内部证据，不依赖 R5，因此不会削弱 STF3 主模型叙事。

## 11. `max_acc` 阈值实验

R7_224_FAKEW12 的验证集 `max_f1`、`max_balanced_acc` 和 `max_acc` 都选择了相同阈值 `0.2507`。这是正常现象，说明该阈值在当前 validation predictions 上同时使多个指标达到最优。

| Objective | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| max_f1 | 0.2507 | 0.9372 | 0.9438 | 0.9860 | 0.9051 | 199 | 27 |
| max_balanced_acc | 0.2507 | 0.9372 | 0.9438 | 0.9860 | 0.9051 | 199 | 27 |
| max_acc | 0.2507 | 0.9372 | 0.9438 | 0.9860 | 0.9051 | 199 | 27 |
| precision_0.95_recall_max | 0.0545 | **0.9430** | **0.9499** | 0.9740 | **0.9271** | **153** | 52 |

解释：

- `max_acc` 在 validation 上选出的阈值到 OOD test 后较为保守，虽然减少 FP，但明显增加 FN。
- 对当前 fake 检测任务，`precision_0.95_recall_max` 在 test 上取得更好的 ACC、F1 和 Recall，因此仍然是更合适的主要选择目标。

## 12. Generator 级关键方法对比

对 fake generator，表中的正确率等于 Recall；对 `real_MSRVTT`，正确率等于 Specificity。

| Method | MorphStudio | Show_1 | Sora | WildScrape | real_MSRVTT |
|---|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 baseline | 0.9914 | 0.9943 | 0.9643 | 0.7804 | 0.9653 |
| Branch LR calibration | 0.9871 | 0.9900 | 0.9107 | **0.7975** | **0.9727** |
| STF3-only ensemble | **0.9929** | **0.9943** | **0.9643** | 0.7866 | 0.9640 |
| 5-clip mean | 0.9829 | 0.9729 | 0.9286 | 0.7165 | 0.9667 |
| 5-clip top-2 mean | 0.9814 | 0.9714 | 0.9286 | 0.7227 | 0.9580 |

对应 FN/FP：

| Method | MorphStudio FN | Show_1 FN | Sora FN | WildScrape FN | real_MSRVTT FP |
|---|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 baseline | 6 | 4 | 2 | 141 | 52 |
| Branch LR calibration | 9 | 7 | 5 | **130** | **41** |
| STF3-only ensemble | **5** | **4** | **2** | 137 | 54 |
| 5-clip mean | 12 | 19 | 4 | 182 | 50 |
| 5-clip top-2 mean | 13 | 20 | 4 | 178 | 63 |

Generator 级解释：

- Branch LR 的总体提升主要来自 WildScrape FN 从 141 降至 130，以及 real FP 从 52 降至 41。
- Branch LR 对 MorphStudio、Show_1 和 Sora 有小幅退化，因此它不是对所有 generator 的全面提升，而是改善了最困难 generator 与 real 类之间的决策平衡。
- STF3-only ensemble 对 MorphStudio、Show_1 和 Sora 保持较好表现，同时将 WildScrape FN 降至 137，但 real FP 增至 54。
- multi-clip 的主要失败来源是 WildScrape：5-clip mean 和 top-2 mean 分别产生 182 和 178 个 WildScrape FN。

## 13. Generator 级 AUC / AP

以下 AUC/AP 使用每个 fake generator 与全部 `real_MSRVTT` 样本组成二分类子集计算，与近期视频生成检测论文的 generator-level 报告方式接近。

| Method | Generator | AUC | AP |
|---|---|---:|---:|
| Baseline | MorphStudio | 0.9991 | 0.9982 |
| Baseline | Show_1 | 0.9980 | 0.9966 |
| Baseline | Sora | 0.9938 | 0.9209 |
| Baseline | WildScrape | 0.9540 | 0.9331 |
| Branch LR | MorphStudio | 0.9984 | 0.9971 |
| Branch LR | Show_1 | 0.9985 | 0.9974 |
| Branch LR | Sora | 0.9787 | 0.8861 |
| Branch LR | WildScrape | **0.9700** | **0.9544** |
| STF3-only ensemble | MorphStudio | **0.9993** | **0.9985** |
| STF3-only ensemble | Show_1 | **0.9987** | **0.9973** |
| STF3-only ensemble | Sora | **0.9956** | **0.9318** |
| STF3-only ensemble | WildScrape | 0.9575 | 0.9343 |
| 5-clip mean | WildScrape | 0.9388 | 0.9059 |
| 5-clip top-2 mean | WildScrape | 0.9352 | 0.8945 |

解释：

- Branch LR 明显改善 WildScrape 的排序能力，但损害 Sora 排序能力。
- STF3-only ensemble 在 MorphStudio、Show_1 和 Sora 上取得最强或接近最强的 AUC/AP，同时对 WildScrape 有小幅提升。
- multi-clip 不只是阈值选择不佳，它同时降低了 WildScrape AUC/AP，说明当前时间裁剪与聚合策略损害了 OOD 排序能力。

## 14. Multi-clip 失败分析

当前 multi-clip 设置：

```text
clip_count = 5
clip_crop_ratio = 0.6
aggregate = mean / top2_mean
```

该方案改变了模型训练与原始评估时使用的完整视频 uniform sampling 分布。虽然多个 clip 能覆盖更多时间位置，但裁剪后的局部 clip 可能缺少稳定的全局时序与频率证据，最终导致 OOD test，尤其 WildScrape，出现明显退化。

因此当前结论不是“multi-clip 一定无效”，而是：

```text
当前 60% temporal crop + probability mean/top-2 mean 的 multi-clip 方案不适合该 STF3 checkpoint。
```

不建议继续直接增加 clip 数量。如果未来重新研究 multi-clip，更合理的方向是保留完整视频 uniform clip，并只改变采样偏移或在特征级融合。

## 15. 当前最佳结果与论文定位

### 最佳单模型

```text
R7_224_FAKEW12 + validation-calibrated threshold
ACC       = 0.9430
F1        = 0.9499
Precision = 0.9740
Recall    = 0.9271
FN        = 153
FP        = 52
```

### 最佳综合后处理结果

```text
R7_224_FAKEW12 + branch-level Logistic Regression calibration
ACC       = 0.9466
F1        = 0.9530
Precision = 0.9794
Recall    = 0.9280
FN        = 151
FP        = 41
```

### 最佳 Recall 后处理结果

```text
R7_224_FAKEW12 + R7_224 STF3-only ensemble
ACC       = 0.9439
F1        = 0.9508
Precision = 0.9731
Recall    = 0.9295
FN        = 148
FP        = 54
```

建议论文表述：

```text
优化后的三分支 STF3 单模型在 OOD test 上取得 0.9430 ACC 和 0.9499 F1。进一步利用验证集进行 STF3 分支级轻量校准后，ACC 和 F1 分别提升至 0.9466 和 0.9530，同时将 real 视频误报数从 52 降至 41。STF3 变体间概率融合则取得最高 Recall 0.9295 和最低 FN 148，说明不同 STF3 优化变体具有一定互补性。简单 multi-clip 概率聚合未带来提升，并在 WildScrape 上出现明显退化。
```

需要保留的限制：

- Branch LR 是验证集拟合的后处理校准器，应称为 `STF3 + branch-level calibration`，不能描述成新的主模型架构。
- STF3-only ensemble 是两个 STF3/R7 checkpoint 的推理融合，增加了推理成本。
- WildScrape 仍然是主要失败来源，当前最佳方案中绝大多数 FN 仍来自 WildScrape。
- 在最终论文结论前，仍需完成 R5_224 与 R7_224 的公平对照实验，以验证三分支 STF3 相比两分支 spatial-temporal 模型的贡献。

## 16. 第 11 模块：保持 STF3 独立性的训练优化结果

第 11 模块的目标是继续优化 STF3 单模型本体，而不是依赖外部检测器、R5 融合或额外后处理模型。因此只测试两个结构不变实验：

```text
R7_224_FAKEW12_WAVEREP_BANK
R7_224_F16_FAKEW12
```

两个实验均以当前最佳 STF3 单模型 `R7_224_FAKEW12` 为基准。

### 16.1 实验配置与训练情况

| Experiment | 结构变化 | num_frames | image_size | fake_loss_weight | WaveRep | best epoch | 训练时间 |
|---|---|---:|---:|---:|---|---:|---:|
| R7_224_FAKEW12 | 无 | 8 | 224 | 1.2 | 实际未生效 | 2 | 275.4 min |
| R7_224_FAKEW12_WAVEREP_BANK | 无 | 8 | 224 | 1.2 | donor-bank, prob=0.1 | 5 | 250.9 min |
| R7_224_F16_FAKEW12 | 无 | 16 | 224 | 1.2 | 关闭 | 5 | 406.1 min |

训练观察：

- WaveRep-bank 的训练 AUC 上升较慢，说明 donor-bank 频带扰动确实增加了训练难度。
- WaveRep-bank 的最佳 checkpoint 来自 epoch 5，而原基准来自 epoch 2，说明增强改变了训练动态。
- 16 帧实验训练时间明显增加，从约 275 min 增至约 406 min，但最终 OOD test 并未换来更高泛化表现。
- 两个实验均未改变 STF3 三分支结构和 branch-token Transformer 融合方式，因此结论可以用于 STF3 单模型内部优化判断。

### 16.2 默认阈值 0.5 对比

| Experiment | ACC | AUC | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 | 0.9275 | 0.9848 | 0.9341 | 0.9930 | 0.8818 | 1487 | 13 | 248 | 1850 |
| R7_224_FAKEW12_WAVEREP_BANK | 0.9125 | 0.9733 | 0.9192 | 0.9950 | 0.8541 | 1491 | 9 | 306 | 1792 |
| R7_224_F16_FAKEW12 | 0.9086 | 0.9638 | 0.9155 | 0.9928 | 0.8494 | 1487 | 13 | 316 | 1782 |

默认阈值结论：

- 两个新实验在默认阈值下都比基准更保守，Recall 明显下降。
- WaveRep-bank 将 FP 从 13 降到 9，但 FN 从 248 增加到 306。
- 16 帧默认阈值下 FN 增至 316，说明单纯增加帧数并没有直接提高 fake 检出能力。

### 16.3 验证集校准阈值对比

阈值选择目标仍为 `precision_0.95_recall_max`。

| Experiment | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 | 0.0545 | **0.9430** | **0.9499** | 0.9740 | **0.9271** | **153** | 52 |
| R7_224_FAKEW12_WAVEREP_BANK | 0.1252 | 0.9283 | 0.9351 | **0.9894** | 0.8866 | 238 | **20** |
| R7_224_F16_FAKEW12 | 0.0074 | 0.9339 | 0.9408 | 0.9834 | 0.9018 | 206 | 32 |

相对基准的变化：

| Experiment | ΔACC | ΔF1 | ΔRecall | ΔFN | ΔFP |
|---|---:|---:|---:|---:|---:|
| R7_224_FAKEW12_WAVEREP_BANK | -0.0147 | -0.0148 | -0.0405 | +85 | -32 |
| R7_224_F16_FAKEW12 | -0.0092 | -0.0091 | -0.0253 | +53 | -20 |

校准阈值结论：

- 两个新实验经过阈值校准后均有所恢复，但仍不如基准。
- WaveRep-bank 的主要效果是降低 FP 和提高 Precision，但代价是 Recall 明显下降。
- 16 帧实验比 WaveRep-bank 更接近基准，但仍增加 53 个 FN。
- 对当前任务而言，减少 FN 比进一步减少 FP 更重要，因此两者都不应替代 R7_224_FAKEW12。

### 16.4 Generator 级 Recall/FN 对比

对 fake generator，表中数值为 Recall；对 `real_MSRVTT`，表中数值为 Specificity。

| Experiment | MorphStudio | Show_1 | Sora | WildScrape | real_MSRVTT |
|---|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 | 0.9914 | **0.9943** | 0.9643 | **0.7804** | 0.9653 |
| R7_224_FAKEW12_WAVEREP_BANK | 0.9886 | 0.9457 | 0.8929 | 0.7103 | **0.9867** |
| R7_224_F16_FAKEW12 | **0.9929** | 0.9714 | **0.9821** | 0.7196 | 0.9787 |

对应 FN/FP：

| Experiment | MorphStudio FN | Show_1 FN | Sora FN | WildScrape FN | real_MSRVTT FP |
|---|---:|---:|---:|---:|---:|
| R7_224_FAKEW12 | 6 | **4** | 2 | **141** | 52 |
| R7_224_FAKEW12_WAVEREP_BANK | 8 | 38 | 6 | 186 | **20** |
| R7_224_F16_FAKEW12 | **5** | 20 | **1** | 180 | 32 |

Generator 级结论：

- WaveRep-bank 的退化主要来自 Show_1、Sora 和 WildScrape 的 FN 增加，其中 WildScrape FN 从 141 增至 186。
- 16 帧实验改善了 MorphStudio 和 Sora，但 Show_1 与 WildScrape 明显退化。
- WildScrape 仍然是关键瓶颈；两个新实验都没有降低 WildScrape FN。
- 两个实验都提高了 real_MSRVTT specificity，说明它们更保守、更不容易误报 real，但这与当前减少 fake 漏检的目标不一致。

### 16.5 Generator 级 AUC / AP 对比

以下 AUC/AP 使用每个 fake generator 与全部 `real_MSRVTT` 组成二分类子集计算。

| Experiment | Generator | AUC | AP |
|---|---|---:|---:|
| R7_224_FAKEW12 | MorphStudio | 0.9991 | 0.9982 |
| R7_224_FAKEW12 | Show_1 | 0.9980 | 0.9966 |
| R7_224_FAKEW12 | Sora | 0.9938 | 0.9209 |
| R7_224_FAKEW12 | WildScrape | **0.9540** | **0.9331** |
| WaveRep-bank | MorphStudio | 0.9986 | 0.9979 |
| WaveRep-bank | Show_1 | 0.9949 | 0.9916 |
| WaveRep-bank | Sora | 0.9922 | 0.9249 |
| WaveRep-bank | WildScrape | 0.9206 | 0.8954 |
| F16 | MorphStudio | 0.9990 | 0.9978 |
| F16 | Show_1 | 0.9974 | 0.9946 |
| F16 | Sora | **0.9964** | **0.9354** |
| F16 | WildScrape | 0.8861 | 0.8754 |

AUC/AP 结论：

- 两个新实验不只是阈值选择问题，它们对 WildScrape 的排序能力也下降。
- WaveRep-bank 将 WildScrape AUC 从 0.9540 降至 0.9206。
- 16 帧将 WildScrape AUC 降至 0.8861，说明更多帧没有带来更稳定的困难生成器排序。
- 16 帧在 Sora 上 AUC/AP 更高，但 Sora 样本数只有 56，且总体 FN 主要来自 WildScrape，因此不足以抵消整体退化。

### 16.6 第 11 模块最终结论

第 11 模块验证了两个合理但最终无效的假设：

```text
有效 WaveRep 增强可以改善 OOD 频域泛化。
完整视频 16 帧采样可以改善 D3/ReStraV 时序泛化。
```

实验结果表明，对当前 STF3 与当前 OOD split：

- WaveRep-bank 使模型更保守，提高 Precision 和 real specificity，但降低 fake Recall。
- 16 帧提高了采样密度，但没有改善 WildScrape，反而降低总体 OOD 排序能力。
- 当前最佳单模型仍是 `R7_224_FAKEW12 + validation-calibrated threshold`。
- 不建议继续运行 `WaveRep-bank + 16F` 联合实验。
- 不建议把第 11 模块结果作为最终主结果，但应作为论文中的负结果和优化边界说明。

推荐论文表述：

```text
To further improve STF3 without introducing external detectors, we evaluated two structure-preserving optimizations: effective WaveRep donor-bank augmentation and full-video 16-frame sampling. Both strategies preserved the original STF3 architecture. However, neither improved the OOD test performance over the 8-frame R7_224_FAKEW12 baseline. WaveRep-bank reduced false positives but increased false negatives, while 16-frame sampling improved a few easier generators but degraded WildScrape ranking. These results suggest that the current STF3 benefits more from the 224-resolution and recall-oriented loss combination than from stronger frequency perturbation or denser temporal sampling under the current training protocol.
```

中文论文表述：

```text
为保证 STF3 主模型的独立性，本文进一步测试了两个不改变网络结构的优化策略：有效 WaveRep donor-bank 频带增强和完整视频 16 帧采样。实验表明，二者均未超过当前最佳 8 帧 STF3 单模型。WaveRep-bank 虽然减少了 real 视频误报，但显著增加 fake 漏检；16 帧采样虽然改善了部分较容易生成器，但未能提升 WildScrape，且总体排序能力下降。因此，当前最可靠的主模型仍为 R7_224_FAKEW12，后续优化应继续围绕 WildScrape 困难样本和训练协议，而不是简单增加频域扰动或采样帧数。
```

## 17. 后续实验建议

当前阶段不建议继续进行简单 multi-clip、盲目增加 epoch，或立即加入外部检测器。后续路线分为“主模型优化”“论文消融收尾”和“增强系统验证”三层。

### 17.1 主模型优化优先级

1. 暂停 `WaveRep-bank + 16F` 联合实验，因为两个单独实验均未超过基准。
2. 保留 `R7_224_FAKEW12` 作为主单模型。
3. 若继续优化，应优先围绕 WildScrape 困难样本做更有针对性的鲁棒增强，而不是继续简单增加帧数或频带扰动。
4. 分别比较默认阈值与验证集校准阈值，避免把概率刻度变化误判为模型能力变化。
5. 对比总体 ACC/F1/Recall、WildScrape FN 和 real_MSRVTT FP。

### 17.2 下一阶段判断标准

新实验不应只凭单一最高分决定是否保留。建议使用以下标准：

| 判断维度 | 通过条件 |
|---|---|
| STF3 独立性 | 不引入外部检测器或额外主模型 |
| 总体表现 | ACC/F1 不明显低于 R7_224_FAKEW12 |
| fake 检测 | Recall 提升或 FN 明显下降 |
| 困难生成器 | WildScrape FN 下降 |
| real 稳定性 | real_MSRVTT FP 增量可接受 |
| 可解释性 | 能明确说明提升来自增强或采样，而不是同时改变过多变量 |

### 17.3 论文消融与收尾实验

1. `R5_224_FAKEW12` 与 `R7_224_FAKEW12` 的强公平对照已经完成，可作为最终内部消融表的核心证据；如果时间允许，再补纯 `R5_224` vs `R7_224` 只作为非 fake-loss 设置的补充。
2. 保留原始 R7、R7_224、R7_FAKEW12、R7_224_FAKEW12，构成分辨率与 fake loss 的组合消融。
3. 将 multi-clip 作为负结果或局限性分析，不作为最终方法。
4. 如果篇幅允许，报告 generator-level Recall/FN，避免只展示总体指标。

### 17.4 增强系统验证

1. 对 Branch LR 做不同随机划分或交叉验证，降低验证集后处理过拟合风险。
2. 保留 STF3-only ensemble 作为 Recall 导向增强方案。
3. 明确区分单模型、STF3 内部后处理和多 checkpoint ensemble，避免论文表述混淆。

## 18. 论文式最终叙事建议

当前研究故事可以概括为：

> 本研究首先在 Random split 上验证 STF3 的基本检测能力，随后通过跨生成器 OOD split 识别其真实泛化瓶颈。原始 OOD 结果显示，STF3 具有较高 Precision，但仍存在明显 fake 漏检。验证集阈值校准证明部分问题来自决策边界，但 generator 级错误分析进一步发现，剩余漏检高度集中于 WildScrape，说明模型仍缺乏对困难未知分布的稳定表示。基于这一诊断，研究分别探索了训练轮数、fake 类损失权重和输入分辨率。单独增加 epoch 或分辨率未带来稳定收益，而 224 分辨率与适度 fake loss weight 的组合同时增强了细粒度伪造证据保留和召回导向学习，形成当前最强 STF3 单模型。进一步的 STF3-only ensemble 与分支级校准证明不同 STF3 变体和内部证据具有互补性；相反，简单 multi-clip 时间裁剪损害了 WildScrape 排序能力，表明完整视频范围内的联合时空证据对该模型更加重要。最后，在保持 STF3 结构独立性的前提下，本文验证了有效 WaveRep donor-bank 增强和完整视频 16 帧采样。二者均未超过 R7_224_FAKEW12，说明当前 STF3 的主要收益来自 224 分辨率与召回导向损失的组合，而不是更强频带扰动或更密时间采样。

这条叙事的核心不是“不断增加模块”，而是：

```text
用 OOD 错误定位问题
→ 用低成本校准区分阈值问题与表示问题
→ 用有针对性的训练设置降低 FN
→ 用失败实验验证模型真正依赖的证据
→ 在保持 STF3 独立性的前提下验证进一步优化，并保留有效结果、记录无效边界
```

## 19. 当前结论边界

目前可以支持的结论：

- R7/STF3 在原始 OOD R1-R7 中具有最均衡的默认阈值表现。
- R7_224_FAKEW12 是当前已完成实验中的最佳 STF3 单模型。
- 224 与 fake loss weight 1.2 存在组合收益。
- 阈值校准能够显著减少 FN，但不能替代模型训练优化。
- WildScrape 是当前最主要的 OOD 失败来源。
- STF3 内部分支与不同 checkpoint 具有互补性。
- 当前 60% temporal crop multi-clip 策略不适合该模型。
- 此前主要实验中的 WaveRep 训练增强实际未生效；第 11 模块已经验证有效 WaveRep-bank，但其结果未超过 R7_224_FAKEW12。
- 完整视频 16 帧采样已经验证，但同样未超过 R7_224_FAKEW12。

目前不能过度声称的结论：

- 不能声称 224 在所有配置下必然优于 112。
- 不能声称增加 epoch 能改善 OOD 泛化。
- 不能声称 multi-clip 方法普遍无效，只能否定当前裁剪与聚合方案。
- 不能将 Branch LR 描述为新的 STF3 主架构。
- 不能声称 frequency 分支在所有训练设置下都必然显著提升；当前最强证据来自 `R5_224_FAKEW12` 与 `R7_224_FAKEW12` 的强公平消融，即相同 224、相同 fake loss、相同阈值校准下三分支 STF3 优于 S+T 双分支。
- 不能声称 WaveRep-bank 或 16F 能提高当前最终指标；现有证据恰好显示它们低于基准。

## 20. R5_224_FAKEW12 正式消融补充：S+T 双分支 vs STF3 三分支

本节补齐最终公平消融表中最关键的一组对照：

```text
R5_224_FAKEW12 = spatial_temporal_new + image_size=224 + fake_loss_weight=1.2
R7_224_FAKEW12 = stf3_new + image_size=224 + fake_loss_weight=1.2
```

这组实验的意义是：在输入分辨率、训练轮数、fake 类损失权重、OOD split、val/test predictions 和阈值选择协议都一致的情况下，比较两分支 S+T 模型与三分支 STF3 模型。它比单纯比较原始 R5/R7 更适合作为最终优化阶段的强公平消融。

### 20.1 文件完整性确认

`R5_224_FAKEW12` 已经具备完整评估文件，可以正式写入结果表：

| 文件 | 状态 |
|---|---|
| `runs/ood_spatial_temporal_new_224_fakew12/best.pt` | 已存在 |
| `runs/ood_spatial_temporal_new_224_fakew12/history.json` | 已存在 |
| `outputs/ood_spatial_temporal_new_224_fakew12/predictions.csv` | 已存在 |
| `outputs/ood_spatial_temporal_new_224_fakew12/metrics.json` | 已存在 |
| `outputs/ood_spatial_temporal_new_224_fakew12_val/predictions.csv` | 已存在 |
| `outputs/ood_spatial_temporal_new_224_fakew12_val/metrics.json` | 已存在 |

### 20.2 默认阈值 0.5 结果

| Model | Branch setting | Split | ACC | AUC | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R5_224_FAKEW12 | S+T | val | 0.9876 | 0.9988 | 0.9838 | 0.9891 | 0.9785 | 1490 | 10 | 20 | 909 |
| R5_224_FAKEW12 | S+T | test | 0.8997 | 0.9802 | 0.9062 | 0.9960 | 0.8313 | 1493 | 7 | 354 | 1744 |
| R7_224_FAKEW12 | S+T+F STF3 | val | 0.9897 | 0.9993 | 0.9866 | 0.9839 | 0.9892 | 1485 | 15 | 10 | 919 |
| R7_224_FAKEW12 | S+T+F STF3 | test | 0.9275 | 0.9848 | 0.9341 | 0.9930 | 0.8818 | 1487 | 13 | 248 | 1850 |

默认阈值下，R7_224_FAKEW12 相比 R5_224_FAKEW12 已经有明显优势：Test ACC 从 0.8997 提升到 0.9275，F1 从 0.9062 提升到 0.9341，Recall 从 0.8313 提升到 0.8818，FN 从 354 降到 248。代价是 FP 从 7 增加到 13，但 Precision 仍保持在 0.9930。

### 20.3 验证集校准阈值结果

阈值选择目标继续使用 `precision_0.95_recall_max`，即在 validation predictions 上优先保证 Precision 不低于约 0.95，并尽量提高 Recall。该策略适合当前内容安全任务，因为主要风险来自 fake 被漏检。

| Model | Branch setting | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R5_224_FAKEW12 | S+T | 0.0089 | 0.9277 | 0.9360 | 0.9679 | 0.9061 | 197 | 63 |
| R7_224_FAKEW12 | S+T+F STF3 | 0.0545 | **0.9430** | **0.9499** | **0.9740** | **0.9271** | **153** | **52** |
| Δ R7 - R5 | +F branch + STF3 fusion | - | **+0.0153** | **+0.0139** | **+0.0060** | **+0.0210** | **-44** | **-11** |

这一结果是目前最适合放进论文或答辩的强公平消融表。它说明 R7 的提升并不是单纯来自 224 分辨率、fake loss weight 或阈值后处理，因为这些因素在 R5 与 R7 中保持一致；在相同协议下，三分支 STF3 仍然取得更高 ACC、F1、Precision 和 Recall，并同时减少 FN 与 FP。

### 20.4 Generator-level 细节

在 `precision_0.95_recall_max` 阈值下，fake generator 报告 Recall/FN，real_MSRVTT 报告 Specificity/FP。

| Model | MorphStudio Recall / FN | Show_1 Recall / FN | Sora Recall / FN | WildScrape Recall / FN | real_MSRVTT Specificity / FP |
|---|---:|---:|---:|---:|---:|
| R5_224_FAKEW12 | 0.9886 / 8 | 0.9757 / 17 | 0.9643 / 2 | 0.7352 / 170 | 0.9580 / 63 |
| R7_224_FAKEW12 | **0.9914 / 6** | **0.9943 / 4** | 0.9643 / 2 | **0.7804 / 141** | **0.9653 / 52** |
| Δ R7 - R5 | +0.0029 / -2 | +0.0186 / -13 | 0.0000 / 0 | +0.0452 / -29 | +0.0073 / -11 |

最重要的变化来自 WildScrape：R7_224_FAKEW12 将 WildScrape FN 从 170 降到 141，减少 29 个。Show_1 也从 17 个 FN 降到 4 个。Sora 样本数较少，二者表现相同。real_MSRVTT 上，R7 还把 FP 从 63 降到 52，说明该提升不是简单地用更多误报换取更高 Recall。

### 20.5 可写入论文的结论

推荐论文或答辩中这样表述：

```text
To isolate the contribution of the frequency branch and branch-token STF3 fusion under the optimized training protocol, we further compare the spatial-temporal model and the full STF3 model using the same 224 input resolution, the same fake-class loss weight, and the same validation-calibrated threshold selection rule. The full STF3 model improves OOD ACC from 0.9277 to 0.9430, F1 from 0.9360 to 0.9499, and Recall from 0.9061 to 0.9271, while reducing false negatives from 197 to 153. Generator-level analysis shows that the largest reduction occurs on WildScrape, the hardest OOD generator, where FN decreases from 170 to 141. These results support that the frequency branch and branch-token fusion provide complementary evidence beyond the spatial-temporal baseline.
```

中文表述可以写为：

```text
为隔离频域分支与 branch-token STF3 融合在最终优化协议下的贡献，本文进一步比较了相同 224 输入分辨率、相同 fake 类损失权重和相同验证集阈值校准策略下的 spatial-temporal 双分支模型与完整 STF3 模型。结果显示，完整 STF3 将 OOD ACC 从 0.9277 提升到 0.9430，F1 从 0.9360 提升到 0.9499，Recall 从 0.9061 提升到 0.9271，并将 FN 从 197 降到 153。Generator-level 分析表明，提升主要来自最困难的 WildScrape 生成源，其 FN 从 170 降到 141。这说明频域分支和 branch-token 融合为 spatial-temporal baseline 提供了互补证据。
```

### 20.6 结论边界

这组实验可以支持“在当前最终优化设置下，R7/STF3 优于 R5/S+T 双分支”。但它不应被过度解释为“频域分支在任何分辨率、任何 loss、任何数据集上都必然提升”。更稳妥的结论是：

```text
在 GenVideo-Val OOD split 上，当输入分辨率提高到 224 且使用适度 fake 类损失权重后，完整 STF3 相比 spatial-temporal 双分支 baseline 表现出更好的跨生成器泛化，尤其降低了 WildScrape 等困难生成源的漏检。
```

