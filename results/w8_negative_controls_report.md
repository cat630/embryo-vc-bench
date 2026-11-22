# W8 阴性对照｜44 个共用目标基因（事后探索性敏感性分析）

> **性质声明（务必随结果一起保留）**
> - **这不是预注册分析**。执行于预注册打分（2026-11-18 06:36:10 UTC）**之后**，
>   属 post hoc exploratory sensitivity analysis。
> - **不产生 p 值、不做显著性检验、不做 BH 校正**；仅描述性统计。
> - **不改变 H2 判定**（仍为「未确认为一致零结果 / 有部分反向证据 / 未正式证伪」），
>   不修改任何已发布分数、任何 `[[R#]]` 槽位或任何注册判据。
> - **不需要第三次 OSF update**：注册的分析计划、假设、判据、种子与已发布分数均未改动。
>   若写入稿件，必须明确标注为 exploratory/post hoc，并与预注册 Results 分开放置。
> - 输出文件全部为新建 `w8_*`，四份被 W6 措辞审计锚定的源文档与 `w5_scores.csv` 均未改动。

## 它回答什么、不回答什么

- **回答**：scGPT 在 DES/PDS 上的方向优势，是不是「不管拿什么基因当目标、或不管怎么配对」
  都会出现的指标假象？如果换随机基因／打乱配对／打乱基因身份后优势消失，
  则优势至少依赖于「正确的目标身份 + 正确的基因身份」。
- **不回答**：这 44 个目标当初由 VCC 选出的选择偏倚；不能据此把 T2 变成独立真实数据集，
  也不能用来补上 H2 证伪规则所缺的「≥2 个独立数据集」前提。

## 设计

固定一个已冻结的评估块：T3_DE_real / seed 0 / VCC 验证 DE 真值，
44 个目标 × 11502 个共享特征，top-k=20，与正式打分完全相同的输入与指标实现。

| 条件 | 做法 |
|---|---|
| A 重现锚 | 真实 44 目标重算，必须与 `w5_scores.csv` 逐位一致 |
| B 随机目标 | 换成按对照细胞平均表达四分位**匹配**抽取的 44 个非目标基因（scGPT 3 组 / Geneformer 1 组） |
| C 配对打乱 | 真实 delta 不变，44 行随机置换 × 20 次 |
| D 基因身份打乱 | 真实 delta 不变，特征列随机置换 × 20 次 |

**重现锚：PASS**，与已发布分数最大绝对差 `0.000e+00`。

## 结果（scGPT，描述性均值）

| 条件 | DES | PDS | MAE | MAE_z |
|---|---|---|---|---|
| A 真实目标（重现） | 0.0534 | 0.8636 | 0.096454 | 0.5549 |
| B 表达匹配随机目标 | 0.0015 | 0.4693 | 0.096558 | 0.5857 |
| C 配对打乱 | 0.0050 | 0.5035 | 0.096554 | 0.5593 |
| D 基因身份打乱 | 0.0018 | 0.5090 | 0.096577 | 0.5576 |
| 参照：no_change 基线 | 0.0000 | 0.5000 | 0.096489 | 0.5073 |

逐次重复、Geneformer 与两条基线的完整数值见 `w8_negative_controls.csv` / `.json`。

## 合规边界（写入稿件时必须照抄口径）

若要把这组对照写进稿件，建议措辞：

> *As a post hoc, exploratory sensitivity analysis conducted after the preregistered scoring run (not part of the registered hypothesis family, with no inferential test), replacing the 44 validation targets with expression-matched non-target genes, or permuting target-to-truth pairing or gene identity, attenuated the scale-free directional advantage. This analysis is descriptive and was not used to adjudicate H2.*

