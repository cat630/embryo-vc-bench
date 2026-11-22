# W8 post hoc sensitivity analysis — submission-ready supplement

> **Status: DRAFT SUPPLEMENT — NOT yet merged into the manuscript.**
> This is a standalone, non-source file. The submission manuscript remains
> `virtual-cell-benchmark-manuscript-v0.md`; that file is SHA256-anchored by the
> W6 wording audit and must not be edited until the final clean-up pass, when the
> audit is refreshed. Merge the English block below at that point.

## What this file is for

W8 answers the single most likely methodological objection from an external
reviewer: *the 44 VCC validation target genes are shared between T2 and T3, so is
the scGPT directional (DES/PDS) advantage merely an artefact of target-set
selection or of the metric itself?*

It is an **exploratory, descriptive, post hoc** analysis executed after the
preregistered scoring run (2026-11-18T06:36:10Z). It is **not** part of the
registered hypothesis family. It produces **no p-value, no inferential test and
no BH correction**, does **not** change H2 adjudication, does **not** modify any
published score, any `[[R#]]` slot or any registered criterion, and therefore
requires **no third OSF update**. If reported, it must be explicitly labelled
post hoc and placed **separately from the preregistered Results**.

## Design (one frozen evaluation block: T3_DE_real / seed 0)

| Condition | Manipulation | Replicates |
|---|---|---|
| A Replication anchor | Real 44 targets, real delta; must reproduce `w5_scores.csv` bit-for-bit | 1 |
| B Random target | 44 expression-quartile-matched non-target genes from the same 17,555-gene axis | scGPT 3, Geneformer 1 |
| C Target shuffle | Real delta, 44 rows permuted (pairing broken) | 20 per model |
| D Feature shuffle | Real delta, 11,502 feature columns permuted (gene identity broken) | 20 per model |

Replication anchor: **PASS**, maximum absolute difference from the published
scores = `0.000e+00` across all four methods and all four metrics. All conditions
use the same inputs, top-k = 20 and `score_all_scaled` implementation as the
preregistered run.

## Descriptive results (scGPT, mean over replicates)

| Condition | DES | PDS | MAE | MAE_z |
|---|---|---|---|---|
| A Real targets (replication) | 0.0534 | 0.8636 | 0.096454 | 0.5549 |
| B Expression-matched random targets | 0.0015 | 0.4693 | 0.096558 | 0.5857 |
| C Target-to-truth pairing shuffled | 0.0050 | 0.5035 | 0.096554 | 0.5593 |
| D Gene identity shuffled | 0.0018 | 0.5090 | 0.096577 | 0.5576 |
| `no_change` mandatory baseline | 0.0000 | 0.5000 | 0.096489 | 0.5073 |

Ranges across replicates: B DES 0.0011–0.0023, PDS 0.4218–0.5095; C DES
0.0023–0.0091, PDS 0.4334–0.5714; D DES 0.0000–0.0057, PDS 0.4313–0.5930.
Geneformer at A: DES 0.0102, PDS 0.5063; B: DES 0.0045, PDS 0.4382.

**Interpretation, stated conservatively:** the scale-free directional advantage
is contingent on both the correct target identity and the correct gene identity;
it is not reproduced by arbitrary expression-matched targets or by broken
pairing. MAE remains indistinguishable from the `no_change` baseline in every
condition, and MAE_z remains worse — consistent with the preregistered H2
adjudication. This analysis **does not** address selection bias in how the 44
targets were originally chosen by VCC, **does not** turn T2 into an independent
real dataset, and **does not** supply the ">=2 independent real datasets"
precondition missing from the H2 falsification rule.

## English block to merge into the manuscript (Discussion / limitations)

> *As a post hoc, exploratory sensitivity analysis conducted after the
> preregistered scoring run — not part of the registered hypothesis family and
> with no inferential test — we replaced the 44 validation targets with
> expression-matched non-target genes and, separately, permuted target-to-truth
> pairing and gene identity on one frozen evaluation block (T3, seed 0). The
> scale-free directional advantage of scGPT was attenuated under all three
> manipulations (DES 0.0534 at the replication anchor versus 0.0015, 0.0050 and
> 0.0018, respectively; PDS 0.8636 versus 0.4693, 0.5035 and 0.5090), while MAE
> remained indistinguishable from the mandatory no-change baseline in every
> condition. The replication anchor reproduced the published scores exactly
> (maximum absolute difference 0.000e+00). This analysis is descriptive, was not
> used to adjudicate H2, does not address selection bias in the original choice
> of validation targets, and does not establish T2 as an independent real
> dataset. Full condition-level values are deposited with the code release.*

A second one-sentence change belongs in **Methods** (evaluation-object
definition): the object of evaluation is *the sensitivity of zero-shot
representations to gene-token ablation and its transferability across atlases*,
not the models' native perturbation-prediction task. Add the equivalent sentence
to Discussion as well.

## Provenance

- Script: `w8_negative_controls.py` (W8, 395 lines), run in `.venv-models`.
- Outputs: `results/w8_negative_controls.csv` (89 rows),
  `results/w8_negative_controls.json`, `results/w8_negative_controls_report.md`.
- Input hashes are recorded in `w8_negative_controls.json`; the preregistered
  score table, summary, manuscript, proposal, engineering README and OSF draft
  were left untouched (verified by timestamp and hash).
- Generated 2026-11-20T07:04:24Z, elapsed 915.8 s.
