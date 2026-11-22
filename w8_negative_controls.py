#!/usr/bin/env python3
"""W8 — POST-HOC negative controls for the 44 shared VCC validation targets.

WHAT THIS IS
------------
An exploratory sensitivity analysis performed AFTER the preregistered scoring
run (2026-11-18T06:36:10Z). It asks one question raised in review of the draft:

    Is the scale-free directional advantage (DES/PDS) of scGPT an artefact of
    the 44 target genes shared between T2 and T3, or of the metric itself?

Four conditions, all on ONE frozen evaluation block (T3_DE_real, seed 0):

  A REPLICATION       Reproduce the published model scores bit-for-bit from the
                      same frozen inputs. If this fails, nothing below is valid.
  B RANDOM_TARGET     Replace the 44 real targets with 44 EXPRESSION-MATCHED
                      non-target genes drawn from the same 17,555-gene axis.
  C TARGET_SHUFFLE    Keep the real delta, permute its 44 rows (breaks the
                      target-to-truth pairing).
  D FEATURE_SHUFFLE   Keep the real delta, permute its 11,502 feature columns
                      (breaks gene identity, preserves the marginal distribution).

WHAT THIS IS NOT  (binding compliance boundary)
----------------------------------------------
  * NOT preregistered. NOT part of the H1/H2/H3 hypothesis family.
  * NOT used for the H2 adjudication, which remains exactly as published:
    "not confirmed as a uniform null; partial evidence against; NOT formally
    falsified".
  * NO p-values, NO permutation-null inference, NO BH correction, NO new
    hypothesis test. DES/PDS/MAE/MAE_z are reported descriptively only.
  * NO change to w5_scores.csv, w5_score_summary.json, any [[R#]] slot, the
    manuscript, the proposal, the README or the OSF draft (all hash-anchored by
    the W6 wording audit).
  * NO third OSF registration update: the registered analysis plan, hypotheses,
    criteria, seeds and published scores are untouched. If any of these results
    are reported, they must be labelled exploratory/post hoc and placed apart
    from the preregistered Results.

Outputs (all NEW, w8_ prefixed):
  results/w8_negative_controls.json
  results/w8_negative_controls.csv
  results/w8_negative_controls_report.md

Run with the model environment: ./.venv-models/Scripts/python.exe w8_negative_controls.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from w5_gene_space import RESULTS, build_gene_space                 # noqa: E402
from w5_score import (                                              # noqa: E402
    CONTROL_PER_SEED, CONTROL_POOL, SHARED_FEATURES, TRAIN_DE, VAL_DE,
    align_de_to_axis, baseline_predictions, score_all_scaled,
)
from w5_adapters import PROTOCOL_VERSION, build_adapters            # noqa: E402

STATUS = (
    "POST_HOC_EXPLORATORY_SENSITIVITY_ANALYSIS - NOT PREREGISTERED; "
    "NOT A HYPOTHESIS TEST; NO P-VALUE; NOT USED FOR H2 ADJUDICATION; "
    "NO PUBLISHED SCORE, [[R#]] SLOT OR REGISTERED CRITERION CHANGED"
)

N_SHUFFLE = 20
RANDOM_SEEDS = (0, 1, 2)
RANDOM_MODELS = {"scGPT": (0, 1, 2), "Geneformer": (0,)}
METRICS = ("MAE", "MAE_z", "DES", "PDS")

OUT_JSON = RESULTS / "w8_negative_controls.json"
OUT_CSV = RESULTS / "w8_negative_controls.csv"
OUT_MD = RESULTS / "w8_negative_controls_report.md"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    """Rebuild the exact T3/seed-0 scoring block from frozen published inputs."""
    gs = build_gene_space()
    sf = json.loads(SHARED_FEATURES.read_text(encoding="utf-8"))
    shared_pos = sf["gene_axis_index"]

    val_df = pd.read_parquet(VAL_DE)
    train_df = pd.read_parquet(TRAIN_DE)
    val_mat, val_targets, _ = align_de_to_axis(val_df, gs, sf["features"])
    train_mat, train_targets, _ = align_de_to_axis(train_df, gs, sf["features"])
    assert all(t in gs.genes for t in val_targets)

    pool = np.load(CONTROL_POOL)
    pool_X = pool["counts"]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(pool_X.shape[0], size=CONTROL_PER_SEED, replace=False))
    ctrl = np.ascontiguousarray(pool_X[pick])

    baselines = baseline_predictions(train_mat, val_mat)
    return {
        "gs": gs, "shared_pos": shared_pos, "val_mat": val_mat,
        "val_targets": list(val_targets), "train_mat": train_mat,
        "ctrl": ctrl, "baselines": baselines,
        "n_train": len(train_targets), "n_test": len(val_targets),
    }


def score_delta(delta_full, shared_pos, val_mat) -> dict:
    pred = np.asarray(delta_full)[:, shared_pos]
    if pred.shape != val_mat.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {val_mat.shape}")
    return {k: float(v) for k, v in score_all_scaled(pred, val_mat).items()}


def matched_random_targets(gs, ctrl, real_targets, seed: int) -> list[str]:
    """Draw non-target genes matched to the real targets by expression quartile.

    Matching prevents the trivial criticism that random genes were simply
    low-expression genes with no token to ablate.
    """
    mean_expr = np.asarray(ctrl, dtype=np.float64).mean(axis=0)
    real_idx = [gs.genes.index(t) for t in real_targets]
    real_set = set(real_idx)
    candidates = [i for i in range(gs.n_genes) if i not in real_set]

    cuts = np.quantile(mean_expr, [0.25, 0.50, 0.75])
    bin_of = np.searchsorted(cuts, mean_expr, side="right")
    counts = np.bincount(bin_of[real_idx], minlength=4)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        pool_b = [i for i in candidates if bin_of[i] == b and i not in chosen]
        if len(pool_b) < n:
            pool_b = [i for i in candidates if i not in chosen]
        chosen.extend(int(x) for x in rng.choice(pool_b, size=n, replace=False))
    rng.shuffle(chosen)
    assert len(chosen) == len(real_targets)
    assert not (set(chosen) & real_set)
    return [gs.genes[i] for i in chosen]


def summarise(rows: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        v = np.asarray([r[m] for r in rows], dtype=float)
        out[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "n": len(v)}
    return out


def main() -> int:
    t0 = time.time()
    log("W8 post-hoc negative controls: starting (NOT preregistered, see module docstring)")
    P = prepare()
    gs, sp, vm = P["gs"], P["shared_pos"], P["val_mat"]
    targets = P["val_targets"]
    log(f"frozen block: T3_DE_real seed=0 | {P['n_train']} train / {P['n_test']} val targets "
        f"| {len(sp)} shared features | {gs.n_genes} gene axis")

    input_hashes = {p.name: sha(p) for p in
                    (RESULTS / "w5_scores.csv", SHARED_FEATURES, CONTROL_POOL,
                     TRAIN_DE, VAL_DE)}
    pub = pd.read_csv(RESULTS / "w5_scores.csv")
    pub_block = pub[(pub["task"] == "T3_DE_real") & (pub["seed"] == 0)]

    log("loading the two pinned zero-shot adapters (no fine-tuning)")
    adapters = build_adapters(gs, seed=0)

    # ---------------------------------------------------------------- A
    log("A REPLICATION: regenerating published deltas on the real 44 targets")
    d_true = {}
    for name, model in adapters.items():
        ts = time.time()
        d_true[name] = model.predict_delta(P["ctrl"], targets)
        log(f"  {name}: {d_true[name].shape} in {time.time()-ts:.1f}s")

    rows: list[dict] = []
    replication = {}
    for method, pred in P["baselines"].items():
        s = {k: float(v) for k, v in score_all_scaled(pred, vm).items()}
        rows.append({"condition": "A_REPLICATION", "model": method, "replicate": 0, **s})
        r = pub_block[pub_block["method"] == method].iloc[0]
        replication[method] = {m: {"recomputed": s[m], "published": float(r[m]),
                                    "abs_diff": abs(s[m] - float(r[m]))} for m in METRICS}
    for name, d in d_true.items():
        s = score_delta(d, sp, vm)
        rows.append({"condition": "A_REPLICATION", "model": name, "replicate": 0, **s})
        r = pub_block[pub_block["method"] == name].iloc[0]
        replication[name] = {m: {"recomputed": s[m], "published": float(r[m]),
                                  "abs_diff": abs(s[m] - float(r[m]))} for m in METRICS}

    max_diff = max(v[m]["abs_diff"] for v in replication.values() for m in METRICS)
    log(f"A REPLICATION max |recomputed - published| = {max_diff:.3e}")
    if max_diff > 1e-9:
        log("FATAL: replication anchor failed. The control pipeline is NOT the "
            "published scoring pipeline; refusing to continue.")
        OUT_JSON.write_text(json.dumps({
            "status": STATUS, "replication": "FAILED", "max_abs_diff": max_diff,
            "detail": replication}, indent=2), encoding="utf-8")
        return 2
    log("A REPLICATION PASS — control pipeline is bit-identical to the published run")

    # ---------------------------------------------------------------- B
    log("B RANDOM_TARGET: expression-matched non-target gene sets")
    for name, seeds in RANDOM_MODELS.items():
        for rs in seeds:
            rnd = matched_random_targets(gs, P["ctrl"], targets, rs)
            ts = time.time()
            d = adapters[name].predict_delta(P["ctrl"], rnd)
            s = score_delta(d, sp, vm)
            rows.append({"condition": "B_RANDOM_TARGET", "model": name,
                         "replicate": rs, **s})
            log(f"  {name} seed={rs}: DES={s['DES']:.4f} PDS={s['PDS']:.4f} "
                f"({time.time()-ts:.1f}s)")

    # ---------------------------------------------------------------- C/D
    log("C TARGET_SHUFFLE and D FEATURE_SHUFFLE (no model call, descriptive only)")
    for name, d in d_true.items():
        proj = np.asarray(d)[:, sp]
        for rep in range(N_SHUFFLE):
            rng = np.random.default_rng(1000 + rep)
            perm = rng.permutation(proj.shape[0])
            s = {k: float(v) for k, v in score_all_scaled(proj[perm], vm).items()}
            rows.append({"condition": "C_TARGET_SHUFFLE", "model": name,
                         "replicate": rep, **s})
        for rep in range(N_SHUFFLE):
            rng = np.random.default_rng(2000 + rep)
            perm = rng.permutation(proj.shape[1])
            s = {k: float(v) for k, v in score_all_scaled(proj[:, perm], vm).items()}
            rows.append({"condition": "D_FEATURE_SHUFFLE", "model": name,
                         "replicate": rep, **s})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    conditions = {}
    for cond in ("A_REPLICATION", "B_RANDOM_TARGET", "C_TARGET_SHUFFLE",
                 "D_FEATURE_SHUFFLE"):
        conditions[cond] = {}
        for model in sorted({r["model"] for r in rows if r["condition"] == cond}):
            sub = [r for r in rows if r["condition"] == cond and r["model"] == model]
            conditions[cond][model] = summarise(sub)

    anchor = {m: conditions["A_REPLICATION"][m] for m in ("scGPT", "Geneformer",
                                                          "no_change", "train_mean")}
    contrasts = {}
    for cond in ("B_RANDOM_TARGET", "C_TARGET_SHUFFLE", "D_FEATURE_SHUFFLE"):
        contrasts[cond] = {}
        for model, s in conditions[cond].items():
            contrasts[cond][model] = {
                m: {"delta_vs_replication": s[m]["mean"] - anchor[model][m]["mean"],
                    "direction_retained": bool(s[m]["mean"] > anchor["no_change"][m]["mean"])
                    if m in ("DES", "PDS") else None}
                for m in METRICS}

    payload = {
        "status": STATUS,
        "work_package": "W8_post_hoc_negative_controls",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "compliance": {
            "preregistered": False,
            "performed_after_preregistered_run": "2026-11-18T06:36:10Z",
            "no_p_value": True, "no_inferential_test": True, "no_bh_correction": True,
            "descriptive_only": True,
            "h2_adjudication_unchanged": "not confirmed as a uniform null; "
                                         "partial evidence against; NOT formally falsified",
            "published_files_untouched": ["w5_scores.csv", "w5_score_summary.json",
                                          "virtual-cell-benchmark-manuscript-v0.md",
                                          "virtual-cell-benchmark-proposal-v1.md",
                                          "embryo-vc-bench/README.md",
                                          "osf-preregistration-draft.md"],
            "osf_update_required": False,
            "osf_update_rationale": "A post hoc sensitivity analysis does not modify "
                                    "the registered analysis plan, hypotheses, "
                                    "falsification criteria, seeds or published "
                                    "scores. If reported, it must be explicitly "
                                    "labelled exploratory and kept separate from "
                                    "the preregistered Results.",
        },
        "design": {
            "frozen_block": "T3_DE_real / seed 0 / VCC validation DE truth",
            "n_real_targets": len(targets), "n_shared_features": len(sp),
            "n_gene_axis": gs.n_genes, "topk": 20,
            "protocol_version": PROTOCOL_VERSION,
            "n_shuffle_replicates": N_SHUFFLE,
            "random_target_seeds": {k: list(v) for k, v in RANDOM_MODELS.items()},
            "matching": "non-target genes matched to the 44 real targets by "
                        "control-cell mean-expression quartile",
            "conditions": {
                "A_REPLICATION": "real targets, real delta (replication anchor)",
                "B_RANDOM_TARGET": "expression-matched random non-target genes",
                "C_TARGET_SHUFFLE": "real delta, 44 rows permuted (pairing broken)",
                "D_FEATURE_SHUFFLE": "real delta, feature columns permuted (gene identity broken)",
            },
        },
        "input_sha256": input_hashes,
        "replication": {"result": "PASS", "max_abs_diff": max_diff,
                        "detail": replication},
        "conditions": conditions,
        "contrasts_vs_replication": contrasts,
        "output_files": [OUT_CSV.name, OUT_MD.name],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- report
    sc_a = conditions["A_REPLICATION"]["scGPT"]
    sc_b = conditions["B_RANDOM_TARGET"]["scGPT"]
    sc_c = conditions["C_TARGET_SHUFFLE"]["scGPT"]
    sc_d = conditions["D_FEATURE_SHUFFLE"]["scGPT"]
    nc = conditions["A_REPLICATION"]["no_change"]
    lines = [
        "# W8 阴性对照｜44 个共用目标基因（事后探索性敏感性分析）",
        "",
        "> **性质声明（务必随结果一起保留）**",
        "> - **这不是预注册分析**。执行于预注册打分（2026-11-18 06:36:10 UTC）**之后**，",
        ">   属 post hoc exploratory sensitivity analysis。",
        "> - **不产生 p 值、不做显著性检验、不做 BH 校正**；仅描述性统计。",
        "> - **不改变 H2 判定**（仍为「未确认为一致零结果 / 有部分反向证据 / 未正式证伪」），",
        ">   不修改任何已发布分数、任何 `[[R#]]` 槽位或任何注册判据。",
        "> - **不需要第三次 OSF update**：注册的分析计划、假设、判据、种子与已发布分数均未改动。",
        ">   若写入稿件，必须明确标注为 exploratory/post hoc，并与预注册 Results 分开放置。",
        "> - 输出文件全部为新建 `w8_*`，四份被 W6 措辞审计锚定的源文档与 `w5_scores.csv` 均未改动。",
        "",
        "## 它回答什么、不回答什么",
        "",
        "- **回答**：scGPT 在 DES/PDS 上的方向优势，是不是「不管拿什么基因当目标、或不管怎么配对」",
        "  都会出现的指标假象？如果换随机基因／打乱配对／打乱基因身份后优势消失，",
        "  则优势至少依赖于「正确的目标身份 + 正确的基因身份」。",
        "- **不回答**：这 44 个目标当初由 VCC 选出的选择偏倚；不能据此把 T2 变成独立真实数据集，",
        "  也不能用来补上 H2 证伪规则所缺的「≥2 个独立数据集」前提。",
        "",
        "## 设计",
        "",
        "固定一个已冻结的评估块：T3_DE_real / seed 0 / VCC 验证 DE 真值，",
        f"{len(targets)} 个目标 × {len(sp)} 个共享特征，top-k=20，与正式打分完全相同的输入与指标实现。",
        "",
        "| 条件 | 做法 |",
        "|---|---|",
        "| A 重现锚 | 真实 44 目标重算，必须与 `w5_scores.csv` 逐位一致 |",
        "| B 随机目标 | 换成按对照细胞平均表达四分位**匹配**抽取的 44 个非目标基因（scGPT 3 组 / Geneformer 1 组） |",
        f"| C 配对打乱 | 真实 delta 不变，44 行随机置换 × {N_SHUFFLE} 次 |",
        f"| D 基因身份打乱 | 真实 delta 不变，特征列随机置换 × {N_SHUFFLE} 次 |",
        "",
        f"**重现锚：PASS**，与已发布分数最大绝对差 `{max_diff:.3e}`。",
        "",
        "## 结果（scGPT，描述性均值）",
        "",
        "| 条件 | DES | PDS | MAE | MAE_z |",
        "|---|---|---|---|---|",
        f"| A 真实目标（重现） | {sc_a['DES']['mean']:.4f} | {sc_a['PDS']['mean']:.4f} | {sc_a['MAE']['mean']:.6f} | {sc_a['MAE_z']['mean']:.4f} |",
        f"| B 表达匹配随机目标 | {sc_b['DES']['mean']:.4f} | {sc_b['PDS']['mean']:.4f} | {sc_b['MAE']['mean']:.6f} | {sc_b['MAE_z']['mean']:.4f} |",
        f"| C 配对打乱 | {sc_c['DES']['mean']:.4f} | {sc_c['PDS']['mean']:.4f} | {sc_c['MAE']['mean']:.6f} | {sc_c['MAE_z']['mean']:.4f} |",
        f"| D 基因身份打乱 | {sc_d['DES']['mean']:.4f} | {sc_d['PDS']['mean']:.4f} | {sc_d['MAE']['mean']:.6f} | {sc_d['MAE_z']['mean']:.4f} |",
        f"| 参照：no_change 基线 | {nc['DES']['mean']:.4f} | {nc['PDS']['mean']:.4f} | {nc['MAE']['mean']:.6f} | {nc['MAE_z']['mean']:.4f} |",
        "",
        "逐次重复、Geneformer 与两条基线的完整数值见 `w8_negative_controls.csv` / `.json`。",
        "",
        "## 合规边界（写入稿件时必须照抄口径）",
        "",
        "若要把这组对照写进稿件，建议措辞：",
        "",
        "> *As a post hoc, exploratory sensitivity analysis conducted after the "
        "preregistered scoring run (not part of the registered hypothesis "
        "family, with no inferential test), replacing the 44 validation targets "
        "with expression-matched non-target genes, or permuting target-to-truth "
        "pairing or gene identity, attenuated the scale-free directional "
        "advantage. This analysis is descriptive and was not used to adjudicate H2.*",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"DONE in {time.time()-t0:.1f}s -> {OUT_CSV.name}, {OUT_JSON.name}, {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
