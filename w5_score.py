#!/usr/bin/env python3
"""W5 preregistered T1/T2/T3 scoring harness.

THREE MODES (strictly separated on purpose)
------------------------------------------
  --mode self-test   Synthetic arrays only. Verifies the algebra and the exact
                     equivalence of the two accelerated routines below against
                     the preregistered core. Touches NO real data, NO model.

  --mode preflight   Real-data alignment + BASELINE-ONLY calibration on the
                     frozen 17,555-gene axis. NO model is loaded, NO model
                     score, NO hypothesis test, NO p-value for H1/H2/H3.
                     Produces the frozen scoring plan and the hESC control-cell
                     pool used later by T3.

  --mode smoke       Executes the ENTIRE scoring loop with deterministic
                     SYNTHETIC adapters (no torch, no checkpoint, no model
                     score). Writes only to w5_smoke_*.csv/json and NEVER to
                     w5_scores.csv. Proves the loop, projection, BH, CIs and
                     summary before the real run. Compliance gate untouched.

  --mode verify      Loads the two REAL preregistered adapters and runs
                     predict_delta on the frozen 44-target set with hESC control
                     cells. Checks shape/finiteness/non-degeneracy/determinism
                     ONLY; records array SHA256, no metric, no baseline, no
                     p-value, no model score. Engineering verification.

  --mode score       The full preregistered run (5 seeds x 2 models x 3
                     baselines, permutation nulls, BH, bootstrap CIs).
                     *** GATED ***: refuses to execute unless
                     results/w5_amendment_decision.json exists with a recorded
                     registrant decision. This makes the compliance gate a
                     property of the code, not of a note in a document.

WHY THE GATE EXISTS
-------------------
The adapter protocol was revised v1 -> v2 (target-gene TOKEN ABLATION instead
of zeroing the expression value) after the v1 run produced identically zero
deltas on real cells, but BEFORE any model score existed. The registrant (the
author/editor) must record whether that implementation-level revision requires
a timestamped OSF registration update, and record the conclusion either way.
Model scores are irreversible with respect to preregistration prospectivity, so
`--mode score` will not run until that decision is on disk. See
``osf-preregistration-draft.md`` section 11.

HONESTY / SCOPE
---------------
Nothing in this file fills a ``[[R#]]`` slot. W4 numbers, W5 engineering
verification and the preflight calibration are NOT model performance and must
never be written into Results. All outputs carry an explicit status label.

Author: (to be filled)   Python: 3.13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from embryo_vc_bench import (
    PRESET, DeltaSet, bh_adjust, bootstrap_ci, check_fold_leakage,
    des_score, leave_one_atlas_out, mae, make_surrogate_effects,
    no_change_baseline, pds_score, permutation_pvalue, train_mean_baseline,
    _topk_genes, _row_cosine,
)
from w5_gene_space import build_gene_space, load_all, DATA, RESULTS

ROOT = Path(__file__).resolve().parent
TRAIN_DE = DATA / "VCC_competition_train_DE_parquet__competition_train.parquet"
VAL_DE = DATA / "VCC_competition_val_DE_parquet__competition_val.parquet"
H5AD = DATA / "VCC_validation_h5ad_GCS__adata_Validation.h5ad"

PLAN_FILE = RESULTS / "w5_scoring_plan.json"
DECISION_FILE = RESULTS / "w5_amendment_decision.json"
CONTROL_POOL = RESULTS / "w5_t3_control_cells.npz"
SHARED_FEATURES = RESULTS / "w5_shared_features.json"

SEEDS = tuple(PRESET["seeds"])
B_PERM = int(PRESET["n_permutations"])
N_BOOT = int(PRESET["n_bootstrap"])
TOPK = int(PRESET["de_topk"])
MIN_FOLDS = int(PRESET["min_atlas_folds"])

CONTROL_POOL_SIZE = 64          # fixed, seed-0 deterministic draw from h5ad
CONTROL_PER_SEED = 8            # cells fed to each adapter call (CPU-bound)

STATUS_CALIBRATION = (
    "BASELINE-ONLY CALIBRATION - NOT A MODEL SCORE, NOT A PREREGISTERED "
    "HYPOTHESIS TEST, NO H2 CONCLUSION, NO [[R#]] SLOT FILLED"
)
STATUS_SELFTEST = (
    "SYNTHETIC SELF-TEST - ALGEBRA VERIFICATION ONLY, NO REAL DATA, "
    "NO MODEL, NO RESULT"
)


# ---------------------------------------------------------------------------
# 1. Memory-safe, algebraically IDENTICAL surrogate construction
# ---------------------------------------------------------------------------

def surrogate_effects_rows(expr: np.ndarray,
                           target_idx,
                           strength: float = 1.0) -> np.ndarray:
    """Exact equivalent of ``make_surrogate_effects`` without the n_gene^2 matrix.

    ``make_surrogate_effects`` calls ``np.corrcoef(x, rowvar=False)``, which
    materialises an (n_gene, n_gene) matrix: at 17,555 genes that is ~2.46 GB
    of float64 plus corrcoef's own temporaries. Only ``n_target`` ROWS of that
    matrix are ever used. Computing just those rows is O(n_target * n_cell *
    n_gene) memory O(n_cell * n_gene) and is algebraically identical:

        corr[g, j] = (c_g . c_j) / (||c_g|| ||c_j||),  c = x - mean(x, axis=0)

    which is exactly what np.corrcoef computes (its ddof cancels in the ratio).
    Zero-variance columns give NaN -> 0.0, matching ``np.nan_to_num`` in the
    core; values are clipped to [-1, 1] as np.corrcoef does.

    Equivalence is asserted against the core function on synthetic data by
    ``verify_surrogate_equivalence`` before any real-data use.
    """
    x = np.asarray(expr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expr must be 2-D (n_cell, n_gene)")
    n_cell, n_gene = x.shape
    c = x - x.mean(axis=0, keepdims=True)
    ss = np.sqrt(np.einsum("ij,ij->j", c, c))
    targets = [int(g) for g in target_idx]
    out = np.zeros((len(targets), n_gene), dtype=np.float64)
    for i, g in enumerate(targets):
        if not (0 <= g < n_gene):
            raise IndexError(f"target_idx out of range: {g}")
        with np.errstate(divide="ignore", invalid="ignore"):
            row = (c[:, g] @ c) / (ss[g] * ss)
        row = np.clip(np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        magnitude = float(np.mean(np.abs(x[:, g])))
        out[i] = -float(strength) * np.abs(row) * magnitude
        out[i, g] = -float(strength) * magnitude
    return out


# ---------------------------------------------------------------------------
# 2. Exact, fast permutation nulls
# ---------------------------------------------------------------------------

def fast_permutation_pvalues(pred: np.ndarray, true: np.ndarray,
                             B: int, seed: int) -> dict[str, float]:
    """Exact equivalent of two ``permutation_pvalue`` calls, ~1000x faster.

    A permutation of the ROWS of ``true`` does not change any row's top-k gene
    set, nor the pairwise cosine matrix; both are merely reordered. We therefore
    precompute them once and only reorder indices for each of the B draws.
    The RNG and the (B+1) correction are identical to the core function, so the
    returned p-values are bit-for-bit the same (asserted in self-test).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("shape mismatch")
    n, g = t.shape
    if n < 2:
        return {"p_perm_DES": float("nan"), "p_perm_PDS": float("nan")}
    k = min(TOPK, g)

    # DES ingredients (mirror des_score exactly)
    pt = [_topk_genes(p[i], k) if np.any(p[i] != 0) else set() for i in range(n)]
    tt = [_topk_genes(t[i], k) for i in range(n)]
    obs_des = float(np.mean([len(pt[i] & tt[i]) / k for i in range(n)])) if k else 0.0

    # PDS ingredients (mirror pds_score exactly)
    sim = _row_cosine(p, t)                    # (n, n), computed ONCE
    def pds_from_row(row: np.ndarray, i: int) -> float:
        if np.allclose(row, row[0], atol=1e-12):
            return 0.5
        return float(np.mean(row[i] > np.delete(row, i)))
    obs_pds = float(np.mean([pds_from_row(sim[i], i) for i in range(n)]))

    rng = np.random.default_rng(seed)
    null_des = np.empty(B, dtype=np.float64)
    null_pds = np.empty(B, dtype=np.float64)
    for b in range(B):
        perm = rng.permutation(n)
        null_des[b] = (float(np.mean([len(pt[i] & tt[perm[i]]) / k for i in range(n)]))
                       if k else 0.0)
        null_pds[b] = float(np.mean(
            [pds_from_row(sim[i, perm], i) for i in range(n)]))
    # Both DES and PDS use higher_is_better=True in evaluate_one_fold.
    return {
        "p_perm_DES": float((np.sum(null_des >= obs_des) + 1) / (B + 1)),
        "p_perm_PDS": float((np.sum(null_pds >= obs_pds) + 1) / (B + 1)),
    }


# ---------------------------------------------------------------------------
# 3. Label harmonisation (T1) - recorded, not silently discarded
# ---------------------------------------------------------------------------

LINEAGE_MAP = {
    "trophectoderm": "TE", "epiblast": "EPI", "primitive endoderm": "PE",
    "TE": "TE", "EPI": "EPI", "PE": "PE",
}
STAGE_MAP = {
    "embryonic day 3": "D3", "embryonic day 4": "D4",
    "embryonic day 5": "D5", "embryonic day 6": "D6",
    "embryonic day 7": "D7",
    "D6": "D6", "D7": "D7", "D8": "D8", "D9": "D9", "D10": "D10",
    "D12": "D12", "D14": "D14",
}


def harmonise_report(atlases: dict) -> dict:
    """Report harmonised label coverage per atlas. No label is dropped silently."""
    out = {}
    for name, a in atlases.items():
        lin = {LINEAGE_MAP.get(str(x).strip(), "") for x in a.lineage}
        lin.discard("")
        stg = {STAGE_MAP.get(str(x).strip(), "") for x in a.stage}
        stg.discard("")
        out[name] = {
            "n_cells": int(a.counts.shape[0]),
            "lineage_labels_raw": sorted({str(x) for x in set(a.lineage) if str(x)}),
            "lineage_labels_harmonised": sorted(lin),
            "lineage_cells_usable": int(sum(
                LINEAGE_MAP.get(str(x).strip(), "") != "" for x in a.lineage)),
            "stage_labels_raw": sorted({str(x) for x in set(a.stage) if str(x)}),
            "stage_labels_harmonised": sorted(stg),
            "stage_cells_usable": int(sum(
                STAGE_MAP.get(str(x).strip(), "") != "" for x in a.stage)),
        }
    usable_lineage = sorted({n for n, v in out.items() if v["lineage_cells_usable"] > 0})
    usable_stage = sorted({n for n, v in out.items() if v["stage_cells_usable"] > 0})
    out["_summary"] = {
        "atlases_with_lineage_labels": usable_lineage,
        "atlases_with_stage_labels": usable_stage,
        "shared_lineage_classes": sorted(set.intersection(*[
            set(out[n]["lineage_labels_harmonised"]) for n in usable_lineage]))
            if len(usable_lineage) >= 2 else [],
        "shared_stage_classes": sorted(set.intersection(*[
            set(out[n]["stage_labels_harmonised"]) for n in usable_stage]))
            if len(usable_stage) >= 2 else [],
    }
    return out


# ---------------------------------------------------------------------------
# 4. T3 alignment: VCC DE tables -> the fixed 17,555-gene axis
# ---------------------------------------------------------------------------

def align_de_to_axis(df: pd.DataFrame, gs,
                     shared_features: list[str]) -> tuple[np.ndarray, list[str], dict]:
    """Pivot a VCC DE table onto ONE shared feature axis; missing cells -> 0.0.

    Both train and val MUST be reindexed onto the same column list, otherwise
    ``train_mean_baseline`` returns a differently shaped prediction and the
    metrics cannot be computed. The shared axis is the intersection of the two
    DE tables' features with the frozen 17,555-gene axis (W4 used the same
    intersection; here it is additionally restricted to the gene axis).

    Missing (target, feature) rows are ABSENT MEASUREMENTS, not biological
    zeros; the fill is explicit and the observed density is reported.
    """
    targets = sorted(df["target"].astype(str).unique())
    feats = set(df["feature"].astype(str))
    wide = df.pivot_table(index="target", columns="feature",
                          values="log2_fold_change", aggfunc="first")
    present = wide.reindex(index=targets, columns=shared_features)
    observed = present.notna().to_numpy().sum()
    mat = present.fillna(0.0).to_numpy(dtype=np.float64)
    info = {
        "n_targets": len(targets),
        "n_features_in_table": len(feats),
        "n_shared_features_used": len(shared_features),
        "features_lost_to_shared_axis": len(feats) - len(shared_features),
        "matrix_shape": list(mat.shape),
        "observed_density": float(observed / mat.size),
        "missing_cells_filled_with": 0.0,
    }
    return mat, targets, info


# ---------------------------------------------------------------------------
# 5. Scoring primitives (identical metric definitions for models AND baselines)
# ---------------------------------------------------------------------------

def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Per-perturbation z-scoring; constant rows stay zero (no NaN)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return np.divide(x - mu, sd, out=np.zeros_like(x), where=sd > 1e-12)


def score_all_scaled(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Report RAW MAE (always, separately), MAE_z, DES and PDS.

    Neither adapter delta is on a log2FC scale and the two are not on a common
    scale with each other or with the CRISPRi ground truth (protocol v2 warning).
    Raw MAE is therefore NOT comparable across methods with different scales;
    it is reported because the preregistration requires MAE to appear separately
    so a composite score cannot mask poor absolute error. MAE_z is computed on
    per-perturbation z-scores and is the scale-free counterpart. DES/PDS are
    scale-free by construction.
    """
    return {
        "MAE": mae(pred, true),
        "MAE_z": mae(zscore_rows(pred), zscore_rows(true)),
        "DES": des_score(pred, true, topk=TOPK),
        "PDS": pds_score(pred, true),
    }


def score_methods(predictions: dict[str, np.ndarray], true: np.ndarray,
                  B: int, seed: int) -> list[dict]:
    """Score every method on identical rows; mandatory baselines enforced."""
    required = {"no_change", "train_mean"}
    if not required.issubset(predictions):
        raise KeyError(f"missing mandatory baselines: {sorted(required - set(predictions))}")
    rows = []
    for method, pred in predictions.items():
        s = score_all_scaled(pred, true)
        pv = fast_permutation_pvalues(pred, true, B=B, seed=seed)
        rows.append({"method": method, **s, **pv})
    # BH across the pre-specified family within this evaluation block
    for col_in, col_out in (("p_perm_DES", "p_adj_DES"),
                            ("p_perm_PDS", "p_adj_PDS")):
        adj = bh_adjust([r[col_in] for r in rows])
        for r, v in zip(rows, adj):
            r[col_out] = float(v)
    return rows


def baseline_predictions(train_effects: np.ndarray,
                         true: np.ndarray) -> dict[str, np.ndarray]:
    """Two mandatory baselines. ``additive`` is NOT APPLICABLE here: no
    combinatorial perturbations exist in either VCC DE table (W4 finding), and
    the core raises rather than imputing missing singles."""
    return {
        "no_change": no_change_baseline(true),
        "train_mean": train_mean_baseline(train_effects, true.shape[0]),
    }


# ---------------------------------------------------------------------------
# 6. Self-tests (synthetic only)
# ---------------------------------------------------------------------------

def verify_surrogate_equivalence(rng) -> dict:
    worst = 0.0
    cases = []
    for n_cell, n_gene, n_target in ((30, 45, 4), (12, 60, 6), (40, 25, 3)):
        x = rng.uniform(0, 8, size=(n_cell, n_gene))
        x[:, 0] = 0.0                     # zero-variance column -> NaN in corrcoef
        x[0, :] = x[0, :] * 0 + 1.0     # near-constant row
        tidx = rng.choice(n_gene, size=n_target, replace=False).tolist()
        a = make_surrogate_effects(x, tidx, strength=1.0)
        b = surrogate_effects_rows(x, tidx, strength=1.0)
        d = float(np.max(np.abs(a - b)))
        cases.append({"shape": [n_cell, n_gene], "n_target": n_target,
                      "max_abs_diff": d, "equal": bool(np.allclose(a, b, atol=1e-10))})
        worst = max(worst, d)
    return {"all_pass": bool(all(c["equal"] for c in cases)),
            "worst_max_abs_diff": worst, "cases": cases}


def verify_permutation_equivalence(rng) -> dict:
    """The fast nulls must reproduce the core's p-values exactly, including RNG."""
    n, g, B = 9, 40, 40
    true = rng.normal(0, 1, size=(n, g))
    pred = true * 0.4 + rng.normal(0, 1, size=(n, g))
    cases = []
    for seed in (0, 3, 7):
        # Core uses higher_is_better=True for both DES and PDS in
        # evaluate_one_fold; reproduce that exact call here.
        core_des = permutation_pvalue(des_score, pred, true, B=B, seed=seed,
                                      higher_is_better=True)
        core_pds = permutation_pvalue(pds_score, pred, true, B=B, seed=seed,
                                      higher_is_better=True)
        fast = fast_permutation_pvalues(pred, true, B=B, seed=seed)
        cases.append({
            "seed": seed,
            "core_DES": core_des, "fast_DES": fast["p_perm_DES"],
            "core_PDS": core_pds, "fast_PDS": fast["p_perm_PDS"],
        })
    return {"all_pass": bool(all(
        c["core_DES"] == c["fast_DES"] and c["core_PDS"] == c["fast_PDS"]
        for c in cases)), "cases": cases,
        "note": "PDS must use higher_is_better=True to match evaluate_one_fold"}


def run_self_test() -> dict:
    rng = np.random.default_rng(20261118)
    sur = verify_surrogate_equivalence(rng)
    perm = verify_permutation_equivalence(rng)

    # leakage check on 3 synthetic atlases
    atl = {n: DeltaSet(rng.normal(0, 1, (4, 6)), [f"t{i}" for i in range(4)],
                       [f"g{i}" for i in range(6)], atlas=n)
           for n in ("A", "B", "C")}
    folds = leave_one_atlas_out(atl, min_folds=MIN_FOLDS, seeds=SEEDS)
    leakage = bool(all(check_fold_leakage(f) for f in folds))

    # baseline + BH + bootstrap smoke test
    tr = rng.normal(0, 1, (6, 6))
    te = rng.normal(0, 1, (4, 6))
    preds = baseline_predictions(tr, te)
    rows = score_methods(preds, te, B=50, seed=0)
    ci = bootstrap_ci([r["DES"] for r in rows], B=200, seed=0)
    checks = {
        "surrogate_equivalence": sur["all_pass"],
        "permutation_equivalence": perm["all_pass"],
        "all_folds_leakage_free": leakage,
        "n_folds": len(folds),
        "n_distinct_test_atlases": len({f.test_atlas for f in folds}),
        "mandatory_baselines_present": {"no_change", "train_mean"}.issubset(preds),
        "no_change_DES_is_zero": bool(rows[0]["DES"] == 0.0),
        "bh_values_in_unit_interval": bool(all(
            0.0 <= r[k] <= 1.0 for r in rows for k in ("p_adj_DES", "p_adj_PDS"))),
        "bootstrap_ci_ordered": bool(np.isfinite(ci).all() and ci[0] <= ci[1]),
    }
    return {
        "status": STATUS_SELFTEST,
        "generated_at": _utc(),
        "checks": checks,
        "all_pass": bool(all(v for k, v in checks.items()
                             if isinstance(v, bool))),
        "surrogate_detail": sur,
        "permutation_detail": perm,
    }


# ---------------------------------------------------------------------------
# 7. Preflight (real data, baselines only)
# ---------------------------------------------------------------------------

def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _lognorm(counts: np.ndarray) -> np.ndarray:
    """log1p(CP10K) for the SURROGATE ground truth only.

    The preregistration requires RAW COUNTS to be retained for MODEL input;
    this normalisation is applied only inside ``make_surrogate_effects``, whose
    own docstring specifies log-normalised counts for real data. Models in
    --mode score always receive raw counts.
    """
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib * 1e4).astype(np.float64)


def build_scoring_plan(gs, targets: list[str], harmon: dict,
                       de_info: dict) -> dict:
    return {
        "plan_version": "w5-scoring-plan-1",
        "status": "FROZEN BEFORE ANY MODEL SCORE; no [[R#]] slot filled",
        "generated_at": _utc(),
        "gene_axis": {
            "n_genes": gs.n_genes,
            "fingerprint": gs.fingerprint,
            "rule": "intersection of ARC gene_names / scGPT vocab / Geneformer V1 tokens",
        },
        "adapter_protocol": "w5-adapter-protocol-2 (results/w5_protocol.json)",
        "targets": {
            "rule": "the 44 VCC 2025 validation CRISPRi targets, all present in the fixed gene axis",
            "n_targets": len(targets),
            "list": targets,
            "added_or_removed_after_freeze": False,
        },
        "seeds": list(SEEDS),
        "permutations_B": B_PERM,
        "bootstrap_B": N_BOOT,
        "alpha": PRESET["alpha"],
        "multiple_testing": PRESET["multiple_testing"],
        "de_topk": TOPK,
        "metrics": {
            "MAE": "raw mean absolute error, ALWAYS reported separately; not "
                   "comparable across methods with different delta scales",
            "DES": "scale-free, mean top-k overlap (framework-internal)",
            "PDS": "scale-free, normalised cosine rank (framework-internal)",
            "MAE_z": "MAE after PER-PERTURBATION z-scoring of both matrices; "
                     "adapter deltas are NOT on a log2FC scale",
        },
        "baselines": {
            "no_change": "mandatory",
            "train_mean": "mandatory, mean effect across training perturbations",
            "additive": "NOT APPLICABLE - no combinatorial perturbations exist in "
                        "the VCC DE tables (W4); missing singles are never imputed",
        },
        "hard_rule": "no model score without its mandatory baselines",
        "T1": {
            "fold_rule": "leave-one-atlas-out over licence-verified atlases",
            "n_independent_atlases": 3,
            "n_distinct_test_atlases": 3,
            "min_folds_required": MIN_FOLDS,
            "H1_status": "DEMOTED TO EXPLORATORY by the preregistered rule "
                         "('If fewer than 5 independent atlases are available, "
                         "H1 is demoted to exploratory')",
            "tau_status": "UNAVAILABLE - W3 produced a CPU feasibility prototype "
                          "only; the preregistered scVI/scANVI shared latent time "
                          "axis tau has NOT been fitted, so the registered H1 "
                          "macro-F1 label-transfer test cannot be executed and "
                          "must not be imputed",
            "label_harmonisation": harmon["_summary"],
        },
        "T2": {
            "ground_truth": "SURROGATE IN-SILICO PERTURBATION, NOT EXPERIMENTAL "
                            "PERTURBATION: target-gene masking propagated along a "
                            "co-expression matrix, on log1p(CP10K) expression",
            "implementation": "row-wise correlation, algebraically identical to "
                              "make_surrogate_effects (self-tested); avoids the "
                              "17,555^2 matrix",
            "cells": "all QC-passing cells of each atlas; raw counts retained for models",
        },
        "T3": {
            "ground_truth": "real H1 hESC CRISPRi log2_fold_change from the CC0 "
                            "VCC 2025 validation DE parquet",
            "model_input": "non-targeting CONTROL cells from the repaired 6.9 GB "
                           "validation h5ad (generic cell-line context, as preregistered)",
            "control_pool_size": CONTROL_POOL_SIZE,
            "control_cells_per_seed": CONTROL_PER_SEED,
            "alignment": "DE tables pivoted onto the fixed 17,555-gene axis; "
                         "absent (target, feature) rows filled 0.0 with density reported",
            "de_alignment": de_info,
        },
        "scale_warning": "Neither adapter delta is on a log2FC scale and the two "
                         "are not on a common scale with each other or with the "
                         "CRISPRi ground truth. DES/PDS/MAE_z are scale-free; "
                         "raw MAE is reported separately but is NOT comparable "
                         "across methods with different delta scales.",
        "external_metrics": "cell-eval 0.8.2 official metrics are NOT computed "
                            "here; framework-internal DES/PDS/MAE/MAE_z must "
                            "never be compared numerically with VCC leaderboard "
                            "values",
    }


def compute_shared_features(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            gs) -> tuple[list[str], list[int]]:
    """Frozen T3 feature axis = train DE features ∩ val DE features ∩ gene axis."""
    common = (set(train_df["feature"].astype(str))
              & set(val_df["feature"].astype(str)) & set(gs.genes))
    feats = [g for g in gs.genes if g in common]       # deterministic gene-axis order
    pos = {g: i for i, g in enumerate(gs.genes)}
    return feats, [pos[g] for g in feats]


def run_preflight() -> dict:
    gs = build_gene_space()
    atlases = load_all(gs)
    harmon = harmonise_report(atlases)

    train_df = pd.read_parquet(TRAIN_DE)
    val_df = pd.read_parquet(VAL_DE)
    shared_feats, shared_pos = compute_shared_features(train_df, val_df, gs)
    SHARED_FEATURES.write_text(json.dumps({
        "status": "FROZEN T3 FEATURE AXIS - no [[R#]] slot filled",
        "generated_at": _utc(),
        "rule": "train DE features INTERSECT val DE features INTERSECT 17,555-gene axis",
        "n_features": len(shared_feats),
        "gene_axis_index": shared_pos,
        "features": shared_feats,
        "sha256": hashlib.sha256("|".join(shared_feats).encode()).hexdigest(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    train_mat, train_targets, train_info = align_de_to_axis(train_df, gs, shared_feats)
    val_mat, val_targets, val_info = align_de_to_axis(val_df, gs, shared_feats)
    assert all(t in gs.genes for t in val_targets), "a val target left the gene axis"
    assert train_mat.shape[1] == val_mat.shape[1] == len(shared_feats)

    target_pos = {g: i for i, g in enumerate(gs.genes)}
    tidx = [target_pos[t] for t in val_targets]

    # T2 surrogate ground truth per atlas (real cells, no model involved)
    surrogate = {}
    t2_info = {}
    for name, a in atlases.items():
        x = _lognorm(a.counts)
        d = surrogate_effects_rows(x, tidx, strength=1.0)
        surrogate[name] = d
        t2_info[name] = {
            "n_cells": int(a.counts.shape[0]),
            "delta_shape": list(d.shape),
            "finite": bool(np.isfinite(d).all()),
            "nonzero_rows": int(np.count_nonzero(np.any(d != 0, axis=1))),
            "delta_sha256": hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest(),
        }
        del x, d

    # T1 folds from the preregistered constructor, on surrogate DeltaSets
    ds = {n: DeltaSet(surrogate[n], val_targets, list(gs.genes), atlas=n,
                      domain="embryo",
                      metadata={"kind": "surrogate in-silico perturbation"})
          for n in surrogate}
    folds = leave_one_atlas_out(ds, min_folds=MIN_FOLDS, seeds=SEEDS)
    leakage_free = bool(all(check_fold_leakage(f) for f in folds))
    distinct_tests = len({f.test_atlas for f in folds})

    # BASELINE-ONLY calibration, explicitly not a model score
    cal_rows = []
    for f in folds:
        true = np.vstack([surrogate[n] for n in sorted(ds)])[f.test_idx]
        train_stack = np.vstack([surrogate[n] for n in f.train_atlases])
        preds = baseline_predictions(train_stack, true)
        for r in score_methods(preds, true, B=B_PERM, seed=f.seed):
            cal_rows.append({"task": "T2_surrogate", **f.as_record(), **r})
    # T3 baseline calibration on the real DE ground truth
    preds3 = baseline_predictions(train_mat, val_mat)
    for seed in SEEDS:
        for r in score_methods(preds3, val_mat, B=B_PERM, seed=seed):
            cal_rows.append({"task": "T3_DE_real", "test_atlas": "VCC_val_DE",
                             "train_atlases": "VCC_train_DE", "n_train": len(train_targets),
                             "n_test": len(val_targets), "seed": seed, **r})

    cal = pd.DataFrame(cal_rows)
    cal.to_csv(RESULTS / "w5_preflight_baseline_calibration.csv", index=False)

    # hESC control-cell pool for T3 model input (data preparation, no model)
    control_info = export_control_pool(gs)

    plan = build_scoring_plan(gs, val_targets, harmon,
                              {"train": train_info, "val": val_info})
    PLAN_FILE.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    out = {
        "status": STATUS_CALIBRATION,
        "generated_at": _utc(),
        "no_model_loaded": True,
        "no_model_score": True,
        "no_p_value_for_any_preregistered_hypothesis": True,
        "no_result_slot_filled": True,
        "gene_space": gs.summary(),
        "label_harmonisation": harmon,
        "de_alignment": {"train": train_info, "val": val_info},
        "t2_surrogate": t2_info,
        "t1_folds": {
            "n_folds": len(folds),
            "n_distinct_test_atlases": distinct_tests,
            "min_required": MIN_FOLDS,
            "all_leakage_free": leakage_free,
            "H1_demoted_to_exploratory": distinct_tests < MIN_FOLDS,
            "tau_available": False,
        },
        "baseline_calibration_summary": {
            "n_rows": int(len(cal)),
            "methods": sorted(cal["method"].unique().tolist()),
            "tasks": sorted(cal["task"].unique().tolist()),
            "mean_MAE_z_by_task_method": {
                f"{t}|{m}": float(cal[(cal.task == t) & (cal.method == m)]["MAE_z"].mean())
                for t in sorted(cal.task.unique()) for m in sorted(cal.method.unique())},
            "note": "these are BASELINE numbers, not model performance; H2 untested",
        },
        "hESC_control_pool": control_info,
        "scoring_plan": str(PLAN_FILE.name),
        "source_sha256": {TRAIN_DE.name: _sha(TRAIN_DE), VAL_DE.name: _sha(VAL_DE)},
    }
    (RESULTS / "w5_preflight.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def export_control_pool(gs) -> dict:
    """Deterministic seed-0 draw of non-targeting hESC control cells -> .npz.

    The repaired 6.9 GB h5ad is read in backed mode; only 64 rows are pulled
    into memory. Raw counts are projected onto the fixed gene axis. This is
    DATA PREPARATION, not a model run: the file is consumed by --mode score.
    """
    import anndata as ad
    a = ad.read_h5ad(H5AD, backed="r")
    is_ctrl = a.obs["target_gene"].astype(str).to_numpy() == "non-targeting"
    idx = np.flatnonzero(is_ctrl)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(idx, size=CONTROL_POOL_SIZE, replace=False))
    X = a[pick].X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    var_pos = {str(g): j for j, g in enumerate(a.var_names)}
    out = np.zeros((len(pick), gs.n_genes), dtype=np.float32)
    mapped = 0
    for i, g in enumerate(gs.genes):
        if g in var_pos:
            out[:, i] = X[:, var_pos[g]]
            mapped += 1
    np.savez_compressed(CONTROL_POOL, counts=out,
                        genes=np.array(gs.genes, dtype="U"),
                        cell_indices=pick)
    return {
        "file": CONTROL_POOL.name,
        "n_control_cells": int(len(pick)),
        "shape": list(out.shape),
        "genes_mapped_from_h5ad": mapped,
        "genes_zero_filled": int(gs.n_genes - mapped),
        "finite": bool(np.isfinite(out).all()),
        "nonnegative": bool((out >= 0).all()),
        "sha256": _sha(CONTROL_POOL),
        "selection": "numpy default_rng(0), sorted, from obs.target_gene == 'non-targeting'",
        "note": "raw counts on the fixed gene axis; DATA PREPARATION ONLY, no model run",
    }


# ---------------------------------------------------------------------------
# 8. Gated full run
# ---------------------------------------------------------------------------

def read_decision() -> dict:
    if not DECISION_FILE.exists():
        raise SystemExit(
            "GATE: results/w5_amendment_decision.json is missing.\n"
            "Preflight found FOUR deviations from the frozen OSF plan that the "
            "registrant (author/editor) must resolve BEFORE the first model "
            "score: (D1) only 3 licence-verified independent atlases -> 3 < 5 "
            "leave-one-atlas-out folds -> H1 demoted to exploratory; (D2) the "
            "shared latent time axis tau was NEVER fitted -> H1 macro-F1 label "
            "transfer and the entire H3 cannot be executed, [[R3]]/[[R5]] stay "
            "placeholders; (D3) no combinatorial perturbations -> additive "
            "baseline NOT APPLICABLE; (D4) official cell-eval ANNDATA_PAIR "
            "metrics are not applicable to adapter representation shifts. "
            "Also disclose adapter protocol v1->v2 and the frozen scoring "
            "specifications. Either conclusion (amendment_filed or "
            "no_amendment_required) is acceptable, but it MUST be recorded "
            "with decided_by / recorded_at / rationale. See "
            "osf-preregistration-draft.md section 11.\n"
            "No model was loaded and no score was produced.")
    d = json.loads(DECISION_FILE.read_text(encoding="utf-8"))
    if d.get("decision") not in ("amendment_filed", "no_amendment_required"):
        raise SystemExit("GATE: decision value is invalid; no score produced.")
    for k in ("decided_by", "recorded_at", "rationale"):
        if not d.get(k):
            raise SystemExit(f"GATE: decision record is missing '{k}'; no score produced.")
    if d["decision"] == "amendment_filed" and not d.get("osf_update_timestamp_utc"):
        raise SystemExit("GATE: amendment_filed requires osf_update_timestamp_utc; "
                         "no score produced.")
    return d


class _SyntheticAdapter:
    """Deterministic stand-in used ONLY by --mode smoke. Never a model score."""

    name = "SYNTHETIC"

    def __init__(self, n_genes: int, offset: float):
        self.n_genes = n_genes
        self.offset = offset

    def predict_delta(self, counts, perturbation):
        rng = np.random.default_rng(777)
        d = rng.normal(0.0, 1.0, size=(len(perturbation), self.n_genes))
        d = d * np.abs(counts.mean(axis=0))[None, :] + self.offset
        return np.ascontiguousarray(d.astype(np.float32))


def run_verify() -> dict:
    """Engineering check that both real adapters accept the frozen 44 targets.

    No metric, no baseline comparison, no permutation, no p-value, no score and
    no ``[[R#]]`` slot. Only array SHA256 fingerprints are persisted.
    """
    from w5_adapters import build_adapters, PROTOCOL_VERSION
    from w5_gene_space import build_gene_space
    gs = build_gene_space()
    sf = json.loads(SHARED_FEATURES.read_text(encoding="utf-8"))
    val_targets = sorted(pd.read_parquet(VAL_DE, columns=["target"])["target"]
                         .astype(str).unique())
    pool = np.load(CONTROL_POOL)
    X = np.ascontiguousarray(pool["counts"][:CONTROL_PER_SEED])
    adapters = build_adapters(gs, seed=0)
    out = {
        "status": ("ENGINEERING VERIFICATION - REAL ADAPTERS, NO MODEL SCORE, "
                   "NO METRIC, NO BASELINE, NO P-VALUE, NO [[R#]] SLOT FILLED"),
        "generated_at": _utc(),
        "protocol_version": PROTOCOL_VERSION,
        "gene_axis": {"n_genes": gs.n_genes, "fingerprint": gs.fingerprint},
        "n_targets": len(val_targets),
        "n_control_cells": int(X.shape[0]),
        "models": {},
    }
    for name, m in adapters.items():
        t0 = time.time()
        d = m.predict_delta(X, val_targets)
        d2 = m.predict_delta(X, val_targets)
        out["models"][name] = {
            "shape": list(d.shape),
            "expected_shape": [len(val_targets), gs.n_genes],
            "shape_ok": bool(d.shape == (len(val_targets), gs.n_genes)),
            "finite": bool(np.isfinite(d).all()),
            "nonzero_rows": int(np.count_nonzero(np.any(d != 0, axis=1))),
            "deterministic_max_abs_diff": float(np.max(np.abs(d - d2))),
            "delta_sha256": hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest(),
            "seconds": round(time.time() - t0, 2),
        }
    out["all_pass"] = bool(all(
        v["shape_ok"] and v["finite"] and v["nonzero_rows"] > 0
        and v["deterministic_max_abs_diff"] == 0.0
        for v in out["models"].values()))
    (RESULTS / "w5_verify_real_adapters.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_score(smoke: bool = False) -> dict:
    """Full scoring loop. The REAL run is reached ONLY after the compliance gate.

    ``smoke=True`` replaces both preregistered models with deterministic
    synthetic stand-ins and writes exclusively to w5_smoke_* outputs. It
    exercises every branch (T2 folds, T3 seeds, projection, permutation, BH,
    bootstrap CIs, summary) so that an eventual real run has no untested code
    path, while producing NO model score and NOT satisfying the gate.
    """
    if smoke:
        decision = {"decision": "N/A_SMOKE", "decided_by": "code",
                    "recorded_at": _utc(), "rationale":
                    "synthetic-adapter pipeline verification; not a registrant decision"}
        protocol_version = "SYNTHETIC_ADAPTER_SMOKE"
    else:
        decision = read_decision()
    if not PLAN_FILE.exists():
        raise SystemExit("GATE: run --mode preflight first (frozen plan missing).")
    if not CONTROL_POOL.exists():
        raise SystemExit("GATE: run --mode preflight first (control pool missing).")
    if not SHARED_FEATURES.exists():
        raise SystemExit("GATE: run --mode preflight first (shared feature axis missing).")

    if smoke:
        from w5_gene_space import build_gene_space, load_all
        build_smoke = lambda gs, seed=0: {
            "SYNTHETIC_A": _SyntheticAdapter(gs.n_genes, 0.1),
            "SYNTHETIC_B": _SyntheticAdapter(gs.n_genes, -0.1)}
        build_adapters = build_smoke
    else:
        from w5_adapters import build_adapters, PROTOCOL_VERSION as protocol_version
        from w5_gene_space import build_gene_space, load_all

    gs = build_gene_space()
    atlases = load_all(gs)
    val_df = pd.read_parquet(VAL_DE)
    train_df = pd.read_parquet(TRAIN_DE)
    sf = json.loads(SHARED_FEATURES.read_text(encoding="utf-8"))
    shared_feats, shared_pos = sf["features"], sf["gene_axis_index"]
    assert shared_feats == compute_shared_features(train_df, val_df, gs)[0], \
        "shared feature axis changed since preflight; refusing to score"
    val_mat, val_targets, _ = align_de_to_axis(val_df, gs, shared_feats)
    train_mat, train_targets, _ = align_de_to_axis(train_df, gs, shared_feats)
    assert all(t in gs.genes for t in val_targets)

    pool = np.load(CONTROL_POOL)
    pool_X = pool["counts"]
    target_pos = {g: i for i, g in enumerate(gs.genes)}
    tidx = [target_pos[t] for t in val_targets]
    surrogate = {n: surrogate_effects_rows(_lognorm(a.counts), tidx)
                 for n, a in atlases.items()}
    ds = {n: DeltaSet(surrogate[n], val_targets, list(gs.genes), atlas=n)
          for n in surrogate}
    folds = leave_one_atlas_out(ds, min_folds=MIN_FOLDS, seeds=SEEDS)
    stacked = np.vstack([surrogate[n] for n in sorted(ds)])

    adapters = build_adapters(gs, seed=0)   # zero-shot, deterministic
    all_rows = []
    audit = {"decision": decision, "protocol_version": protocol_version,
             "started_at": _utc(), "models": sorted(adapters),
             "n_folds": len(folds), "errors": [], "smoke": bool(smoke)}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ctrl_pick = np.sort(rng.choice(pool_X.shape[0],
                                       size=CONTROL_PER_SEED, replace=False))
        ctrl = np.ascontiguousarray(pool_X[ctrl_pick])
        # T3 model deltas from hESC CONTROL cells (generic cell-line context)
        try:
            model_delta_t3 = {n: m.predict_delta(ctrl, val_targets)
                              for n, m in adapters.items()}
        except Exception as exc:                       # disclosure, never a silent drop
            audit["errors"].append({"seed": seed, "task": "T3", "error": repr(exc)})
            model_delta_t3 = {}
        # Score ALL methods in ONE block: otherwise the mandatory baselines are
        # recomputed once per model and would be duplicated in the summary.
        preds = baseline_predictions(train_mat, val_mat)
        for name, d3 in model_delta_t3.items():
            # Adapter deltas live on the full 17,555-gene axis; T3 ground truth
            # lives on the frozen shared-feature axis. Project explicitly.
            preds[name] = np.asarray(d3)[:, shared_pos]
            assert preds[name].shape == val_mat.shape
        for r in score_methods(preds, val_mat, B=B_PERM, seed=seed):
            all_rows.append({
                "task": "T3_DE_real", "seed": seed,
                "test_atlas": "VCC_val_DE", "train_atlases": "VCC_train_DE",
                "n_train": len(train_targets), "n_test": len(val_targets),
                **r})

        # T2 model deltas on the held-out atlas' own cells
        for f in [x for x in folds if x.seed == seed]:
            a = atlases[f.test_atlas]
            cell_pick = np.sort(rng.choice(a.counts.shape[0],
                                           size=min(CONTROL_PER_SEED, a.counts.shape[0]),
                                           replace=False))
            cells = np.ascontiguousarray(a.counts[cell_pick])
            try:
                md = {n: m.predict_delta(cells, val_targets)
                      for n, m in adapters.items()}
            except Exception as exc:
                audit["errors"].append({"seed": seed, "task": "T2",
                                        "test_atlas": f.test_atlas, "error": repr(exc)})
                md = {}
            true = stacked[f.test_idx]
            train_stack = np.vstack([surrogate[n] for n in f.train_atlases])
            preds = baseline_predictions(train_stack, true)
            preds.update(md)
            for r in score_methods(preds, true, B=B_PERM, seed=f.seed):
                all_rows.append({"task": "T2_surrogate", **f.as_record(), **r})

    df = pd.DataFrame(all_rows)
    score_csv = RESULTS / ("w5_smoke_scores.csv" if smoke else "w5_scores.csv")
    df.to_csv(score_csv, index=False)

    # bootstrap CIs over seeds/folds, per task x method x metric
    ci_rows = []
    for (task, method), sub in df.groupby(["task", "method"]):
        rec = {"task": task, "method": method, "n": int(len(sub))}
        for met in ("MAE", "MAE_z", "DES", "PDS"):
            vals = sub[met].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals, B=N_BOOT, seed=0)
            rec[f"{met}_mean"] = float(np.mean(vals))
            rec[f"{met}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{met}_ci95"] = [lo, hi]
        ci_rows.append(rec)
    summary_path = RESULTS / ("w5_smoke_summary.json" if smoke else "w5_score_summary.json")
    summary_path.write_text(json.dumps({
        "status": ("SYNTHETIC-ADAPTER PIPELINE SMOKE TEST - NO MODEL SCORE, "
                   "NO [[R#]] SLOT FILLED, COMPLIANCE GATE UNCHANGED"
                   if smoke else
                   "PREREGISTERED MODEL SCORES - draft, not yet released"),
        "generated_at": _utc(),
        "decision": decision,
        "audit": audit,
        "by_task_method": ci_rows,
        "h1_status": "demoted to exploratory (3 independent atlases < 5 folds)",
        "h2_status": "computed from the table above; no conclusion drafted here",
        "scale_warning": "DES/PDS scale-free; MAE_z is per-perturbation z-scored",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    audit["finished_at"] = _utc()
    return audit


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("self-test", "preflight", "verify",
                                        "smoke", "score"), default="self-test")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    if args.mode == "self-test":
        out = run_self_test()
        (RESULTS / "w5_self_test.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"mode": "self-test", "all_pass": out["all_pass"],
                          "checks": out["checks"]}, indent=2))
        return 0 if out["all_pass"] else 1
    if args.mode == "preflight":
        out = run_preflight()
        print(json.dumps({
            "mode": "preflight", "status": out["status"],
            "gene_axis": out["gene_space"]["n_genes"],
            "t1": out["t1_folds"],
            "de_alignment": out["de_alignment"],
            "baseline_rows": out["baseline_calibration_summary"]["n_rows"],
            "control_pool": out["hESC_control_pool"]["shape"],
            "elapsed_s": round(time.time() - t0, 1)}, indent=2))
        return 0
    if args.mode == "verify":
        out = run_verify()
        print(json.dumps({"mode": "verify", "all_pass": out["all_pass"],
                          "n_targets": out["n_targets"],
                          "models": {k: {kk: v[kk] for kk in
                                    ("shape", "finite", "nonzero_rows",
                                     "deterministic_max_abs_diff", "seconds")}
                                    for k, v in out["models"].items()},
                          "elapsed_s": round(time.time() - t0, 1)}, indent=2))
        return 0 if out["all_pass"] else 1
    if args.mode == "smoke":
        audit = run_score(smoke=True)
        print(json.dumps({"mode": "smoke", "status": audit["protocol_version"],
                          "models": audit["models"], "errors": audit["errors"],
                          "elapsed_s": round(time.time() - t0, 1)}, indent=2))
        return 0
    audit = run_score(smoke=False)
    print(json.dumps({"mode": "score", "audit": audit,
                      "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
