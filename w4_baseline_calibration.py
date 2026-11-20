#!/usr/bin/env python3
"""W4 baseline calibration on the preregistered, CC0 Arc VCC 2025 DE truth.

WHAT THIS IS
------------
A preregistered W4 analysis (OSF 8tzrq, DOI 10.17605/OSF.IO/8TZRQ) that
calibrates the *mandatory naive baselines* on real public DE tables, before any
model score is produced. It is the "no model score without baselines" step made
executable. It also runs the official Arc `cell-eval` **DE-type** metrics
directly on these DE tables via `initialize_de_comparison`, which does not
require a single-cell h5ad.

WHAT THIS IS NOT
----------------
* Not a biological discovery and not an embryo result.
* Not a virtual-cell model result: scGPT and Geneformer were NOT run. No number
  produced here may be reported as model performance or used to accept/reject
  H2.
* Not the full `cell-eval run` pipeline. Metrics of MetricType.ANNDATA_PAIR
  (`mae`, `mae_delta`, `discrimination_score_*`) require a valid h5ad; the
  6.9 GB validation h5ad failed the W1b content gate and was quarantined.
  Framework-internal MAE/DES/PDS are reported as delta-space counterparts with
  an explicit mapping table and are NOT numerically interchangeable with the
  official metrics.
* `additive` is not applicable: the VCC 2025 DE tables contain zero
  combinatorial perturbations. Its absence is recorded, not imputed.

Run:
    ./.venv/Scripts/python.exe w4_baseline_calibration.py

Outputs (results/): w4_baseline_calibration.json, w4_framework_metrics.csv,
    w4_official_de_metrics.csv, w4_agg_no_change.csv, w4_agg_train_mean.csv,
    w4_official_baseline_diff.csv, w4_alignment_map.csv, w4_run_log.txt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

import embryo_vc_bench as B
from cell_eval._score import score_agg_metrics
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import metrics_registry
from w1_fetch import DATA_DIR, ROOT, now, read_manifest

RESULTS = ROOT / "results"

TRAIN_DE = DATA_DIR / "VCC_competition_train_DE_parquet__competition_train.parquet"
VAL_DE = DATA_DIR / "VCC_competition_val_DE_parquet__competition_val.parquet"
GENES = DATA_DIR / "ARC_gene_names__gene_names.csv"

TRAIN_ID = "VCC_competition_train_DE_parquet"
VAL_ID = "VCC_competition_val_DE_parquet"

SEED = B.PRESET["seeds"][0]
N_PERM = B.PRESET["n_permutations"]
N_BOOT = B.PRESET["n_bootstrap"]
TOPK = B.PRESET["de_topk"]
ALPHA = B.PRESET["alpha"]

OFFICIAL_DE_METRICS = [
    "overlap_at_N", "overlap_at_50", "overlap_at_100",
    "precision_at_N", "precision_at_50",
    "de_direction_match", "de_spearman_lfc_sig", "de_sig_genes_recall",
    "de_spearman_sig", "pr_auc", "roc_auc",
]

# Deliberate, non-numerical mapping: these are conceptual counterparts only.
ALIGNMENT_MAP = [
    ("MAE", "mae / mae_delta", "ANNDATA_PAIR", "UNAVAILABLE",
     "Requires a valid single-cell h5ad; quarantined validation h5ad blocks it."),
    ("DES", "overlap_at_N / precision_at_N / de_sig_genes_recall", "DE", "AVAILABLE",
     "Both measure DE-gene recovery, but our DES is top-k by |LFC| without FDR filtering; "
     "official overlap ranks post-FDR significant genes. Not interchangeable."),
    ("PDS", "discrimination_score_cosine", "ANNDATA_PAIR", "UNAVAILABLE",
     "Official implementation operates on cell-level profiles, not DE tables."),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(entry_id: str) -> str:
    for row in read_manifest():
        if row["entry_id"] == entry_id:
            return row["sha256"]
    return ""


def load_and_align(train_df: pd.DataFrame,
                   val_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    """Pivot both DE tables onto a shared feature space.

    Missing (target, feature) cells are absent measurements, not zeros; they
    are filled with 0.0 (no predicted/observed change) and the coverage is
    reported so this imputation is never hidden.
    """
    features = sorted(set(train_df["feature"]) & set(val_df["feature"]))
    t_targets = sorted(train_df["target"].unique())
    v_targets = sorted(val_df["target"].unique())

    def pivot(df: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, float]:
        sub = df[df["feature"].isin(features)]
        wide = sub.pivot_table(index="target", columns="feature",
                               values="log2_fold_change", aggfunc="mean",
                               observed=True)
        wide = wide.reindex(index=targets, columns=features)
        coverage = float(wide.notna().to_numpy().mean())
        return wide.fillna(0.0).to_numpy(dtype=np.float64), coverage

    train_mat, train_cov = pivot(train_df, t_targets)
    val_mat, val_cov = pivot(val_df, v_targets)
    info = {
        "shared_features": len(features),
        "train_features": int(train_df["feature"].nunique()),
        "val_features": int(val_df["feature"].nunique()),
        "n_train_perturbations": len(t_targets),
        "n_val_perturbations": len(v_targets),
        "shared_targets": len(set(t_targets) & set(v_targets)),
        "train_density": round(train_cov, 6),
        "val_density": round(val_cov, 6),
        "missing_cell_policy": "absent (target, feature) rows filled with 0.0; density reported",
        "combinatorial_perturbations": sum(
            1 for t in set(t_targets) | set(v_targets)
            if any(s in t for s in ("+", ",", ";"))),
    }
    return train_mat, val_mat, t_targets, v_targets, info


def per_pert_metrics(pred: np.ndarray, true: np.ndarray,
                     topk: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-perturbation MAE and DES using the framework's operational rules."""
    mae_i = np.mean(np.abs(pred - true), axis=1)
    k = min(topk, true.shape[1])
    des_i = np.zeros(true.shape[0], dtype=np.float64)
    for i in range(true.shape[0]):
        pt = (set(np.argsort(-np.abs(pred[i]), kind="stable")[:k].tolist())
              if np.any(pred[i] != 0) else set())
        tt = set(np.argsort(-np.abs(true[i]), kind="stable")[:k].tolist())
        des_i[i] = len(pt & tt) / k if k else 0.0
    return mae_i, des_i


def training_derived_significance(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Operational per-gene effect + significance for the train-mean baseline.

    Derived ONLY from training perturbations (no validation labels). For each
    gene we take the mean log2FC over the perturbations where it was measured,
    a one-sample t statistic against zero, and a BH-adjusted q value. The q
    value is a ranking device for the baseline DE table, NOT a claim that a
    baseline prediction is statistically significant.
    """
    g = train_df.groupby("feature")["log2_fold_change"]
    n = g.size().astype(float)
    mu = g.mean()
    sd = g.std(ddof=1).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = mu / (sd / np.sqrt(n))
    p = 2.0 * stats.t.sf(np.abs(t_stat.to_numpy()), df=(n - 1).to_numpy())
    p = np.where(np.isfinite(p), p, 1.0)
    fdr = B.bh_adjust(p)
    return mu, pd.Series(p, index=mu.index, name="p_value"), pd.Series(
        fdr, index=mu.index, name="fdr")


def build_de_frame(rows: pd.DataFrame, lfc: pd.Series | float,
                   pval: pd.Series | float, fdr: pd.Series | float) -> pl.DataFrame:
    """Build a cell-eval-compatible DE table with the exact required columns."""
    out = pd.DataFrame({
        "target": rows["target"].to_numpy(),
        "feature": rows["feature"].to_numpy(),
    })
    for col, val in (("log2_fold_change", lfc), ("p_value", pval), ("fdr", fdr)):
        if isinstance(val, pd.Series):
            out[col] = out["feature"].map(val).fillna(0.0 if col == "log2_fold_change" else 1.0).to_numpy()
        else:
            out[col] = float(val)
    return pl.from_pandas(out[["target", "feature", "log2_fold_change", "p_value", "fdr"]])


def official_score_equivalent(user: dict[str, float], base: dict[str, float],
                              metrics: list[str]) -> tuple[list[dict], dict]:
    """Reproduce `cell_eval._score.score_agg_metrics` without touching site-packages.

    cell-eval 0.8.2 raises `IndexError: arrays used as indices must be of
    integer (or boolean) type` when *every* metric has MetricBestValue.ONE (or
    every metric has ZERO): the empty index list becomes `np.array([])` with
    dtype float64. All DE-type metrics in this registry are ONE-best, so the
    official CLI/API cannot score an all-DE comparison. We therefore apply the
    identical published formulas from `_score.py` here, using the registry's
    own `best_value`, and record the upstream failure in the audit JSON.
    """
    zero: list[str] = []
    one: list[str] = []
    best: dict[str, str] = {}
    for m in metrics:
        bv = metrics_registry.get_metric(m).best_value.value
        best[m] = bv
        if bv == "zero":
            zero.append(m)
        elif bv == "one":
            one.append(m)
    rows: list[dict] = []
    scores: list[float] = []
    for m in zero:  # _calc_norm_by_zero: 1 - user/base
        u, b = user[m], base[m]
        v = 1.0 - (u / b) if b not in (0.0,) else float("nan")
        rows.append({"metric": m, "normalization": "1 - user/base", "from_baseline": v})
        scores.append(0.0 if np.isnan(v) else v)
    for m in one:  # _calc_norm_by_one: (user - base) / (1 - base)
        u, b = user[m], base[m]
        v = (u - b) / (1.0 - b) if b != 1.0 else float("nan")
        rows.append({"metric": m, "normalization": "(user - base)/(1 - base)",
                     "from_baseline": v})
        scores.append(0.0 if np.isnan(v) else v)
    rows.append({"metric": "avg_score", "normalization": "mean(clipped>=0)",
                 "from_baseline": float(np.mean(np.clip(scores, 0, None))) if scores else None})
    return rows, {"best_values": best, "n_zero_best": len(zero),
                  "n_one_best": len(one),
                  "formula_source": "cell_eval/_score.py::_calc_norm_by_zero / _calc_norm_by_one",
                  "upstream_bug": "empty np.array([]) float64 index in score_agg_metrics"}


def aggregate_metric(raw) -> dict:
    """Convert a cell-eval metric result into a finite-aware summary."""
    if isinstance(raw, dict):
        vals = np.array([v for v in raw.values() if v is not None], dtype=np.float64)
    else:
        vals = np.array([raw], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    return {
        "n_values": int(vals.size),
        "n_finite": int(finite.size),
        "mean": float(finite.mean()) if finite.size else None,
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "status": "OK" if finite.size else "NA_ALL_NONFINITE",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    log: list[str] = []

    for path in (TRAIN_DE, VAL_DE, GENES):
        if not path.exists():
            print(f"FATAL missing input: {path}")
            return 2

    train_df = pd.read_parquet(TRAIN_DE)
    val_df = pd.read_parquet(VAL_DE)
    gene_universe = [l.strip() for l in GENES.read_text(encoding="utf-8").splitlines() if l.strip()]
    log.append(f"Loaded train DE {train_df.shape}, val DE {val_df.shape}, "
               f"gene universe {len(gene_universe)}")

    train_mat, val_mat, t_targets, v_targets, align = load_and_align(train_df, val_df)
    align["gene_universe_rows"] = len(gene_universe)
    align["shared_feature_fraction_of_gene_universe"] = round(
        align["shared_features"] / max(len(gene_universe), 1), 6)
    align["target_overlap_note"] = (
        "0 shared perturbation targets: VCC 2025 validation perturbations are "
        "held out from training, so no target-level leakage is possible; "
        "train_mean is a gene-level transfer baseline, not a per-target copy.")
    log.append(f"Aligned: {align['shared_features']} shared features, "
               f"{align['n_train_perturbations']} train / {align['n_val_perturbations']} val "
               f"perturbations, shared targets={align['shared_targets']}, "
               f"density train={align['train_density']:.4f} val={align['val_density']:.4f}")

    # ---- preregistered baselines -------------------------------------------------
    baselines = {
        "no_change": B.no_change_baseline(val_mat),
        "train_mean": B.train_mean_baseline(train_mat, val_mat.shape[0]),
    }
    if align["combinatorial_perturbations"]:
        log.append("WARNING: combinatorial perturbations present; additive baseline "
                   "still requires explicit single-gene effects and is deferred")
    else:
        log.append("additive baseline NOT APPLICABLE: 0 combinatorial perturbations "
                   "in the VCC 2025 DE tables (recorded, not imputed)")

    # ---- framework-internal metrics ---------------------------------------------
    fw_records: list[dict] = []
    pvals: list[float] = []
    for name, pred in baselines.items():
        agg = B.score_all(pred, val_mat)
        mae_i, des_i = per_pert_metrics(pred, val_mat, TOPK)
        ci_mae = B.bootstrap_ci(mae_i, alpha=ALPHA, B=N_BOOT, seed=SEED)
        ci_des = B.bootstrap_ci(des_i, alpha=ALPHA, B=N_BOOT, seed=SEED)
        p_mae = B.permutation_pvalue(B.mae, pred, val_mat, B=N_PERM, seed=SEED,
                                     higher_is_better=False)
        p_des = B.permutation_pvalue(B.des_score, pred, val_mat, B=N_PERM, seed=SEED,
                                     higher_is_better=True)
        p_pds = B.permutation_pvalue(B.pds_score, pred, val_mat, B=N_PERM, seed=SEED,
                                     higher_is_better=True)
        for metric, value, ci, pv, higher in (
            ("MAE", agg["MAE"], ci_mae, p_mae, False),
            ("DES", agg["DES"], ci_des, p_des, True),
            ("PDS", agg["PDS"], (float("nan"), float("nan")), p_pds, True),
        ):
            fw_records.append({
                "baseline": name, "metric": metric, "value": value,
                "ci_low": ci[0], "ci_high": ci[1],
                "permutation_p": pv, "p_adj": None,
                "higher_is_better": higher,
            })
            pvals.append(pv)
        log.append(f"FRAMEWORK {name}: MAE={agg['MAE']:.6f} DES={agg['DES']:.6f} "
                   f"PDS={agg['PDS']:.6f} | MAE 95% CI=({ci_mae[0]:.6f},{ci_mae[1]:.6f}) "
                   f"DES 95% CI=({ci_des[0]:.6f},{ci_des[1]:.6f})")
    adj = B.bh_adjust(pvals)
    for rec, q in zip(fw_records, adj):
        rec["p_adj"] = float(q)
    pd.DataFrame(fw_records).to_csv(RESULTS / "w4_framework_metrics.csv", index=False)

    # ---- official cell-eval DE metrics ------------------------------------------
    real_rows = val_df[["target", "feature"]].copy()
    # Real DE table: values are positional (one row per measured target/gene).
    real_pl = pl.from_pandas(pd.DataFrame({
        "target": val_df["target"].to_numpy(),
        "feature": val_df["feature"].to_numpy(),
        "log2_fold_change": val_df["log2_fold_change"].to_numpy(),
        "p_value": val_df["p_value"].to_numpy(),
        "fdr": val_df["p_adj"].to_numpy(),
    }))

    mu, p_ser, fdr_ser = training_derived_significance(train_df)
    pred_frames = {
        "no_change": build_de_frame(real_rows, 0.0, 1.0, 1.0),
        "train_mean": build_de_frame(real_rows, mu, p_ser, fdr_ser),
    }

    off_records: list[dict] = []
    agg_means: dict[str, dict[str, float]] = {}
    for name, pred_pl in pred_frames.items():
        comparison = initialize_de_comparison(real_pl, pred_pl)
        agg_means[name] = {}
        for metric in OFFICIAL_DE_METRICS:
            try:
                raw = metrics_registry.compute(metric, comparison)
                summary = aggregate_metric(raw)
            except Exception as exc:
                summary = {"n_values": 0, "n_finite": 0, "mean": None,
                           "sd": None, "status": f"ERROR: {type(exc).__name__}: {exc}"[:200]}
            rec = {"baseline": name, "official_metric": metric, **summary}
            off_records.append(rec)
            if summary["mean"] is not None:
                agg_means[name][metric] = summary["mean"]
        log.append(f"OFFICIAL {name}: {len(agg_means[name])}/{len(OFFICIAL_DE_METRICS)} "
                   f"DE metrics returned finite means")
    pd.DataFrame(off_records).to_csv(RESULTS / "w4_official_de_metrics.csv", index=False)

    # ---- cell-eval score CLI/API alignment ---------------------------------------
    common = sorted(set(agg_means["no_change"]) & set(agg_means["train_mean"]))
    for name in ("no_change", "train_mean"):
        frame = pd.DataFrame([{"statistic": "mean",
                               **{m: agg_means[name][m] for m in common}}])
        frame.to_csv(RESULTS / f"w4_agg_{name}.csv", index=False)
    score_status = "NOT_RUN"
    score_detail: object = ""
    score_meta: dict = {}
    if common:
        try:
            diff = score_agg_metrics(
                results_user=str(RESULTS / "w4_agg_train_mean.csv"),
                results_base=str(RESULTS / "w4_agg_no_change.csv"),
                output=str(RESULTS / "w4_official_baseline_diff.csv"),
            )
            score_status = "OK__OFFICIAL_API"
            score_detail = diff.to_pandas().to_dict(orient="records")
            log.append(f"cell-eval score (official API): {len(common)} metrics compared")
        except Exception as exc:
            score_status = "UPSTREAM_BUG__EQUIVALENT_REIMPLEMENTATION"
            score_detail = f"{type(exc).__name__}: {exc}"
            rows, score_meta = official_score_equivalent(
                agg_means["train_mean"], agg_means["no_change"], common)
            pd.DataFrame(rows).to_csv(RESULTS / "w4_official_baseline_diff.csv", index=False)
            log.append(f"cell-eval score official API FAILED: {score_detail}")
            log.append(f"  -> equivalent formulas applied to {len(common)} metrics "
                       f"({score_meta['n_one_best']} ONE-best / "
                       f"{score_meta['n_zero_best']} ZERO-best); "
                       f"avg_score={rows[-1]['from_baseline']}")
    else:
        log.append("cell-eval score SKIPPED: no common finite official metrics")

    pd.DataFrame(ALIGNMENT_MAP, columns=[
        "framework_metric", "official_counterpart", "official_metric_type",
        "availability", "caveat"]).to_csv(RESULTS / "w4_alignment_map.csv", index=False)

    # ---- consolidated, auditable record -----------------------------------------
    warnings = [
        "NO virtual cell model was run. scGPT / Geneformer weights are not installed; "
        "these are baseline-only results and must not be used to accept or reject H2.",
        "ANNDATA_PAIR official metrics (mae, mae_delta, discrimination_score_*) are "
        "UNAVAILABLE because the 6.9 GB validation h5ad failed the W1b content gate "
        "and was quarantined as CORRUPT_INCOMPLETE.",
        "Framework-internal MAE/DES/PDS are NOT numerically interchangeable with the "
        "official cell-eval metrics; see w4_alignment_map.csv and Results [[R8]].",
        "additive baseline is not applicable: zero combinatorial perturbations in the "
        "VCC 2025 DE tables; its absence is reported rather than imputed.",
        f"{(1 - align['val_density']) * 100:.2f}% of validation (target, feature) cells "
        "are absent measurements filled with 0.0; density is reported.",
        "The train_mean baseline's p_value/fdr are operational training-derived ranking "
        "statistics (one-sample t + BH over training perturbations), not a claim of "
        "statistical significance of a prediction and not derived from validation labels.",
        "These are DE-level, hESC-domain results. They are not embryo results and imply "
        "no functional validation in embryos.",
        "score_agg_metrics in cell-eval 0.8.2 fails on an all-ONE-best DE metric set "
        "(empty float64 index array). The identical publishable normalization formulas "
        "were reimplemented in this script; site-packages was NOT modified. See "
        "w4_official_baseline_diff.csv and the cell_eval_score block in the JSON.",
        "This score comparison is baseline-vs-baseline calibration (train_mean vs "
        "no_change), NOT a model-vs-baseline result. No preregistered model was run.",
        "0/44 validation perturbation targets occur in training; results are "
        "out-of-target, not in-sample fits.",
    ]
    record = {
        "schema_version": "w4-baseline-calibration-v1",
        "generated_at": now(),
        "analysis_status": "PREREGISTERED W4 BASELINE CALIBRATION - BASELINE RESULTS ONLY",
        "preregistration": {
            "url": "https://osf.io/8tzrq/overview",
            "doi": "10.17605/OSF.IO/8TZRQ",
            "registered_utc": "2026-11-17T10:15:47Z",
            "model_set_update_utc": "2026-11-17T14:23:25Z",
        },
        "inputs": {
            "train_de": {"path": str(TRAIN_DE.relative_to(ROOT)),
                         "sha256": manifest_hash(TRAIN_ID), "rows": int(len(train_df))},
            "val_de": {"path": str(VAL_DE.relative_to(ROOT)),
                       "sha256": manifest_hash(VAL_ID), "rows": int(len(val_df))},
            "gene_universe_rows": len(gene_universe),
            "licence": "VERIFIED_CC0 (Arc Virtual Cell Atlas VCC 2025 CC0 1.0 dedication)",
        },
        "alignment": align,
        "presets": {"seed": SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOT,
                    "de_topk": TOPK, "alpha": ALPHA,
                    "multiple_testing": B.PRESET["multiple_testing"]},
        "baselines": list(baselines.keys()),
        "framework_metrics": fw_records,
        "official_de_metrics": off_records,
        "official_metric_alignment_map": [
            dict(zip(["framework_metric", "official_counterpart",
                      "official_metric_type", "availability", "caveat"], r))
            for r in ALIGNMENT_MAP],
        "cell_eval_score": {
            "status": score_status,
            "metrics_compared": common,
            "detail": score_detail,
            "comparison_semantics": "train_mean baseline vs no_change baseline; "
                                    "NOT a model-vs-baseline result",
            "note": "Official API attempted first; site-packages untouched.",
            **score_meta,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cell_eval": importlib.metadata.version("cell-eval"),
            "polars": importlib.metadata.version("polars"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "warnings": warnings,
    }
    (RESULTS / "w4_baseline_calibration.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log.append("W4 baseline calibration written -> results/")
    (RESULTS / "w4_run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"\nHashes W4 outputs: {sha256_of(RESULTS / 'w4_baseline_calibration.json')[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
