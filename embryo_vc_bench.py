"""
embryo_vc_bench
===============

Dependency-light, runnable core of the benchmark described in the manuscript
v0 "An open and reproducible benchmark for virtual cell models in human
preimplantation embryo single-cell atlases".

HONESTY / SCOPE STATEMENT (keep this in the published Method section)
--------------------------------------------------------------------
This module implements ONLY the components that require no downloaded data,
no GPU and no deep-learning stack:

  * three mandatory naive baselines
  * three evaluation metrics (MAE, DES, PDS) as explicit operational definitions
  * permutation nulls, bootstrap confidence intervals, BH correction
  * the evaluation protocol (T1 leave-one-atlas-out, T2 surrogate perturbation,
    T3 out-of-domain generalisation) and its leakage checks

It deliberately does NOT implement scVI/scANVI harmonisation or foundation-model
adapters. Those are declared as `typing.Protocol` interfaces with optional
dependencies (anndata, scanpy, scvi-tools, torch) and raise informative errors
until W3/W5 are executed.

CRITICAL: the DES and PDS functions below are self-contained, reproducible
OPERATIONAL DEFINITIONS. They are NOT claimed to be numerically identical to the
Arc Virtual Cell Challenge evaluation code. Work package W4 must align them
against the official implementation and report any discrepancy in the
manuscript. Until that alignment is done, scores produced here must be labelled
"framework-internal" and must not be compared numerically with leaderboard
values.

No experimental results are contained in this file. All numbers produced by
running it on synthetic data are self-test artefacts, not biological findings.

Author: (to be filled)   License: MIT (to be confirmed before release)
Python: 3.13 (tested)    Requires: numpy, pandas, scipy, scikit-learn
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "PRESET",
    "DeltaSet",
    "Fold",
    "no_change_baseline",
    "train_mean_baseline",
    "additive_baseline",
    "all_baselines",
    "mae",
    "des_score",
    "pds_score",
    "score_all",
    "permutation_pvalue",
    "bootstrap_ci",
    "bh_adjust",
    "leave_one_atlas_out",
    "make_surrogate_effects",
    "check_fold_leakage",
    "ModelAdapter",
    "LatentTimeModel",
    "OPTIONAL_DEPENDENCIES",
]

# ---------------------------------------------------------------------------
# Pre-specified design parameters. These are DESIGN VALUES, not results.
# Freeze these in the OSF pre-registration before any real-data run.
# ---------------------------------------------------------------------------

PRESET: dict[str, Any] = {
    "alpha": 0.05,
    "multiple_testing": "benjamini-hochberg",
    "n_permutations": 1000,
    "n_permutations_extended": 10000,
    "n_bootstrap": 2000,
    "seeds": (0, 1, 2, 3, 4),
    "delta_1_f1": 0.10,          # H1 minimum macro-F1 improvement
    "delta_2_effect": 0.05,      # H2 minimum DES/PDS effect size
    "de_topk": 20,               # top-k genes for the operational DE-overlap score
    "min_atlas_folds": 5,        # below this, H1 is demoted to exploratory
    "tau_range": (0.0, 1.0),
}

OPTIONAL_DEPENDENCIES = (
    "anndata", "scanpy", "scvi-tools", "torch",
    "scgpt", "geneformer", "scfoundation", "gears", "cpa",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DeltaSet:
    """A matrix of perturbation effects (delta = perturbed minus control).

    Attributes
    ----------
    delta : np.ndarray, shape (n_perturbation, n_gene)
    keys : perturbation identifiers, length n_perturbation
    genes : gene identifiers, length n_gene
    atlas : source atlas / dataset identifier (used for fold assignment)
    domain : "embryo" | "hESC" | "blastoid" | ... (used for T3)
    metadata : free-form provenance fields
    """
    delta: np.ndarray
    keys: Sequence[str]
    genes: Sequence[str]
    atlas: str = "unspecified"
    domain: str = "embryo"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delta = np.asarray(self.delta, dtype=np.float64)
        if self.delta.ndim != 2:
            raise ValueError("delta must be 2-D (n_perturbation, n_gene)")
        if len(self.keys) != self.delta.shape[0]:
            raise ValueError("keys length must equal n_perturbation")
        if len(self.genes) != self.delta.shape[1]:
            raise ValueError("genes length must equal n_gene")
        if not np.all(np.isfinite(self.delta)):
            raise ValueError("delta contains NaN or Inf")

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256(self.delta.tobytes()).hexdigest()[:16]
        return f"{self.atlas}:{self.domain}:{self.delta.shape[0]}x{self.delta.shape[1]}:{h}"


@dataclass
class Fold:
    """One leave-one-atlas-out split."""
    train_atlases: tuple[str, ...]
    test_atlas: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int

    def as_record(self) -> dict[str, Any]:
        return {
            "test_atlas": self.test_atlas,
            "train_atlases": "|".join(self.train_atlases),
            "n_train": int(len(self.train_idx)),
            "n_test": int(len(self.test_idx)),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Mandatory naive baselines
# ---------------------------------------------------------------------------

def no_change_baseline(true_delta: np.ndarray) -> np.ndarray:
    """Predict no effect: delta_hat = 0. Always available, always reported."""
    return np.zeros_like(np.asarray(true_delta, dtype=np.float64))


def train_mean_baseline(train_delta: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the training-set mean perturbation effect, ignoring perturbation identity.

    NOTE ON NOMENCLATURE: "cell-mean" / "training-mean" baselines are named
    inconsistently across the literature (cf. Ahlmann-Eltze et al., 2025; the
    Arc VCC naive baseline). A baseline that predicts the mean of CONTROL cells
    is algebraically identical to no-change in delta space. To avoid a
    degenerate duplicate, we implement the non-trivial version: the mean effect
    across training perturbations. W4 must state explicitly which published
    baseline this corresponds to and report both if they differ.
    """
    td = np.asarray(train_delta, dtype=np.float64)
    if td.ndim != 2 or td.shape[0] == 0:
        raise ValueError("train_delta must be a non-empty 2-D array")
    return np.tile(td.mean(axis=0, keepdims=True), (n_test, 1))


def additive_baseline(
    pair_keys: Sequence[tuple[str, str]],
    single_delta: Mapping[str, np.ndarray],
    n_genes: int,
) -> np.ndarray:
    """Two-gene perturbation = sum of the two single-gene effects.

    Missing singles are NOT silently imputed: they raise, so the caller must
    report coverage explicitly.
    """
    missing = [k for pair in pair_keys for k in pair if k not in single_delta]
    if missing:
        raise KeyError(f"missing single-gene effects for: {sorted(set(missing))}")
    out = np.zeros((len(pair_keys), n_genes), dtype=np.float64)
    for i, (a, b) in enumerate(pair_keys):
        out[i] = np.asarray(single_delta[a], dtype=np.float64) + np.asarray(
            single_delta[b], dtype=np.float64
        )
    return out


def all_baselines(
    train_delta: np.ndarray,
    true_delta: np.ndarray,
    pair_keys: Sequence[tuple[str, str]] | None = None,
    single_delta: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return every applicable baseline. `no_change` and `train_mean` are mandatory."""
    out = {
        "no_change": no_change_baseline(true_delta),
        "train_mean": train_mean_baseline(train_delta, true_delta.shape[0]),
    }
    if pair_keys is not None and single_delta is not None:
        out["additive"] = additive_baseline(pair_keys, single_delta, true_delta.shape[1])
    return out


# ---------------------------------------------------------------------------
# Metrics (operational definitions -- see module docstring)
# ---------------------------------------------------------------------------

def mae(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Mean absolute error over the full expression profile. Lower is better."""
    p = np.asarray(pred_delta, dtype=np.float64)
    t = np.asarray(true_delta, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    return float(np.mean(np.abs(p - t)))


def _topk_genes(delta_row: np.ndarray, k: int) -> set[int]:
    """Genes with the largest absolute effect; ties broken by index (deterministic)."""
    k = min(int(k), delta_row.size)
    if k <= 0:
        return set()
    return set(np.argsort(-np.abs(delta_row), kind="stable")[:k].tolist())


def des_score(pred_delta: np.ndarray, true_delta: np.ndarray, topk: int = PRESET["de_topk"]) -> float:
    """Operational Differential Expression Score: mean top-k DE-gene overlap.

    For each perturbation, take the top-k genes by |true delta| and by
    |predicted delta| and compute |intersection| / k. Mean over perturbations.
    Range [0, 1]; higher is better. A no-change prediction yields 0 by
    construction (empty top-k), which is intentional and must be reported.

    THIS IS NOT THE ARC VCC DES IMPLEMENTATION. See module docstring.
    """
    p = np.asarray(pred_delta, dtype=np.float64)
    t = np.asarray(true_delta, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    if t.shape[0] == 0:
        return float("nan")
    k = min(int(topk), t.shape[1])
    scores = []
    for i in range(t.shape[0]):
        pt = _topk_genes(p[i], k) if np.any(p[i] != 0) else set()
        tt = _topk_genes(t[i], k)
        scores.append(len(pt & tt) / k if k else 0.0)
    return float(np.mean(scores))


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row of a against each row of b."""
    eps = 1e-12
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return an @ bn.T


def pds_score(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Operational Perturbation Discrimination Score.

    For each perturbation i, rank cos(pred_i, true_i) among
    cos(pred_i, true_j) for all j, normalised to [0, 1] (1 = correct match is
    the most similar). Mean over perturbations. A constant prediction makes all
    similarities tied; we then return the expected chance value (0.5) so that
    baselines are not artificially rewarded.

    THIS IS NOT THE ARC VCC PDS IMPLEMENTATION. See module docstring.
    """
    p = np.asarray(pred_delta, dtype=np.float64)
    t = np.asarray(true_delta, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    n = t.shape[0]
    if n < 2:
        return float("nan")
    sim = _row_cosine(p, t)                       # (n_pred, n_true)
    ranks = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = sim[i]
        if np.allclose(row, row[0], atol=1e-12):  # degenerate / tied prediction
            ranks[i] = 0.5
            continue
        # fraction of distractors strictly less similar than the true match
        ranks[i] = float(np.mean(row[i] > np.delete(row, i)))
    return float(np.mean(ranks))


def score_all(pred_delta: np.ndarray, true_delta: np.ndarray) -> dict[str, float]:
    """All three metrics at once. MAE must never be hidden behind a composite score."""
    return {
        "MAE": mae(pred_delta, true_delta),
        "DES": des_score(pred_delta, true_delta),
        "PDS": pds_score(pred_delta, true_delta),
    }


# ---------------------------------------------------------------------------
# Uncertainty: permutation nulls, bootstrap CIs, multiplicity
# ---------------------------------------------------------------------------

def permutation_pvalue(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    B: int = PRESET["n_permutations"],
    seed: int = 0,
    higher_is_better: bool = True,
) -> float:
    """Permute perturbation labels of the true matrix to build a null distribution.

    Returns the (B+1)-corrected p-value, always in (0, 1].
    """
    t = np.asarray(true_delta, dtype=np.float64)
    p = np.asarray(pred_delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = metric_fn(p, t)
    if not np.isfinite(observed):
        return float("nan")
    null = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        null[b] = metric_fn(p, t[rng.permutation(t.shape[0])])
    if higher_is_better:
        extreme = int(np.sum(null >= observed))
    else:
        extreme = int(np.sum(null <= observed))
    return float((extreme + 1) / (int(B) + 1))


def bootstrap_ci(
    values: Sequence[float],
    alpha: float = PRESET["alpha"],
    B: int = PRESET["n_bootstrap"],
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Returns (lower, upper)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(int(B), x.size))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same length, same order)."""
    p = np.asarray(pvals, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank_from_end, i in enumerate(order[::-1]):
        k = n - rank_from_end
        val = min(prev, p[i] * n / k)
        adj[i] = val
        prev = val
    return np.clip(adj, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------

def leave_one_atlas_out(
    atlases: Mapping[str, DeltaSet],
    min_folds: int = PRESET["min_atlas_folds"],
    seeds: Sequence[int] = PRESET["seeds"],
) -> list[Fold]:
    """T1: build leave-one-atlas-out folds with an explicit leakage check.

    Each atlas object is treated as an indivisible unit: no perturbation from
    the test atlas may appear in the training set.
    """
    names = sorted(atlases)
    if len(names) < 2:
        raise ValueError("leave-one-atlas-out needs >= 2 independent atlases")
    folds: list[Fold] = []
    offset = 0
    index: dict[str, np.ndarray] = {}
    for name in names:
        n = atlases[name].delta.shape[0]
        index[name] = np.arange(offset, offset + n)
        offset += n
    for seed in seeds:
        for test in names:
            train = tuple(x for x in names if x != test)
            tr = np.concatenate([index[x] for x in train])
            folds.append(Fold(train, test, tr, index[test], seed))
    if len({f.test_atlas for f in folds}) < min_folds:
        # Not an error: H1 is demoted to exploratory, as pre-registered.
        pass
    return folds


def check_fold_leakage(fold: Fold) -> bool:
    """True iff train and test index sets are disjoint."""
    return bool(np.intersect1d(fold.train_idx, fold.test_idx).size == 0)


def make_surrogate_effects(
    expr: np.ndarray,
    target_idx: Sequence[int],
    corr: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """T2: construct SURROGATE in-silico perturbation effects.

    WARNING: this is a computational proxy obtained by masking a gene and,
    optionally, propagating the mask along a co-expression matrix. It is NOT
    an experimental perturbation, NOT a knockdown, and must be labelled
    "surrogate in-silico perturbation, not experimental perturbation"
    everywhere it appears in the manuscript.

    `expr` : (n_cell, n_gene) expression matrix. Log-normalised counts (>= 0)
    should be used with real data. If a z-scored matrix is supplied, the gene
    magnitude is taken as the mean absolute value so that the surrogate remains
    a well-defined down-regulation and never flips sign.
    Returns (n_target, n_gene) pseudo-delta matrix.
    """
    x = np.asarray(expr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expr must be 2-D (n_cell, n_gene)")
    n_genes = x.shape[1]
    if corr is None:
        corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
    corr = np.asarray(corr, dtype=np.float64)
    if corr.shape != (n_genes, n_genes):
        raise ValueError("corr must be (n_gene, n_gene)")
    out = np.zeros((len(target_idx), n_genes), dtype=np.float64)
    for i, g in enumerate(target_idx):
        g = int(g)
        if not (0 <= g < n_genes):
            raise IndexError(f"target_idx out of range: {g}")
        # Magnitude must be non-negative; use abs to stay valid for z-scored input.
        magnitude = float(np.mean(np.abs(x[:, g])))
        propagated = np.abs(corr[g]) * magnitude
        out[i] = -float(strength) * propagated
        out[i, g] = -float(strength) * magnitude
    return out


def evaluate_one_fold(
    fold: Fold,
    stacked_delta: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    B: int = PRESET["n_permutations"],
) -> pd.DataFrame:
    """Score every prediction method (models AND baselines) on one fold.

    Baselines must be present in `predictions`; a model score is never reported
    without its baselines (pre-specified hard rule).
    """
    required = {"no_change", "train_mean"}
    if not required.issubset(predictions):
        raise KeyError(f"a fold must contain the mandatory baselines {sorted(required)}")
    true = np.asarray(stacked_delta, dtype=np.float64)[fold.test_idx]
    rows = []
    for method, pred_all in predictions.items():
        pred = np.asarray(pred_all, dtype=np.float64)[fold.test_idx]
        s = score_all(pred, true)
        rows.append({
            "test_atlas": fold.test_atlas,
            "seed": fold.seed,
            "method": method,
            **s,
            "p_perm_DES": permutation_pvalue(des_score, pred, true, B=B, seed=fold.seed),
            "p_perm_PDS": permutation_pvalue(pds_score, pred, true, B=B, seed=fold.seed),
        })
    df = pd.DataFrame(rows)
    df["p_adj_DES"] = bh_adjust(df["p_perm_DES"].to_numpy())
    df["p_adj_PDS"] = bh_adjust(df["p_perm_PDS"].to_numpy())
    return df


# ---------------------------------------------------------------------------
# Optional-dependency interfaces (W3 / W5). Not runnable on a bare environment.
# ---------------------------------------------------------------------------

class ModelAdapter(Protocol):
    """Uniform interface for zero-shot foundation-model representations."""
    name: str

    def embed(self, counts: np.ndarray, genes: Sequence[str]) -> np.ndarray:
        """Return (n_cell, d) embeddings; no fine-tuning by default."""
        ...

    def predict_delta(self, counts: np.ndarray, perturbation: Sequence[str]) -> np.ndarray:
        """Return (n_perturbation, n_gene) predicted effects, if supported."""
        ...


class LatentTimeModel(Protocol):
    """Uniform interface for the shared latent developmental time axis tau."""
    def fit(self, counts: np.ndarray, batch: Sequence[str], covariates: Mapping[str, Sequence[Any]]) -> None:
        ...

    def tau(self) -> np.ndarray:
        """Return tau in [0, 1] with an uncertainty estimate."""
        ...


def _require_optional(package: str) -> None:
    raise NotImplementedError(
        f"W3/W5 component requires optional dependency '{package}'. "
        "Install the locked environment (environment.yml / Dockerfile) before running. "
        "Absence of this dependency must be disclosed in the manuscript; it must "
        "NOT be used as a reason to silently drop a model from the benchmark."
    )
