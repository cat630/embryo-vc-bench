#!/usr/bin/env python
"""W6-3: preregistered `cell-eval --ceiling` data-noise upper bound.

Registration basis
------------------
OSF 8tzrq §5 ("Indices / metrics implementation") and RQ4 explicitly require
`--ceiling` to separate "the model failed" from "the data do not contain enough
information". W4 never executed it (see results/w4_baseline_calibration.json:
no ceiling field), and README §3.4 recorded that it became technically possible
only after the W1b repair of the 6.9 GB validation h5ad. This script executes
that preregistered, descriptive, model-free measurement.

Hard engineering constraint
---------------------------
The validation h5ad stores X as CSR with 864,793,017 non-zero float32 values
(data + int32 indices ~= 6.9 GB). `MetricsEvaluator.compute_ceiling` materialises
two disjoint half copies ON TOP of the loaded matrix, so full-data peak memory is
roughly 3x the matrix. The host has 16.9 GB total / 4.9 GB available. A full-data
run is therefore not executable here.

We use a DETERMINISTIC, seed-0, per-perturbation cell subsample of the SAME
dataset. This is not a change to the registered dataset inclusion rule, not a new
model score, not a hypothesis test and not a new inferential procedure. It is
reported as a memory-constrained implementation detail with the subsampling rule,
seed and hashes, and it is used ONLY for RQ4's descriptive inference boundary.
H1/H2/H3 adjudication is untouched.

Usage
-----
  python w6_ceiling.py --smoke     # tiny subset, verifies the CLI path
  python w6_ceiling.py --run       # full deterministic subsample
  python w6_ceiling.py --collect   # parse outputs -> results/w6_ceiling.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
H5AD = DATA / "VCC_validation_h5ad_GCS__adata_Validation.h5ad"
SUBSET = DATA / "w6_ceiling_subset.h5ad"
SMOKE_SUBSET = DATA / "w6_ceiling_smoke_subset.h5ad"
OUTDIR = RESULTS / "w6_ceiling_out"
SMOKE_OUTDIR = RESULTS / "w6_ceiling_smoke_out"
REPORT = RESULTS / "w6_ceiling.json"
CELL_EVAL = ROOT / ".venv" / "Scripts" / "cell-eval.exe"

SEED = 0
TARGET_CELLS = 200          # per perturbation (all 50 have >= 161; cap at available)
CONTROL_CELLS = 2000        # larger reference so split-half DE is not degenerate
SMOKE_TARGETS = 3
SMOKE_TARGET_CELLS = 40
SMOKE_CONTROL_CELLS = 200


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_categorical(f: h5py.File, name: str):
    g = f["obs"][name]
    cats = g["categories"][:].astype(str)
    codes = g["codes"][:]
    return np.asarray(cats, dtype=object)[codes]


def build_subset(dest: Path, target_cells: int, control_cells: int,
                 max_targets: int | None = None) -> dict:
    """Deterministic seed-0 per-perturbation subsample, written as a small h5ad."""
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    with h5py.File(H5AD, "r") as f:
        labels = read_categorical(f, "target_gene")
        var_names = f["var"]["_index"][:].astype(str)
        indptr = f["X"]["indptr"][:]
        n_rows = len(labels)
        ctrl_mask = labels == "non-targeting"
        targets = sorted(set(labels[~ctrl_mask].tolist()))
        if max_targets is not None:
            targets = targets[:max_targets]

        picked: list[int] = []
        for t in targets:
            idx = np.flatnonzero(labels == t)
            k = min(target_cells, idx.size)
            picked.extend(rng.choice(idx, size=k, replace=False).tolist())
        ctrl_idx = np.flatnonzero(ctrl_mask)
        k = min(control_cells, ctrl_idx.size)
        picked.extend(rng.choice(ctrl_idx, size=k, replace=False).tolist())
        picked = np.array(sorted(picked), dtype=np.int64)

        # Row-slice a CSR matrix without materialising the 6.9 GB full matrix.
        data_parts, indices_parts, new_indptr = [], [], [0]
        dset = f["X"]["data"]
        iset = f["X"]["indices"]
        for r in picked:
            s, e = int(indptr[r]), int(indptr[r + 1])
            data_parts.append(dset[s:e])
            indices_parts.append(iset[s:e])
            new_indptr.append(new_indptr[-1] + (e - s))
        data = np.concatenate(data_parts)
        indices = np.concatenate(indices_parts)
        del data_parts, indices_parts
        X = csr_matrix((data, indices, np.array(new_indptr, dtype=np.int32)),
                       shape=(len(picked), len(var_names)))
        del data, indices

    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(labels[picked].astype(str))},
        index=np.array([f"w6_{i}" for i in range(len(picked))], dtype=object),
    )
    var = pd.DataFrame(index=pd.Index(var_names, dtype=object))
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.write_h5ad(dest, compression="gzip")
    info = {
        "source_h5ad": H5AD.name,
        "source_h5ad_sha256": sha(H5AD),
        "subset_h5ad": dest.name,
        "subset_h5ad_sha256": sha(dest),
        "seed": SEED,
        "n_source_cells": int(n_rows),
        "n_source_genes": int(len(var_names)),
        "n_source_perturbations": int(len(set(labels.tolist()))),
        "n_targets_in_subset": len(targets),
        "target_cells_per_perturbation": target_cells,
        "control_cells": k,
        "n_subset_cells": int(a.n_obs),
        "n_subset_nonzeros": int(a.X.nnz),
        "subset_matrix_mb": round((a.X.data.nbytes + a.X.indices.nbytes) / 1e6, 1),
        "build_seconds": round(time.time() - t0, 1),
    }
    del a, X
    return info


def run_cell_eval(subset: Path, outdir: Path, profile: str = "de",
                  threads: int = 4) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CELL_EVAL), "run",
        "-ar", str(subset),
        "--ceiling",
        "--ceiling-seed", str(SEED),
        "--profile", profile,
        "--num-threads", str(threads),
        "--allow-discrete",
        "-o", str(outdir),
    ]
    t0 = time.time()
    print("CMD", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = {
        "command": " ".join(cmd),
        "returncode": p.returncode,
        "seconds": round(time.time() - t0, 1),
        "stdout_tail": (p.stdout or "")[-4000:],
        "stderr_tail": (p.stderr or "")[-4000:],
    }
    (outdir / "w6_cli_log.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RC", p.returncode, "seconds", out["seconds"], flush=True)
    if p.returncode != 0:
        print("STDERR", out["stderr_tail"][-1500:], flush=True)
    return out


def collect(smoke: bool = False) -> dict:
    import pandas as pd
    outdir = SMOKE_OUTDIR if smoke else OUTDIR
    agg = outdir / "agg_ceiling_results.csv"
    raw = outdir / "ceiling_results.csv"
    report = {
        "schema_version": "w6-ceiling-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "smoke" if smoke else "preregistered_RQ4_descriptive",
        "preregistration_basis": [
            "OSF 8tzrq §5 Indices / metrics implementation: --ceiling for the "
            "data-noise-limited upper bound",
            "RQ4 (measurement objective, no hypothesis, no confirmatory test)",
            "README §3.4: ceiling unavailable at W4 because the 6.9 GB h5ad was "
            "corrupt; W1b repaired it, so W6 executes it",
        ],
        "status": None,
        "outdir": str(outdir.relative_to(ROOT)),
        "raw_ceiling_csv_present": raw.exists(),
        "agg_ceiling_csv_present": agg.exists(),
        "interpretation_boundary": (
            "Official cell-eval DE-type metrics are NOT numerically "
            "interchangeable with the framework-internal DES/PDS/MAE reported in "
            "§3.3. This ceiling bounds the official DE-type metrics under the "
            "noise of the REAL data only. It is descriptive, model-free, and is "
            "NOT used to adjudicate H1/H2/H3, to re-scale any model score, or to "
            "change any preregistered criterion."
        ),
        "compliance": {
            "no_model_loaded": True,
            "no_new_model_score": True,
            "no_new_inferential_test": True,
            "no_preregistered_criterion_changed": True,
            "third_osf_update_required": False,
            "reason_no_update": (
                "The registered dataset set, model set, metrics, thresholds, "
                "fold count and seeds are unchanged. The cell subsample is a "
                "memory-constrained implementation detail WITHIN one already "
                "registered dataset, disclosed with seed and hashes in the "
                "manuscript Methods; it produces no model score and no "
                "hypothesis conclusion."
            ),
        },
    }
    if not agg.exists():
        report["status"] = "NOT EXECUTABLE - no agg_ceiling_results.csv produced"
        return report
    a = pd.read_csv(agg)
    report["agg_columns"] = list(a.columns)
    report["agg_records"] = json.loads(a.to_json(orient="records"))
    if raw.exists():
        r = pd.read_csv(raw)
        report["raw_rows"] = int(len(r))
        report["raw_columns"] = list(r.columns)
        report["raw_perturbations"] = (
            sorted(r["perturbation"].astype(str).unique().tolist())
            if "perturbation" in r.columns else None
        )
        report["raw_csv_sha256"] = sha(raw)
    report["agg_csv_sha256"] = sha(agg)
    report["status"] = "COMPUTED - descriptive data-noise ceiling"
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        info = build_subset(SMOKE_SUBSET, SMOKE_TARGET_CELLS,
                            SMOKE_CONTROL_CELLS, max_targets=SMOKE_TARGETS)
        print("SMOKE SUBSET", json.dumps(info, ensure_ascii=False), flush=True)
        cli = run_cell_eval(SMOKE_SUBSET, SMOKE_OUTDIR, threads=2)
        rep = collect(smoke=True)
        rep["subset"] = info
        rep["cli"] = cli
        (RESULTS / "w6_ceiling_smoke.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("SMOKE STATUS", rep["status"], flush=True)
        return 0

    if args.run:
        info = build_subset(SUBSET, TARGET_CELLS, CONTROL_CELLS)
        print("SUBSET", json.dumps(info, ensure_ascii=False), flush=True)
        cli = run_cell_eval(SUBSET, OUTDIR, threads=4)
        rep = collect(smoke=False)
        rep["subset"] = info
        rep["cli"] = cli
        REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("STATUS", rep["status"], flush=True)
        return 0 if cli["returncode"] == 0 else 1

    if args.collect:
        rep = collect(smoke=False)
        REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("STATUS", rep["status"], flush=True)
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
