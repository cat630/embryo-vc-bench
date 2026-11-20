#!/usr/bin/env python3
"""W5 engineering verification of the two preregistered model adapters.

HARD SCOPE LIMIT
----------------
This script does NOT run the preregistered benchmark. It embeds a small,
fixed subset of REAL E-MTAB-3929 cells and checks that both adapters return
finite, non-degenerate, deterministic arrays of the declared shape. It computes
NO baseline comparison, NO DES/PDS/MAE score, NO permutation test, NO p-value
and NO figure value. No `[[R#]]` slot may be filled from anything here.

Run with the isolated model environment:

    ./.venv-models/Scripts/python.exe w5_adapter_test.py
    ./.venv-models/Scripts/python.exe w5_adapter_test.py --n-cells 4 --n-targets 2

Outputs: results/w5_adapter_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import time

import numpy as np

from w5_adapters import PROTOCOL_VERSION, RESULTS, build_adapters, write_protocol
from w5_gene_space import build_gene_space, load_emtab3929

STATUS = ("ENGINEERING VERIFICATION - NOT A PREREGISTERED ANALYSIS; "
          "NO MODEL SCORE, NO BASELINE COMPARISON, NO P-VALUE, "
          "NO RESULT SLOT FILLED")

CANDIDATE_TARGETS = ("POU5F1", "NANOG", "SOX2", "CDX2", "GATA3", "TEAD4")


def sha_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--n-targets", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    gs = build_gene_space()
    atlas = load_emtab3929(gs)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(atlas.counts.shape[0], size=args.n_cells, replace=False))
    X = np.ascontiguousarray(atlas.counts[idx])
    targets = [g for g in CANDIDATE_TARGETS if g in gs.genes][:args.n_targets]
    if len(targets) < args.n_targets:
        raise RuntimeError("not enough candidate target genes in the fixed axis")

    adapters = build_adapters(gs, seed=args.seed)
    records = {}
    for name, adapter in adapters.items():
        t1 = time.time()
        emb = adapter.embed(X, gs.genes)
        emb_rep = adapter.embed(X, gs.genes)
        delta = adapter.predict_delta(X, targets)
        delta_rep = adapter.predict_delta(X, targets)
        checks = {
            "embed_shape": list(emb.shape),
            "embed_expected_shape": [args.n_cells, adapter.embedding_dim],
            "embed_shape_ok": emb.shape == (args.n_cells, adapter.embedding_dim),
            "embed_finite": bool(np.isfinite(emb).all()),
            "embed_nonconstant": bool(np.std(emb) > 1e-6),
            "embed_deterministic_max_abs_diff": float(np.max(np.abs(emb - emb_rep))),
            "delta_shape": list(delta.shape),
            "delta_expected_shape": [len(targets), gs.n_genes],
            "delta_shape_ok": delta.shape == (len(targets), gs.n_genes),
            "delta_finite": bool(np.isfinite(delta).all()),
            "delta_nonzero_rows": int(np.count_nonzero(np.any(delta != 0, axis=1))),
            "delta_deterministic_max_abs_diff": float(np.max(np.abs(delta - delta_rep))),
            "seconds": round(time.time() - t1, 2),
            "embed_sha256": sha_array(emb),
            "delta_sha256": sha_array(delta),
            "adapter": adapter.describe(),
        }
        records[name] = checks
        ok = (checks["embed_shape_ok"] and checks["embed_finite"]
              and checks["embed_nonconstant"]
              and checks["embed_deterministic_max_abs_diff"] < 1e-5
              and checks["delta_shape_ok"] and checks["delta_finite"]
              and checks["delta_nonzero_rows"] == len(targets)
              and checks["delta_deterministic_max_abs_diff"] < 1e-5)
        checks["PASS"] = bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name:12s} emb={checks['embed_shape']} "
              f"delta={checks['delta_shape']} "
              f"nonzero_rows={checks['delta_nonzero_rows']} "
              f"det={checks['embed_deterministic_max_abs_diff']:.2e}/{checks['delta_deterministic_max_abs_diff']:.2e} "
              f"{checks['seconds']}s", flush=True)

    versions = {}
    for p in ("torch", "transformers", "scgpt", "numpy", "pandas", "anndata"):
        try:
            versions[p] = md.version(p)
        except Exception:
            versions[p] = "NOT_INSTALLED"
    audit = {
        "status": STATUS,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "gene_space": gs.summary(),
        "input": {
            "atlas": atlas.name,
            "atlas_fingerprint": atlas.fingerprint,
            "n_cells": args.n_cells,
            "cell_indices": idx.tolist(),
            "counts_sha256": sha_array(X),
            "counts_dtype": str(X.dtype),
            "targets": targets,
            "stage_labels": [str(s) for s in atlas.stage[idx]],
        },
        "versions": versions,
        "adapters": records,
        "all_pass": bool(all(r["PASS"] for r in records.values())),
        "models_evaluated": sorted(records),
        "preregistered_model_set_complete": set(records) == {"scGPT", "Geneformer"},
        "wall_clock_seconds": round(time.time() - t0, 2),
        "declaration": ("Embedding and delta arrays are written only as SHA256 "
                         "fingerprints; no biological or performance number from "
                         "this file may be reported in the manuscript Results."),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "w5_adapter_test.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_protocol({
        "adapter_verification": {
            "result_file": "results/w5_adapter_test.json",
            "all_pass": audit["all_pass"],
            "models_evaluated": audit["models_evaluated"],
            "note": "engineering verification only; no preregistered score",
        }})
    print(f"\n{STATUS}\nall_pass={audit['all_pass']} "
          f"models={audit['models_evaluated']} "
          f"wall={audit['wall_clock_seconds']}s -> results/w5_adapter_test.json")
    return 0 if audit["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
