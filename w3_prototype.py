#!/usr/bin/env python3
"""W3 technical feasibility prototype — NOT a preregistered analysis.

Purpose: verify that scVI 1.5.2 actually trains on CPU under Python 3.13 /
torch 2.15 with the real E-MTAB-3929 count matrix, before committing to W3.
It uses a deliberately small, arbitrary subset (first 2,000 genes / 500 cells)
and 5 epochs. No QC thresholds, no filtering decisions, no latent time axis,
no biological interpretation and no performance number from this script may
appear in the manuscript Results.

Run with the project virtual environment:
    ./.venv/Scripts/python.exe w3_prototype.py
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scvi
import torch
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
COUNTS = ROOT / "data" / "BS_E-MTAB-3929_counts__counts.txt"
RESULTS = ROOT / "results"
N_GENES = 2000
N_CELLS = 500
EPOCHS = 5
SEED = 0


def main() -> int:
    if not COUNTS.exists():
        raise SystemExit(f"missing input: {COUNTS}")
    RESULTS.mkdir(exist_ok=True)
    scvi.settings.seed = SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    df = pd.read_csv(COUNTS, sep="\t", index_col=0)
    full_shape = list(df.shape)  # genes x cells
    sub = df.iloc[:N_GENES, :N_CELLS]
    cell_ids = list(sub.columns)
    stages = [".".join(str(c).split(".")[:2]) for c in cell_ids]
    X = np.asarray(sub.T.values, dtype=np.float32)  # cells x genes
    obs = pd.DataFrame({"cell_id": cell_ids, "stage": stages}, index=cell_ids)
    var = pd.DataFrame(index=[str(g) for g in sub.index])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    t_load = time.time() - t0

    t1 = time.time()
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=10, n_layers=1)
    model.train(max_epochs=EPOCHS, batch_size=128)
    latent = model.get_latent_representation()
    t_train = time.time() - t1

    history = {}
    try:
        for k, v in model.history.items():
            history[k] = [float(x) for x in np.asarray(v).ravel()[-EPOCHS:]]
    except Exception as e:
        history = {"unavailable": f"{type(e).__name__}: {e}"}

    payload = {
        "status": "TECHNICAL FEASIBILITY PROTOTYPE - NOT A PREREGISTERED ANALYSIS",
        "warning": "No value in this file may be reported as a biological or "
                   "benchmark result. Arbitrary subset, arbitrary epoch count.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(COUNTS.relative_to(ROOT)),
        "full_matrix_shape_genes_x_cells": full_shape,
        "subset_genes": N_GENES, "subset_cells": N_CELLS,
        "epochs": EPOCHS, "n_latent": 10, "seed": SEED,
        "stages_observed": sorted(set(stages)),
        "latent_shape": list(latent.shape),
        "latent_mean_abs": float(np.abs(latent).mean()),
        "load_seconds": round(t_load, 2),
        "train_seconds": round(t_train, 2),
        "history_tail": history,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "anndata": version("anndata"), "scanpy": version("scanpy"),
            "scvi-tools": version("scvi-tools"), "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    out = RESULTS / "w3_prototype.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"W3 prototype OK: full matrix {full_shape}, subset {X.shape}, "
          f"latent {latent.shape}, load {t_load:.1f}s, train {t_train:.1f}s")
    print(f"-> {out.name} (NOT for manuscript Results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
