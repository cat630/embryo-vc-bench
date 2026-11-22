# embryo-vc-bench

Code and source data accompanying the manuscript (draft, not yet submitted).

> **Status: DRAFT MANUSCRIPT — a versioned, archived code release exists.**
> Public repository: `https://github.com/cat630/embryo-vc-bench`. Code licence: **MIT**.
> Restrictions on non-academic use: **none**. Zenodo concept DOI
> `10.5281/zenodo.24433661` (cite this); `v0.1.0` version DOI `10.5281/zenodo.24433662`;
> release date 2026-11-22. The manuscript itself is still a draft and has not been
> submitted. `[[PENDING_*]]` strings in `results/w7_release_summary.json` are a
> frozen evidence snapshot, deliberately preserved for hash integrity.

## Layout

- `w1_*.py` … `w7_release.py` — pipeline stages: W1 data provenance and
  integrity repair, W2 QC audit, W3 prototype, W4 baseline calibration,
  W5 adapters and preregistered scoring, W6 orthogonal validation, W7 release
  packaging.
- `results/` — preregistered score rows and validation outputs.
- `release/` — reproducible release package: environment locks, figure and
  table source data, schema.org metadata, a read-only replay script and
  notebook, a FAIR self-check and a SHA256 manifest. See `release/README.md`.
- `data_manifest.csv`, `model_manifest.csv` — dataset and model provenance.

## Replay the released tables and figure

No model is loaded and no score is recomputed:

```
python release/reproduce.py
```

or open `release/reproduce.ipynb`.

Raw datasets, model weights and virtual environments are **not** included;
accessions, versions and exact dependency pins are recorded in the manifests
and `release/environment/`.

## Integrity notes

Framework-internal DES/PDS/MAE are **not** on a log2FC scale and must not be
compared numerically with Virtual Cell Challenge leaderboard values. The W6
ceiling is a subsample-depth bound, not a full-data bound.
