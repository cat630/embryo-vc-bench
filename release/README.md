# embryo-vc-bench — W7 reproducible release (draft, 18 November 2026)

This directory contains the **locally completable** part of W7: environment
locks, machine-readable figure/table source data, schema.org metadata, a
read-only replay notebook/script, a FAIR self-check and a SHA256 manifest.

> **This is not yet a published release.** The public repository is
> `https://github.com/cat630/embryo-vc-bench` and the code is licensed **MIT** (restrictions on non-academic
> use: **none**). The Zenodo release DOI and the release date remain
> explicit `[[PENDING_*]]` placeholders; no identifier has been invented.
> Manuscript slot `[[R9]]` stays unfilled until the DOI and the author
> declarations exist.

## Layout

```
release/
├── README.md                     this file
├── environment/
│   ├── requirements-main.lock.txt    pip freeze, .venv (110 pinned distributions)
│   ├── requirements-models.lock.txt  pip freeze, .venv-models (102 pinned distributions)
│   ├── environment.yml               optional Conda skeleton (pip section = main lock)
│   ├── Dockerfile                    optional Linux path, NOT CI-tested
│   └── platform.json                 hardware / OS / Python actually used
├── figure_source_data/
│   ├── fig4_t2_t3_means.csv          8 task x method rows = manuscript 3.3 table
│   ├── fig4_fold_level_scores.csv    all 80 preregistered rows
│   ├── fig4_paired_contrasts.csv     model-minus-baseline deltas + bootstrap 95% CI
│   ├── fig_t2_by_heldout_atlas.csv   cross-platform stratification
│   ├── table1_datasets.csv           42 provenance rows (licence status included)
│   ├── table3_validation_layers.csv  three-layer validation table
│   ├── ceiling_de_metrics.csv        W6 official DE-type ceiling (subsample depth)
│   └── SOURCE_INDEX.md               provenance, seed, protocol and SHA256 per file
├── metadata.jsonld               schema.org SoftwareSourceCode + Dataset
├── CITATION.cff                  CFF 1.2.0
├── LICENSE.md                    MIT licence; upstream terms and non-academic-use statement
├── FAIR_selfcheck.md             FAIR + journal-specific self-assessment
├── reproduce.ipynb               read-only replay of main tables/figure
├── reproduce.py                  same replay, plain Python (no notebook deps)
└── RELEASE_MANIFEST.json         SHA256 of every file in this directory
```

## Reproduce the main figure and tables (no model, no GPU, no rescoring)

```bash
# From the repository root, using the main environment:
./.venv/Scripts/python.exe release/reproduce.py
```

The replay reads only the released, hash-verified artefacts in `results/`.
It recomputes the 8-row mean table, the paired model-vs-baseline contrasts
with 2000-resample bootstrap CIs and the cross-atlas stratification, verifies
every SHA256 in `RELEASE_MANIFEST.json`, and (if matplotlib is available)
writes `figure_source_data/fig4_des_by_method.png`. It **never** loads scGPT or
Geneformer, never recomputes a preregistered score, and never re-runs a W6
generator.

Full from-scratch re-scoring (hours, CPU, pinned weights) is documented in
`../README.md` §3.9. It is not needed to reproduce the released figures.

## Reconstruct the two environments

```bash
python -m venv .venv          && .venv/Scripts/pip install -r release/environment/requirements-main.lock.txt
python -m venv .venv-models   && .venv-models/Scripts/pip install -r release/environment/requirements-models.lock.txt
```

The split is mandatory: installing the model dependencies into `.venv`
downgrades scvi-tools. `scgpt==0.2.4` is available only on the official PyPI
index and must be installed with `--no-deps` into `.venv-models`. See
`environment/Dockerfile` for the Linux caveats.

## Refresh the manifest after editing anything here

```bash
python w7_release.py --refresh
```

Filling the GitHub URL, Zenodo DOI, release date or licence changes file
content and therefore **all** SHA256 values. `--refresh` recomputes
`RELEASE_MANIFEST.json` and `results/w7_release_summary.json`. If the
manuscript has already cited the manifest hash, re-sync that citation. The
`results/w7_release_summary.json` hardware/runtime record is intentionally
independent of the pending identifiers, so the `[[R7]]` citation stays stable.

## Blocking items — status updated 20 November 2026

1. **[DONE] Code licence** — the author chose **MIT**; `LICENSE.md` now holds
   the full MIT text. Restrictions on non-academic use are stated explicitly
   as **`"none"`**.
2. **[DONE] Public repository** — `https://github.com/cat630/embryo-vc-bench` (public, `main`), 95
   files pushed; the repository placeholder has been replaced in
   `metadata.jsonld` / `CITATION.cff`.
3. **[PENDING] Zenodo release DOI** — archive the tagged release, fill
   `[[PENDING_ZENODO_DOI]]` and `[[PENDING_RELEASE_DATE]]`, then run
   `python w7_release.py --refresh`.
4. **Manuscript `[[R9]]`** — repository URL, DOI, licence, OS, reproduction
   commands, plus CRediT, Funding, generative-AI disclosure and the COI
   two-way choice (editorial recusal vs standard declaration).
5. **Manuscript `[[R1]]`** — institutional confirmation of the 21
   `VERIFIED_POLICY` rows in `data_manifest.csv` before Table 1 is filled.
6. **Reference DOIs** — Crossref / Europe PMC verification of every cited
   entry before submission.
7. **Remove internal markers** — DECISION RECORD, H2 ADJUDICATION RECORD,
   appendices A/B and the Chinese COI note; change `w5_score_summary.json`
   status from `draft, not yet released` only after the author signs off.

## Integrity rules carried over from W5/W6

- `[[R2]]`, `[[R3]]`, `[[R5]]` are **permanent placeholders** (τ never fitted).
  Never fill them with imputed values.
- No post-scoring test was added, no preregistered criterion was changed, and
  no third OSF update was required. The only registration updates are the two
  approved timestamped ones (2026-11-17 14:23:25 UTC and 2026-11-18 05:18:50 UTC).
- Framework-internal DES/PDS/MAE are **not** on a log2FC scale and must not be
  compared numerically with Virtual Cell Challenge leaderboard values.
- The W6 ceiling is a **subsample-depth** bound (11,961 cells, seed 0), not a
  full-data bound. It is descriptive, model-free and used for no hypothesis
  decision.
