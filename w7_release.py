"""W7 reproducible-release builder for embryo-vc-bench.

Generates, from already released and hash-verified artefacts only:

  release/environment/requirements-main.lock.txt     (pip freeze, .venv)
  release/environment/requirements-models.lock.txt   (pip freeze, .venv-models)
  release/environment/environment.yml                (Conda skeleton)
  release/environment/platform.json                  (hardware, OS, Python)
  release/figure_source_data/*.csv                   (main figure/table source data)
  release/figure_source_data/SOURCE_INDEX.md         (provenance + SHA256)
  release/reproduce.py                               (plain-Python read-only replay)
  release/reproduce.ipynb                            (notebook replay)
  release/RELEASE_MANIFEST.json                      (SHA256 of every release file)
  results/w7_release_summary.json                    (R7 evidence, stable)

HARD RULES
  * No foundation model is loaded and no preregistered score is recomputed.
  * No W6 generator is re-run (w6_orthogonal.py / w6_ceiling.py): those files
    carry generated_at fields and their hashes are cited in manuscript 3.5.
  * No identifier is invented. GitHub URL, Zenodo DOI, release date and the
    code licence remain explicit [[PENDING_*]] placeholders.
  * results/w7_release_summary.json intentionally contains no timestamp and no
    release-file hash, so the manuscript's [[R7]] citation stays stable when
    the author later fills the pending identifiers.

Usage
  python w7_release.py --build     # first build (idempotent for stable files)
  python w7_release.py --refresh   # recompute manifest + source index + summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "release"
ENV = RELEASE / "environment"
FIG = RELEASE / "figure_source_data"
RESULTS = ROOT / "results"
MAIN_PY = ROOT / ".venv" / "Scripts" / "python.exe"
MODELS_PY = ROOT / ".venv-models" / "Scripts" / "python.exe"

RECORD_DATE = "2026-11-18"
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_B = 2000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_if_changed(path: Path, text: str) -> bool:
    """Return True if the file was written. Keeps hashes stable on rebuild."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def csv_text(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n",
                       extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in columns})
    return buf.getvalue()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# 1. Environment locks and platform record
# --------------------------------------------------------------------------
def build_environment() -> dict:
    ENV.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    for label, exe, dest, note in [
        ("main", MAIN_PY, ENV / "requirements-main.lock.txt",
         "Main analysis environment: framework, cell-eval 0.8.2, scvi-tools, scanpy."),
        ("models", MODELS_PY, ENV / "requirements-models.lock.txt",
         "Isolated zero-shot model environment: scGPT + Geneformer + transformers."),
    ]:
        if not exe.exists():
            raise SystemExit(f"missing interpreter: {exe}")
        p = subprocess.run([str(exe), "-m", "pip", "freeze"],
                           capture_output=True, text=True, check=True)
        lines = [ln.rstrip() for ln in p.stdout.splitlines() if ln.strip()]
        header = [
            f"# embryo-vc-bench {label} environment lock",
            f"# {note}",
            "# Produced by `python -m pip freeze` on the machine that ran the analyses.",
            "# Platform: Windows 11 (10.0.26200), AMD64, CPython 3.13.14.",
            f"# {len(lines)} pinned distributions. Do not edit by hand.",
        ]
        write_if_changed(dest, "\n".join(header + lines) + "\n")
        out[label] = {"file": str(dest.relative_to(ROOT)).replace("\\", "/"),
                      "n_pinned": len(lines), "sha256": sha(dest)}

    # Conda skeleton: the pip section IS the main lock, verbatim.
    lock_lines = [ln for ln in (ENV / "requirements-main.lock.txt")
                  .read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.startswith("#")]
    yml = [
        "name: embryo-vc-bench",
        "channels:",
        "  - conda-forge",
        "dependencies:",
        "  - python=3.13.14",
        "  - pip>=24",
        "  - pip:",
    ] + [f"      - {ln}" for ln in lock_lines]
    yml.insert(1, "# Conda skeleton. The authoritative byte-exact lock is")
    yml.insert(2, "# requirements-main.lock.txt; this file embeds it verbatim.")
    yml.insert(3, "# The model environment MUST stay separate: its pins downgrade scvi-tools.")
    write_if_changed(ENV / "environment.yml", "\n".join(yml) + "\n")

    try:
        import psutil
        ram_total = psutil.virtual_memory().total
        cpu_logical = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
    except Exception:
        ram_total, cpu_logical, cpu_physical = 16856768512, 12, 8

    plat = {
        "schema_version": "w7-platform-v1",
        "record_date": RECORD_DATE,
        "note": "Static hardware record. No timestamp field: this file is cited by hash.",
        "os": platform.platform(),
        "os_release": "Windows 11",
        "os_build": "10.0.26200",
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical": cpu_logical,
        "cpu_physical": cpu_physical,
        "ram_total_bytes": ram_total,
        "ram_total_gib": round(ram_total / (1024 ** 3), 2),
        "ram_available_at_scoring_bytes": 4900000000,
        "ram_available_at_scoring_note": "Approximately 4.9 GB available when --mode score ran; this is why the W6 full-data ceiling was not executable and a deterministic seed-0 subsample was used.",
        "gpu": "none used; all model inference was CPU-only",
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "disk_free_at_decision_gb": 161,
        "two_environments": {
            "main": out["main"],
            "models": out["models"],
            "reason": "Installing model dependencies into the main environment downgrades scvi-tools; scgpt==0.2.4 is installed with --no-deps from the official PyPI index only.",
        },
        "linux_docker_status": "provided as an interoperability aid in environment/Dockerfile; NOT executed or CI-tested, and not the environment in which published numbers were produced",
    }
    write_if_changed(ENV / "platform.json",
                     json.dumps(plat, ensure_ascii=False, indent=2) + "\n")
    return plat


# --------------------------------------------------------------------------
# 2. Figure / table source data
# --------------------------------------------------------------------------
def build_figure_source() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    meta: dict = {}

    scores = read_csv(RESULTS / "w5_scores.csv")
    metrics = ["MAE", "MAE_z", "DES", "PDS"]

    # (a) 8-row task x method mean table = manuscript 3.3
    grouped: dict = defaultdict(list)
    for r in scores:
        grouped[(r["task"], r["method"])].append(r)
    means = []
    import statistics as st
    for (task, method), rows in sorted(grouped.items()):
        rec = {"task": task, "method": method, "n_instances": len(rows)}
        for m in metrics:
            vals = [float(r[m]) for r in rows]
            rec[m + "_mean"] = round(st.mean(vals), 6)
            rec[m + "_sd"] = round(st.pstdev(vals), 6)
        means.append(rec)
    cols = ["task", "method", "n_instances"] + [
        m + s for m in metrics for s in ("_mean", "_sd")]
    write_if_changed(FIG / "fig4_t2_t3_means.csv", csv_text(means, cols))
    meta["fig4_t2_t3_means.csv"] = {
        "rows": len(means),
        "source_script": "w5_score.py --mode score",
        "source_file": "results/w5_scores.csv",
        "seeds": "0-4 (T3) / 15 fold instances x 5 seeds (T2)",
        "protocol": "w5-adapter-protocol-2; fixed 17,555-gene axis; B=1000 permutation; BH across 4 methods per block",
        "note": "Exactly the 8 task x method rows reported in manuscript 3.3. Values are framework-internal and NOT on a log2FC scale.",
    }

    # (b) all 80 fold-level rows (verbatim snapshot)
    with (RESULTS / "w5_scores.csv").open(encoding="utf-8", newline="") as f:
        raw = f.read()
    write_if_changed(FIG / "fig4_fold_level_scores.csv", raw)
    meta["fig4_fold_level_scores.csv"] = {
        "rows": len(scores),
        "source_script": "w5_score.py --mode score",
        "source_file": "results/w5_scores.csv",
        "seeds": "0-4",
        "protocol": "verbatim snapshot; 60 T2 rows (15 fold instances x 4 methods) + 20 T3 rows (5 seeds x 4 methods)",
        "note": "Use this file to redraw forest plots or confidence intervals. Zero nulls, zero errors.",
    }

    # (c) paired model-vs-baseline contrasts with bootstrap CI
    import numpy as np
    idx: dict = defaultdict(dict)
    for r in scores:
        key = ((r["seed"], r["test_atlas"]) if r["task"] == "T2_surrogate"
               else r["seed"])
        idx[(r["task"], key)][r["method"]] = r
    contrasts = []
    for task in ("T2_surrogate", "T3_DE_real"):
        for model in ("scGPT", "Geneformer"):
            for baseline in ("no_change", "train_mean"):
                for metric in metrics:
                    d = np.array([
                        float(v[model][metric]) - float(v[baseline][metric])
                        for (tk, _), v in idx.items() if tk == task and model in v
                    ], dtype=float)
                    rng = np.random.default_rng(0)
                    bs = np.array([rng.choice(d, len(d), replace=True).mean()
                                   for _ in range(BOOTSTRAP_B)])
                    contrasts.append({
                        "task": task, "model": model, "baseline": baseline,
                        "metric": metric, "n_instances": len(d),
                        "mean_delta": round(float(d.mean()), 6),
                        "ci95_low": round(float(np.percentile(bs, 2.5)), 6),
                        "ci95_high": round(float(np.percentile(bs, 97.5)), 6),
                        "min_delta": round(float(d.min()), 6),
                        "max_delta": round(float(d.max()), 6),
                    })
    ccols = ["task", "model", "baseline", "metric", "n_instances",
             "mean_delta", "ci95_low", "ci95_high", "min_delta", "max_delta"]
    write_if_changed(FIG / "fig4_paired_contrasts.csv",
                     csv_text(contrasts, ccols))
    meta["fig4_paired_contrasts.csv"] = {
        "rows": len(contrasts),
        "source_script": "w7_release.py (descriptive replay only; no new inferential test)",
        "source_file": "results/w5_scores.csv",
        "seeds": "bootstrap RNG seed 0; B=" + str(BOOTSTRAP_B),
        "protocol": "paired per-instance difference model minus baseline; percentile bootstrap CI",
        "note": "These are descriptive paired effect sizes. They are NOT p-values and were NOT added after scoring as a new significance test. The registered p_perm tests each method against permuted rows, not the model-vs-baseline contrast.",
    }

    # (d) cross-atlas stratification (verbatim W6 output)
    t2 = read_csv(RESULTS / "w6_t2_by_heldout_atlas.csv")
    tcols = list(t2[0].keys())
    write_if_changed(FIG / "fig_t2_by_heldout_atlas.csv", csv_text(t2, tcols))
    meta["fig_t2_by_heldout_atlas.csv"] = {
        "rows": len(t2),
        "source_script": "w6_orthogonal.py --mode atlas",
        "source_file": "results/w6_t2_by_heldout_atlas.csv",
        "seeds": "0-4", "protocol": "verbatim W6 output; descriptive re-stratification of already released scores",
        "note": "Three held-out atlases with three distinct gene-identifier systems. Supports 'externally replicable across independent atlases' only.",
    }

    # (e) Table 1 dataset-level provenance (licence status is explicit)
    dm = read_csv(ROOT / "data_manifest.csv")
    dcols = ["entry_id", "accession", "source", "kind", "version",
             "licence_status", "licence_evidence_url", "citation_requirement",
             "status", "sha256", "notes"]
    for r in dm:
        r["entry_id"] = r.get("\ufeffentry_id", r.get("entry_id", ""))
    write_if_changed(FIG / "table1_datasets.csv", csv_text(dm, dcols))
    from collections import Counter
    lic = Counter(r["licence_status"] for r in dm)
    meta["table1_datasets.csv"] = {
        "rows": len(dm),
        "source_script": "w1_fetch.py / w1_fetch_large.py / w1_validate.py",
        "source_file": "data_manifest.csv",
        "seeds": "not applicable", "protocol": "verbatim licence/provenance columns",
        "licence_status_counts": dict(lic),
        "note": "13 UNVERIFIED rows are BLOCKED_LICENCE and excluded. The 21 VERIFIED_POLICY rows require institutional confirmation before manuscript [[R1]] / Table 1 can be filled; this file does NOT assert that clearance.",
    }

    # (f) Table 3 three-layer validation table, machine readable
    layers = [
        {"layer": "L1_internal", "evidence_type": "permutation null, bootstrap CI, mandatory naive baselines",
         "evidence_available": "B=1000 row permutations; BH across 4 methods per block; 2000-resample bootstrap CIs; no_change DES=0/PDS=0.5 zero anchor on 20 instances",
         "permitted_wording": "internally stable",
         "prohibited_wording": "experimentally validated",
         "source_file": "results/w6_null_audit.json"},
        {"layer": "L2_cross_atlas", "evidence_type": "leave-one-atlas-out re-stratification across three licence-verified independent embryo atlases",
         "evidence_available": "scGPT directional advantage present on all three held-out atlases; raw MAE unimproved on all three",
         "permitted_wording": "externally replicable across independent atlases",
         "prohibited_wording": "validated in human embryos",
         "source_file": "results/w6_atlas_replication.json"},
        {"layer": "L3_orthogonal", "evidence_type": "published out-of-domain CRISPRi differential-expression table plus a model-free data-noise ceiling",
         "evidence_available": "44 VCC validation target genes; official cell-eval DE-type ceiling from a deterministic seed-0 11,961-cell subsample",
         "permitted_wording": "orthogonal support from published perturbation experiments",
         "prohibited_wording": "causal proof / virtual embryo / clinically actionable",
         "source_file": "results/w6_ceiling.json"},
    ]
    lcols = ["layer", "evidence_type", "evidence_available",
             "permitted_wording", "prohibited_wording", "source_file"]
    write_if_changed(FIG / "table3_validation_layers.csv",
                     csv_text(layers, lcols))
    meta["table3_validation_layers.csv"] = {
        "rows": len(layers),
        "source_script": "w6_orthogonal.py",
        "source_file": "results/w6_validation_table3.md",
        "seeds": "0-4 / ceiling seed 0", "protocol": "structured form of the W6 Table 3",
        "note": "Wording audit PASS with 0 unresolved assertions; see results/w6_wording_audit.json.",
    }

    # (g) official DE-type ceiling (verbatim aggregate)
    agg = read_csv(RESULTS / "w6_ceiling_out" / "agg_ceiling_results.csv")
    acols = list(agg[0].keys())
    write_if_changed(FIG / "ceiling_de_metrics.csv", csv_text(agg, acols))
    meta["ceiling_de_metrics.csv"] = {
        "rows": len(agg),
        "source_script": "w6_ceiling.py --run (cell-eval 0.8.2 run --ceiling)",
        "source_file": "results/w6_ceiling_out/agg_ceiling_results.csv",
        "seeds": "ceiling seed 0", "protocol": "profile de; --allow-discrete; 11,961-cell deterministic subsample",
        "note": "SUBSAMPLE-DEPTH bound only. Full-data ceiling was not executable at 4.9 GB available RAM (compute_ceiling needs ~3x the 6.9 GB CSR matrix). Model-free, descriptive, used for no hypothesis decision.",
    }

    # Source index with hashes (written after all CSVs exist)
    lines = [
        "# Figure and table source data — provenance index",
        "",
        "Every file below is derived from an already released, hash-verified",
        "artefact in `results/`. No model was loaded and no preregistered score",
        "was recomputed to produce them. Framework-internal DES/PDS/MAE are not",
        "on a log2FC scale and must not be compared numerically with Virtual",
        "Cell Challenge leaderboard values.",
        "",
        "| File | Rows | SHA256 | Source script | Seeds / protocol | Note |",
        "|---|---|---|---|---|---|",
    ]
    for name, info in sorted(meta.items()):
        lines.append(
            f"| `{name}` | {info['rows']} | `{sha(FIG / name)}` | "
            f"`{info['source_script']}` | {info['seeds']}; {info['protocol']} | "
            f"{info['note']} |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        "- `results/w6_orthogonal_summary.json` and `results/w6_wording_audit.json`",
        "  carry `generated_at` fields and are cited by SHA256 in manuscript 3.5.",
        "  Do NOT re-run `w6_orthogonal.py --mode all` after submission drafting.",
        "- `w5_scores.csv` SHA256 is cited in manuscript 3.3 and must stay stable.",
        "- Re-run `python w7_release.py --refresh` after changing any file in",
        "  `release/`, then re-sync any manuscript hash citation that points here.",
        "",
    ]
    write_if_changed(FIG / "SOURCE_INDEX.md", "\n".join(lines))
    return meta


# --------------------------------------------------------------------------
# 3. Read-only replay (plain Python + notebook)
# --------------------------------------------------------------------------
REPLAY_PY = '''"""Read-only replay of the released main tables and figure.

No model is loaded, no preregistered score is recomputed, no W6 generator is
re-run. Run from anywhere: paths are resolved relative to this file.
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
from collections import defaultdict
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RELEASE = ROOT / "release"
FIG = RELEASE / "figure_source_data"


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_table():
    scores = rows(RESULTS / "w5_scores.csv")
    g = defaultdict(list)
    for r in scores:
        g[(r["task"], r["method"])].append(r)
    out = []
    for (task, method), rs in sorted(g.items()):
        rec = {"task": task, "method": method, "n": len(rs)}
        for m in ("MAE", "MAE_z", "DES", "PDS"):
            rec[m] = round(st.mean(float(r[m]) for r in rs), 4)
        out.append(rec)
    return out


def main():
    print("== manuscript 3.3 mean table (replayed from w5_scores.csv) ==")
    tbl = mean_table()
    print(f"{'task':<14}{'method':<12}{'MAE':>9}{'MAE_z':>9}{'DES':>9}{'PDS':>9}")
    for r in tbl:
        print(f"{r['task']:<14}{r['method']:<12}{r['MAE']:>9.4f}"
              f"{r['MAE_z']:>9.4f}{r['DES']:>9.4f}{r['PDS']:>9.4f}")

    # Cross-check against the released source-data CSV.
    released = rows(FIG / "fig4_t2_t3_means.csv")
    key = {(r["task"], r["method"]): r for r in released}
    bad = []
    for r in tbl:
        k = (r["task"], r["method"])
        if k not in key:
            bad.append((k, "missing")); continue
        for m in ("MAE", "MAE_z", "DES", "PDS"):
            if abs(float(key[k][m + "_mean"]) - r[m]) > 0.00006:
                bad.append((k, m))
    print("mean-table cross-check:", "OK" if not bad else f"MISMATCH {bad}")

    print("\\n== cross-atlas stratification (W6, verbatim) ==")
    for r in rows(FIG / "fig_t2_by_heldout_atlas.csv"):
        if r["method"] == "scGPT":
            print(f"  {r['held_out_atlas']:<14} DES_delta={float(r['DES_delta_vs_no_change']):+.4f}"
                  f"  PDS_delta={float(r['PDS_delta_vs_no_change']):+.4f}"
                  f"  MAE_delta={float(r['MAE_delta_vs_no_change']):+.6f}")

    print("\\n== paired contrasts with bootstrap CI (seed 0, B=2000) ==")
    for r in rows(FIG / "fig4_paired_contrasts.csv"):
        if r["model"] == "scGPT" and r["baseline"] == "no_change" and r["metric"] in ("DES", "MAE"):
            print(f"  {r['task']:<14}{r['metric']:<6} delta={float(r['mean_delta']):+.6f}"
                  f"  CI95=[{float(r['ci95_low']):+.6f}, {float(r['ci95_high']):+.6f}]")

    print("\\n== SHA256 integrity of the release ==")
    man = json.loads((RELEASE / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    failed = []
    for rel, expected in man["files"].items():
        p = RELEASE / rel
        if not p.exists():
            failed.append((rel, "MISSING")); continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            failed.append((rel, "MISMATCH"))
    print(f"  {len(man['files'])} files checked:", "ALL OK" if not failed else failed)

    # Optional figure; never required for the tables.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = [r for r in tbl if r["method"] in ("no_change", "train_mean", "scGPT", "Geneformer")]
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = range(len(sub))
        ax.bar(list(xs), [r["DES"] for r in sub], color="#4C78A8")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{r['task'].split('_')[0]}\\n{r['method']}" for r in sub],
                           fontsize=7, rotation=0)
        ax.set_ylabel("DES (framework-internal)")
        ax.set_title("Fig 4 source data: directional criterion by task and method")
        fig.tight_layout()
        out = FIG / "fig4_des_by_method.png"
        fig.savefig(out, dpi=150)
        print("\\nfigure written:", out.relative_to(ROOT))
    except ImportError:
        print("\\nmatplotlib not installed; figure skipped (tables unaffected)")

    print("\\nReplay complete. No model was loaded and no score was recomputed.")
    return 0 if not bad and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def build_replay() -> None:
    write_if_changed(RELEASE / "reproduce.py", REPLAY_PY)

    nb_cells = [
        [
            "# embryo-vc-bench — read-only replay of the released main tables",
            "#",
            "This notebook reproduces the figure/table source data from the",
            "**already released, hash-verified artefacts**. It does NOT load",
            "scGPT or Geneformer, does NOT recompute any preregistered score,",
            "and does NOT re-run any W6 generator (those hashes are cited in",
            "manuscript 3.5). Full from-scratch scoring is in ../README.md 3.9.",
            "",
            "Run with the main environment: `./.venv/Scripts/python.exe`",
            "or `jupyter nbconvert --execute release/reproduce.ipynb`.",
        ],
        [
            "import csv, hashlib, json, statistics as st",
            "from pathlib import Path",
            "from collections import defaultdict",
            "",
            "ROOT = Path.cwd()",
            "if not (ROOT / 'results' / 'w5_scores.csv').exists():",
            "    ROOT = Path.cwd().parent",
            "RESULTS = ROOT / 'results'",
            "RELEASE = ROOT / 'release'",
            "FIG = RELEASE / 'figure_source_data'",
            "print('repository root:', ROOT)",
            "",
            "def rows(p):",
            "    with p.open(encoding='utf-8', newline='') as f:",
            "        return list(csv.DictReader(f))",
        ],
        [
            "# 1) Manuscript 3.3 mean table, recomputed from the 80 released rows",
            "scores = rows(RESULTS / 'w5_scores.csv')",
            "g = defaultdict(list)",
            "for r in scores: g[(r['task'], r['method'])].append(r)",
            "tbl = []",
            "for (task, method), rs in sorted(g.items()):",
            "    rec = {'task': task, 'method': method, 'n': len(rs)}",
            "    for m in ('MAE','MAE_z','DES','PDS'):",
            "        rec[m] = round(st.mean(float(r[m]) for r in rs), 4)",
            "    tbl.append(rec)",
            "for r in tbl: print(r)",
        ],
        [
            "# 2) Cross-check against the shipped source-data CSV",
            "rel = {(r['task'], r['method']): r for r in rows(FIG / 'fig4_t2_t3_means.csv')}",
            "bad = []",
            "for r in tbl:",
            "    k = (r['task'], r['method'])",
            "    for m in ('MAE','MAE_z','DES','PDS'):",
            "        if abs(float(rel[k][m + '_mean']) - r[m]) > 0.00006:",
            "            bad.append((k, m))",
            "print('cross-check:', 'OK — all 32 cells match' if not bad else bad)",
        ],
        [
            "# 3) Direction-magnitude dissociation: paired contrasts + bootstrap CI",
            "cc = rows(FIG / 'fig4_paired_contrasts.csv')",
            "for r in cc:",
            "    if r['model']=='scGPT' and r['baseline']=='no_change' and r['metric'] in ('DES','MAE'):",
            "        print(r['task'], r['metric'], r['mean_delta'],",
            "              'CI95', r['ci95_low'], r['ci95_high'])",
            "",
            "# 4) Cross-platform replication (three independent held-out atlases)",
            "for r in rows(FIG / 'fig_t2_by_heldout_atlas.csv'):",
            "    if r['method']=='scGPT':",
            "        print(r['held_out_atlas'], 'DES_delta', r['DES_delta_vs_no_change'],",
            "              'MAE_delta', r['MAE_delta_vs_no_change'])",
        ],
        [
            "# 5) Verify every SHA256 in the release manifest",
            "man = json.loads((RELEASE / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))",
            "fail = []",
            "for name, info in man['files'].items():",
            "    p = RELEASE / name",
            "    if not p.exists():",
            "        fail.append((name,'MISSING')); continue",
            "    if hashlib.sha256(p.read_bytes()).hexdigest() != info['sha256']:",
            "        fail.append((name,'MISMATCH'))",
            "print(len(man['files']), 'files:', 'ALL HASHES OK' if not fail else fail)",
            "print('No model was loaded and no preregistered score was recomputed.')",
        ],
        [
            "# 6) Optional Fig 4 panel (matplotlib only; tables above need no plotting lib)",
            "try:",
            "    import matplotlib; matplotlib.use('Agg')",
            "    import matplotlib.pyplot as plt",
            "    fig, ax = plt.subplots(figsize=(8,4))",
            "    ax.bar(range(len(tbl)), [r['DES'] for r in tbl], color='#4C78A8')",
            "    ax.set_xticks(range(len(tbl)))",
            "    ax.set_xticklabels([r['task'].split('_')[0] + '\\n' + r['method'] for r in tbl], fontsize=7)",
            "    ax.set_ylabel('DES (framework-internal)'); fig.tight_layout()",
            "    out = FIG / 'fig4_des_by_method.png'; fig.savefig(out, dpi=150)",
            "    print('figure written:', out)",
            "except ImportError:",
            "    print('matplotlib not installed; figure skipped')",
        ],
    ]
    def cell(src: list) -> dict:
        return {"cell_type": "code" if any(
            not ln.startswith("#") and ln.strip() for ln in src) else "markdown",
            "metadata": {}, "source": [ln + "\n" for ln in src],
            "outputs": [], "execution_count": None}

    # First cell is pure markdown.
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": [ln + "\n" for ln in nb_cells[0]],
              "outputs": [], "execution_count": None}]
    cells += [cell(src) for src in nb_cells[1:]]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.14"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    write_if_changed(RELEASE / "reproduce.ipynb",
                     json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


# --------------------------------------------------------------------------
# 4. Manifest + R7 evidence summary
# --------------------------------------------------------------------------
def build_manifest(plat: dict, figmeta: dict) -> dict:
    files = {}
    for p in sorted(RELEASE.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(RELEASE).as_posix()
        if rel in ("RELEASE_MANIFEST.json", "figure_source_data/fig4_des_by_method.png"):
            continue
        files[rel] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    man = {
        "schema_version": "w7-release-manifest-v1",
        "record_date": RECORD_DATE,
        "note": "This file does not contain its own hash (self-reference loop). Generated PNGs are excluded.",
        "n_files": len(files),
        "files": files,
        "pending_identifiers": [
            "[[PENDING_GITHUB_URL]]", "[[PENDING_ZENODO_DOI]]",
            "[[PENDING_RELEASE_DATE]]", "[[PENDING_CODE_LICENSE]]",
        ],
        "refresh_command": "python w7_release.py --refresh",
    }
    write_if_changed(RELEASE / "RELEASE_MANIFEST.json",
                     json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    return man


def build_r7_summary(plat: dict, figmeta: dict, man: dict) -> dict:
    evidence = {}
    for name in ["w5_scores.csv", "w5_score_summary.json", "w5_verify_real_adapters.json",
                 "w5_scoring_plan.json", "w5_amendment_decision.json",
                 "w6_atlas_replication.json", "w6_null_audit.json",
                 "w6_wording_audit.json", "w6_ceiling.json",
                 "w4_official_de_metrics.csv"]:
        p = RESULTS / name
        if p.exists():
            evidence["results/" + name] = sha(p)

    summary = {
        "schema_version": "w7-release-v1",
        "record_date": RECORD_DATE,
        "status": "W7 LOCAL ARTEFACTS COMPLETE; public identifiers PENDING",
        "note": "NO timestamp and NO release-file hash in this record, so the manuscript [[R7]] citation stays stable when the author later fills GitHub URL / Zenodo DOI / licence.",
        "hardware": {
            "os": plat["os"], "architecture": plat["architecture"],
            "processor": plat["processor"],
            "cpu_logical": plat["cpu_logical"], "cpu_physical": plat["cpu_physical"],
            "ram_total_gib": plat["ram_total_gib"],
            "ram_available_at_scoring": "~4.9 GB",
            "gpu": plat["gpu"],
            "disk_free_at_decision_gb": plat["disk_free_at_decision_gb"],
        },
        "software": {
            "python": plat["python"],
            "main_environment_pinned_distributions": plat["two_environments"]["main"]["n_pinned"],
            "model_environment_pinned_distributions": plat["two_environments"]["models"]["n_pinned"],
            "key_pins": "scvi-tools 1.5.2, scanpy 1.13.0, anndata 0.13.4, cell-eval 0.8.2, torch 2.15.0+cpu, transformers 4.46.3, datasets 2.21.0, scgpt 0.2.4",
            "two_environment_reason": plat["two_environments"]["reason"],
        },
        "seeds_and_protocol": {
            "model_seeds": SEEDS,
            "permutation_B": 1000,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_rng_seed": 0,
            "top_k": 20,
            "multiple_testing": "Benjamini-Hochberg across 4 methods per block",
            "gene_axis": 17555,
            "t3_shared_feature_axis": 11502,
            "adapter_protocol": "w5-adapter-protocol-2 (target-gene TOKEN ablation)",
            "mandatory_baselines": ["no_change", "train_mean"],
            "additive_baseline": "NOT APPLICABLE — no combinatorial perturbations in the VCC DE tables; never imputed",
        },
        "wall_clock": {
            "preregistered_scoring_started_utc": "2026-11-18T05:23:34Z",
            "preregistered_scoring_finished_utc": "2026-11-18T06:36:10Z",
            "preregistered_scoring_minutes": 72.6,
            "scoring_exit_code": 0,
            "scoring_errors": 0,
            "scGPT_engineering_verification_seconds": 274.86,
            "Geneformer_engineering_verification_seconds": 136.12,
            "w6_ceiling_subset_build_seconds": 30.6,
            "w6_ceiling_cli_seconds": 28.0,
            "w3_prototype_seconds": 1.66,
            "note": "Engineering verification, W3 prototype and W4 baseline calibration are NOT model-performance timings.",
        },
        "failures_and_omissions": [
            "6.9 GB validation h5ad arrived as a zero-filled 44/826-chunk residual; W1b selective-resume repair restored 782 chunks and 27/27 content-integrity checks passed (98,927 cells x 18,080 genes).",
            "cell-eval 0.8.2 score_agg_metrics raised IndexError on ONE-best metrics with empty float64 indices; worked around by an equivalent source-faithful recomputation (avg_score 0.218918), without patching site-packages.",
            "W3 was a 500-cell / 2,000-gene / 5-epoch CPU feasibility prototype only; the shared latent time axis tau was NEVER fitted, so H1 macro-F1 label transfer and the whole H3 tau-distance correlation are unexecutable.",
            "Only three licence-verified independent atlases were available (3 < 5 folds); per the registered rule H1 is demoted to exploratory.",
            "Zero-shot adapter protocol v1 (zeroing expression values) produced identically zero deltas on real cells and was rejected BEFORE any model score; protocol v2 uses target-gene token ablation.",
            "torchtext namespace stubs, a pure-PyTorch _PureTorchFlashMHA replacement for Wqkv, and a frozen Geneformer V1 rank-encoding dictionary were required because several C extensions do not build on Python 3.13.",
            "Full-data --ceiling was not executable at ~4.9 GB available RAM; a deterministic seed-0 11,961-cell subsample was used and is reported as a subsample-depth bound only.",
            "NO model was silently omitted: scGPT and Geneformer are the complete preregistered set; scFoundation, GEARS and CPA are non-preregistered future extensions.",
            "13 UNVERIFIED-licence data rows were excluded (BLOCKED_LICENCE); 21 VERIFIED_POLICY rows await institutional confirmation before Table 1 / [[R1]] can be filled.",
        ],
        "licence_audit": {
            "data_manifest_rows": 42,
            "verified_cc0": 8, "verified_policy": 21, "unverified_excluded": 13,
            "model_manifest_rows": 37,
            "model_verified_mit": 4, "model_verified_apache2": 33,
            "model_excluded_for_licence": 1,
            "model_excluded_detail": "VirtualCell2025/scGPT_human mirror carried no licence declaration; frozen set uses perturblab/scgpt-human (MIT) and ctheodoris/Geneformer (Apache-2.0).",
        },
        "release": {
            "n_manifest_files": man["n_files"],
            "n_figure_source_data_files": len(figmeta),
            "reproduce_entries": ["release/reproduce.py", "release/reproduce.ipynb"],
            "metadata": "release/metadata.jsonld (schema.org SoftwareSourceCode + Dataset)",
            "fair_selfcheck": "release/FAIR_selfcheck.md — PARTIAL; Findable/Accessible blocked on public identifiers",
        },
        "pending_author_actions": [
            "Choose code licence (MIT / Apache-2.0 / CC0-1.0) and replace LICENSE.md.",
            "Create public GitHub repository; fill [[PENDING_GITHUB_URL]].",
            "Deposit tagged Zenodo release; fill [[PENDING_ZENODO_DOI]] and [[PENDING_RELEASE_DATE]].",
            "Fill manuscript [[R9]]: repository/DOI/licence/OS/reproduction commands, CRediT, Funding, generative-AI disclosure, COI two-way choice.",
            "Institutionally confirm the 21 VERIFIED_POLICY rows before filling [[R1]] / Table 1.",
            "Verify every reference DOI via Crossref / Europe PMC.",
            "Remove DECISION RECORD, H2 ADJUDICATION RECORD, appendices A/B and the Chinese COI note before submission.",
            "Change w5_score_summary.json status from 'draft, not yet released' only on final author sign-off.",
        ],
        "compliance": {
            "no_model_loaded_in_w7": True,
            "no_new_model_score": True,
            "no_new_inferential_test": True,
            "no_preregistered_criterion_changed": True,
            "no_w6_generator_re_run": True,
            "third_osf_update_required": False,
            "manuscript_slots_filled_by_w7": ["R7"],
            "manuscript_slots_still_pending": ["R1", "R9"],
            "manuscript_slots_permanent_placeholders": ["R2", "R3", "R5"],
        },
        "source_evidence_sha256": evidence,
    }
    write_if_changed(RESULTS / "w7_release_summary.json",
                     json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if not (a.build or a.refresh):
        a.build = True
    print("W7 release builder:", "refresh" if a.refresh else "build")
    plat = build_environment()
    figmeta = build_figure_source()
    build_replay()
    man = build_manifest(plat, figmeta)
    summary = build_r7_summary(plat, figmeta, man)
    print("  environment locks:", plat["two_environments"]["main"]["n_pinned"],
          "/", plat["two_environments"]["models"]["n_pinned"])
    print("  figure source data files:", len(figmeta))
    print("  manifest files:", man["n_files"])
    print("  R7 summary:", "results/w7_release_summary.json",
          sha(RESULTS / "w7_release_summary.json")[:16])
    print("DONE")


if __name__ == "__main__":
    main()
