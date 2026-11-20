"""Read-only replay of the released main tables and figure.

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

    print("\n== cross-atlas stratification (W6, verbatim) ==")
    for r in rows(FIG / "fig_t2_by_heldout_atlas.csv"):
        if r["method"] == "scGPT":
            print(f"  {r['held_out_atlas']:<14} DES_delta={float(r['DES_delta_vs_no_change']):+.4f}"
                  f"  PDS_delta={float(r['PDS_delta_vs_no_change']):+.4f}"
                  f"  MAE_delta={float(r['MAE_delta_vs_no_change']):+.6f}")

    print("\n== paired contrasts with bootstrap CI (seed 0, B=2000) ==")
    for r in rows(FIG / "fig4_paired_contrasts.csv"):
        if r["model"] == "scGPT" and r["baseline"] == "no_change" and r["metric"] in ("DES", "MAE"):
            print(f"  {r['task']:<14}{r['metric']:<6} delta={float(r['mean_delta']):+.6f}"
                  f"  CI95=[{float(r['ci95_low']):+.6f}, {float(r['ci95_high']):+.6f}]")

    print("\n== SHA256 integrity of the release ==")
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
        ax.set_xticklabels([f"{r['task'].split('_')[0]}\n{r['method']}" for r in sub],
                           fontsize=7, rotation=0)
        ax.set_ylabel("DES (framework-internal)")
        ax.set_title("Fig 4 source data: directional criterion by task and method")
        fig.tight_layout()
        out = FIG / "fig4_des_by_method.png"
        fig.savefig(out, dpi=150)
        print("\nfigure written:", out.relative_to(ROOT))
    except ImportError:
        print("\nmatplotlib not installed; figure skipped (tables unaffected)")

    print("\nReplay complete. No model was loaded and no score was recomputed.")
    return 0 if not bad and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
