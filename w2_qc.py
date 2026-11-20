#!/usr/bin/env python3
"""W2 data-readiness audit: format, shape, identifier type and hash provenance.

This is NOT a biological result and performs NO filtering, normalisation,
imputation, clustering, differential expression or statistical testing. It only
answers: what did we actually download, is the file intact, and can it be read?

Standard library only, so it runs in the base environment as well as .venv.
Outputs (relative to this file's directory):
    results/w2_qc_report.json    full per-file audit
    results/w2_qc_summary.csv    one row per downloaded file
    results/w2_run_log.txt       command, timestamp, Python version, seeds

Usage:
    python w2_qc.py                      # audit all manifest-tracked Downloads
    python w2_qc.py --max-lines 5000    # cap rows scanned per text matrix
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data_manifest.csv"
RESULTS = ROOT / "results"
SEED = 0  # documented: this audit is deterministic and uses no random sampling

FIELDS = ["entry_id", "accession", "kind", "licence_status", "file",
          "size_bytes", "sha256_matches_manifest", "format", "n_rows",
          "n_cols", "delimiter", "header_preview", "gene_id_type",
          "zero_fraction_sample", "note"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_gene_ids(first_column_values: list[str]) -> str:
    """Guess the identifier convention from the first column of a matrix."""
    vals = [v.strip().strip('"') for v in first_column_values if v.strip()]
    if not vals:
        return "unknown"
    ensg = sum(bool(re.match(r"^ENS[GTP]\d+", v, re.I)) for v in vals)
    numeric = sum(bool(re.fullmatch(r"[\d.]+", v)) for v in vals)
    symbol = sum(bool(re.match(r"^[A-Za-z][A-Za-z0-9._-]*$", v)) for v in vals)
    if ensg / len(vals) > 0.8:
        return "Ensembl-like (ENSG/ENST)"
    if numeric / len(vals) > 0.8:
        return "numeric"
    if symbol / len(vals) > 0.8:
        return "gene symbol-like"
    return "mixed/unknown"


def sniff_delimiter(line: str) -> str:
    if "\t" in line:
        return "tab"
    if line.count(",") > line.count(";"):
        return "comma"
    if ";" in line:
        return "semicolon"
    return "unknown"


def audit_text(path: Path, gz: bool, max_lines: int) -> dict:
    """Read a delimited text matrix, counting rows and sampling zero fraction."""
    opener = gzip.open if gz else open
    n_rows = 0
    n_cols = 0
    delimiter = "unknown"
    header = ""
    first_col: list[str] = []
    numeric_total = numeric_zero = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if n_rows == 0:
                header = line.rstrip("\n\r")[:300]
                delimiter = sniff_delimiter(line)
                sep = "\t" if delimiter == "tab" else "," if delimiter == "comma" else ";"
                n_cols = len(line.rstrip("\n\r").split(sep))
            n_rows += 1
            if n_rows > 1 and len(first_col) < 200:
                sep = "\t" if delimiter == "tab" else "," if delimiter == "comma" else ";"
                parts = line.rstrip("\n\r").split(sep)
                if parts:
                    first_col.append(parts[0])
                if numeric_total < 20000:
                    for cell in parts[1:21]:
                        try:
                            v = float(cell)
                            numeric_total += 1
                            numeric_zero += int(v == 0)
                        except ValueError:
                            pass
            if max_lines and n_rows >= max_lines:
                break
    return {
        "format": "gzip-delimited-text" if gz else "delimited-text",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "delimiter": delimiter,
        "header_preview": header,
        "gene_id_type": classify_gene_ids(first_col),
        "zero_fraction_sample": round(numeric_zero / numeric_total, 4) if numeric_total else None,
        "note": f"scanned up to {max_lines} rows; no filtering applied",
    }


def audit_tar(path: Path) -> dict:
    with tarfile.open(path) as t:
        members = t.getmembers()
        names = [m.name for m in members]
    return {
        "format": "tar",
        "n_rows": len(members),
        "n_cols": None,
        "delimiter": "n/a",
        "header_preview": "; ".join(names[:5]),
        "gene_id_type": "n/a (archive)",
        "zero_fraction_sample": None,
        "note": f"{len(members)} members; not extracted",
    }


def audit_parquet(path: Path) -> dict:
    """Read parquet metadata only. pyarrow is optional: if absent, fall back."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        return {
            "format": "parquet (metadata unavailable)", "n_rows": None,
            "n_cols": None, "delimiter": "n/a", "header_preview": "",
            "gene_id_type": "n/a", "zero_fraction_sample": None,
            "note": f"pyarrow unavailable: {type(e).__name__}: {e}",
        }
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    return {
        "format": "parquet", "n_rows": pf.metadata.num_rows,
        "n_cols": len(cols), "delimiter": "n/a",
        "header_preview": ", ".join(cols[:14]),
        "gene_id_type": "n/a (DE table)", "zero_fraction_sample": None,
        "note": f"{pf.metadata.num_row_groups} row group(s); columns read from "
                f"Arrow schema, no expression values inspected",
    }


def audit_opaque(path: Path) -> dict:
    with path.open("rb") as fh:
        head = fh.read(200)
    if head.startswith(b"PAR1"):
        return audit_parquet(path)  # content sniff, independent of file extension
    kind = "json" if head.lstrip().startswith(b"{") else (
        "xml" if head.lstrip().startswith(b"<") else "binary/opaque")
    return {
        "format": kind, "n_rows": None, "n_cols": None,
        "delimiter": "n/a", "header_preview": head.decode("utf-8", "replace")[:180],
        "gene_id_type": "n/a", "zero_fraction_sample": None,
        "note": "metadata / opaque payload; not parsed as a matrix",
    }


def audit_file(row: dict, max_lines: int) -> dict | None:
    rel = row.get("local_path", "")
    if not rel:
        return None
    p = ROOT / rel
    if not p.exists():
        return None
    name = p.name
    base = {
        "entry_id": row["entry_id"], "accession": row["accession"],
        "kind": row["kind"], "licence_status": row["licence_status"],
        "file": rel, "size_bytes": p.stat().st_size,
        "sha256_matches_manifest": sha256_of(p) == row.get("sha256", ""),
    }
    try:
        if name.endswith(".parquet"):
            detail = audit_parquet(p)
        elif name.endswith(".tar"):
            detail = audit_tar(p)
        elif name.endswith(".gz"):
            detail = audit_text(p, True, max_lines)
        elif name.endswith((".txt", ".csv", ".tsv")):
            detail = audit_text(p, False, max_lines)
        else:
            detail = audit_opaque(p)
    except Exception as e:
        detail = {
            "format": "ERROR", "n_rows": None, "n_cols": None,
            "delimiter": "n/a", "header_preview": "",
            "gene_id_type": "n/a", "zero_fraction_sample": None,
            "note": f"{type(e).__name__}: {e}",
        }
    return {**base, **detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-lines", type=int, default=20000,
                    help="cap rows scanned per text matrix (0 = unlimited)")
    args = ap.parse_args()
    if not MANIFEST.exists():
        raise SystemExit("data_manifest.csv missing; run w1_fetch.py init first")
    RESULTS.mkdir(exist_ok=True)
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    targets = [r for r in rows
               if r["status"].startswith("DOWNLOADED") and r["local_path"]]
    reports = []
    for r in targets:
        out = audit_file(r, args.max_lines)
        if out:
            reports.append(out)
            print(f"  {out['entry_id'][:44]:44s} {out['format']:20s} "
                  f"rows={out['n_rows']} cols={out['n_cols']} "
                  f"hash={'OK' if out['sha256_matches_manifest'] else 'MISMATCH'}")
    payload = {
        "generated_utc": now(),
        "purpose": "W2 data-readiness audit; NOT a biological result",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "max_lines_per_matrix": args.max_lines,
        "n_files_audited": len(reports),
        "n_hash_mismatches": sum(not r["sha256_matches_manifest"] for r in reports),
        "files": reports,
    }
    (RESULTS / "w2_qc_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "w2_qc_summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(reports)
    (RESULTS / "w2_run_log.txt").write_text(
        f"command: python w2_qc.py --max-lines {args.max_lines}\n"
        f"timestamp_utc: {now()}\n"
        f"python: {platform.python_version()}\n"
        f"platform: {platform.platform()}\n"
        f"seed: {SEED} (no stochastic step)\n"
        f"manifest: data_manifest.csv\n"
        f"files_audited: {len(reports)}\n",
        encoding="utf-8")
    print(f"W2 audit: {len(reports)} files -> {RESULTS.name}/; "
          f"hash mismatches = {payload['n_hash_mismatches']}")
    return 0 if payload["n_hash_mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
