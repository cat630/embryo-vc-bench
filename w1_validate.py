#!/usr/bin/env python3
"""W1b - deep, format-aware integrity gate for W1 downloads.

Why this exists
---------------
`data_manifest.csv` stores a *local* SHA256. It proves a file has not changed
since registration; it does NOT prove the download matched the remote object.
A pre-allocated, partially written 6.9 GB HDF5 file can have the right size,
the right superblock magic bytes, and a self-consistent SHA256 while being
unreadable. This script opens every downloaded file with its native reader and
refuses to let a structurally invalid file reach W2 or W4.

Run with the project virtualenv (h5py / pyarrow / anndata live there):

    ./.venv/Scripts/python.exe w1_validate.py
    ./.venv/Scripts/python.exe w1_validate.py --quarantine

Outputs (all reproducible, no biological analysis is performed here):
    results/w1_integrity_report.json
    results/w1_integrity_summary.csv
    results/w1_integrity_log.txt
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

from w1_fetch import DATA_DIR, MANIFEST, ROOT, now, read_manifest, write_manifest

RESULTS = ROOT / "results"
QUARANTINE = DATA_DIR / "_quarantine"
FIELDS = ["entry_id", "licence_status", "format", "sha256_ok", "content_ok",
          "verdict", "detail", "size_bytes", "local_path"]


def sha256_of(path: Path) -> str:
    """Return the SHA256 of a local file (local integrity only, not authenticity)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gzip(path: Path) -> tuple[bool, str]:
    """Fully decompress to verify the gzip CRC, not just the header."""
    n = 0
    with gzip.open(path, "rb") as fh:
        while True:
            block = fh.read(4 * 1024 * 1024)
            if not block:
                break
            n += len(block)
    return True, f"gzip CRC OK; {n:,} uncompressed bytes"


def check_tar(path: Path) -> tuple[bool, str]:
    """List all members; a truncated tar fails here or on member access."""
    with tarfile.open(path, "r:") as tf:
        members = tf.getmembers()
        total = sum(m.size for m in members)
    return True, f"{len(members)} members, {total:,} bytes uncompressed"


def check_parquet(path: Path) -> tuple[bool, str]:
    """Read parquet metadata plus one row group (footer integrity)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    n_cols = len(pf.schema_arrow.names)
    n_groups = pf.metadata.num_row_groups
    if n_groups:
        pf.read_row_group(0)
    return True, (f"{pf.metadata.num_rows:,} rows x {n_cols} cols, "
                  f"{n_groups} row group(s)")


def check_h5ad(path: Path) -> tuple[bool, str]:
    """Open with h5py and read the AnnData structure in backed mode."""
    import h5py
    import anndata as ad
    with h5py.File(path, "r") as fh:
        keys = list(fh.keys())
    if not keys:
        return False, "HDF5 root group has no members"
    a = ad.read_h5ad(path, backed="r")
    return True, (f"shape={a.shape}, root_keys={keys[:6]}, "
                  f"obs_cols={list(a.obs.columns)[:6]}")


def check_text(path: Path) -> tuple[bool, str]:
    """Minimal readability check for provenance metadata / plain matrices."""
    with path.open("rb") as fh:
        head = fh.read(65536)
    if not head:
        return False, "empty file"
    head.decode("utf-8", errors="replace")
    return True, f"first {len(head):,} bytes readable as text"


def validate(path: Path) -> tuple[str, bool, str]:
    """Dispatch to a native reader by magic bytes first, filename second.

    Returns (detected_format, content_ok, detail).
    """
    with path.open("rb") as fh:
        head = fh.read(512)
    name = path.name.lower()
    fmt, checker = "unknown", check_text
    if head.startswith(b"\x89HDF\r\n\x1a\n") or ".h5ad" in name:
        fmt, checker = "h5ad", check_h5ad
    elif head.startswith(b"PAR1") or name.endswith(".parquet"):
        fmt, checker = "parquet", check_parquet
    elif head[:2] == b"\x1f\x8b" or name.endswith(".gz"):
        fmt, checker = "gzip", check_gzip
    elif head[257:262] == b"ustar" or name.endswith(".tar"):
        fmt, checker = "tar", check_tar
    try:
        return fmt, *checker(path)
    except Exception as exc:  # a broken reader must be recorded, not hidden
        return fmt, False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quarantine", action="store_true",
                   help="move failing files to data/_quarantine and mark the manifest")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows = read_manifest()
    records: list[dict] = []
    log: list[str] = []

    for row in rows:
        rel = row["local_path"]
        if not rel or not row["status"].startswith("DOWNLOADED"):
            continue
        path = ROOT / rel
        if not path.exists():
            rec = dict(entry_id=row["entry_id"], licence_status=row["licence_status"],
                       format="missing", sha256_ok=False, content_ok=False,
                       verdict="MISSING", detail="local_path does not exist",
                       size_bytes=0, local_path=rel)
            records.append(rec)
            continue
        digest = sha256_of(path)
        hash_ok = (digest == row["sha256"])
        fmt, content_ok, detail = validate(path)
        verdict = "PASS" if (hash_ok and content_ok) else "FAIL"
        rec = dict(entry_id=row["entry_id"], licence_status=row["licence_status"],
                   format=fmt, sha256_ok=hash_ok, content_ok=content_ok,
                   verdict=verdict, detail=detail, size_bytes=path.stat().st_size,
                   local_path=rel)
        records.append(rec)
        log.append(f"{verdict:7s} {row['entry_id']:48s} {fmt:9s} "
                   f"sha={'OK' if hash_ok else 'BAD'} content={'OK' if content_ok else 'BAD'} "
                   f"| {detail[:110]}")

        if args.quarantine and verdict == "FAIL":
            QUARANTINE.mkdir(exist_ok=True)
            target = QUARANTINE / path.name
            if target.exists():
                target.unlink()
            shutil.move(str(path), str(target))
            # Also move any chunk-progress sidecar, so a later resume cannot
            # trust a partial write that belongs to this quarantined file.
            for sidecar in path.parent.glob(path.name + ".part.progress"):
                shutil.move(str(sidecar), str(QUARANTINE / sidecar.name))
            row["status"] = "CORRUPT_INCOMPLETE"
            row["sha256"] = ""          # never let a self-hash re-validate it
            row["local_path"] = ""      # no analysis script can pick it up
            row["notes"] += ("; W1b content gate FAILED (" + detail[:200] +
                             f"); quarantined at {target.relative_to(ROOT)}")
            log.append(f"  -> QUARANTINED {target.relative_to(ROOT)}; "
                       f"manifest status=CORRUPT_INCOMPLETE")

    if args.quarantine:
        write_manifest(rows)

    n_pass = sum(r["verdict"] == "PASS" for r in records)
    n_fail = len(records) - n_pass
    summary = {
        "generated_at": now(),
        "script": "w1_validate.py",
        "purpose": "W1b deep format-aware integrity gate (no biological analysis)",
        "sha256_semantics": "local immutability only; NOT upstream authenticity",
        "files_checked": len(records),
        "pass": n_pass,
        "fail": n_fail,
        "quarantine_applied": bool(args.quarantine),
        "records": records,
    }
    (RESULTS / "w1_integrity_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (RESULTS / "w1_integrity_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)
    log.append(f"W1b integrity: {n_pass}/{len(records)} PASS, {n_fail} FAIL")
    (RESULTS / "w1_integrity_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log[-max(1, len(log) - 40):]))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
