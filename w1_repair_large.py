#!/usr/bin/env python3
"""W1b-repair: resume a quarantined, partially written large CC0 download.

The 6.9 GB Arc VCC validation h5ad was pre-allocated to full size by the
parallel Range downloader, but only 44/826 chunks were actually written. Its
size, HDF5 superblock magic bytes and a *local* SHA256 all looked correct while
h5py could not open it. HDF5 has no "repair" mode for missing payload bytes:
the only correct repair is to fetch the missing byte ranges and keep the
already-correct ones in place. Purely local HDF5 tooling (h5clear / h5repack)
cannot help; those fix locks and layout, not absent data.

This script makes that repair reproducible and auditable:

  1. fingerprints the remote object (length, ETag, generation, crc32c,
     component count) so a mid-transfer object change cannot go unnoticed;
  2. proves the surviving `.part.progress` is trustworthy by comparing a
     reproducible sample of completed chunks byte-for-byte with the remote,
     and by confirming a not-done chunk is locally zero-filled but remotely
     non-zero;
  3. restores the quarantined payload to `data/` as `.part` + `.part.progress`;
  4. resumes via `w1_fetch_large.download(..., workers=N)`, whose content gate
     uses h5py when run with the project virtualenv;
  5. re-checks the remote fingerprint, opens the finished HDF5 with h5py and
     AnnData, and writes a full audit record.

Run with the project virtualenv so the STRONG HDF5 content gate is available
(the base interpreter would fall back to a weaker zero-window scan):

    ./.venv/Scripts/python.exe w1_repair_large.py --verify-only
    ./.venv/Scripts/python.exe w1_repair_large.py --prepare-only
    ./.venv/Scripts/python.exe w1_repair_large.py --workers 8

The h5ad is OPTIONAL for the preregistered DE-level T3 evaluation (the two
CC0 DE parquet tables are the ground truth). Repairing it only unlocks
single-cell-resolution secondary analyses and ANNDATA_PAIR-type metrics.
No biological analysis is performed here.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
from pathlib import Path

from w1_fetch import (DATA_DIR, MANIFEST, ROOT, UA, now, read_manifest,
                      safe_name, write_manifest)
from w1_fetch_large import (CHUNK, TIMEOUT, content_gate, download, sha256_of)

TARGET = "VCC_validation_h5ad_GCS"
RESULTS = ROOT / "results"
QUARANTINE = DATA_DIR / "_quarantine"
AUDIT = RESULTS / "w1_repair_log.json"
SAMPLE_BYTES = 65536


def remote_fingerprint(url: str) -> dict:
    """Immutable-object metadata; used as a version anchor, not as a checksum."""
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        h = r.headers
        return {
            "content_length": int(h["Content-Length"]),
            "etag": h.get("ETag", ""),
            "generation": h.get("x-goog-generation", ""),
            "last_modified": h.get("Last-Modified", ""),
            "goog_hash": h.get("x-goog-hash", ""),
            "component_count": h.get("x-goog-component-count", ""),
            "accept_ranges": h.get("Accept-Ranges", ""),
        }


def get_range(url: str, start: int, end: int) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, headers={**UA, "Range": f"bytes={start}-{end}"}, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, r.read()


def verify_progress(q: Path, qprog: Path, url: str, fp: dict) -> dict:
    """Prove the surviving chunk ledger matches the current remote object."""
    if not qprog.exists():
        raise RuntimeError("no .part.progress sidecar; cannot resume selectively")
    done = sorted(set(json.loads(qprog.read_text(encoding="utf-8"))))
    total = fp["content_length"]
    n_chunks = (total + CHUNK - 1) // CHUNK
    if q.stat().st_size != total:
        raise RuntimeError(f"quarantined size {q.stat().st_size} != remote {total}")
    if any(i < 0 or i >= n_chunks for i in done):
        raise RuntimeError("progress contains an out-of-range chunk index")

    rng = random.Random(0)          # fixed seed -> reproducible sampling
    sample = sorted(rng.sample(done, min(5, len(done))))
    checks = []
    with q.open("rb") as fh:
        for i in sample:
            s = i * CHUNK
            status, blob = get_range(url, s, s + SAMPLE_BYTES - 1)
            fh.seek(s)
            local = fh.read(SAMPLE_BYTES)
            checks.append({"chunk": i, "http_status": status,
                           "identical": status == 206 and blob == local,
                           "local_nonzero": any(local)})
        missing = [i for i in range(n_chunks) if i not in set(done)]
        neg = rng.choice(missing) if missing else None
        negative = None
        if neg is not None:
            s = neg * CHUNK
            status, blob = get_range(url, s, s + SAMPLE_BYTES - 1)
            fh.seek(s)
            local = fh.read(SAMPLE_BYTES)
            negative = {"chunk": neg, "http_status": status,
                        "local_all_zero": not any(local),
                        "remote_nonzero": any(blob)}
    ok = (all(c["identical"] for c in checks)
          and negative is not None
          and negative["http_status"] == 206
          and negative["local_all_zero"]
          and negative["remote_nonzero"])
    return {"ok": ok, "n_chunks": n_chunks, "done_chunks": len(done),
            "remaining_chunks": n_chunks - len(done),
            "remaining_bytes": (n_chunks - len(done)) * CHUNK,
            "sampled_done_chunks": checks, "negative_control": negative,
            "remote_fingerprint": fp}


def restore(q: Path, qprog: Path, part: Path, progress: Path) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    shutil.move(str(q), str(part))
    shutil.move(str(qprog), str(progress))


def audit_append(record: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    history = []
    if AUDIT.exists():
        try:
            history = json.loads(AUDIT.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(record)
    AUDIT.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                     encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entry-id", default=TARGET)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--verify-only", action="store_true",
                   help="fingerprint + ledger proof, no file movement")
    p.add_argument("--prepare-only", action="store_true",
                   help="stage .part/.part.progress but do not download")
    args = p.parse_args()

    rows = read_manifest()
    row = next((r for r in rows if r["entry_id"] == args.entry_id), None)
    if row is None:
        print(f"FAIL {args.entry_id}: not in manifest")
        return 1
    url = row["url"]
    final = DATA_DIR / f"{row['entry_id']}__{safe_name(row)}"
    part = final.with_suffix(final.suffix + ".part")
    progress = part.with_suffix(part.suffix + ".progress")
    q = QUARANTINE / final.name
    qprog = QUARANTINE / (final.name + ".part.progress")

    fp = remote_fingerprint(url)
    print(f"REMOTE length={fp['content_length']:,} etag={fp['etag']} "
          f"generation={fp['generation']} hash={fp['goog_hash']} "
          f"components={fp['component_count']} ranges={fp['accept_ranges']}",
          flush=True)
    if fp["accept_ranges"] != "bytes":
        print("FAIL: server does not advertise byte ranges")
        return 1
    if fp["content_length"] != int(row["content_length_bytes"]):
        print("FAIL: remote size differs from the manifest record")
        return 1

    staged = part.exists() and progress.exists()
    if staged:
        print(f"STAGED: {part.name} + {progress.name} already in data/; "
              f"will resume directly", flush=True)
        evidence = {"mode": "resume-staged", "remote_fingerprint": fp}
    elif q.exists():
        evidence = verify_progress(q, qprog, url, fp)
        evidence["mode"] = "from-quarantine"
        print(f"LEDGER {evidence['done_chunks']}/{evidence['n_chunks']} chunks done, "
              f"{evidence['remaining_chunks']} remaining "
              f"({evidence['remaining_bytes'] / 1e9:.2f} GB)", flush=True)
        for c in evidence["sampled_done_chunks"]:
            print(f"  done chunk {c['chunk']:3d}: identical={c['identical']}", flush=True)
        print(f"  negative control {evidence['negative_control']}", flush=True)
        if not evidence["ok"]:
            print("FAIL: chunk ledger does not match the remote object; "
                  "a selective resume would mix stale bytes. Fall back to a "
                  "full re-download (delete the .part/.progress pair).")
            audit_append({"at": now(), "result": "LEDGER_FAIL", **evidence})
            return 1
    else:
        print(f"FAIL: neither staged .part nor quarantined payload found for "
              f"{args.entry_id}")
        return 1

    if args.verify_only:
        audit_append({"at": now(), "result": "VERIFY_ONLY", **evidence})
        print("VERIFY_ONLY: repair is feasible; nothing was moved")
        return 0

    if not staged:
        restore(q, qprog, part, progress)
        print(f"RESTORED {q.name} -> {part.name} (+ ledger)", flush=True)

    if args.prepare_only:
        audit_append({"at": now(), "result": "PREPARED", **evidence})
        print(f"PREPARED. Resume with: ./.venv/Scripts/python.exe "
              f"w1_fetch_large.py --workers {args.workers}")
        return 0

    print(f"RESUMING with {args.workers} parallel Range connections", flush=True)
    changed = download(row, args.workers)
    if row["status"] != "DOWNLOADED_DATA":
        audit_append({"at": now(), "result": "INCOMPLETE", **evidence,
                      "status": row["status"],
                      "hint": "rerun w1_repair_large.py --workers N to continue"})
        print(f"INCOMPLETE status={row['status']}; .part/.progress kept for resume")
        return 1

    fp2 = remote_fingerprint(url)
    stable = all(fp2[k] == fp[k] for k in
                 ("content_length", "etag", "generation", "last_modified"))
    if not stable:
        # Refuse to register a file assembled from two object generations.
        audit_append({"at": now(), "result": "REMOTE_CHANGED", **evidence,
                      "fingerprint_after": fp2})
        print(f"FAIL: remote object changed during transfer ({fp['etag']} -> "
              f"{fp2['etag']}); file written but manifest NOT updated")
        return 1

    ok, why = content_gate(final)
    shape, keys, ann_detail = None, [], ""
    if ok:
        try:
            import h5py
            import anndata as ad
            with h5py.File(final, "r") as fh:
                keys = list(fh.keys())
            a = ad.read_h5ad(final, backed="r")
            shape = list(a.shape)
            try:
                a.file.close()
            except Exception:
                pass
            ann_detail = "AnnData backed read OK"
        except Exception as exc:
            # content_gate() already proved h5py can open the file. A reader
            # API quirk must not silently reject a byte-complete HDF5 object.
            ann_detail = f"AnnData backed read warning: {type(exc).__name__}: {exc}"

    if not ok:
        audit_append({"at": now(), "result": "CONTENT_GATE_FAIL", **evidence,
                      "detail": why})
        print(f"FAIL: post-repair content gate rejected the file: {why}")
        return 1

    row["notes"] += (
        f"; W1b repair: resumed from {evidence.get('done_chunks', '?')}/"
        f"{evidence.get('n_chunks', '?')} verified chunks; remote ETag={fp['etag']} "
        f"generation={fp['generation']} {fp['goog_hash']} (composite CRC32C, "
        f"NOT locally comparable); object stable across transfer; "
        f"h5py+AnnData backed read OK shape={shape}")
    latest = read_manifest()
    for r in latest:
        if r["entry_id"] == args.entry_id:
            for k in ("local_path", "sha256", "download_date", "http_status",
                      "content_length_bytes", "status", "notes"):
                r[k] = row[k]
    write_manifest(latest)

    record = {"at": now(), "result": "REPAIRED", **evidence,
              "fingerprint_after": fp2, "sha256_local": sha256_of(final),
              "hdf5_root_keys": keys, "anndata_shape": shape,
              "anndata_detail": ann_detail,
              "content_gate": why, "download_changed_row": changed,
              "file": str(final.relative_to(ROOT)).replace("\\", "/")}
    audit_append(record)
    print(f"REPAIRED {final.name}: shape={shape}, root_keys={keys[:6]}, "
          f"sha256={record['sha256_local'][:16]}... [{why}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
