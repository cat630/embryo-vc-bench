#!/usr/bin/env python3
"""Resumable HTTP-Range downloader for multi-GB, explicitly CC0 datasets.

Standard library only (native readers are used opportunistically when the
virtualenv packages are importable). Companion to w1_fetch.py, which
deliberately refuses files above its ordinary size ceiling. Only rows with
licence_status VERIFIED_CC0 and status REQUIRES_LARGE_FETCH / DOWNLOADED_DATA
are eligible.

    python w1_fetch_large.py --workers 8           # validation h5ad (6.9 GB)
    python w1_fetch_large.py --all --workers 8     # also the 15.5 GB train h5ad
    python w1_fetch_large.py --verify-only         # hash + content gate

Resume
------
An interrupted parallel run leaves a pre-allocated `.part` plus a
`.part.progress` file listing completed chunk indices. Only that progress file
is authoritative. A full-sized `.part` WITHOUT a progress file is NOT trusted:
it may be a pre-allocated, mostly zero-filled file. This rule was added after
a 6.9 GB HDF5 file was wrongly registered as complete on 2026-11-17.

Content gate
------------
Size, magic bytes and a self-recorded SHA256 are necessary but not sufficient.
Before registration every download must pass `content_gate()`: HDF5 must open
and expose a non-empty root group (or, without h5py, pass a zero-window scan).
A rejected file is left as `.part`, never registered, never analysed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from w1_fetch import (DATA_DIR, MANIFEST, UA, VERIFIED, now, read_manifest,
                      safe_name, write_manifest)

CHUNK = 8 * 1024 * 1024          # 8 MiB per request
MAX_RETRIES = 5
TIMEOUT = 120
DEFAULT_TARGET = "VCC_validation_h5ad_GCS"
OPTIONAL_TARGET = "VCC_train_h5ad_GCS"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def sha256_of(path: Path) -> str:
    """Return the local SHA256 (immutability, not upstream authenticity)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sparse_scan(path: Path, windows: int = 24) -> tuple[bool, str]:
    """Reject a pre-allocated file that is still mostly zero-filled.

    Samples `windows` evenly spaced 64 KiB blocks. A valid HDF5/Parquet/binary
    object will not be >50% all-zero windows; a truncated pre-allocated file
    will be. Used as the fallback when native readers are unavailable.
    """
    size = path.stat().st_size
    if size == 0:
        return False, "empty file"
    zero = 0
    with path.open("rb") as fh:
        for i in range(windows):
            fh.seek(int(size * (i + 0.5) / windows))
            block = fh.read(65536)
            if block and not any(block):
                zero += 1
    frac = zero / windows
    if frac > 0.5:
        return False, (f"{zero}/{windows} sampled windows are all-zero "
                       f"({frac:.0%}) - pre-allocated / incomplete")
    return True, f"{zero}/{windows} sampled windows all-zero"


def content_gate(path: Path) -> tuple[bool, str]:
    """Structurally validate a download before it may be registered.

    Returns (ok, human-readable reason). h5py / pyarrow are optional: when they
    cannot be imported the standard-library zero-window scan is used instead,
    and this is stated in the reason so the weaker check is never hidden.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
        name = path.name.lower()
        if head.startswith(HDF5_MAGIC) or ".h5ad" in name:
            if not head.startswith(HDF5_MAGIC):
                return False, "missing HDF5 superblock signature"
            try:
                import h5py
            except ImportError:
                ok, why = sparse_scan(path)
                return ok, f"HDF5, h5py unavailable (weak check): {why}"
            try:
                with h5py.File(path, "r") as fh:
                    keys = list(fh.keys())
                if not keys:
                    return False, "HDF5 root group is empty"
                return True, f"HDF5 opens, root keys={keys[:6]}"
            except Exception as exc:
                return False, f"HDF5 open failed: {type(exc).__name__}: {exc}"
        if head.startswith(b"PAR1"):
            try:
                import pyarrow.parquet as pq
            except ImportError:
                ok, why = sparse_scan(path)
                return ok, f"parquet, pyarrow unavailable (weak check): {why}"
            try:
                pf = pq.ParquetFile(path)
                return True, f"parquet opens, rows={pf.metadata.num_rows:,}"
            except Exception as exc:
                return False, f"parquet open failed: {type(exc).__name__}: {exc}"
        return sparse_scan(path)
    except Exception as exc:
        return False, f"content gate error: {type(exc).__name__}: {exc}"


def head_size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int) -> tuple[int, bytes]:
    """Return (HTTP status, body). 206 = Range honoured; 200 = server ignored
    Range and returned the whole file (some NCBI endpoints do this)."""
    req = urllib.request.Request(
        url, headers={**UA, "Range": f"bytes={start}-{end}"}, method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, r.read()


def download_parallel(row: dict, url: str, final: Path, part: Path,
                      total: int, workers: int) -> bool:
    """Fetch independent byte ranges concurrently into one pre-allocated file.

    A `.progress` file records completed chunk indices, so an interrupted run
    resumes without re-downloading finished chunks. Each thread opens its own
    file handle and seeks to its own offset, so there is no shared seek state.
    """
    n_chunks = (total + CHUNK - 1) // CHUNK
    progress = part.with_suffix(part.suffix + ".progress")
    done: set[int] = set()
    if progress.exists():
        try:
            done = set(json.loads(progress.read_text(encoding="utf-8")))
        except Exception:
            done = set()
    # Pre-allocate so seeks beyond EOF are valid; existing bytes are preserved.
    with part.open("r+b" if part.exists() else "wb") as fh:
        fh.truncate(total)
    lock = threading.Lock()
    remaining = [i for i in range(n_chunks) if i not in done]
    print(f"PARALLEL {row['entry_id']}: {n_chunks} chunks, {len(done)} done, "
          f"{len(remaining)} remaining, workers={workers}", flush=True)

    def one(i: int) -> tuple[int, bool]:
        start = i * CHUNK
        end = min(start + CHUNK, total) - 1
        for attempt in range(MAX_RETRIES):
            try:
                status, blob = fetch_range(url, start, end)
                if status != 206 or len(blob) != end - start + 1:
                    raise RuntimeError(f"bad chunk {i}: status={status} len={len(blob)}")
                with part.open("r+b") as fh:
                    fh.seek(start)
                    fh.write(blob)
                with lock:
                    done.add(i)
                    progress.write_text(json.dumps(sorted(done)), encoding="utf-8")
                return i, True
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    return i, False
                time.sleep(min(2 ** attempt, 8))
        return i, False

    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, ok in ex.map(one, remaining):
            if ok:
                finished += 1
            if finished % 50 == 0 and finished:
                print(f"  {finished}/{len(remaining)} chunks "
                      f"({(len(done) / n_chunks) * 100:.1f}% of file)", flush=True)
    failed = n_chunks - len(done)
    if failed:
        print(f"PARTIAL {row['entry_id']}: {failed} chunk(s) failed; "
              f"rerun to resume from .progress", flush=True)
        return False
    ok, why = content_gate(part)
    if not ok:
        print(f"INVALID {row['entry_id']}: content gate failed ({why}); "
              f".part and .progress kept, NOT registered", flush=True)
        progress.write_text(json.dumps([]), encoding="utf-8")
        return False
    progress.unlink(missing_ok=True)
    part.replace(final)
    digest = sha256_of(final)
    row["local_path"] = str(final.relative_to(MANIFEST.parent)).replace("\\", "/")
    row["sha256"] = digest
    row["download_date"] = now()
    row["http_status"] = "200"
    row["content_length_bytes"] = str(total)
    row["status"] = "DOWNLOADED_DATA"
    row["notes"] += "; parallel HTTP Range + content gate passed"
    print(f"DONE {row['entry_id']}: {final.name}, sha256={digest} [{why}]",
          flush=True)
    return True


def download(row: dict, workers: int = 1) -> bool:
    """Download or validate one manifest row. Returns True if the row changed."""
    url = row["url"]
    if not url.startswith("https://"):
        print(f"SKIP {row['entry_id']}: non-HTTPS")
        return False
    if row["licence_status"] not in VERIFIED:
        print(f"SKIP {row['entry_id']}: licence_status={row['licence_status']}")
        return False
    DATA_DIR.mkdir(exist_ok=True)
    final = DATA_DIR / f"{row['entry_id']}__{safe_name(row)}"
    part = final.with_suffix(final.suffix + ".part")
    progress = part.with_suffix(part.suffix + ".progress")
    qdir = DATA_DIR / "_quarantine"

    if final.exists():
        digest = sha256_of(final)
        if digest == row["sha256"] and not progress.exists():
            ok, why = content_gate(final)
            if ok:
                print(f"KEPT {row['entry_id']}: hash + content gate passed [{why}]")
                return False
            reason = f"self-consistent hash but invalid content: {why}"
        elif progress.exists():
            reason = "stale .part.progress proves the file was never finished"
        else:
            reason = "existing file hash mismatch"
        print(f"QUARANTINE {row['entry_id']}: {reason}")
        qdir.mkdir(exist_ok=True)
        target = qdir / final.name
        if target.exists():
            target.unlink()
        final.replace(target)
        for sidecar in DATA_DIR.glob(final.name + ".part.progress"):
            sidecar.unlink()
        part.unlink(missing_ok=True)
        row["sha256"] = ""
        row["local_path"] = ""
        row["status"] = "CORRUPT_INCOMPLETE"
        row["notes"] += (f"; quarantined by content gate ({reason[:180]}); "
                         f"file at {target.relative_to(MANIFEST.parent)}")
        return True

    total = head_size(url)
    # A progress file always means chunk-wise writes: use the chunk path even
    # with workers=1. Never let a full-sized .part masquerade as a serial file.
    if progress.exists() or (workers > 1 and total > 512 * 1024 * 1024):
        existing = part.stat().st_size if part.exists() else 0
        n_full = existing // CHUNK
        if 0 < existing < total and n_full and not progress.exists():
            progress.write_text(json.dumps(list(range(n_full))), encoding="utf-8")
            print(f"  seeded .progress with {n_full} contiguous chunks from .part",
                  flush=True)
        return download_parallel(row, url, final, part, total, max(workers, 1))

    offset = part.stat().st_size if part.exists() else 0
    if offset == total and total:
        print(f"WARN {row['entry_id']}: full-sized .part without .progress is "
              f"NOT trusted (pre-allocated?); restarting from byte 0")
        part.unlink()
        offset = 0
    if offset > total:
        print(f"WARN {row['entry_id']}: .part larger than remote, restarting")
        part.unlink()
        offset = 0
    print(f"START {row['entry_id']}: {total:,} bytes, resume from {offset:,} "
          f"({offset / total * 100:.1f}%)")

    mode = "ab" if offset else "wb"
    with part.open(mode) as out:
        while offset < total:
            end = min(offset + CHUNK, total) - 1
            for attempt in range(MAX_RETRIES):
                try:
                    status, blob = fetch_range(url, offset, end)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"FAIL {row['entry_id']} at byte {offset:,}: "
                              f"{type(e).__name__}: {e}; .part kept for resume")
                        return False
                    wait = 2 ** attempt
                    print(f"  retry {attempt + 1}/{MAX_RETRIES} after {wait}s "
                          f"({type(e).__name__})")
                    time.sleep(wait)
            if status == 200 and offset == 0:
                if len(blob) != total:
                    print(f"FAIL full-body length {len(blob):,} != {total:,}")
                    return False
                out.write(blob)
                out.flush()
                offset = len(blob)
                break
            if status != 206:
                print(f"FAIL unexpected HTTP {status} at byte {offset:,}")
                return False
            expected = end - offset + 1
            if len(blob) != expected:
                print(f"FAIL short chunk at {offset:,}: got {len(blob)} expected {expected}")
                return False
            out.write(blob)
            out.flush()
            offset += len(blob)
            if offset % (256 * 1024 * 1024) < CHUNK or offset == total:
                print(f"  {offset:,}/{total:,} ({offset / total * 100:.1f}%)", flush=True)

    ok, why = content_gate(part)
    if not ok:
        print(f"FAIL {row['entry_id']}: content gate rejected .part ({why}); "
              f"NOT registered")
        row["status"] = "CORRUPT_INCOMPLETE"
        row["notes"] += f"; rejected by content gate: {why[:180]}"
        return True
    part.replace(final)
    digest = sha256_of(final)
    row["local_path"] = str(final.relative_to(MANIFEST.parent)).replace("\\", "/")
    row["sha256"] = digest
    row["download_date"] = now()
    row["http_status"] = "200"
    row["content_length_bytes"] = str(total)
    row["status"] = "DOWNLOADED_DATA"
    row["notes"] += f"; serial HTTP Range + content gate passed [{why[:120]}]"
    print(f"DONE {row['entry_id']}: {final.name}, sha256={digest}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="also fetch the optional training h5ad")
    p.add_argument("--entry-ids", nargs="*", default=[],
                   help="additional verified manifest rows to download with Range resume")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel Range connections for files >512 MiB (try 8)")
    p.add_argument("--verify-only", action="store_true",
                   help="recheck hash AND structural content, without downloading")
    args = p.parse_args()

    rows = read_manifest()
    by_id = {r["entry_id"]: r for r in rows}
    # Large, fast CC0 files first; slower NCBI supplementary files last, so a
    # slow endpoint can never delay the T3 ground truth.
    targets = ([DEFAULT_TARGET] + ([OPTIONAL_TARGET] if args.all else [])
               + list(args.entry_ids))
    if args.verify_only:
        ok = 0
        for tid in targets:
            r = by_id.get(tid)
            if not r or not r["local_path"]:
                print(f"MISSING {tid}: no local copy")
                continue
            f = MANIFEST.parent / r["local_path"]
            if not f.exists():
                print(f"MISSING {tid}: {f}")
                continue
            good = sha256_of(f) == r["sha256"]
            content_ok, why = content_gate(f)
            good = good and content_ok
            ok += int(good)
            print(f"{'PASS' if good else 'FAIL'} {tid} sha256={r['sha256'][:16]} "
                  f"content={'OK' if content_ok else 'BAD'} [{why}]")
        print(f"verify-only: {ok}/{len(targets)} pass hash + content gate")
        return 0 if ok == len(targets) else 1

    changed = False
    for tid in targets:
        r = by_id.get(tid)
        if r is None:
            print(f"SKIP {tid}: not in manifest; run w1_fetch.py init first")
            continue
        changed |= download(r, args.workers)
    if changed:
        # Re-read the manifest before writing: another w1_fetch.py run may have
        # finished while a multi-GB download was in progress. Merge only the
        # rows this script actually touched, so no concurrent result is lost.
        latest = read_manifest()
        touched = {tid: by_id[tid] for tid in targets if tid in by_id}
        for r in latest:
            src = touched.get(r["entry_id"])
            if not src:
                continue
            for k in ("local_path", "sha256", "download_date", "http_status",
                      "content_length_bytes", "status", "notes"):
                r[k] = src[k]
        write_manifest(latest)
        print(f"manifest updated: {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
