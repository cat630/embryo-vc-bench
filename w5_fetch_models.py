#!/usr/bin/env python3
"""W5: freeze and verify the two PREREGISTERED model artifacts.

OSF 8tzrq (DOI 10.17605/OSF.IO/8TZRQ, model-set update 2026-11-17T14:23:25Z)
freezes the model set to exactly scGPT + Geneformer. This script pins the
concrete implementation of each model *before any model score is computed*:
Hugging Face repository, immutable commit SHA, licence evidence, and the exact
file list. Pinning here prevents post-result model/version shopping (HARKing).

Nothing in this script runs a model. It only downloads, hashes and records.

Why these particular artifacts
------------------------------
* scGPT: `perturblab/scgpt-human` states in its own model card that the weights
  were originally downloaded from the official `bowang-lab/scGPT` repository,
  carries `license: mit`, and points `base_model` at bowang-lab/scGPT. The
  official scGPT GitHub repository is MIT-licensed (LICENSE + README badge).
  The card therefore supplies both provenance and an explicit reuse licence;
  the unlabelled `VirtualCell2025/scGPT_human` mirror was NOT used.
* Geneformer: the official `ctheodoris/Geneformer` repository carries
  `license: apache-2.0`. The V1-10M checkpoint is pinned (Genecorpus-30M,
  vocab_size 25,426) together with the gc30M dictionaries required by its
  rank-value tokenizer. V2 checkpoints are deliberately NOT pinned: adding a
  larger model after seeing results would be a preregistration violation.

All downloads go through the reachable HF mirror (hf-mirror.com); the primary
huggingface.co host times out from this machine. URLs and commit SHAs in the
manifest still name the canonical HF repositories, so the record is portable.

Usage
-----
    python w5_fetch_models.py fetch      # download missing artifacts + hashes
    python w5_fetch_models.py verify     # re-hash everything on disk
    python w5_fetch_models.py status     # summary

Outputs: models/**, model_manifest.csv, results/w5_model_freeze.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "model_manifest.csv"
MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "embryo-vc-bench/W5 model-artifact freeze (reproducible research)"}
TIMEOUT = 180
CHUNK = 4 * 1024 * 1024

FIELDS = [
    "entry_id", "model", "repo", "revision", "artifact", "canonical_url",
    "download_url", "local_path", "http_status", "content_length_bytes",
    "size_bytes", "sha256", "licence_status", "licence_evidence_url",
    "download_date", "status", "notes",
]

SCGPT_REV = "571a0445d68fa48381f863ff75dd4f6d0eae3dfc"
GF_REV = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"

SCGPT_FILES = ["README.md", "args.json", "vocab.json", "best_model.pt"]

GF_FILES = [
    "README.md", "setup.py", "requirements.txt",
    "Geneformer-V1-10M/config.json",
    "Geneformer-V1-10M/model.safetensors",
    "Geneformer-V1-10M/training_args.bin",
    # rank-value tokenizer dictionaries for the pinned V1 (Genecorpus-30M) model
    "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
    "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl",
    "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
    "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl",
    # importable package source at the same immutable revision
    "geneformer/__init__.py",
    "geneformer/tokenizer.py",
    "geneformer/emb_extractor.py",
    "geneformer/in_silico_perturber.py",
    "geneformer/in_silico_perturber_stats.py",
    "geneformer/perturber_utils.py",
    "geneformer/pretrainer.py",
    "geneformer/classifier.py",
    "geneformer/classifier_utils.py",
    "geneformer/collator_for_classification.py",
    "geneformer/evaluation_utils.py",
    "geneformer/mtl_classifier.py",
    "geneformer/mtl/__init__.py",
    "geneformer/mtl/collators.py",
    "geneformer/mtl/data.py",
    "geneformer/mtl/eval_utils.py",
    "geneformer/mtl/model.py",
    "geneformer/mtl/train.py",
    "geneformer/mtl/utils.py",
    "geneformer/token_dictionary_gc104M.pkl",
    "geneformer/gene_name_id_dict_gc104M.pkl",
    "geneformer/gene_median_dictionary_gc104M.pkl",
    "geneformer/ensembl_mapping_dict_gc104M.pkl",
]

VERIFIED = ("VERIFIED_MIT", "VERIFIED_APACHE2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_rows() -> list[dict]:
    specs = [
        ("scGPT", "perturblab/scgpt-human", SCGPT_REV, SCGPT_FILES,
         "VERIFIED_MIT",
         "https://hf-mirror.com/perturblab/scgpt-human (cardData license: mit; "
         "base_model bowang-lab/scGPT; MIT LICENSE in official GitHub repo)",
         "Weights re-uploaded from official bowang-lab/scGPT; human whole-body "
         "checkpoint. Alternative unlabelled mirrors deliberately excluded."),
        ("Geneformer", "ctheodoris/Geneformer", GF_REV, GF_FILES,
         "VERIFIED_APACHE2",
         "https://hf-mirror.com/ctheodoris/Geneformer (cardData license: "
         "apache-2.0, tag license:apache-2.0)",
         "V1-10M checkpoint pinned with Genecorpus-30M dictionaries; V2 "
         "variants not pinned and must not be added post hoc."),
    ]
    rows = []
    for model, repo, rev, files, lic, evidence, note in specs:
        for path in files:
            enc = quote(path, safe="/")
            canonical = f"https://huggingface.co/{repo}/blob/{rev}/{enc}"
            dl = f"{MIRROR}/{repo}/resolve/{rev}/{enc}"
            local = MODELS / model / path
            eid = f"{model}__{path.replace('/', '_')}"
            rows.append({
                "entry_id": eid, "model": model, "repo": repo, "revision": rev,
                "artifact": path, "canonical_url": canonical, "download_url": dl,
                "local_path": str(local.relative_to(ROOT)).replace("\\", "/"),
                "http_status": "", "content_length_bytes": "", "size_bytes": "",
                "sha256": "", "licence_status": lic,
                "licence_evidence_url": evidence, "download_date": "",
                "status": "PENDING", "notes": note,
            })
    return rows


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(rows: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(MANIFEST)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download(row: dict) -> dict:
    dest = ROOT / row["local_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and row.get("sha256"):
        return {**row, "status": "VERIFIED_PRESENT", "size_bytes": str(dest.stat().st_size)}

    if row["licence_status"] not in VERIFIED:
        return {**row, "status": "BLOCKED_LICENCE"}

    start = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(row["download_url"], headers=headers, method="GET")
    mode = "ab" if start else "wb"
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, part.open(mode) as out:
        status = r.status
        clen = r.headers.get("Content-Length")
        if status == 200 and start:
            # Server ignored Range: restart from scratch to avoid concatenation.
            out.seek(0)
            out.truncate()
            start = 0
        total = 0
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
        out.flush()
    expected = int(clen) if clen and clen.isdigit() else None
    if status == 206 and start:
        expected = start + (expected or total)
    size = part.stat().st_size
    if expected is not None and size != expected:
        return {**row, "http_status": str(status),
                "content_length_bytes": str(expected or ""),
                "size_bytes": str(size), "status": "INCOMPLETE",
                "notes": (row["notes"] + f" | partial {size}/{expected} bytes kept for resume")}
    part.replace(dest)
    return {**row, "http_status": str(status),
            "content_length_bytes": str(expected or ""),
            "size_bytes": str(size), "sha256": sha256(dest),
            "download_date": now(), "status": "DOWNLOADED_VERIFIED"}


def cmd_fetch() -> int:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    fresh = build_rows()
    old = {r["entry_id"]: r for r in read_manifest()}
    rows = []
    for row in fresh:
        prev = old.get(row["entry_id"], {})
        merged = {**row, **{k: v for k, v in prev.items()
                            if k in ("sha256", "download_date", "size_bytes",
                                     "http_status", "content_length_bytes", "status")
                            and v}}
        # A pinned revision is immutable: a changed revision invalidates the hash.
        if prev.get("revision") and prev["revision"] != row["revision"]:
            merged["status"] = "REVISION_CHANGED_REFETCH_REQUIRED"
            merged["sha256"] = ""
        rows.append(merged)

    changed = False
    for i, row in enumerate(rows, 1):
        if row["status"] in ("DOWNLOADED_VERIFIED", "VERIFIED_PRESENT"):
            continue
        print(f"[{i:02d}/{len(rows)}] {row['model']}: {row['artifact']}", flush=True)
        rows[i - 1] = download(row)
        print(f"        -> {rows[i-1]['status']} "
              f"{rows[i-1].get('size_bytes', '')} bytes", flush=True)
        changed = True
    write_manifest(rows)
    freeze = {
        "schema_version": "w5-model-freeze-v1",
        "frozen_at": now(),
        "preregistration": {
            "url": "https://osf.io/8tzrq/overview",
            "doi": "10.17605/OSF.IO/8TZRQ",
            "model_set_update_utc": "2026-11-17T14:23:25Z",
            "frozen_model_set": ["scGPT", "Geneformer"],
        },
        "policy": "Revisions and file lists are hard-coded BEFORE any model "
                  "score exists. Changing them after seeing results would be "
                  "a preregistration violation and requires a timestamped OSF "
                  "registration update plus an explicit note in the manuscript.",
        "pinned": [
            {"model": "scGPT", "repo": "perturblab/scgpt-human",
             "revision": SCGPT_REV, "licence": "MIT",
             "checkpoint": "best_model.pt (official bowang-lab/scGPT human weights)",
             "n_artifacts": len(SCGPT_FILES)},
            {"model": "Geneformer", "repo": "ctheodoris/Geneformer",
             "revision": GF_REV, "licence": "Apache-2.0",
             "checkpoint": "Geneformer-V1-10M/model.safetensors (vocab_size 25426)",
             "n_artifacts": len(GF_FILES)},
        ],
        "download_endpoint": MIRROR,
        "canonical_endpoint": "https://huggingface.co",
        "endpoint_note": "huggingface.co times out from this machine; the mirror "
                         "serves byte-identical files at a pinned commit. "
                         "Canonical URLs are recorded for portability.",
        "no_model_was_run": True,
        "dependency_conflict_note": (
            "scgpt==0.2.4 pins scvi-tools<1.0 (would downgrade the main .venv "
            "from 1.5.2 to 0.20.3) and pulls torchtext 0.6.0/jax; Geneformer "
            "pins transformers==4.46. Model inference therefore MUST run in a "
            "separate .venv-models environment. The main .venv stays frozen "
            "for W1-W4 reproducibility."),
        "artifacts": [{k: r[k] for k in ("entry_id", "model", "artifact",
                                         "sha256", "size_bytes", "licence_status",
                                         "status")} for r in rows],
    }
    (RESULTS / "w5_model_freeze.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmanifest: {len(rows)} artifacts | changed={changed}")
    return 0


def cmd_verify() -> int:
    rows = read_manifest()
    if not rows:
        print("no manifest; run fetch first")
        return 2
    ok = bad = 0
    for r in rows:
        p = ROOT / r["local_path"]
        if not r["sha256"] or not p.exists():
            print(f"MISSING  {r['entry_id']}")
            bad += 1
            continue
        if sha256(p) == r["sha256"]:
            ok += 1
        else:
            print(f"MISMATCH {r['entry_id']}")
            r["status"] = "HASH_MISMATCH"
            bad += 1
    write_manifest(rows)
    print(f"W5 verify: {ok}/{len(rows)} OK, {bad} bad")
    return 0 if bad == 0 else 1


def cmd_status() -> int:
    rows = read_manifest()
    from collections import Counter
    print(f"W5 model artifacts: {len(rows)} rows")
    for key in ("status", "model", "licence_status"):
        print(" ", key, dict(Counter(r[key] for r in rows)))
    total = sum(int(r["size_bytes"]) for r in rows if r["size_bytes"].isdigit())
    print(f"  total bytes on record: {total:,} ({total/1e6:.1f} MB)")
    for r in rows:
        if r["sha256"]:
            print(f"  {r['model']:11s} {r['artifact'][:52]:52s} {r['sha256'][:12]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "verify", "status"])
    a = ap.parse_args()
    return {"fetch": cmd_fetch, "verify": cmd_verify, "status": cmd_status}[a.command]()


if __name__ == "__main__":
    sys.exit(main())
