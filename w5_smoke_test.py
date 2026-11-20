#!/usr/bin/env python3
"""W5 loadability smoke test for the two PREREGISTERED model artifacts.

This checks ONLY that the pinned files can be opened by their native readers in
an isolated environment. It does NOT embed a single cell, does NOT score T1/T2/T3,
and does NOT produce any number that may appear in the manuscript Results.

Run in the isolated model environment (NOT the W1-W4 `.venv`):
    ./.venv-models/Scripts/python.exe w5_smoke_test.py

Output: results/w5_smoke_test.json
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
OUT = RESULTS / "w5_smoke_test.json"

SCGPT_CKPT = MODELS / "scGPT" / "best_model.pt"
SCGPT_VOCAB = MODELS / "scGPT" / "vocab.json"
GF_DIR = MODELS / "Geneformer" / "Geneformer-V1-10M"
GF_DICTS = [
    MODELS / "Geneformer" / "geneformer" / "gene_dictionaries_30m" /
    "token_dictionary_gc30M.pkl",
    MODELS / "Geneformer" / "geneformer" / "gene_dictionaries_30m" /
    "gene_name_id_dict_gc30M.pkl",
    MODELS / "Geneformer" / "geneformer" / "gene_dictionaries_30m" /
    "gene_median_dictionary_gc30M.pkl",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def manifest_hash(local_path: str) -> str:
    import csv
    with (ROOT / "model_manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["local_path"] == local_path:
                return r["sha256"]
    return ""


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def ver(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "NOT_INSTALLED"


def check(path: Path) -> dict:
    if not path.exists():
        return {"path": rel(path), "status": "MISSING"}
    h = sha256(path)
    expected = manifest_hash(rel(path))
    return {"path": rel(path), "bytes": path.stat().st_size,
            "sha256": h, "hash_matches_manifest": h == expected,
            "status": "OK" if h == expected else "HASH_MISMATCH"}


def test_scgpt() -> dict:
    out: dict = {"model": "scGPT", "checks": [check(SCGPT_CKPT), check(SCGPT_VOCAB)]}
    if any(c["status"] != "OK" for c in out["checks"]):
        out["status"] = "BLOCKED_HASH"
        return out
    try:
        import torch
        # weights_only=True is the safe path; scGPT checkpoints contain an
        # argparse Namespace alongside tensors, so a failure here is expected
        # and must be recorded, not bypassed silently.
        try:
            obj = torch.load(SCGPT_CKPT, map_location="cpu", weights_only=True)
            out["checkpoint_reader"] = "torch.load(weights_only=True)"
            out["checkpoint_type"] = type(obj).__name__
            if isinstance(obj, dict):
                out["checkpoint_keys"] = sorted(obj.keys())[:12]
                out["n_tensors"] = sum(1 for v in obj.values()
                                       if hasattr(v, "shape"))
            out["status"] = "LOADABLE"
        except Exception as exc:
            out["checkpoint_reader"] = "torch.load(weights_only=True)"
            out["weights_only_error"] = f"{type(exc).__name__}: {exc}"[:300]
            out["status"] = "PARTIAL__NON_TENSOR_METADATA_PRESENT"
            out["note"] = ("Checkpoint integrity and hash are verified; full "
                           "deserialisation is deferred to the adapter step, "
                           "which will use weights_only=False only on this "
                           "hash-verified local artifact.")
        with SCGPT_VOCAB.open(encoding="utf-8") as fh:
            vocab = json.load(fh)
        out["vocab_entries"] = len(vocab)
        out["vocab_sample_keys"] = list(vocab)[:5]
    except Exception as exc:
        out["status"] = f"ERROR: {type(exc).__name__}: {exc}"[:300]
    return out


def test_geneformer() -> dict:
    out: dict = {"model": "Geneformer",
                 "checks": [check(GF_DIR / "config.json"),
                            check(GF_DIR / "model.safetensors")] +
                            [check(p) for p in GF_DICTS]}
    if any(c["status"] != "OK" for c in out["checks"]):
        out["status"] = "BLOCKED_HASH"
        return out
    try:
        from transformers import BertForMaskedLM
        with (GF_DIR / "config.json").open(encoding="utf-8") as fh:
            cfg = json.load(fh)
        out["config"] = {k: cfg.get(k) for k in
                         ("architectures", "hidden_size", "num_hidden_layers",
                          "max_position_embeddings", "vocab_size")}
        model = BertForMaskedLM.from_pretrained(
            str(GF_DIR), local_files_only=True)
        n_params = sum(p.numel() for p in model.parameters())
        out["n_parameters"] = int(n_params)
        out["model_class"] = type(model).__name__
        out["status"] = "LOADABLE"
        del model
    except Exception as exc:
        out["status"] = f"ERROR: {type(exc).__name__}: {exc}"[:300]
        return out
    # Dictionaries are pickled Python objects from a pinned, hash-verified
    # official commit; they are read here only to confirm they deserialize.
    try:
        sizes = {}
        for p in GF_DICTS:
            with p.open("rb") as fh:
                sizes[p.name] = len(pickle.load(fh))
        out["dictionary_sizes"] = sizes
    except Exception as exc:
        out["dictionary_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    record = {
        "schema_version": "w5-smoke-test-v1",
        "analysis_status": "LOADABILITY CHECK ONLY - NO MODEL WAS RUN, "
                            "NO CELL WAS EMBEDDED, NO SCORE WAS COMPUTED",
        "preregistration": {"url": "https://osf.io/8tzrq/overview",
                            "doi": "10.17605/OSF.IO/8TZRQ",
                            "frozen_model_set": ["scGPT", "Geneformer"]},
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "is_isolated_env": ".venv-models" in sys.executable.replace("\\", "/"),
            "versions": {n: ver(n) for n in
                         ("torch", "transformers", "datasets", "safetensors",
                          "huggingface_hub", "scgpt", "geneformer")},
        },
        "models": [test_scgpt(), test_geneformer()],
        "warnings": [
            "This is not an analysis. Nothing here may be reported as a result "
            "or used to accept/reject H1, H2 or H3.",
            "The main .venv (W1-W4) must remain frozen; model inference runs "
            "only in .venv-models.",
            "Weights were downloaded from hf-mirror.com at immutable commits "
            "and are hash-verified against model_manifest.csv.",
        ],
    }
    OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    for m in record["models"]:
        print(f"{m['model']:11s} -> {m['status']}")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
