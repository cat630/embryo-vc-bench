#!/usr/bin/env python3
"""W5 unified gene space and provenance-anchored atlas loaders.

Why this module exists
----------------------
The preregistered benchmark compares two foundation models, three naive
baselines and a CRISPRi ground truth on ONE comparable gene axis. The public
files do not share gene identifiers:

  * Arc VCC 2025 (T3 ground truth) and E-MTAB-3929 use gene SYMBOLS
  * GSE66507 uses Ensembl gene IDs (ENSG...)
  * GSE136447 uses "ENSG...|SYMBOL"
  * scGPT's `vocab.json` is keyed by SYMBOL (60,697 entries)
  * Geneformer V1's `token_dictionary_gc30M.pkl` is keyed by Ensembl ID

We therefore define the preregistered analysis universe as the **intersection**
of (i) the 18,080 Arc gene symbols, (ii) the scGPT vocabulary and (iii) the
Geneformer V1 token dictionary reachable through its own
`gene_name_id_dict_gc30M.pkl` symbol->Ensembl mapping. Every dataset is
projected onto this fixed, sorted gene list. Genes outside it are dropped,
never imputed; the count of retained genes is logged per atlas.

No normalization, filtering or imputation of biological values is performed
here beyond reindexing (missing genes -> 0 counts). All W5 model adapters and
all baseline scoring use this same axis, so DES/PDS/MAE are comparable.

Determinism: gene order is sorted(); all hashes are reported in the audit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

ARC_GENES = DATA / "ARC_gene_names__gene_names.csv"
SCGPT_VOCAB = MODELS / "scGPT" / "vocab.json"
GF_DICT_DIR = MODELS / "Geneformer" / "geneformer" / "gene_dictionaries_30m"

EMTAB_COUNTS = DATA / "BS_E-MTAB-3929_counts__counts.txt"
EMTAB_SDRF = DATA / "BS_E-MTAB-3929_sdrf__E-MTAB-3929.sdrf.txt"
GSE66507 = DATA / "GEO_GSE66507_human_blastocyst_rnaseq_counts__GSE66507_human_blastocyst_rnaseq_counts.txt.gz"
GSE136447 = DATA / "GEO_GSE136447_555sample_gene_count_matrix__GSE136447_555sample_gene_count_matrix.txt.gz"

GF_PAD_TOKEN = 0
GF_MASK_TOKEN = 1


@dataclass
class GeneSpace:
    """The single fixed gene axis used by every W5 computation."""
    genes: tuple[str, ...]
    symbol_to_ensg: dict[str, str]
    scgpt_id: np.ndarray          # (n_genes,) int32 token ids
    gf_token: np.ndarray           # (n_genes,) int32 Geneformer token ids (0 = absent)
    sources: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert len(self.genes) == len(self.scgpt_id) == len(self.gf_token)

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def fingerprint(self) -> str:
        payload = ("|".join(self.genes) + "|" +
                   hashlib.sha256(self.scgpt_id.tobytes()).hexdigest() + "|" +
                   hashlib.sha256(self.gf_token.tobytes()).hexdigest())
        return hashlib.sha256(payload.encode()).hexdigest()

    def summary(self) -> dict:
        return {
            "n_genes": self.n_genes,
            "fingerprint": self.fingerprint,
            "first_genes": list(self.genes[:5]),
            "scgpt_tokens_resolved": int((self.scgpt_id >= 0).sum()),
            "geneformer_tokens_resolved": int((self.gf_token > GF_MASK_TOKEN).sum()),
            "sources": self.sources,
        }


@dataclass
class Atlas:
    """One public atlas, projected onto the fixed GeneSpace."""
    name: str
    counts: np.ndarray             # (n_cell, n_gene) float32, RAW counts
    stage: np.ndarray              # (n_cell,) strings, "" if unavailable
    lineage: np.ndarray            # (n_cell,) strings, "" if unavailable
    source_files: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.counts.astype(np.float32).tobytes())
        h.update("|".join(self.stage.tolist()).encode())
        h.update("|".join(self.lineage.tolist()).encode())
        return h.hexdigest()

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_cells": int(self.counts.shape[0]),
            "n_genes": int(self.counts.shape[1]),
            "genes_detected_in_universe": int((self.counts > 0).any(axis=0).sum()),
            "stage_counts": pd.Series(self.stage).value_counts().to_dict(),
            "lineage_counts": pd.Series(self.lineage).value_counts().to_dict(),
            "fingerprint": self.fingerprint,
            "source_files": self.source_files,
            **self.meta,
        }


def build_gene_space() -> GeneSpace:
    arc = [ln.strip() for ln in ARC_GENES.read_text(encoding="utf-8").splitlines()
           if ln.strip()]
    vocab = json.loads(SCGPT_VOCAB.read_text(encoding="utf-8"))
    with (GF_DICT_DIR / "token_dictionary_gc30M.pkl").open("rb") as fh:
        gf_tok = pickle.load(fh)
    with (GF_DICT_DIR / "gene_name_id_dict_gc30M.pkl").open("rb") as fh:
        name_to_id = pickle.load(fh)

    gf_ensg = {k for k, v in gf_tok.items()
               if isinstance(k, str) and k.startswith("ENSG") and int(v) > GF_MASK_TOKEN}
    symbols = sorted({g for g in arc
                      if g in vocab and g in name_to_id and name_to_id[g] in gf_ensg})
    if len(symbols) < 1000:
        raise RuntimeError(f"gene universe collapsed to {len(symbols)} genes")

    s2e = {g: name_to_id[g] for g in symbols}
    return GeneSpace(
        genes=tuple(symbols),
        symbol_to_ensg=s2e,
        scgpt_id=np.array([vocab[g] for g in symbols], dtype=np.int32),
        gf_token=np.array([int(gf_tok[s2e[g]]) for g in symbols], dtype=np.int32),
        sources={
            "arc_gene_names": ARC_GENES.name,
            "n_arc": len(arc),
            "scgpt_vocab": SCGPT_VOCAB.name,
            "n_scgpt_vocab": len(vocab),
            "geneformer_token_dict": "token_dictionary_gc30M.pkl",
            "n_geneformer_tokens": len(gf_tok),
            "geneformer_name_map": "gene_name_id_dict_gc30M.pkl",
        },
    )


def _project(frame: pd.DataFrame, gs: GeneSpace) -> np.ndarray:
    """frame: genes (index) x cells (columns) -> cells x fixed gene axis."""
    f = frame.copy()
    f.index = f.index.astype(str)
    f = f[~f.index.duplicated(keep="first")]
    aligned = f.reindex(gs.genes).fillna(0.0)
    return np.ascontiguousarray(aligned.T.to_numpy(dtype=np.float32))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_emtab3929(gs: GeneSpace) -> Atlas:
    """Petropoulos 2016: 1,529 cells, SYMBOL rows, sdrf supplies real labels."""
    df = pd.read_csv(EMTAB_COUNTS, sep="\t", index_col=0)
    sdrf = pd.read_csv(EMTAB_SDRF, sep="\t", dtype=str).fillna("")
    sid = "Source Name"
    stage_map = dict(zip(sdrf[sid], sdrf["Characteristics[developmental stage]"]))
    lin_map = dict(zip(sdrf[sid], sdrf["Characteristics[inferred lineage]"]))
    cells = [str(c) for c in df.columns]
    stage = np.array([stage_map.get(c, "") for c in cells], dtype=object)
    lineage = np.array([lin_map.get(c, "") for c in cells], dtype=object)
    lineage = np.where(lineage == "not applicable", "", lineage)
    return Atlas(
        name="E-MTAB-3929",
        counts=_project(df, gs),
        stage=stage, lineage=lineage,
        source_files=[EMTAB_COUNTS.name, EMTAB_SDRF.name],
        meta={"citation": "Petropoulos et al., Cell 2016; E-MTAB-3929",
              "label_source": "sdrf Characteristics[developmental stage] / [inferred lineage]",
              "source_sha256": {EMTAB_COUNTS.name: _sha(EMTAB_COUNTS),
                                EMTAB_SDRF.name: _sha(EMTAB_SDRF)}},
    )


def load_gse66507(gs: GeneSpace) -> Atlas:
    """Blakeley 2015: blastocyst lineages, Ensembl rows, TE/EPI/PE in column names."""
    with gzip.open(GSE66507, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    ensg_to_symbol = {v: k for k, v in gs.symbol_to_ensg.items()}
    df.index = [ensg_to_symbol.get(str(i).split(".")[0], str(i)) for i in df.index]
    cells = [str(c) for c in df.columns]
    lineage = np.array([c.split(".")[0].split("_")[0].rstrip("0123456789")
                        for c in cells], dtype=object)
    lineage = np.array([x if x in ("TE", "EPI", "PE") else "" for x in lineage],
                       dtype=object)
    return Atlas(
        name="GSE66507",
        counts=_project(df, gs),
        stage=np.full(len(cells), "", dtype=object), lineage=lineage,
        source_files=[GSE66507.name],
        meta={"citation": "Blakeley et al., Development 2015; GSE66507",
              "label_source": "column-name lineage prefix (TE/EPI/PE)",
              "id_mapping": "Ensembl -> symbol via Geneformer gene_name_id_dict_gc30M.pkl",
              "source_sha256": {GSE66507.name: _sha(GSE66507)}},
    )


def load_gse136447(gs: GeneSpace) -> Atlas:
    """Xiang 2020: D6/D7 samples; gene_id is 'ENSG...|SYMBOL'."""
    with gzip.open(GSE136447, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    def sym(i: str) -> str:
        s = str(i)
        return s.split("|")[-1] if "|" in s else s
    df.index = [sym(i) for i in df.index]
    cells = [str(c) for c in df.columns]
    # Column names include D6N1B23, D7N1..., D14_3S15, ... so the day must
    # be parsed as the FULL integer after 'D'; taking c[1] would turn D14 into D1.
    import re
    def day(c: str) -> str:
        m = re.match(r"D(\d+)", c)
        return "D" + m.group(1) if m else ""
    stage = np.array([day(c) for c in cells], dtype=object)
    return Atlas(
        name="GSE136447",
        counts=_project(df, gs),
        stage=stage, lineage=np.full(len(cells), "", dtype=object),
        source_files=[GSE136447.name],
        meta={"citation": "Xiang et al., Nature 2020; GSE136447",
              "label_source": "column-name day prefix (D6/D7)",
              "id_mapping": "suffix after '|' in gene_id",
              "source_sha256": {GSE136447.name: _sha(GSE136447)}},
    )


LOADERS = {
    "E-MTAB-3929": load_emtab3929,
    "GSE66507": load_gse66507,
    "GSE136447": load_gse136447,
}


def load_all(gs: GeneSpace, names=None) -> dict[str, Atlas]:
    names = names or list(LOADERS)
    return {n: LOADERS[n](gs) for n in names}


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    gs = build_gene_space()
    atlases = load_all(gs)
    out = {"gene_space": gs.summary(),
           "atlases": {n: a.summary() for n, a in atlases.items()}}
    (RESULTS / "w5_gene_space.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gene_space": gs.summary()["n_genes"],
                      "fingerprint": gs.fingerprint[:16],
                      "atlases": {n: (a.counts.shape, len(set(a.stage)))
                                  for n, a in atlases.items()}}, indent=2))
