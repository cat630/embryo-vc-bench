#!/usr/bin/env python3
"""W5 ModelAdapter implementations for the TWO preregistered models.

This module turns the `ModelAdapter` Protocol in `embryo_vc_bench.py` into
runnable code. Both models are used strictly ZERO-SHOT: no fine-tuning, no
weight update, no gradient. Every artifact is the pinned checkpoint recorded in
`model_manifest.csv`.

PROTOCOL (fixed before any model score; exported to results/w5_protocol.json)
---------------------------------------------------------------------------
1. One gene axis. All data are projected onto `w5_gene_space.GeneSpace`
   (17,555 symbols, sorted). Genes outside it are dropped, never imputed.
2. scGPT tokenization: library-size normalize to 1e4, log1p, per-cell
   quantile bin nonzero values into 1..51 (zeros -> 0), then take the top-1200
   genes by log-normalized expression in that cell (deterministic; the
   checkpoint's own stochastic HVG sampling is not used). No CLS token:
   this checkpoint was trained with `no_cls=True`, so the original
   `cell_emb_style="cls"` would simply read the first gene's position. We
   therefore construct the same network with `cell_emb_style="avg-pool"`,
   which changes no parameter and only changes the pooling rule.
3. Geneformer V1 tokenization: raw counts divided by that gene's
   Genecorpus-30M median, rank-descending within each cell, top-2048 tokens,
   mapped through `token_dictionary_gc30M.pkl`; pad/missing -> token 0.
4. `embed`: scGPT -> `output["cell_emb"]` (512-d); Geneformer -> mean of the
   last hidden state over non-pad tokens (256-d).
5. `predict_delta`: adapter-defined IN-SILICO PERTURBATION = TARGET-GENE
   TOKEN ABLATION, never an experimental perturbation. This matches the
   preregistered T2 wording ("masking a target gene").
   - Both models force each target gene into the fixed sequence when it fell
     outside that cell's top-L, replacing the LOWEST-expression token and
     keeping the gene's true (possibly zero) value. No expression is imputed.
     Without this step a low-expressed target would have no token to ablate.
   - scGPT: replace the target gene's token with `<pad>` in the FIXED sequence
     (sequence length/order held constant so the two forward passes are
     position-wise comparable), subtract `mlm_output`, average over cells and
     scatter back onto the fixed gene axis.
   - Geneformer: replace the target gene's token with `<mask>` (id 1), take
     the cell-embedding shift averaged over cells, and project it onto the
     model's OWN gene-token embedding matrix to obtain a gene-axis ranking.
     Geneformer has no gene-expression decoder, so this is an explicit
     construct, not a claim that the model predicts expression.
6. SCALE WARNING. Neither delta is on a log2FC scale and the two deltas are
   not on the same scale as each other or as the CRISPRi ground truth.
   DES (top-k overlap) and PDS (cosine rank) are scale-free and are valid;
   MAE is only interpretable after per-perturbation z-scoring, and must be
   reported with that transformation stated. This is recorded in the audit.

HONESTY: this file computes NO preregistered result, no baseline comparison,
no p-value and no figure. It only implements the interface. Running
`w5_adapter_test.py` verifies shapes/finiteness/determinism on a fixed subset
of real cells; those checks are engineering verification, NOT findings.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from w5_gene_space import (GF_DICT_DIR, MODELS, RESULTS, GeneSpace,
                           GF_MASK_TOKEN, GF_PAD_TOKEN, build_gene_space)

SCGPT_DIR = MODELS / "scGPT"
GF_CKPT = MODELS / "Geneformer" / "Geneformer-V1-10M"

PROTOCOL_VERSION = "w5-adapter-protocol-2"
PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "revision_history": [
        {"version": "w5-adapter-protocol-1",
         "status": "SUPERSEDED before any model score existed",
         "delta": "set the target gene's binned value to 0",
         "reason_rejected": ("a zero/low-expressed target outside top-L produced "
                            "an identical input and an identically zero delta "
                            "for both models on real E-MTAB-3929 cells")},
        {"version": "w5-adapter-protocol-2",
         "status": "ACTIVE",
         "delta": "target-gene token ablation (<pad> / <mask>) with forced inclusion",
         "rationale": ("matches the preregistered T2 wording 'masking a target "
                       "gene'; always well-defined; no expression imputed")},
    ],
    "models": ["scGPT", "Geneformer"],
    "zero_shot": True,
    "gene_axis": "17,555-symbol intersection of Arc gene_names / scGPT vocab / Geneformer V1 tokens",
    "scgpt": {
        "checkpoint": "models/scGPT/best_model.pt",
        "checkpoint_revision": "perturblab/scgpt-human@571a0445d68fa48381f863ff75dd4f6d0eae3dfc",
        "normalization": "library size -> 1e4, log1p",
        "binning": "per-cell quantile bins 1..51 for nonzero values, zero -> 0",
        "sequence": "deterministic top-1200 genes by log-normalized value; no CLS token",
        "pooling": "avg-pool over non-pad tokens (checkpoint trained with no_cls=True)",
        "embedding_dim": 512,
        "attention": ("checkpoint trained with fast_transformer=True (Wqkv layout); "
                       "loaded on CPU via _PureTorchFlashMHA, an exact-parameter-name "
                       "stand-in using torch.nn.Linear + scaled_dot_product_attention. "
                       "The compiled flash-attn package is NOT used."),
        "checkpoint_load_disclosure": ("10 cls_decoder tensors are absent and randomly "
                                        "initialised but never called; flag_encoder.weight "
                                        "is unused; all 144 transformer tensors and the "
                                        "encoder/decoder weights load exactly"),
        "delta": ("target gene TOKEN ABLATION -> <pad> on the FIXED sequence; "
                  "mlm_output(pert) - mlm_output(base); cell-mean; scatter to gene axis"),
        "force_target_into_sequence": ("targets outside a cell's top-L replace the "
                                        "lowest-expression token, keeping their true "
                                        "binned value; no expression is imputed"),
    },
    "geneformer": {
        "checkpoint": "models/Geneformer/Geneformer-V1-10M/model.safetensors",
        "checkpoint_revision": "ctheodoris/Geneformer@1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5",
        "dictionaries": "Genecorpus-30M: token_dictionary, gene_median_dictionary, gene_name_id_dict",
        "normalization": "raw counts / per-gene Genecorpus-30M median",
        "sequence": "rank-descending within cell, top-2048 Ensembl tokens, pad -> 0",
        "pooling": "mean of last hidden state over non-pad tokens",
        "embedding_dim": 256,
        "delta": ("target gene TOKEN ABLATION -> <mask>(1); cell-embedding shift "
                  "(cell-mean) projected onto the model's gene-token embedding matrix"),
        "force_target_into_sequence": ("targets outside a cell's top-2048 replace the "
                                      "lowest-ranked token and are included in attention; "
                                      "no expression is imputed"),
    },
    "delta_scale_warning": (
        "Neither adapter delta is on a log2FC scale and the two are not on a "
        "common scale with each other or with the CRISPRi ground truth. "
        "DES/PDS are scale-free; MAE requires per-perturbation z-scoring "
        "and must be labelled as such."
    ),
    "preregistration_note": (
        "Implements the preregistered ModelAdapter interface (embed, "
        "predict_delta) for the two frozen models. No hypothesis, metric, "
        "fold, seed, dataset inclusion rule or falsification threshold is "
        "changed. The concrete tokenization/pooling/delta construction are "
        "implementation details of the adapter and are disclosed here; "
        "whether a timestamped registration amendment is required is a "
        "decision for the author/editor, not for this code."
    ),
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class _PureTorchFlashMHA(torch.nn.Module):
    """CPU-loadable stand-in for flash-attn 1.x `FlashMHA`.

    The pinned scGPT-human checkpoint was trained with `fast_transformer=True`,
    so its attention tensors are named `Wqkv` / `out_proj`. scgpt 0.2.4 can
    only build that layout when the compiled `flash_attn` package is present,
    which does not exist for Python 3.13 / CPU. This class reproduces the
    EXACT parameter names and shapes with `torch.nn.Linear` and computes the
    same multi-head attention with PyTorch's native scaled_dot_product_attention.
    It has no learnable parameters beyond the checkpoint's own tensors, so
    `load_state_dict` loads all 144 transformer tensors byte-for-byte.

    `key_padding_mask` follows the scGPT fast-transformer convention: True
    means a token to KEEP (the inverse of the PyTorch Transformer convention).
    """

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True,
                 attention_dropout=0.0, causal=False, device=None, dtype=None, **kw):
        super().__init__()
        self.Wqkv = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(self, x, key_padding_mask=None, need_weights=False, **kw):
        import torch.nn.functional as F
        b, s, d = x.shape
        h, dh = self.num_heads, x.shape[-1] // self.num_heads
        qkv = self.Wqkv(x).view(b, s, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        am = None
        if key_padding_mask is not None and key_padding_mask.dtype == torch.bool:
            am = key_padding_mask[:, None, None, :].expand(b, 1, s, s)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=0.0)
        o = o.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(o), None


def _import_scgpt_transformer_model():
    """Load `scgpt/model/model.py` without executing `scgpt/__init__.py`.

    The package initializer imports the tokenizer, which hard-depends on
    `torchtext` (no wheel for Python 3.13). We do not use that tokenizer:
    `vocab.json` is a plain dict and binning is implemented here. Registering
    the two parent packages as namespace stubs lets the model's own relative
    imports (`.dsbn`, `.grad_reverse`) resolve while skipping the initializer.
    We then substitute the pure-PyTorch FlashMHA above so the fast-transformer
    checkpoint can be loaded on CPU without the compiled flash-attn package.
    """
    import importlib
    import importlib.util
    import sys
    import types

    spec = importlib.util.find_spec("scgpt")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("scgpt package not found in this environment")
    root = Path(list(spec.submodule_search_locations)[0])
    parent = types.ModuleType("scgpt")
    parent.__path__ = [str(root)]
    sys.modules["scgpt"] = parent
    model_pkg = types.ModuleType("scgpt.model")
    model_pkg.__path__ = [str(root / "model")]
    sys.modules["scgpt.model"] = model_pkg
    mod = importlib.import_module("scgpt.model.model")
    mod.FlashMHA = _PureTorchFlashMHA
    mod.flash_attn_available = True
    return mod.TransformerModel


def _bin_row(row: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """Per-cell quantile binning of nonzero values into 1..n_bins; zeros stay 0."""
    out = np.zeros(row.shape, dtype=np.float32)
    nz = row > 0
    vals = row[nz]
    if vals.size == 0:
        return out
    if vals.size <= n_bins:
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(vals.size, dtype=np.float32)
        ranks[order] = np.arange(1, vals.size + 1, dtype=np.float32)
        bins = np.ceil(ranks / vals.size * n_bins)
    else:
        qs = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(qs, vals, side="right").astype(np.float32) + 1.0
    out[nz] = np.clip(bins, 1, n_bins)
    return out


class _AdapterBase:
    name = "base"
    embedding_dim = 0

    def __init__(self, gs: GeneSpace, seed: int = 0):
        self.gs = gs
        self.seed = seed
        self.load_seconds: float | None = None
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _check_counts(self, counts: np.ndarray, genes=None) -> np.ndarray:
        x = np.asarray(counts, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.gs.n_genes:
            raise ValueError(f"counts must be (n_cell, {self.gs.n_genes}); got {x.shape}")
        if not np.all(np.isfinite(x)) or np.any(x < 0):
            raise ValueError("counts must be finite and non-negative")
        if genes is not None and tuple(genes) != self.gs.genes:
            raise ValueError("genes must equal the fixed GeneSpace axis")
        return x

    def embed(self, counts: np.ndarray, genes=None) -> np.ndarray:
        raise NotImplementedError

    def predict_delta(self, counts: np.ndarray, perturbation) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name,
                "embedding_dim": self.embedding_dim,
                "protocol_version": PROTOCOL_VERSION,
                "load_seconds": self.load_seconds}


class ScGPTAdapter(_AdapterBase):
    """Zero-shot scGPT (continuous, 12-layer, 512-d) on the fixed gene axis."""

    name = "scGPT"
    embedding_dim = 512
    MAX_SEQ_LEN = 1200

    def __init__(self, gs: GeneSpace, seed: int = 0):
        super().__init__(gs, seed)
        t0 = time.time()
        warnings.filterwarnings("ignore", message="flash_attn is not installed")
        TransformerModel = _import_scgpt_transformer_model()

        self.vocab = json.loads((SCGPT_DIR / "vocab.json").read_text(encoding="utf-8"))
        self.args = json.loads((SCGPT_DIR / "args.json").read_text(encoding="utf-8"))
        pad_id = self.vocab["<pad>"]
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.args["embsize"],
            nhead=self.args["nheads"],
            d_hid=self.args["d_hid"],
            nlayers=self.args["nlayers"],
            nlayers_cls=self.args["n_layers_cls"],
            n_cls=self.args["n_cls"],
            vocab=self.vocab,
            dropout=self.args["dropout"],
            pad_token="<pad>",
            pad_value=self.args["pad_value"],
            do_mvc=bool(self.args["MVC"]),
            do_dab=bool(self.args["DAB"]),
            use_batch_labels=bool(self.args["use_batch_labels"]),
            domain_spec_batchnorm=self.args["domain_spec_batchnorm"],
            input_emb_style=self.args["input_emb_style"],
            n_input_bins=self.args["n_bins"],
            cell_emb_style="avg-pool",       # see protocol note 2
            mvc_decoder_style=self.args["mvc_decoder_style"],
            ecs_threshold=self.args["ecs_threshold"],
            explicit_zero_prob=bool(self.args["explicit_zero_prob"]),
            use_fast_transformer=True,      # checkpoint uses FlashMHA layout
            pre_norm=bool(self.args["pre_norm"]),
        )
        sd = torch.load(SCGPT_DIR / "best_model.pt", map_location="cpu",
                        weights_only=True)
        inc = self.model.load_state_dict(sd, strict=False)
        # Two classes of non-loaded tensors are expected and MUST be disclosed:
        #  * cls_decoder.* is absent from this checkpoint (trained with
        #    USE_CLS=False) and is never called (we never pass CLS=True);
        #  * flag_encoder.weight belongs to the generative/perturbation-flag
        #    path, which this wheel does not define and we never use.
        # Any OTHER missing/unexpected tensor is a real mismatch and must fail.
        bad_missing = [k for k in inc.missing_keys if not k.startswith("cls_decoder.")]
        bad_extra = [k for k in inc.unexpected_keys if k != "flag_encoder.weight"]
        if bad_missing or bad_extra:
            raise RuntimeError(
                f"scGPT checkpoint mismatch: unexpected missing={bad_missing[:5]} "
                f"extra={bad_extra[:5]}")
        self.load_report = {
            "checkpoint_tensors": len(sd),
            "loaded_tensors": len(sd) - len(inc.unexpected_keys),
            "missing_keys": list(inc.missing_keys),
            "unexpected_keys": list(inc.unexpected_keys),
            "disclosure": ("cls_decoder (10 tensors) is randomly initialised but "
                           "never called; flag_encoder.weight is unused; all "
                           "encoder/transformer/decoder tensors are loaded exactly"),
        }
        self.model.eval()
        torch.set_num_threads(4)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.load_seconds = round(time.time() - t0, 3)

    def _tokenize(self, x: np.ndarray, force_symbols=()):
        lib = x.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        logx = np.log1p(x / lib * 1e4).astype(np.float32)
        binned = np.vstack([_bin_row(r) for r in logx])
        L = min(self.MAX_SEQ_LEN, x.shape[1])
        order = np.argsort(-logx, axis=1, kind="stable")[:, :L]
        src = self.gs.scgpt_id[order].astype(np.int64).copy()
        values = np.take_along_axis(binned, order, axis=1).astype(np.float32).copy()
        # A target gene outside a cell's top-L has no token to ablate. Replace
        # the LOWEST-expression tokens with the targets (using their true
        # binned values, never an imputed value) so every intervention is
        # defined on the same fixed sequence. See protocol note 5.
        forced = [(g, int(self.gs.scgpt_id[self.gs.genes.index(g)]))
                  for g in force_symbols]
        for k, (gene, tok) in enumerate(forced, start=1):
            pos = -k
            gi = self.gs.genes.index(gene)
            for i in range(x.shape[0]):
                if tok not in src[i]:
                    src[i, pos] = tok
                    values[i, pos] = binned[i, gi]
        return src, values

    def _pad_id(self) -> int:
        return int(self.vocab["<pad>"])

    def _forward(self, src: np.ndarray, values: np.ndarray) -> dict:
        with torch.no_grad():
            s = torch.from_numpy(src)
            v = torch.from_numpy(values)
            mask = torch.zeros(s.shape, dtype=torch.bool)
            return self.model(s, v, mask)

    def embed(self, counts: np.ndarray, genes=None) -> np.ndarray:
        x = self._check_counts(counts, genes)
        src, values = self._tokenize(x)
        out = self._forward(src, values)
        emb = out["cell_emb"].cpu().numpy().astype(np.float32)
        if emb.shape != (x.shape[0], self.embedding_dim):
            raise RuntimeError(f"unexpected scGPT embedding shape {emb.shape}")
        return emb

    def predict_delta(self, counts: np.ndarray, perturbation) -> np.ndarray:
        x = self._check_counts(counts)
        # Force all target genes into the sequence, then ablate the target TOKEN
        # (<pad>) rather than setting its value to zero: a low/zero-expressed
        # target would otherwise produce an identical input and a zero delta.
        src, values = self._tokenize(x, force_symbols=perturbation)
        base = self._forward(src, values)["mlm_output"].cpu().numpy()
        n_cell, L = src.shape
        token_to_axis = {int(self.gs.scgpt_id[i]): i for i in range(self.gs.n_genes)}
        pad_id = self._pad_id()
        deltas = []
        for gene in perturbation:
            if gene not in self.gs.genes:
                raise KeyError(f"target gene {gene!r} is outside the fixed gene axis")
            gi = self.gs.genes.index(gene)
            tok = int(self.gs.scgpt_id[gi])
            hit = src == tok
            if not hit.any():
                raise RuntimeError(f"target {gene} was not forced into the sequence")
            pert_src = src.copy()
            pert_src[hit] = pad_id
            pert_values = values.copy()
            pert_values[hit] = 0.0
            pert = self._forward(pert_src, pert_values)["mlm_output"].cpu().numpy()
            d = (pert - base).astype(np.float32)
            acc = np.zeros(self.gs.n_genes, dtype=np.float64)
            cnt = np.zeros(self.gs.n_genes, dtype=np.float64)
            for i in range(n_cell):
                for j in range(L):
                    idx = token_to_axis.get(int(src[i, j]))
                    if idx is not None:
                        acc[idx] += d[i, j]
                        cnt[idx] += 1.0
            delta = np.divide(acc, cnt,
                              out=np.zeros(self.gs.n_genes, dtype=np.float64),
                              where=cnt > 0)
            deltas.append(delta.astype(np.float32))
        return np.vstack(deltas)

    def describe(self) -> dict:
        return {**super().describe(), **self.load_report,
                "checkpoint_sha256": _sha(SCGPT_DIR / "best_model.pt"),
                "vocab_sha256": _sha(SCGPT_DIR / "vocab.json")}


class GeneformerAdapter(_AdapterBase):
    """Zero-shot Geneformer V1-10M (BertForMaskedLM, 256-d) on the fixed axis."""

    name = "Geneformer"
    embedding_dim = 256
    MAX_SEQ_LEN = 2048

    def __init__(self, gs: GeneSpace, seed: int = 0):
        super().__init__(gs, seed)
        t0 = time.time()
        from transformers import BertForMaskedLM
        self.model = BertForMaskedLM.from_pretrained(
            str(GF_CKPT), local_files_only=True)
        self.model.eval()
        torch.set_num_threads(4)
        with (GF_DICT_DIR / "token_dictionary_gc30M.pkl").open("rb") as fh:
            self.token_dict = pickle.load(fh)
        with (GF_DICT_DIR / "gene_median_dictionary_gc30M.pkl").open("rb") as fh:
            med = pickle.load(fh)
        self.median = np.array(
            [float(med.get(gs.symbol_to_ensg[g], 1.0)) for g in gs.genes],
            dtype=np.float32)
        self.median[self.median <= 0] = 1.0
        with torch.no_grad():
            self.gene_embedding = (
                self.model.bert.embeddings.word_embeddings
                .weight[torch.from_numpy(gs.gf_token.astype(np.int64))]
                .cpu().numpy().astype(np.float32))
        self.load_seconds = round(time.time() - t0, 3)

    def _tokenize(self, x: np.ndarray, force_symbols=()):
        norm = x / self.median[None, :]
        L = min(self.MAX_SEQ_LEN, x.shape[1])
        order = np.argsort(-norm, axis=1, kind="stable")[:, :L]
        ids = self.gs.gf_token[order].astype(np.int64).copy()
        keep = ((ids > GF_MASK_TOKEN)
                & (norm[np.arange(x.shape[0])[:, None], order] > 0)).astype(np.float32)
        forced = [(g, int(self.gs.gf_token[self.gs.genes.index(g)]))
                  for g in force_symbols]
        for k, (gene, tok) in enumerate(forced, start=1):
            pos = -k
            for i in range(x.shape[0]):
                if tok not in ids[i]:
                    ids[i, pos] = tok
                    keep[i, pos] = 1.0     # include the forced token in attention
        ids = np.where(keep > 0, ids, GF_PAD_TOKEN)
        return ids, keep

    def _cell_emb(self, ids: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            ii = torch.from_numpy(ids)
            am = torch.from_numpy(keep.astype(np.int64))
            h = self.model.bert(input_ids=ii, attention_mask=am).last_hidden_state
            w = am.unsqueeze(-1).to(h.dtype)
            emb = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return emb.cpu().numpy().astype(np.float32), h

    def embed(self, counts: np.ndarray, genes=None) -> np.ndarray:
        x = self._check_counts(counts, genes)
        ids, keep = self._tokenize(x)
        emb, _ = self._cell_emb(ids, keep)
        if emb.shape != (x.shape[0], self.embedding_dim):
            raise RuntimeError(f"unexpected Geneformer embedding shape {emb.shape}")
        return emb

    def predict_delta(self, counts: np.ndarray, perturbation) -> np.ndarray:
        x = self._check_counts(counts)
        ids, keep = self._tokenize(x, force_symbols=perturbation)
        base, _ = self._cell_emb(ids, keep)
        deltas = []
        for gene in perturbation:
            if gene not in self.gs.genes:
                raise KeyError(f"target gene {gene!r} is outside the fixed gene axis")
            gi = self.gs.genes.index(gene)
            tok = int(self.gs.gf_token[gi])
            hit = ids == tok
            if not hit.any():
                raise RuntimeError(f"target {gene} was not forced into the sequence")
            pert_ids = ids.copy()
            pert_ids[hit] = GF_MASK_TOKEN
            pert, _ = self._cell_emb(pert_ids, keep)
            shift = (pert - base).mean(axis=0)                 # (256,)
            delta = self.gene_embedding @ shift                 # (n_genes,)
            deltas.append(delta.astype(np.float32))
        return np.vstack(deltas)

    def describe(self) -> dict:
        return {**super().describe(),
                "checkpoint": str(GF_CKPT.relative_to(MODELS.parent)),
                "n_tokens": len(self.token_dict),
                "gene_embedding_shape": list(self.gene_embedding.shape),
                "checkpoint_sha256": _sha(GF_CKPT / "model.safetensors")}


def build_adapters(gs: GeneSpace, seed: int = 0) -> dict[str, _AdapterBase]:
    """Both preregistered models. The set must never be silently reduced."""
    return {"scGPT": ScGPTAdapter(gs, seed),
            "Geneformer": GeneformerAdapter(gs, seed)}


def write_protocol(extra: dict | None = None) -> Path:
    RESULTS.mkdir(exist_ok=True)
    payload = {**PROTOCOL, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **(extra or {})}
    p = RESULTS / "w5_protocol.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


if __name__ == "__main__":
    gs = build_gene_space()
    adapters = build_adapters(gs)
    out = {"gene_space": gs.summary(),
           "protocol": PROTOCOL,
           "adapters": {n: a.describe() for n, a in adapters.items()}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "w5_adapters.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_protocol({"adapters": {n: a.describe() for n, a in adapters.items()}})
    print(json.dumps({n: a.describe() for n, a in adapters.items()}, indent=2))
