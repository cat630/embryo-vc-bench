#!/usr/bin/env python3
"""W1 data provenance, licence gate, download and integrity verification.

Self-contained: Python standard library only. No anndata / scanpy / scvi needed.

Workflow
--------
    python w1_fetch.py init      # create data_manifest.csv from the frozen source list
    python w1_fetch.py probe     # HEAD every URL, record status / size / last-modified
    python w1_fetch.py fetch     # download provenance-only metadata files (small, gated)
    python w1_fetch.py fetch --allow-data   # also fetch verified expression files
    python w1_fetch_large.py    # resumable multi-GB CC0 download (T3 ground truth)
    python w1_fetch.py verify    # recompute SHA256 of downloaded files vs manifest
    python w1_fetch.py clean     # delete local files no longer tracked by the manifest
    python w1_fetch.py status    # print the licence / download summary

Design rules (fixed by the OSF preregistration 8tzrq)
-----------------------------------------------------
* Every dataset needs accession, version, download date, SHA256, licence and
  citation requirement. All six are columns of data_manifest.csv.
* A dataset whose licence cannot be verified is EXCLUDED from analysis by
  default. Plain `fetch` retrieves only *provenance-only* metadata files
  (SOFT / study JSON / SDRF), used for accession and citation checking.
  `fetch --allow-data` additionally retrieves expression files from official
  upstream repositories whose distribution policy has been verified and
  quoted (GEO / BioStudies). Third-party mirrors with no licence statement
  (KI curated files, ModelScope "other", Arc VCC without a LICENSE file)
  remain blocked.
* Files larger than MAX_BYTES are excluded by size regardless of licence.
* Nothing downloaded here is opened or parsed. No QC, no analysis.

Statuses
--------
    NOT_DOWNLOADED        not yet probed / fetched
    PROBED                metadata URL verified, no local copy
    AVAILABLE_DATA        verified expression file, fetch with --allow-data
    REQUIRES_LARGE_FETCH  multi-GB CC0 file, use w1_fetch_large.py (Range resume)
    BLOCKED_LICENCE       data body withheld pending licence confirmation
    EXCLUDED_SIZE         size exceeds MAX_BYTES
    DOWNLOADED_PROVENANCE metadata file stored, NOT usable as analysis input
    DOWNLOADED_DATA       expression file stored under a verified policy
    UNREACHABLE           HEAD failed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data_manifest.csv"
DATA_DIR = ROOT / "data"
UA = {"User-Agent": "WorkBuddy-W1/1.0 (mailto:research@example.org)"}
MAX_BYTES = 256 * 1024 * 1024  # 256 MiB ceiling for ordinary downloads;
# multi-GB CC0 files are handled by w1_fetch_large.py with HTTP Range resume.
VERIFIED = ("VERIFIED_POLICY", "VERIFIED_CC0")
TIMEOUT = 120  # large upstream files (100 MB) need a longer read timeout

FIELDS = [
    "entry_id", "accession", "source", "kind", "version",
    "url", "local_path", "http_status", "content_length_bytes",
    "last_modified", "licence_status", "licence_evidence_url",
    "citation_requirement", "download_date", "sha256", "probe_date",
    "status", "notes",
]

KI = "https://petropoulos-lanner-labs.clintec.ki.se/resource/human_embryo_stem_datasets/"
KI_PAGE = "https://petropoulos-lanner-labs.clintec.ki.se/dataset.download.html"
GEO_POLICY = "https://www.ncbi.nlm.nih.gov/geo/info/overview.html"
BIOSTUDIES_ABOUT = "https://www.ebi.ac.uk/biostudies/about"
MODELSCOPE_PAGE = "https://www.modelscope.cn/datasets/arcinstitute/PIE_arc_vcc_25"

NO_LICENCE = (
    "No explicit licence statement found on the source page at probe time. "
    "Preregistered rule: exclude from analysis until licence is confirmed. "
    "Institution-level confirmation required."
)

# Verbatim policy evidence, checked 2026-11-17 (UTC). These are policy statements
# about free distribution and citation, NOT creative-commons licences; they are
# recorded as VERIFIED_POLICY rather than as an explicit reuse licence.
GEO_POLICY_QUOTE = (
    "NCBI GEO Overview, checked 2026-11-17: 'GEO is an international public "
    "repository that archives and freely distributes ... high-throughput "
    "functional genomics data submitted by the research community.' No additional "
    "reuse restriction stated; cite the accession and the original publication."
)
BIOSTUDIES_POLICY_QUOTE = (
    "EMBL-EBI BioStudies 'About', checked 2026-11-17: 'Please include your dataset "
    "accession number and the URL to the BioStudies home page.' No additional reuse "
    "restriction stated for deposited data; cite accession and original publication."
)
ARC_CC0_EVIDENCE = "https://arcinstitute.org/tools/virtualcellatlas"
ARC_CC0_QUOTE = (
    "Arc Virtual Cell Atlas official page, checked 2026-11-17, under 'Virtual Cell "
    "Challenge 2025': '© 2025. This work is openly licensed via CC0 1.0.' This is an "
    "explicit public-domain dedication, NOT merely a distribution policy."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source(entry_id, accession, src, kind, url, version, evidence, citation,
           notes="", max_bytes=MAX_BYTES, licence_status="UNVERIFIED", large=False):
    return {
        "entry_id": entry_id,
        "accession": accession,
        "source": src,
        "kind": kind,
        "version": version,
        "url": url,
        "local_path": "",
        "http_status": "",
        "content_length_bytes": "",
        "last_modified": "",
        "licence_status": licence_status,
        "licence_evidence_url": evidence,
        "citation_requirement": citation,
        "download_date": "",
        "sha256": "",
        "probe_date": "",
        "status": "NOT_DOWNLOADED",
        "notes": "; ".join(x for x in [
            NO_LICENCE if licence_status == "UNVERIFIED" else "", notes
        ] if x),
        "_max_bytes": max_bytes,
        "_large": large,
    }


def build_sources():
    out = []

    # --- Curated embryonic atlases: Petropoulos & Lanner Labs -----------------
    ki = [
        ("Yan_2013_PMID_23934149", "GSE36552", "Yan et al., 2013, PMID 23934149"),
        ("Petropoulos_2016_PMID_27062923", "E-MTAB-3929", "Petropoulos et al., 2016, PMID 27062923"),
        ("Xiang_2020_PMID_31830756", "GSE136447", "Xiang et al., 2020, PMID 31830756"),
        ("Meistermann_2021_PMID_34004179", "PRJEB30442", "Meistermann et al., 2021, PMID 34004179"),
        ("Zhou_2019_PMID_31435013", "GSE109555", "Zhou et al., 2019, PMID 31435013"),
        ("Tyser_2021_PMID_34789876", "GSE150578", "Tyser et al., 2021, PMID 34789876"),
    ]
    for stem, acc, cite in ki:
        out.append(source(
            f"KI_{stem}_meta", acc, "KI curated", "provenance_metadata",
            KI + stem + ".meta.tsv", "static curated file, no version stamp",
            KI_PAGE, f"cite {cite} and the Petropoulos & Lanner Labs curated resource",
            "Metadata only; cell/sample annotation, not an expression matrix.",
        ))
        out.append(source(
            f"KI_{stem}_counts", acc, "KI curated", "expression_matrix",
            KI + stem + ".counts.gz", "static curated file, no version stamp",
            KI_PAGE, f"cite {cite} and the Petropoulos & Lanner Labs curated resource",
            "Expression matrix; WITHHELD pending licence confirmation.",
        ))
    out.append(source(
        "KI_integrated_seurat_rds", "multi-atlas", "KI curated", "expression_matrix",
        KI + "psd.R3.6.em.seurat.ob.rds", "R 3.6 Seurat object, no version stamp",
        KI_PAGE, "cite the six source studies and the curated resource",
        "Integrated object; WITHHELD pending licence confirmation.",
    ))

    # --- GEO SOFT metadata (small, provenance only) --------------------------
    geo = [
        ("GSE36552", "GSE36nnn"),
        ("GSE66507", "GSE66nnn"),
        ("GSE44183", "GSE44nnn"),
        ("GSE136447", "GSE136nnn"),
        ("GSE109555", "GSE109nnn"),
        ("GSE150578", "GSE150nnn"),
    ]
    for acc, prefix in geo:
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/soft/{acc}_family.soft.gz"
        out.append(source(
            f"GEO_{acc}_SOFT", acc, "NCBI GEO", "provenance_metadata", url,
            "GEO accession (no explicit version)", GEO_POLICY,
            f"cite the original publication linked to {acc}; state GEO accession",
            "SOFT family metadata only; no expression values used in analysis. " + GEO_POLICY_QUOTE,
            licence_status="VERIFIED_POLICY",
        ))

    # --- ArrayExpress / BioStudies and ENA study records ---------------------
    for acc in ["E-MTAB-3929", "E-MTAB-8060", "E-MTAB-9388"]:
        out.append(source(
            f"BS_{acc}_record", acc, "EMBL-EBI BioStudies", "provenance_metadata",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}",
            "BioStudies record (release date in record)", BIOSTUDIES_ABOUT,
            f"cite the original publication linked to {acc}; state accession",
            "Study-level JSON metadata only. " + BIOSTUDIES_POLICY_QUOTE,
            licence_status="VERIFIED_POLICY",
        ))
    out.append(source(
        "ENA_PRJEB30442_record", "PRJEB30442", "EMBL-EBI ENA", "provenance_metadata",
        "https://www.ebi.ac.uk/ena/browser/api/xml/PRJEB30442",
        "ENA project record (no explicit version)", BIOSTUDIES_ABOUT,
        "cite Meistermann et al., 2021, PMID 34004179; state PRJEB30442",
        "Project-level XML metadata only. " + BIOSTUDIES_POLICY_QUOTE,
        licence_status="VERIFIED_POLICY",
    ))

    # --- Official upstream expression files ----------------------------------
    # These avoid the KI mirror, whose page carries no licence statement.
    bs_files = [
        ("counts", "counts.txt"),
        ("ercc_counts", "ercc.counts.txt"),
        ("sdrf", "E-MTAB-3929.sdrf.txt"),
        ("idf", "E-MTAB-3929.idf.txt"),
    ]
    for tag, fname in bs_files:
        out.append(source(
            f"BS_E-MTAB-3929_{tag}", "E-MTAB-3929", "EMBL-EBI BioStudies files",
            "expression_matrix" if tag in ("counts", "ercc_counts") else "provenance_metadata",
            f"https://www.ebi.ac.uk/biostudies/files/E-MTAB-3929/{fname}",
            "BioStudies deposited file (Last-Modified recorded by probe)", BIOSTUDIES_ABOUT,
            "cite Petropoulos et al., 2016, PMID 27062923; state E-MTAB-3929",
            "Official upstream file, KI mirror deliberately bypassed. " + BIOSTUDIES_POLICY_QUOTE,
            licence_status="VERIFIED_POLICY",
        ))
    # Exact supplementary filenames were read from each GEO Series record
    # (acc.cgi?targ=self&form=text&view=brief) rather than guessed.
    geo_suppl = [
        ("GSE36552", "GSE36nnn", "GSE36552_RAW.tar"),
        ("GSE44183", "GSE44nnn", "GSE44183_RAW.tar"),
        ("GSE66507", "GSE66nnn", "GSE66507_human_blastocyst_rnaseq_counts.txt.gz"),
        ("GSE136447", "GSE136nnn", "GSE136447_555-samples-fpkm.txt.gz"),
        ("GSE136447", "GSE136nnn", "GSE136447_555sample_gene_count_matrix.txt.gz"),
        ("GSE109555", "GSE109nnn", "GSE109555_RAW.tar"),
        ("GSE150578", "GSE150nnn", "GSE150578_RAW.tar"),
    ]
    for acc, prefix, fname in geo_suppl:
        tag = fname[len(acc) + 1:].replace(".txt.gz", "").replace(".tar", "")
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/suppl/{fname}"
        out.append(source(
            f"GEO_{acc}_{tag}", acc, "NCBI GEO supplementary", "expression_matrix", url,
            "GEO supplementary archive (Last-Modified recorded by probe)", GEO_POLICY,
            f"cite the original publication linked to {acc}; state GEO accession",
            "Official upstream supplementary file, filename verified against the GEO record. "
            + GEO_POLICY_QUOTE,
            licence_status="VERIFIED_POLICY",
        ))

    # --- Arc VCC 2025 official CC0 resources ---------------------------------
    arc_base = ("https://storage.googleapis.com/arc-institute-virtual-cell-atlas/"
                "virtual-cell-challenge/2025/")
    arc_small = [
        ("gene_names", "gene_names.csv"),
        ("pert_counts_train", "train/pert_counts_Training.csv"),
        ("pert_counts_validation", "validation/pert_counts_Validation.csv"),
        ("pert_counts_test", "test/pert_counts_Test.csv"),
    ]
    for tag, rel in arc_small:
        out.append(source(
            f"ARC_{tag}", "ARC-VCC-2025", "Arc Virtual Cell Atlas (official GCS)",
            "provenance_metadata", arc_base + rel, "VCC 2025 frozen release",
            ARC_CC0_EVIDENCE,
            "cite Roohani et al., 2025; CC0 requires no attribution but cite anyway",
            "Official upstream file. " + ARC_CC0_QUOTE,
            licence_status="VERIFIED_CC0",
        ))
    out.append(source(
        "VCC_validation_h5ad_GCS", "ARC-VCC-2025",
        "Arc Virtual Cell Atlas (official GCS)", "perturbation_table",
        arc_base + "validation/adata_Validation.h5ad", "VCC 2025 validation split",
        ARC_CC0_EVIDENCE,
        "cite Roohani et al., 2025; CC0 requires no attribution but cite anyway",
        "Official real CRISPRi expression matrix (6.9 GB). OPTIONAL and NOT required "
        "for the DE-level T3 evaluation: `cell-eval run -dr` accepts the 76 MB CC0 DE "
        "parquet instead. Download with w1_fetch_large.py only if single-cell resolution "
        "is later needed; HTTP Range resume. " + ARC_CC0_QUOTE,
        licence_status="VERIFIED_CC0", large=True,
    ))

    # --- Arc VCC 2025 perturbation resources ---------------------------------
    ms = "https://www.modelscope.cn/api/v1/datasets/arcinstitute/PIE_arc_vcc_25/repo?Revision=master&FilePath="
    for split in ["competition_train", "competition_val"]:
        out.append(source(
            f"VCC_{split}_DE_parquet", "ARC-VCC-2025", "ModelScope mirror", "perturbation_table",
            ms + f"de/{split}.parquet", "ModelScope Revision=master", MODELSCOPE_PAGE,
            "cite Roohani et al., 2025 (Virtual Cell Challenge) and the Arc Institute dataset",
            "DE tables (146 train / 44 validation perturbations) derived from the official "
            "Arc VCC 2025 H1 hESC CRISPRi data. " + ARC_CC0_QUOTE + " CC0 permits "
            "redistribution and derivative tables, so the CC0 upstream dedication governs "
            "these DE tables even though the ModelScope platform metadata field reads "
            "'other' (recorded here for transparency). These tables are the -dr/--de-real "
            "input of `cell-eval run` and make T3 evaluable WITHOUT the multi-GB h5ad.",
            licence_status="VERIFIED_CC0",
        ))
    out.append(source(
        "VCC_train_h5ad_GCS", "ARC-VCC-2025",
        "Arc Virtual Cell Atlas (official GCS)", "perturbation_table",
        "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/"
        "virtual-cell-challenge/2025/train/adata_Training.h5ad",
        "VCC 2025 frozen training split", ARC_CC0_EVIDENCE,
        "cite Roohani et al., 2025; CC0 requires no attribution but cite anyway",
        "Official 15.5 GB training split, CC0. NOT required by this zero-shot benchmark "
        "(no model training step); optional via w1_fetch_large.py --all. " + ARC_CC0_QUOTE,
        licence_status="VERIFIED_CC0", large=True,
    ))
    return out


def read_manifest():
    if not MANIFEST.exists():
        raise SystemExit("data_manifest.csv missing - run: python w1_fetch.py init")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(rows):
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def cmd_init(_args):
    rows = [{k: v for k, v in s.items() if not k.startswith("_")} for s in build_sources()]
    # `init` syncs the frozen source list; it preserves provenance fields of
    # rows that already have a verified local copy, including multi-GB files
    # downloaded by w1_fetch_large.py.
    preserved = 0
    if MANIFEST.exists():
        old = {r["entry_id"]: r for r in read_manifest()}
        keep = ("local_path", "sha256", "download_date", "status", "http_status",
                "content_length_bytes", "last_modified", "probe_date")
        for r in rows:
            o = old.get(r["entry_id"])
            if not o:
                continue
            for k in keep:
                if o.get(k):
                    r[k] = o[k]
            preserved += 1
    write_manifest(rows)
    n_ver = sum(1 for r in rows if r["licence_status"] in VERIFIED)
    print(f"init: {len(rows)} source rows -> {MANIFEST.name}; "
          f"preserved {preserved} existing row(s)")
    print(f"licence_status: verified={n_ver}, UNVERIFIED={len(rows) - n_ver}. "
          "No download occurs here.")


def head(url):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.headers.get("Content-Length", ""), r.headers.get("Last-Modified", "")
    except urllib.error.HTTPError as e:
        return e.code, "", ""
    except Exception as e:
        return type(e).__name__, "", ""


def cmd_probe(_args):
    rows = read_manifest()
    src = {s["entry_id"]: s for s in build_sources()}
    for row in rows:
        code, size, modified = head(row["url"])
        row["http_status"] = str(code)
        row["content_length_bytes"] = str(size)
        row["last_modified"] = modified
        row["probe_date"] = now()
        cap = src[row["entry_id"]]["_max_bytes"]
        is_large = src[row["entry_id"]]["_large"]
        exceeded = size.isdigit() and int(size) > cap
        already = bool(row["sha256"] and row["local_path"]
                       and (ROOT / row["local_path"]).exists()
                       and row["status"].startswith("DOWNLOADED"))
        if already:
            # keep DOWNLOADED_PROVENANCE / DOWNLOADED_DATA status on re-probe
            pass
        elif code != 200:
            row["status"] = "UNREACHABLE"
        elif is_large:
            row["status"] = "REQUIRES_LARGE_FETCH"
        elif exceeded:
            row["status"] = "EXCLUDED_SIZE"
        elif row["licence_status"] not in VERIFIED:
            row["status"] = "BLOCKED_LICENCE"
        elif row["kind"] == "provenance_metadata":
            row["status"] = "PROBED"
        else:
            row["status"] = "AVAILABLE_DATA"
        print(f"  {row['entry_id'][:42]:42s} {str(code):5s} {str(size):>13s}  {row['status']}")
    write_manifest(rows)
    print(f"probe: manifest updated, {now()}")


def safe_name(row):
    parsed = urlparse(row["url"])
    # ModelScope serves files from a URL ending in `/repo` with the real name in
    # the FilePath query parameter; prefer that so the extension survives.
    q = parse_qs(parsed.query).get("FilePath", [])
    if q and Path(unquote(q[0])).name:
        base = Path(unquote(q[0])).name
    else:
        base = Path(unquote(parsed.path)).name
        if not base or base in ("/", "."):
            base = "file.bin"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in base)


def cmd_fetch(args):
    rows = read_manifest()
    wanted = set(args.ids) if args.ids else None
    DATA_DIR.mkdir(exist_ok=True)
    fetched = 0
    for row in rows:
        if wanted and row["entry_id"] not in wanted:
            continue
        is_data = row["kind"] != "provenance_metadata"
        if row["status"] == "REQUIRES_LARGE_FETCH":
            print(f"WAIT  {row['entry_id']}: multi-GB file; use python w1_fetch_large.py")
            continue
        if is_data and not args.allow_data:
            print(f"WAIT  {row['entry_id']}: data body; needs fetch --allow-data")
            continue
        if is_data and row["licence_status"] not in VERIFIED:
            print(f"SKIP  {row['entry_id']}: licence_status={row['licence_status']}")
            continue
        allowed = ("AVAILABLE_DATA", "DOWNLOADED_DATA") if is_data else (
            "PROBED", "NOT_DOWNLOADED", "DOWNLOADED_PROVENANCE")
        if row["status"] not in allowed:
            print(f"SKIP  {row['entry_id']}: status={row['status']}")
            continue
        if row["local_path"] and row["sha256"] and (ROOT / row["local_path"]).exists():
            h = hashlib.sha256()
            with (ROOT / row["local_path"]).open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() == row["sha256"]:
                print(f"KEPT  {row['entry_id']}: local copy already verified")
                continue
        size = int(row["content_length_bytes"] or 0)
        dest = DATA_DIR / f"{row['entry_id']}__{safe_name(row)}"
        # Recover a file that finished downloading before the process was
        # interrupted but was never registered in the manifest. An incomplete
        # partial file never matches the remote Content-Length and is ignored.
        if dest.exists() and not row["sha256"] and size and dest.stat().st_size == size:
            h = hashlib.sha256()
            with dest.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            row["local_path"] = str(dest.relative_to(ROOT)).replace("\\", "/")
            row["sha256"] = h.hexdigest()
            row["download_date"] = now()
            row["status"] = "DOWNLOADED_DATA" if is_data else "DOWNLOADED_PROVENANCE"
            print(f"RECOVERED {row['entry_id']}: complete local file registered")
            continue
        if size and size > MAX_BYTES:
            print(f"SKIP  {row['entry_id']}: size ceiling")
            continue
        if not row["url"].startswith("https://"):
            print(f"SKIP  {row['entry_id']}: non-HTTPS")
            continue
        dest = DATA_DIR / f"{row['entry_id']}__{safe_name(row)}"
        h = hashlib.sha256()
        total = 0
        try:
            with urllib.request.urlopen(urllib.request.Request(row["url"], headers=UA), timeout=TIMEOUT) as r, dest.open("wb") as out:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise RuntimeError("stream exceeded size ceiling")
                    out.write(chunk)
                    h.update(chunk)
        except Exception as e:
            if dest.exists():
                dest.unlink()
            print(f"FAIL  {row['entry_id']}: {type(e).__name__}: {e}")
            continue
        row["local_path"] = str(dest.relative_to(ROOT)).replace("\\", "/")
        row["sha256"] = h.hexdigest()
        row["download_date"] = now()
        row["status"] = "DOWNLOADED_DATA" if is_data else "DOWNLOADED_PROVENANCE"
        row["notes"] += (
            "; fetched from official upstream under VERIFIED_POLICY" if is_data
            else "; downloaded for accession/citation checking only, NOT analysis input"
        )
        fetched += 1
        print(f"OK    {row['entry_id']:42s} {total:>9d} B  sha256={h.hexdigest()[:16]}...")
    write_manifest(rows)
    print(f"fetch: {fetched} file(s) stored in {DATA_DIR.name}/")


def cmd_verify(_args):
    rows = read_manifest()
    checked = passed = 0
    for row in rows:
        if not row["local_path"] or not row["sha256"]:
            continue
        p = ROOT / row["local_path"]
        checked += 1
        if not p.exists():
            print(f"MISSING {row['entry_id']}: {p}")
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        ok = h.hexdigest() == row["sha256"]
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {row['entry_id']}")
    print(f"verify: {passed}/{checked} hashes match")


def cmd_clean(_args):
    """Remove local files that the manifest no longer tracks (e.g. excluded sources)."""
    rows = read_manifest()
    tracked = {Path(r["local_path"]).name for r in rows if r["local_path"]}
    if not DATA_DIR.exists():
        print("clean: no data/ directory")
        return
    removed = 0
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file() and p.name not in tracked:
            p.unlink()
            removed += 1
            print(f"  removed orphan: {p.name}")
    print(f"clean: {removed} orphan file(s) removed; {len(tracked)} tracked file(s) kept")


def cmd_status(_args):
    rows = read_manifest()
    print(f"W1 manifest: {MANIFEST.name} | {len(rows)} rows | {now()}")
    for key in ("licence_status", "status", "kind"):
        counts = {}
        for r in rows:
            counts[r[key]] = counts.get(r[key], 0) + 1
        print(f"  {key:16s} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    n_meta = sum(1 for r in rows if r["status"] == "DOWNLOADED_PROVENANCE")
    n_data = sum(1 for r in rows if r["status"] == "DOWNLOADED_DATA")
    print(f"  downloaded provenance files = {n_meta}; downloaded expression files = {n_data}")
    n_unverified = sum(1 for r in rows if r["licence_status"] == "UNVERIFIED")
    n_cc0 = sum(1 for r in rows if r["licence_status"] == "VERIFIED_CC0")
    print(f"  UNVERIFIED rows = {n_unverified} (withheld from any analysis); "
          f"verified rows = {len(rows) - n_unverified} (of which CC0 1.0 = {n_cc0})")
    print("  NOTE: VERIFIED_POLICY is citation/free-distribution evidence, not an explicit CC "
          "licence; institution-level confirmation is still required before submission. "
          "VERIFIED_CC0 is an explicit public-domain dedication.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("probe").set_defaults(func=cmd_probe)
    f = sub.add_parser("fetch")
    f.add_argument("--ids", nargs="*", help="limit to these entry_id values")
    f.add_argument("--allow-data", action="store_true",
                   help="also download expression files whose policy is verified")
    f.set_defaults(func=cmd_fetch)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("clean").set_defaults(func=cmd_clean)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
