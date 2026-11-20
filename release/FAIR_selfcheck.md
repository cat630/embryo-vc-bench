# FAIR self-check — embryo-vc-bench W7 release (18 November 2026)

Self-assessment against the FAIR principles cited as [14] in the manuscript.
**Overall: PARTIAL — Interoperable and Reusable are fully met; Findable and
Accessible remain blocked on the Zenodo release DOI.** The public repository
(`https://github.com/cat630/embryo-vc-bench`) and the MIT licence now exist, but no persistent identifier has
been minted yet. No criterion is marked met without a file that can be
inspected now.

| Principle | Criterion | Status | Evidence |
|---|---|---|---|
| **F**indable | F1 (meta)data assigned a globally unique, persistent identifier | **PENDING** | `metadata.jsonld`, `CITATION.cff` hold explicit `[[PENDING_ZENODO_DOI]]`; the OSF registration DOI `10.17605/OSF.IO/8TZRQ` identifies the *plan*, not the code |
| **F** | F2 data described with rich metadata | **MET** | `metadata.jsonld` (`SoftwareSourceCode` + `Dataset`), `CITATION.cff`, `RELEASE_MANIFEST.json` with per-file SHA256 and provenance |
| **F** | F3 metadata clearly include the identifier of the data they describe | **PENDING** | requires the Zenodo DOI to be filled in all three metadata files |
| **F** | F4 (meta)data registered or indexed in a searchable resource | **PARTIAL** | public GitHub repository `https://github.com/cat630/embryo-vc-bench` is indexed; Zenodo deposit pending |
| **A**ccessible | A1 retrievable by identifier using a standardised protocol | **PARTIAL** | HTTPS retrieval from `https://github.com/cat630/embryo-vc-bench` works; persistent identifier pending |
| **A** | A1.1 protocol open, free, universally implementable | **MET** | HTTPS to the public GitHub repository; no embargo, no authentication, no paywall |
| **A** | A1.2 protocol allows authentication/authorisation where necessary | **N/A** | no restricted data; all sources are public |
| **A** | A2 metadata accessible even when data are no longer available | **PARTIAL** | `metadata.jsonld` and `CITATION.cff` are local; persistence requires the Zenodo deposit |
| **I**nteroperable | I1 (meta)data use a formal, accessible, shared language | **MET** | CSV (RFC 4180), JSON, JSON-LD/schema.org, CFF 1.2.0, YAML; UTF-8 throughout |
| **I** | I2 vocabularies follow FAIR principles | **MET** | schema.org terms; SPDX `MIT` in `metadata.jsonld` / `CITATION.cff` / `LICENSE.md`; accession/PMID fields in `data_manifest.csv` |
| **I** | I3 qualified references to other (meta)data | **MET** | `data_manifest.csv` (accession + SHA256 + licence evidence URL), `model_manifest.csv` (repo + immutable commit + SHA256), OSF DOI in every metadata file |
| **R**eusable | R1 rich, accurate, plural attributes with provenance | **MET** | `RELEASE_MANIFEST.json`, `figure_source_data/SOURCE_INDEX.md`, `results/w7_release_summary.json` (hardware, wall-clock, seeds, failures) |
| **R** | R1.1 clear and accessible data usage licence | **MET** | `LICENSE.md` holds the full MIT text; upstream licences are recorded separately and are not overridden |
| **R** | R1.2 detailed provenance | **MET** | W1/W1b/W2/W3/W4/W5/W6 scripts and logs; 27/27 content-integrity pass after the 6.9 GB h5ad selective-resume repair; every filled result slot carries source file, seed and SHA256 |
| **R** | R1.3 domain-relevant community standards | **MET** | fixed-seed preregistered protocol, BH-corrected permutation nulls, bootstrap CIs, mandatory naive baselines, OSF Secondary Data Preregistration v3 |

## Journal-specific items (Frontiers Technology and Code)

| Requirement | Status | Where |
|---|---|---|
| Project link | **MET** | `https://github.com/cat630/embryo-vc-bench` in `metadata.jsonld`, `CITATION.cff`, `release/README.md`; manuscript `[[R9]]` pending DOI |
| Operating system stated | **MET** | `environment/platform.json`, manuscript §2.11 |
| Programming language stated | **MET** | Python 3.13.14, manuscript §2.11 |
| Restrictions on non-academic use | **MET** | `LICENSE.md` states `"none"` explicitly (author decision, 20 November 2026) |
| Metadata file with `SoftwareSourceCode` / `Dataset` | **MET** | `metadata.jsonld` |
| Exact commands reproducing every main figure/table | **MET** | `reproduce.ipynb`, `reproduce.py`, this README |
| Third party can re-run main figures | **PARTIAL** | replay from released artefacts works locally; full from-scratch re-scoring requires the pinned model weights and is not required for the released figure source data |
| Data availability statement with accessions | **PARTIAL** | `data_manifest.csv` complete; Table 1 (`[[R1]]`) blocked on 21 institutional licence confirmations |
| CRediT / Funding / AI disclosure / COI choice | **PENDING (author only)** | `[[R9]]`; see `LICENSE.md` for the AI-disclosure note |

## Honest limitations of this self-check

- The Docker path in `environment/Dockerfile` has **not** been executed on
  Linux. All published numbers were produced on Windows 11 / CPython 3.13.14.
  The Linux path is an interoperability aid, not a validation claim.
- The replay deliberately does **not** reload the foundation models or
  recompute scores. It replays the released, hash-verified artefacts so a third
  party can reproduce the figures without a GPU or the model weights. Full
  re-scoring commands are documented in `../README.md` §3.9.
- Do **not** re-run `w6_orthogonal.py --mode all` after the manuscript has
  cited its artefacts: that file carries a `generated_at` field and its hash,
  plus the hash of `w6_orthogonal_summary.json`, are cited in §3.5. Re-running
  would invalidate 13 cited SHA256 values.
