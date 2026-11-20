# ⚠️ NOT YET A LICENCE — AUTHOR DECISION REQUIRED

This file is a **placeholder**, not a grant of rights. No code licence has been
chosen by the author. Publishing this repository without replacing this file
with a real licence text would leave the code legally unusable by third parties
and would fail the Reusable criterion of the FAIR self-check.

## What must be decided

Choose one SPDX identifier, write the matching full licence text into this
file (rename to `LICENSE` if the repository convention requires), and replace
`[[PENDING_CODE_LICENSE]]` in `metadata.jsonld`, `CITATION.cff` and the
manuscript's Code availability / Declarations sections.

| Option | SPDX | Consequence for this article |
|---|---|---|
| MIT | `MIT` | Permissive, simplest; compatible with both upstream model licences. Recommended if no institutional policy dictates otherwise. |
| Apache-2.0 | `Apache-2.0` | Permissive with an explicit patent grant; matches the Geneformer upstream licence. |
| CC0-1.0 | `CC0-1.0` | Public-domain dedication; strongest reuse, but some institutions prefer a copyright-bearing licence for software. |

**Restrictions on non-academic use:** none is currently intended, but this is
an author decision. Frontiers requires the statement to be explicit — write
`"none"` rather than leaving it blank. It belongs in the README project-link
section and the manuscript Declarations.

## Upstream licences are separate and must NOT be overridden

The code licence applies only to the repository's own source. It cannot
relicense the data or model weights:

| Manifest | Rows | Verified | Excluded / pending |
|---|---|---|---|
| `data_manifest.csv` | 42 | 8 `VERIFIED_CC0`, 21 `VERIFIED_POLICY` | 13 `UNVERIFIED` → `BLOCKED_LICENCE`, excluded from analysis |
| `model_manifest.csv` | 37 | 4 `VERIFIED_MIT`, 33 `VERIFIED_APACHE2` | 0 excluded |

The **21 `VERIFIED_POLICY`** rows in `data_manifest.csv` were verified against
the primary repository's stated policy, but the project's own rule requires
institutional confirmation of the citation/reuse conditions before Table 1
(`[[R1]]`) can be filled. Until that confirmation exists, `[[R1]]` stays a
placeholder and those datasets are not claimed as cleared in the manuscript.

One licence-unclear mirror was excluded from the model set: the
`VirtualCell2025/scGPT_human` image carried no licence declaration. The
frozen set instead uses `perturblab/scgpt-human` (MIT) and
`ctheodoris/Geneformer` (Apache-2.0), each pinned to an immutable commit with
per-file SHA256 in `model_manifest.csv`.

## AI-assistance disclosure

Frontiers requires generative-AI use in writing or code development to be
disclosed in the manuscript. This is an author statement and is **not**
supplied by this repository. The `[[R9]]` slot remains unfilled until it is
provided.
