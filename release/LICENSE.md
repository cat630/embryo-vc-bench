# MIT Licence — embryo-vc-bench

**SPDX identifier:** `MIT`
**Restrictions on non-academic use: none.** The authors impose no additional
field-of-use restriction beyond the MIT terms below; commercial and
non-academic reuse are permitted under those terms.

> Author-name note: the copyright holder below is taken from the public OSF
> contributor record of registration `8tzrq` (`ZHAO Mingpeng`). Confirm the
> spelling, order and any additional copyright holders before submission.

---

MIT License

Copyright (c) 2026 Mingpeng Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Upstream licences are separate and must NOT be overridden

The MIT licence above applies only to the original source code and
documentation written for this repository. It does not relicense the data or
the model weights:

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
supplied by this repository. The manuscript `[[R9]]` slot remains unfilled
until it is provided.

## Frozen evidence snapshot

`results/w7_release_summary.json` is a byte-frozen 18 November 2026 record
whose SHA256 is cited in manuscript section 3.6. Its
`pending_author_actions` checklist is a historical snapshot of that date and
is deliberately **not** rewritten when identifiers are filled, so the cited
hash stays valid. Current status is documented in `release/README.md`.
