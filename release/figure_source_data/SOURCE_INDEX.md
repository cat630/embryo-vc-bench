# Figure and table source data — provenance index

Every file below is derived from an already released, hash-verified
artefact in `results/`. No model was loaded and no preregistered score
was recomputed to produce them. Framework-internal DES/PDS/MAE are not
on a log2FC scale and must not be compared numerically with Virtual
Cell Challenge leaderboard values.

| File | Rows | SHA256 | Source script | Seeds / protocol | Note |
|---|---|---|---|---|---|
| `ceiling_de_metrics.csv` | 1 | `fa5edcb15f180701f22de6322d20e9a0f413c0d0119a7fc9ceeaa9c5a7f0ae62` | `w6_ceiling.py --run (cell-eval 0.8.2 run --ceiling)` | ceiling seed 0; profile de; --allow-discrete; 11,961-cell deterministic subsample | SUBSAMPLE-DEPTH bound only. Full-data ceiling was not executable at 4.9 GB available RAM (compute_ceiling needs ~3x the 6.9 GB CSR matrix). Model-free, descriptive, used for no hypothesis decision. |
| `fig4_fold_level_scores.csv` | 80 | `397ce31864631c1868a1dfb2ed7421578fb1bd6558c47b81e1b3bd8dbfbf2ed9` | `w5_score.py --mode score` | 0-4; verbatim snapshot; 60 T2 rows (15 fold instances x 4 methods) + 20 T3 rows (5 seeds x 4 methods) | Use this file to redraw forest plots or confidence intervals. Zero nulls, zero errors. |
| `fig4_paired_contrasts.csv` | 32 | `39a556f800d8e15be5f31b06fef23867c116146503a0920ac38f4e6f9c09767d` | `w7_release.py (descriptive replay only; no new inferential test)` | bootstrap RNG seed 0; B=2000; paired per-instance difference model minus baseline; percentile bootstrap CI | These are descriptive paired effect sizes. They are NOT p-values and were NOT added after scoring as a new significance test. The registered p_perm tests each method against permuted rows, not the model-vs-baseline contrast. |
| `fig4_t2_t3_means.csv` | 8 | `c218e33b40146d785111ddaee3657fb9ed67a75cd39de6f1d9b8e209f4c83a1b` | `w5_score.py --mode score` | 0-4 (T3) / 15 fold instances x 5 seeds (T2); w5-adapter-protocol-2; fixed 17,555-gene axis; B=1000 permutation; BH across 4 methods per block | Exactly the 8 task x method rows reported in manuscript 3.3. Values are framework-internal and NOT on a log2FC scale. |
| `fig_t2_by_heldout_atlas.csv` | 12 | `e2536b90774787fed02822a255360f3cbe2754190a9123fa710a084b8b96d5dd` | `w6_orthogonal.py --mode atlas` | 0-4; verbatim W6 output; descriptive re-stratification of already released scores | Three held-out atlases with three distinct gene-identifier systems. Supports 'externally replicable across independent atlases' only. |
| `table1_datasets.csv` | 42 | `1853c8ec2666e9cff791d4b429b634dfa045adbee8f27353f7b5905cd12b2249` | `w1_fetch.py / w1_fetch_large.py / w1_validate.py` | not applicable; verbatim licence/provenance columns | 13 UNVERIFIED rows are BLOCKED_LICENCE and excluded. The 21 VERIFIED_POLICY rows require institutional confirmation before manuscript [[R1]] / Table 1 can be filled; this file does NOT assert that clearance. |
| `table3_validation_layers.csv` | 3 | `56b5c89a89a562c507beed458cde65b875b1c0161a4e21301c5921cf6026e1e8` | `w6_orthogonal.py` | 0-4 / ceiling seed 0; structured form of the W6 Table 3 | Wording audit PASS with 0 unresolved assertions; see results/w6_wording_audit.json. |

## Integrity

- `results/w6_orthogonal_summary.json` and `results/w6_wording_audit.json`
  carry `generated_at` fields and are cited by SHA256 in manuscript 3.5.
  Do NOT re-run `w6_orthogonal.py --mode all` after submission drafting.
- `w5_scores.csv` SHA256 is cited in manuscript 3.3 and must stay stable.
- Re-run `python w7_release.py --refresh` after changing any file in
  `release/`, then re-sync any manuscript hash citation that points here.
