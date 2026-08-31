# Expressivity FT weight-deltas (.expr) — MANIFEST

`.expr` files are large (≈180–690MB each; this folder is ~8GB) → **git-ignored** (`.gitignore`
`presets/expr/*.expr`) and hosted on HuggingFace / GH-release, NOT committed raw. **This MANIFEST is the
tracked record so the wins are never lost** (a validated .expr WAS lost on 2026-06-15 to a gitignore+overwrite —
see .gitignore note). Apply with `--expr <file> --expr-weight W`. All 1.7B unless noted.

## ✅ VALIDATED WINS (keep + host)
| file | what | recipe / verdict |
|---|---|---|
| `italian_csp_topk6.expr` | IT emotion CSP-FT, blocks 22-27 (k6), 10ep | **THE IT default** (cleaner than k4 on clones) |
| `german_csp_k6.expr` | native German emotion CSP-FT (emodb) | first native DE FT |
| `french_csp_k6.expr` | native French emotion CSP-FT (cafe) | first native FR FT |
| `italian_l1626_dense.expr` | IT dense 11-layer L16-26 FT ("WOW" 06-15) | moves clones more than CSP on some emotions; CV-intact loads only |

## ❌ DEAD / SUPERSEDED experiments (candidates to DELETE/ARCHIVE — pending user OK)
- Para CSP-FT (all FAILED): `paraling_csp_*` (k4/k6/k8, ep5/8, no0_mid/wide).
- Para LoRA (all FAILED): `para_nonverbal_b*`, `para_exp2a_*`, `para_smoke_lang`.
- Old para LoRA (weak): `paralinguistic_{ep0,ep1,ep2,ep8,v2,aug}`.
- Emotion-vector experiments (partial/superseded by CSP): `italian_emovec_{float,tau,wide}`, `italian_l0_27_dense`,
  `italian_multi_l1626_dense`, `italian_multitag_l1626_dense`, `italian_csp.expr` (pre-k4), `italian_l1626_r{32,64}`,
  `italian_csp_topk6_15ep` (= k6@10 by ear, redundant), `italian_csp_topk4` (superseded by k6, keep only if A/B wanted).

> Para events are NOT solved via .expr (decoder/pronunciation-layer ceiling) — the shipped para win is the
> STEERING VECTOR (`presets/steer/paraling/`), not an .expr.
