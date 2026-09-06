# EPIC AMX-NATIVE — analysis and evidence base

Reference host: GCP Emerald Rapids, Xeon Platinum 8581C, 1 socket, 12 physical cores,
SMT OFF for measurement, 1 NUMA, L3 260 MiB, AMX-INT8 + AMX-BF16 + AVX-512 VNNI.
Canonical topology 2 workers x 6 threads, core-major masks 0-5 / 6-11.

## 0. Scope — precision regimes, not one kernel

In scope from the start so today's decisions do not foreclose them: AMX INT8/W8A8 · AMX
BF16 · existing W4/INT4 · future W4A16 / W4A8 / mixed W4-W8. No new quantization framework
is built in this epic, and AutoRound is not implemented unless a narrow diagnostic needs it.
Numbering follows PLAN (AMX-0a, AMX-0, AMX-1, AMX-2, AMX-2b, AMX-3 BF16, AMX-4 dataflow,
AMX-5 feeding, AMX-6 lanes, AMX-7 1.7B-vs-0.6B, AMX-8 baseline, AMX-9 qualification).

## 1. The objective, stated so it can be falsified

NOT "every linear must use AMX". On tiny M the tile setup, the activation pack and the
int32 rescale can cost more than they save, and that is a measurement, not a belief.

The objective is: **on an AMX host the serving architecture must differ from the VNNI
serving architecture.** Today it does not — AMX is an extra kernel behind a gate on a
VNNI-shaped engine. The falsifiable form: if we disable AMX entirely and lose almost
nothing, the machine is not being used for the reason it was chosen.

Subordinate to the streaming mission (PLAN, AMX-MISSION): once audio is flowing it must
keep flowing. Every AMX change is judged on BOTH first-audio latency and steady-state
generation rate, and a TTFA win that pushes STREAM_RTF toward 1.0 is a loss.

## 2. AMX-1 — evidence already in the tree (archaeology, done 2026-09-06)

| Finding | Where | Result then | Status now |
|---|---|---|---|
| INT8 AMX gate min_b | `g_mm_gate[QWEN_MMK_INT8_AMX]`, `qwen_tts_kernels.c` | lowered 4 -> 3 | CURRENT |
| rows/thread >= 256 discriminator | `qwen_amx_int8_rows_ok_nt()` | agrees with 23/24 measured cells; the rows>=cols rule it replaced was WRONG (unpaired arms, 14% drift) | CURRENT |
| Fused QKV judged on q_dim + 2*kv_dim | `qwen_mm_use_()` | shipped | CURRENT |
| Per-worker batch width in the real server | X86-2 | B 0.9-3.8 across C=1..8 | CURRENT — and see F1 |
| Persistent packed RHS | X86-3 | +7% for +4.3 GB RSS; `QWEN_AMX_PREPACK` defaults 0 | CURRENT as a *result*, RETEST as a *design* (lifetime/layout may have been wrong, not the idea) |
| AMX activation-pack duplicated nt times | X86-5, `qwen_region_i8_run` | real, worth ~5-10% of AMX projection time, worthless at today's B | RETEST once B rises |
| B=32 two-accumulator AMX prototype | `tests/prefill_bench.c` | SIGILL + numerically wrong (2.18 rel) | REJECTED as written, behind `QWEN_PB_AMX_PROTO=1` |
| BF16 dpbf16 -21% at -j1 | v0.18.0, EPYC Zen5 | real | OBSOLETE AS EVIDENCE FOR THIS EPIC — that is AVX-512 `VDPBF16PS`, a VNNI-class instruction, NOT AMX `TDPBF16PS`. Do not cite it as an AMX-BF16 result. |
| oneDNN / benchdnn oracle | P5.2, P5.5, P5.8 | never run | SUBSUMED by AMX-2 |
| AMX kernel healthy under the pool | box probe during X86-5 | engine AMX matmat fine with tile permission requested before or after the pool exists | CURRENT |

## 3. Three structural findings that reshape the epic

### F1 — cross-slot fusion has almost no headroom left; the batcher already does it

The proposal "fuse independent ready streams to raise M" describes something the server
already performs: continuous batching. The measured B of 0.9-3.8 per prefork worker at
C=1..8 IS the fused width. With 2 workers, C=8 ideally yields B=4 per worker, and we
measure 3.8 — near the ceiling, not a scheduling failure.

So M cannot be materially raised by fusing better ACROSS requests. It is bounded by how
many requests can be admitted at all, and C=6/C=8 already underrun (18/18 and 24/24
requests starved). Raising concurrency to raise M is circular.

Consequence: AMX-5 is reformulated in PLAN — cross-stream fusion is sequenced LAST, not first. The headroom is in raising M *within* one request —
CP's 16 sequential steps and 15 lm_heads over one frame, multi-frame decode, the MTP
projection — where rows exist that are today executed as separate small calls. That is a
different, harder, and more honest question than cross-slot fusion.

### F2 — the largest block in the request never reaches the AMX dispatcher at all

From the C=4 deep pass: `decoder.total` 1610 ms/request, of which `decoder.conv_stack`
1526 ms — 94.8%. Reading the code:

- the fp32 decoder path is `im2col` + `cblas_sgemm` (OpenBLAS);
- the int8 path is `sd_gemm_panel` -> `sd_tile_2x4` / `sd_tile_1xN`, a hand-written
  register-tile GEMM;
- `qwen_tts_speech_decoder.c` contains **zero** calls to `qwen_mm_use` or the matmat
  dispatcher.

So the single biggest, most naturally matrix-shaped work in the request is structurally
outside every AMX decision we have made. Its shape is favourable: M = out_ch (512-1024),
K = in_ch * kernel, N = panel columns. Any statement of the form "AMX coverage is X%" is
meaningless until the decoder is in the denominator.

This is the most likely large win in the epic and it is not in the original task list.

### F3 — the census machinery has an unresolved dump-coverage defect

The batched-CP instrumentation (`8f950bd`) produced zero: `cp.decode.total` stayed 76.1%
unattributed. Leading hypothesis, unverified: prefork children exit via `_exit()`, which
does not run `atexit` handlers, and/or pool worker threads never register their TLS block.
The identical symptom already appeared for `units`.

Any AMX census reads the same machinery from the same worker threads. Fix or disprove this
FIRST, or AMX-0 will report an AMX share of zero and we will believe it.

## 4. Metrics, defined once

- `amx_call_share`  — fraction of INT8 linear CALLS dispatched to an AMX kernel.
- `amx_mac_share`   — same, weighted by MACs. The honest coverage number.
- `amx_wall_share`  — same, weighted by measured wall time inside the linear.
- `amx_eligible_share` — of the work NOT on AMX, the split between
  (a) shape-eligible but gated off, (b) shape-ineligible, (c) structurally outside the
  dispatcher (today: the whole decoder). This is what separates a kernel problem from a
  feeding problem BEFORE any oneDNN run, and it is the cheapest discriminator we have.
- `stream_margin = 1.0 - STREAM_RTF`. RTF 0.95 is 5% margin and is not a production win.

Denominators are per request and per frame, never per process.

## 5. AMX-0 census schema (one row per linear site)

    component  site  B/M  N  K  calls/req  MACs/req  wall_ms/req  kernel_selected
    gate_reason_if_not_amx  quant_ms  pack_ms  kernel_ms  epilogue_ms

Components: Talker prefill, Talker decode, CP prefill, CP decode (the REAL batched path
`qwen_batch_cp_predict`), lm_heads, decoder conv stack, decoder transformer.

## 6. AMX-2 shape matrix and the decision gate

M sweep 1,2,3,4,6,8,12,16,24,32 — including 3 and 6, which are the widths the server
actually produces. Arms: native VNNI, native AMX INT8, oneDNN INT8 AMX, native AMX BF16
if present, oneDNN BF16 AMX. Cost split: activation prep / pack / kernel / epilogue / total.
Paired and interleaved inside one process (`qwen_mm_force()`); cross-process arms swing 14%.

Decision gate:

- **A** — our AMX is far below the oneDNN oracle on favourable shapes -> AMX-3 is kernel,
  layout and blocking work.
- **B** — our kernels are healthy and the engine never presents adequate work -> the effort
  is engine architecture: F2 (bring the decoder in), then intra-request row aggregation.
- **C** — both.
- **D** — the oracle itself does not win until M is far above anything we can produce ->
  stop pursuing AMX on the decode path, and the epic reduces to F2 plus prefill.

oneDNN is an ORACLE, never a serving dependency.

## 7. Do not redo

The 14% cross-process drift; the rows>=cols rule; the B>=4 gate; the B=32 prototype as
written; the packed-RHS experiment in the same lifetime and layout; citing AVX-512
`VDPBF16PS` results as AMX-BF16 evidence.

## 8. AMX-0 RESULT (2026-09-06, Emerald Rapids, 1.7B --int8, AMX build, 2x6, C=4)

`amx_mac_share` **14.0%**, `amx_call_share` 10.2%, over 1212.1 GMAC / 16525 calls
(CALL rows only; WRAPPER and SLICE rows re-record MACs already counted underneath, and a
first aggregate that included them inflated the total by 14%).

| component | MAC share | compute-wall share | AMX inside it |
|---|---|---|---|
| decoder | 81.2% | 46.3% | 0.0% |
| talker  | 16.4% | 39.1% | 84.8% |
| cp      |  2.3% | 14.6% | 0.0% |

Why both metrics: CP is 2.3% of the MACs and 14.6% of the compute wall — a 6x divergence.
It is B=1 GEMV re-reading weights, bandwidth-bound, and no AMX kernel changes that.

`amx_eligible_share` of the 86% not on AMX: **94.1% structurally outside the dispatcher**
(the decoder), 5.3% B=1 GEMV with no AMX form, **0.5% shape-eligible but gated to VNNI**.

Consequence, stated plainly: every gate parameter we have tuned — B>=3, rows/thread>=256,
QKV judged on q+2kv — operates on half a percent of the weighted work. Where AMX is
applicable at all it is already being selected (Talker 84.8%). This is a FEEDING problem,
class B, and AMX-2's oracle sweep can raise at most the 14% already on AMX.

Top non-AMX sites: `decoder_conv_int8` 804.1 GMAC (66.3% of all MACs, hand-written 2x4
register tile), `decoder_sgemm` 157.8 GMAC (13.0%, fp32 OpenBLAS), then Talker/CP B=1 GEMV.

## 9. AMX-0 CORRECTED (second pass, merged HEAD) — the first number was wrong

`amx_mac_share` is **2.0%**, not the 14.0% reported in section 8. Section 8 stands only as a
record of the error.

Two defects in the method, both inflating AMX:
1. `kmask` is an OR of every kernel a shape ever used. A shape that took AMX once and VNNI a
   thousand times sets both bits, and classifying on "does the mask contain AMX" counted it as
   fully AMX. A share cannot be read off a mask.
2. `t_census_cur` is never cleared, so `qwen_matmat_stats_note` attributes a kernel to whichever
   row that thread opened last. This produced `decoder_conv_int8 -> VNNI` rows, which contradict
   the code (the decoder makes no dispatcher call) and should have been investigated the moment
   they appeared rather than reported.

Fixed: per-row `kmacs[]`/`kcalls[]` per kernel, and the path `kind` emitted by the engine so the
aggregator stops guessing which paths are wrappers (that hardcoded list was also wrong, which is
why the denominator moved 1212.1 -> 1403.5 GMAC).

| kernel | GMAC | % of all |
|---|---|---|
| **INT8 AMX** | **0.0** | **0.0% — it never runs** |
| int8 VNNI vpdpbusd | 172.6 | 12.3% |
| int8 GEMV | 149.4 | 10.6% |
| bf16 AMX tiles | 27.9 | 2.0% |
| never reached the dispatcher | — | 75.0% |

**The finding: INT8 AMX executes zero MACs under the real server.** Its gate requires B>=3 AND
rows/thread>=256. The batched server measures B 0.9-3.8 per worker, and CP rows over 6 threads
fall below 256. Every parameter tuned on that gate governs a path that does not run. The only
AMX in production is BF16 tiles in the Talker prefill, 2.0% of all MACs.

This strengthens the class-B classification rather than weakening it, since both method errors
were in AMX's favour.
