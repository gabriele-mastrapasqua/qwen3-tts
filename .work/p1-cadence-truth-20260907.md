# P1 cadence truth — 2026-09-07

Task · P1 Cadence truth (CT-1 through CT-5) on the current Tier-A serving binary.

Question · Is playback loss primarily caused by the audio quantum, decoder fixed
overhead, inline admission, or insufficient Talker batching, and which next task has
the strongest evidence?

Known facts · The canonical source was `40487610d2747798d6911d7ac73943a0f73caf17`.
The measured binary carried source fingerprint `4048761:clean`, binary SHA-256 prefix
`0894f126b794e630`, and was built with `SIMD=amx`. The reference machine was GCP
`c4-standard-24`, Xeon Platinum 8581C, one NUMA node, CPUs 0–11 online, SMT off,
2x6 prefork, engine-owned decoder pool, decoder batch cap 2, 1.7B INT8 Design D.

Unknowns · Client receive marks remain upper bounds on server-side lateness. No P1 KPI
run enabled intrusive census/profile output. CT-2 is a diagnostic trace fit, not a
serving KPI. CT-3 is one matched admission/control pair; it establishes magnitude but
does not prove that every overlapping gap was caused by admission. None of these runs
is a five-minute qualification.

Files/functions inspected · `tests/playback_sim.py`, `tests/serve_parallel_wave.py`,
`tests/serve_soak.py`, `tests/soak_client.py`, `tests/load_test.py`, `qwen_tts.c`
(`[ITER]`, `[DECODE]`, `[SDPHASE]`, `[TTFA2]`, serve profile and gang policy),
`tools/check_plan.py`, and the P1 runbook in `PLAN.md`. Raw run artifacts remain in
the private evidence area; this addendum records only reviewed aggregates.

## Evidence

### Provenance and controls

All runs used the same 1.7B model, English bank with 21 texts, speaker `ryan`, seed
base 42, temperature 0, 2x6/SMT-off, batch cap 2, engine decoder pool, ragged threshold
2, prefix cache, INT8 Design D enabled and BF16 disabled. CT-1 used `QWEN_TTFA_TRACE=1`
only; CT-2 additionally used `QWEN_SD_PHASE=1` in separate diagnostic runs. CT-3 used
one temporary client probe plus a matched three-stream no-admission control. CT-4 used
serve profile in separate 1x6 B1/B2 diagnostics. CT-5 had no trace/profile/census.

The clean CT-5 runner explicitly recorded `dirty=no`. CT-1 and CT-4 used a synchronized
source snapshot without a `.git` directory, so their harness fallback printed
`dirty=yes`; the embedded binary fingerprint and SHA were unchanged. Those profile
arms are diagnostic, not qualification evidence.

### CT-1 — quantum discriminator

WAVE results are `p50/p95`; fixed-buffer columns are stall rates and prebuffer-le rates
at 100/250/500/1000 ms, respectively. `prebuffer` is required prebuffer and `safe` is
per-request `safe_play_start`.

| arm | C | STREAM_RTF | TTFA ms | prebuffer ms | safe ms | stall rates | prebuffer-le rates | max gap ms | coalesced reads |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| A q8/q8, gang default | 3 | 0.674/0.774 | 353/556 | 197/329 | 550/863 | 0.67/0.17/0/0 | 0.33/0.67/1/1 | 504/525 | 5.1% |
| A q8/q8, gang default | 4 | 0.784/0.826 | 433/547 | 267/337 | 632/825 | 0.94/0.25/0/0 | 0/0.50/1/1 | 521/554 | 5.1% |
| B q32/q32, gang default | 3 | 0.673/0.767 | 351/549 | 1618/1898 | 1858/2339 | 0.92/0.92/0.92/0.83 | 0.08/0.08/0.08/0.17 | 1904/2146 | 35.3% |
| B q32/q32, gang default | 4 | 0.783/0.944 | 426/538 | 1910/2173 | 2142/2487 | 0.94/0.94/0.88/0.81 | 0/0.06/0.12/0.19 | 2078/2210 | 36.5% |
| C q8/q8, `GANG_MIN=64` | 3 | 0.676/0.786 | 351/556 | 226/357 | 577/914 | 0.67/0.17/0/0 | 0.33/0.67/1/1 | 492/535 | 5.1% |
| C q8/q8, `GANG_MIN=64` | 4 | 0.781/0.813 | 434/610 | 246/326 | 628/911 | 1/0.12/0/0 | 0/0.56/1/1 | 534/594 | 5.1% |

All WAVE requests completed with zero errors, rejects and timeouts. Arm A and C have
usable receive fidelity. Arm B's 33–37% coalesced-read share makes its exact cadence
percentiles non-citation-grade; the high share is itself evidence of queued/bursty
delivery, and the same direction was reproduced by the short SOAK.

The short closed-loop C4 SOAK used 15 s warmup and four 15 s windows, with no intrusive
profile/census. It completed 24 q8 and 23 q32 KPI requests, with zero errors/rejects/
timeouts. The analyzer rejected both runs only because the short completed-request
class mix drifted; this is not a server failure.

| arm | STREAM_RTF | TTFA ms | prebuffer ms | safe ms | stall rates @100/250/500/1000 | max gap ms | coalesced |
|---|---:|---:|---:|---:|---|---:|---:|
| q8 | 0.868/0.901 | 254/549 | 391/726 | 702/1023 | 1/0.79/0.08/0 | 614/964 | 4.6% |
| q32 | 0.897/1.008 | 199/513 | 2129/2544 | 2451/2844 | 1/1/0.96/0.87 | 2305/2469 | 33.0% |

### CT-2 — decoder intercept and slope

`[SDPHASE]` runs were separate diagnostics. OLS fits use decoder total duration versus
frames; ranges are across the two worker processes, with R² around 0.90–0.98.

| quantum | total fixed intercept | total slope | conv-up intercept | conv-up slope |
|---|---:|---:|---:|---:|
| q8 | 23–28 ms | 9.3–9.6 ms/frame | 17–22 ms | 8.4–8.7 ms/frame |
| q32 | 8–13 ms | 12.5–13.3 ms/frame | 1–5 ms | 11.9–12.7 ms/frame |

The fits are sensitive to worker/shape mix and trace overhead, but neither quantum
simultaneously satisfies the demotion rule `intercept <10 ms` and `slope <8 ms/frame`.
The fixed decoder cost is real enough to preserve SQ-1 as the next implementation
candidate; it is not evidence that a new scheduler alone will solve cadence.

### CT-3 — admission interference

Three established long requests started together. One new long request was sent at
about +5 s. The server trace recorded that request admitted at +5.026 s, inline prefill
completed at +5.335 s (308.5 ms), and its first audio arrived at +5.380 s (client TTFA
380 ms). Its required prebuffer was 466 ms and its max gap was 1061 ms.

For established-stream chunk gaps around the event, the matched no-admission control
had p95 gaps of 548/548/541 ms in windows 4–5 s, 5–5.4 s, and 5.4–7 s. The admission
run had 783/783/783 ms. One 783 ms gap began before admission and overlapped the
prefill; the other active streams remained near 506–520 ms in the event window. Thus
inline prefill consumes roughly 309 ms of safety budget and can coincide with a roughly
235 ms larger gap, but this one pair is not sufficient to claim a fully causal
per-stream stall model. LS-4 remains an early P3 follow-up, not a P1 implementation.

Both probes completed without errors. The admission probe was intentionally diagnostic
and used the same client monotonic clock domain as `[TTFA2]`/`[ITER]` for correlation.

### CT-4 — Talker B1 versus B2

Separate 1x6 diagnostic processes with five synchronized waves and serve profile:

| effective work | frames | slot-frames | mean active slots | Talker total | per frame | per slot-frame |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 621 | 621 | 1.00 | 13,406 ms | 21.6 ms | 21.6 ms |
| B2 mix | 959 | 1264 | 1.32 | 22,769 ms | 23.7 ms | 18.0 ms |

B2 occupied 31.8% of iterations (B1 68.2%). The B2/B1 step-wall ratio is 1.10,
well below the `>1.6` kill criterion; normalized per active slot B2 is 0.83x B1.
EO-2 batching is therefore retained behind cadence/slice work. This is a diagnostic
amortization result, not a claim that the current 2x6 production scheduler already
forms one global Talker engine.

### CT-5 — current playback envelope

Clean q8/threshold2 WAVE, four waves per concurrency, no intrusive diagnostics:

| C | STREAM_RTF | TOTAL_RTF | TTFA ms | prebuffer ms | safe ms | stall rates @100/250/500/1000 | max gap ms | receive fidelity | status |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 2 | 0.600/0.623 | 0.613/0.638 | 160/366 | 33/47 | 189/398 | 0/0/0/0 | 394/421 | 5.1% | GOOD |
| 3 | 0.664/0.786 | 0.685/0.867 | 444/547 | 244/370 | 739/814 | 0.67/0.17/0/0 | 492/532 | 5.1% | GOOD |
| 4 | 0.793/0.856 | 0.845/1.066 | 429/544 | 397/596 | 816/978 | 0.94/0.75/0.25/0 | 541/586 | 5.1% | MARGINAL |

All 36 requests completed with zero errors, rejects and timeouts. C4's STREAM_RTF p95
is below one but its required-prebuffer p95 is 596 ms and 25% of requests stall with a
500 ms fixed buffer. It is not qualified. C2 and C3 satisfy the provisional playback
envelope on this short WAVE; a full SOAK is still required before promotion.

## Conclusion

1. CT-1 confirms the cadence law. q8 versus q32 changes C4 required-prebuffer p95 from
   337 ms to 2173 ms in the WAVE and 726 ms to 2544 ms in the short SOAK. STREAM_RTF
   changes less decisively and q32 does not produce a serving win. q32 is rejected as a
   production streaming quantum. The exact q32 cadence numbers are caveated by high
   receive coalescence.
2. `GANG_MIN=64` does not materially improve q8 cadence: C4 STREAM_RTF p95 changes
   0.826→0.813 while TTFA p95 worsens 547→610 ms and max-gap p95 worsens 554→594 ms.
   “Gang joining alone causes the cadence loss” is not supported.
3. CT-2 fails the SQ-1 demotion rule. The next architectural implementation is the
   small-quantum/strip decoder, with fixed per-call work removed or amortized and its
   parity/cadence measured again.
4. CT-3 shows a material inline-prefill event but not a clean causal attribution for
   every established-stream gap. Keep LS-4 early in P3; do not redesign admission during
   P1.
5. CT-4 does not kill EO-2. B2 amortizes the Talker step, but EO-2 remains downstream
   of the decoder cadence decision.

Next action · Mark P1 complete in `PLAN.md`; implement only P2 SQ-1 in the next runtime
cycle. Re-run the same playback-aware C4 screen after a bounded strip slice. Do not run
C5/C6, reopen BF16/W4, or call C4 qualified from this evidence.

Verdict · KEEP current q8/threshold2 as the short-wave control; REJECT q32 as a
production cadence policy; GO for SQ-1; RETAIN EO-2; KEEP LS-4 as measured P3 follow-up.
