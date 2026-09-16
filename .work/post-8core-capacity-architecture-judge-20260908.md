# Post-8-core capacity and architecture judge — 2026-09-08

Task · Second-opinion, READ-ONLY review of what the 2026-09-07/08 campaign and the
completed GCP `c4-highcpu-16` (8 physical AMX cores) qualification actually proved,
and what the next 16-physical-core AMD/Turin VNNI slot must measure.

Question · Is the 8-core result (1.7B → C2 full-envelope, 0.6B → C3) a physical
capacity wall, an engine/orchestration wall, or a mixture, and what does that imply
for the next host, the economic test and the architecture backlog?

Known facts · Repository HEAD `1a83327` (`feature/x86-amx-vnni-oss`), clean except
this file and the untracked PF-2 note. All numbers below are quoted from the tracked
`.work/*.md` addenda, `docs/*.md` references and the current source; nothing was run,
built or launched for this review. Frame = 1920 samples at 24 kHz = 80 ms; q4 = four
frames (320 ms of audio) per decoder call.

Unknowns · Whole-request AMX wall share; per-slot per-iteration stage timings in
steady state; DRAM traffic on the 8-core host during gaps; hourly prices for the GCP
hosts; every Turin number in this note is a prediction until measured.

Files/functions inspected · `PLAN.md`, `ENGINEERING.md`, `AGENTS.md`;
`qwen_tts.c` (`qwen_tts_serve_continuous`, `ADMIT_PREFILL`/`ADMIT_INSTALL`,
`SAMPLE_SLOT`, `RECORD_FRAME_AND_EMBED`, gang decode, `[ITER]`/`[TTFA2]`),
`qwen_tts_server.c` (`sink_next_job`, `sink_on_chunk`, `send_pcm_chunk`, prefork
parent, `sink_step_allowed`, `batch_job_t`), `qwen_tts_talker.c`
(`qwen_batch_talker_step_ragged`, `tk_region_run`), `qwen_tts_code_predictor.c`
(`qwen_batch_cp_predict`, `cp_frame_region_run`), `qwen_tts_speech_decoder.c`
(`sd_stream_batch_body`, `sd_batch_fallback`, `rag_conv1d_amx`),
`qwen_tts_kernels.c` (`g_mm_gate`), `qwen_tts_thread.c` (`qwen_parallel`),
`tools/doctor.py`, `tests/playback_sim.py`, `tests/serve_parallel_wave.py`,
`tests/serve_soak.py`, `tests/ls4_admission_probe.py`, `configs/perf/*.json`, and
the addenda: QL-1/QL-2/product-capacity, PF-2, SL-1/AR-2, decoder quantum floor,
lead gate, prefill helper, output isolation, same-pool consumer, fused residual,
F1/F2/F-cap3/F3, LS-4, prefork admission bound, MT-4, P1 cadence truth, P2
checkpoint, AR-1/AR-1b/Codex audit, post-P4 synthesis, cross-ISA audits,
`docs/reference-aws-c8a-16c-vnni.md`, `docs/runtime-map-c8a-c4.md`,
`docs/reference-gcp-c4-standard-24.md`, `docs/reference-gcp-c3d-8c-vnni.md`.

Evidence · Sections 2–13. Conclusion · Section 1 and Section 15. Next action ·
Section 16 (Codex handoff).

---

# 1. Executive verdict

1. **The 8-core box is limited by a per-stream marginal cost that does not depend on
   model size.** On `1x8`, each admitted stream adds ≈12 ms to every ~80 ms iteration
   (1.7B short bank: C1/C2/C3/C4 iteration wall ≈ 37/49/60/74 ms, derived from
   STREAM_RTF p50 .466/.617/.751/.925). Shrinking the Talker 1.7B → 0.6B saves only
   ≈7–9 ms per iteration (C3 iteration 60 → 53 ms), which is less than one stream's
   marginal cost. That is why full-envelope capacity moved only C2 → C3.
2. **The marginal cost is dominated by the speech decoder and per-slot step work, not
   by the Talker weight pass.** Measured on the 12-core host at 6 threads: Talker+CP
   iteration p50 is flat at ≈40 ms for B=1..3, while the q4 decoder call grows
   59 → 89 → 117 ms for B=1/2/3 (+29 ms per slot per call). The decoder is identical
   for both models. This is the Amdahl floor.
3. **The wall is MIXED, with the engine shape dominant.** Physically the host is at
   its DRAM roof only during the Talker weight pass (~15–20 ms of each iteration) and
   nowhere near AMX or VNNI compute peak in the decoder (~2.5 GMAC per frame per
   stream in ~7 ms on 6 cores ≈ a few percent of AMX peak). The cadence failure comes
   from executing the per-slot cost as a serial, uninterruptible decoder burst on one
   scheduler thread, which stalls every other stream's Talker/CP and leaves ~1.3–2
   cores idle exactly while the deadline is being missed.
4. **Average core-equivalents (6.7/8) are not spare capacity, but they are not
   proof of saturation either.** The iteration has phases at the DRAM roof (Talker),
   phases that are barrier/latency bound (CP: ~700 barriers per frame), phases that are
   glue bound (decoder) and serial windows where seven workers are parked (sampling,
   embedding, VQ, PCM write, gather/scatter). The census in Section 6 is required to
   attribute gaps to these phases; nothing in the current KPI runs can.
5. **At the 1.7B product point (C2) the "8 AMX cores" use AMX only in the decoder
   and the BF16 prefill.** `g_mm_gate` sets INT8 AMX `min_b=3`; at B≤2 Talker and CP
   run VNNI. AMX Talker/CP work exists only at C3+. The whole-request AMX wall share
   remains UNKNOWN. A 16-core VNNI host is therefore mostly being compared on core
   count, per-core bandwidth and topology, not on tile throughput.
6. **SL-1 removed the long-prefill confounder; the remaining C3 (1.7B) and C4 (0.6B)
   failures are steady-state tail margin.** After SL-1 the C3 SOAK windows were
   1.024/.909/.913/1.006/.874: the steady mean is close, the tail is not. That tail is
   the decoder burst plus admission of new streams into an already loaded loop.
7. **Two workers beat one whenever a worker can still reach the memory roof.** The
   12-core and 16-core hosts chose 2 workers; the 8-core host chose `1x8` only because
   a 4-thread Emerald Rapids worker pulls ~55 GB/s of the 110 GB/s roof (≈14.5 GB/s per
   thread). On Zen 5 a single core pulls ~44 GB/s and 4 threads reach ~100 GB/s. The
   topology answer is host-specific and must be measured, not copied.
8. **The general pattern is real:** every change that removed work from the serial
   critical path won (ragged parallel decode, threshold 2, fused residual, SL-1,
   allocator hygiene); every change that added a second submitter, a gate or a parking
   policy on the same pool lost (same-pool consumer, hard lead gate, helper+LOW, cap3,
   LS-4); explicit overload boundaries (fail-fast) helped by making waits visible.
9. **Expected Turin result, before measuring (PREDICTED):** 1.7B `2x8` lands at C3
   full-envelope with C4 as a screen point, roughly where the 12-core AMX host sits;
   0.6B `2x8` is the first configuration with a plausible C5–C6, and a 0.6B `4x4`
   arm is worth one screen if the synthetic chain test in Section 10 permits it.
   Neither prediction is evidence.
10. **The economic test is decided by price, not by a heroic C8.** With the only
    grounded prices in the repo (AWS list, 2026-09-03: `c8a.4xlarge` $0.862/h,
    `c8i.4xlarge` $0.7497/h), a Turin 1.7B C3 costs $0.287 per GOOD stream-hour and
    C4 costs $0.216. The GCP 8-core price is UNKNOWN in the repository and must be
    attached before any cost claim.

# 2. What the 8-core experiment actually proved

**PROVEN (measured, repeated, clean binary)**

- `1x8` beats `2x4` on this host at every C, and C4 fails the realtime margin on both
  (STREAM p95 .974/.983) even though fixed-buffer stalls at 250/500 ms are zero.
- 1.7B: C2 passes the full envelope including a corrected five-minute SOAK (worst
  window STREAM p95 .816, TTFA p95 155 ms, stall@250/500 = 0); C3 passes isolated
  short/medium/long/mixed screens but drifts across SOAK windows with pooled p95 just
  over one.
- 0.6B on the same host and serving generation: C3 passes the full envelope (SOAK
  worst window STREAM p95 .861, safe-start p95 471 ms); C4 tails reach ≈.943 on
  medium/long; C5 is not realtime (short STREAM p95 1.077, one third of requests stall
  at 250 ms).
- SL-1 causally removed the length-dependent startup term for known text: C1 long
  TTFA p95 329 → 65 ms, C3 long 855 → 158 ms, accepted 2+1 injection max-gap
  513–538 → 245–289 ms, no short/medium regression.
- Fail-fast admission converts hidden multi-second listener backlog into immediate
  503 (F2: >97% of the C5 tail was before `accept()`).
- Doctor rho (predicted .75 for 1x8 C4) is not a capacity number: measured .974.

**STRONGLY SUGGESTED (consistent across hosts, not isolated on this host)**

- The per-slot marginal cost is model-size invariant and decoder-dominated
  (Section 3). The 8-core marginal of ≈12 ms per slot per iteration is derived from
  STREAM p50 values, not from a per-phase timer on this host.
- The cadence tail at q4 is `gap ≈ 4·T_step(B) + D(B)` where `D(B)` is the ragged
  decoder call; at C4 1x8 max-gap p95 was 330 ms against 320 ms of audio per call.
- The 0.6B superlinear knee at C5 (iteration 86 ms vs 75 ms at C4) indicates a second
  term (attention over longer KV on the loop thread, cache pressure, or phase
  alignment of decoder bursts) that the linear model does not contain.

**UNKNOWN**

- Whole-request AMX wall share (all SQ-3 quantities except the eligible-panel subset).
- Per-slot per-iteration phase timings in steady state; pool wait per phase; DRAM
  traffic during gap events on this host (no counters were sampled).
- 0.6B C1/C2 iteration walls on this host (not reported).
- Whether the C3 SOAK drift is admission-phase coupling or thermal/noisy-neighbor
  variance: the five windows were not correlated with admission events.
- Hourly price of `c4-highcpu-16`; the economic denominator.

**DISPROVEN**

- "AMX makes the 8-core box a C4 host for 1.7B." No.
- "Average utilization below 8 cores means a third or fourth stream is affordable."
  LS-4 admitted at 37 ms iteration intervals and still broke the established four.
- "The C3 limit is the long-prefill startup term." After SL-1 the long class has
  TTFA p95 158 ms at C3 and the SOAK still crosses one.
- "A wider single loop is better than two workers on larger hosts." 1x12 (12-core)
  and 1x16 (c8a) both lost to 2 workers at C≥3.

# 3. 1.7B vs 0.6B natural experiment

Per-iteration wall derived from STREAM_RTF p50 × 80 ms (DERIVED; short bank unless
stated):

| host / model | C1 | C2 | C3 | C4 | C5 | marginal per slot |
|---|---:|---:|---:|---:|---:|---:|
| 8-core `1x8`, 1.7B | 37 | 49 | 60 | 74 | — | ≈12 ms |
| 8-core `1x8`, 0.6B | UNKNOWN | UNKNOWN | 53 | ≈75 (medium/long) | 86 | ≈12–17 ms |
| 12-core `2x6`, 1.7B, per worker | ≈40 (B1) | — | — | ≈66 (B2, q4 fused) | — | — |

Measured decomposition on the 12-core host at 6 threads (F-cap3 diagnostic, `[ITER]`
and `[DECODE]`, 1.7B, q4):

| quantity | B=1 | B=2 | B=3 | source |
|---|---:|---:|---:|---|
| Talker+CP iteration p50 (no decoder in the interval) | 38.7 | 40.8 | 40.2 | `.work/f-cap3-c5-capacity-20260908.md` §6 |
| iteration p95 (includes decoder-call iterations) | 98 | 147 | 198 | same |
| decoder call, q4, ragged | 59 | 89 | 117 | same |
| decoder intercept + slope at q8 | 23–28 ms + 9.3–9.6 ms/frame | | | `.work/p1-cadence-truth-20260907.md` CT-2 |

Reading: the batched Talker+CP step is nearly flat in B (weights are read once for
all slots; the CP transformer and both heads are batched since `110ff48`/`b152946`),
while the decoder call grows ≈29 ms per slot per call, i.e. ≈7.3 ms per slot per
frame at q4. The remaining ≈4–6 ms per slot of the 8-core marginal is per-slot work
that is not batched: sampling, `RECORD_FRAME_AND_EMBED`, per-slot attention in the
non-region Talker path, VQ scalar loops, gather/scatter and the PCM write, all on
the scheduler thread with the pool parked (`qwen_tts.c` lines cited in Section 5).

Model-size arithmetic (1.7B → 0.6B): the only model-dependent term in the iteration
is the Talker weight pass, ≈1.42 GB INT8 for 1.7B versus ≈0.44 GB for 0.6B. At the
measured 110 GB/s roof that is ≈13 ms versus ≈4 ms, so ≈9 ms recovered per
iteration; measured C3 recovery is 7 ms. One slot costs ≈12 ms. Hence 0.6B buys
≈0.6–0.75 of a slot, and the envelope moves one point (C2 → C3) only because C3 for
1.7B was already close.

The implication is direct: **the CP and the decoder form a model-invariant floor**.
CP re-reads ≈1.3 GB of INT8 weights per frame (L3-resident on the 260 MiB Emerald
Rapids L3, ~16 sequential codebook stages, ~700 barriers per frame) and the decoder
processes ≈2.5 GMAC per frame per stream through im2col, quantization, snake and
~110 small SGEMM/conv calls. Neither term shrinks with the Talker. A 0.6B server on
this architecture is a "cheaper Talker" server, not a "half-cost" server.

Where the additive model breaks: 0.6B C5 (86 ms) is ≈11 ms above the linear
extrapolation. Candidate causes, all UNMEASURED: per-slot attention over long KV on
the loop thread (medium/long classes), L2/L3 pressure of five decoder states plus
CP, and bursts from slots that reach their decoder boundary in different iterations
(each producing a separate `D(1)` ≈ 59 ms call rather than one ragged call).

# 4. Physical wall vs engine wall

**Evidence for PHYSICAL**

- The Talker weight pass runs at the DRAM roof while it runs: 1.42 GB in ≈13–15 ms
  on the 8-core host is ≈100 GB/s against a measured 111 GB/s read roof. On the c8a
  it was measured at 89% of the per-CCX read roof. No number of extra cores on the
  same DRAM channels shortens this phase.
- A 4-thread Emerald Rapids worker cannot reach the roof (≈55 GB/s at 4 threads), so
  `2x4` loses C3 outright (STREAM p95 1.004). This is a hardware property of the
  per-core memory parallelism of this CPU family.
- CP is L3-bandwidth bound per frame (≈1.3 GB per frame per worker), so a second
  worker on the same L3 doubles that traffic.

**Evidence for ENGINE**

- One scheduler thread executes, in order and without yielding: admit(prefill) →
  head → sample → CP → embed → decode(+PCM write) → Talker (`qwen_tts.c:3167–3616`).
  The decoder call is a complete, uninterruptible call whose duration grows +29 ms per
  slot; during it, no stream advances a Talker/CP frame.
- Idle capacity exists during the bursts: on c8a the per-worker pool was asleep ≈25%
  of wall and 2.0 of 8 cores were idle at C4; on the 8-core host 6.7 core-equivalents
  were busy at C4 while STREAM p95 was .974. Those idle cores were not exploitable by
  any tested mechanism that shared the pool, which is an ownership property, not a
  silicon property.
- Cadence fails before the mean saturates: C4 1x8 STREAM p50 .925 with max-gap p95
  330 ms ≥ 320 ms per q4 chunk. The p95 of the iteration interval on the 12-core host
  is 2.5–5× its p50 (98/147/198 vs 39/41/40) purely because of the decoder burst.
- The decoder runs far below any compute roof (≈2.5 GMAC per frame per stream in
  ≈7 ms on 6 cores; the P2/AR-1 audits describe it as glue-bound: im2col, quant,
  barriers, snake, small SGEMM packing).
- Two independent loops beat one wide loop on 12 and 16 cores (2x6 > 1x12, 2x8 > 1x16
  at C≥3) because the second loop's bursts overlap the first loop's steps. That is
  stage overlap by process duplication, at the price of reading the weights twice.

**Evidence for MIXED**

- Both sets are true simultaneously in different phases of the same iteration; the
  weight pass is at roof ≈20% of the time and the decoder burst is serial ≈25–40% of
  the time at C3–C4.

**Ranking (ordinal, not probabilities):** MIXED > ENGINE > PHYSICAL.

The defensible sentence is: *the per-stream cost is physically real work, but its
serial execution on one loop turns moderate average utilization into deadline misses
while cores are idle; more cores help only through more independent chains or through
stage overlap, not through a wider pool.* The next host answers this directly: if
`2x8` on Turin gives ≈2× the 8-core `1x8` capacity per model (C4 for 1.7B, C6 for
0.6B), the wall was chain-shaped; if it gives ≈1.3×, the shared roof dominates.

# 5. Current critical path

One `1x8` worker, default product flags (q4, ragged decoder, synchronous output).
Note that with `--prefork 1` the server takes `qwen_tts_serve_batched` directly
(`main.c:2916–2923`): there is no parent process, and admission cap is `jq_push`
(`qwen_tts_server.c:1662–1677`, running + queued ≥ slots + `--max-queue`), so the F2
parent-backlog term and the LS-4 parent page do not exist on the 8-core reference.
They return on any `2xN` topology.

```
client ──HTTP──▶ srv-read-N  parse, jq_push (cap = slots+max_queue → 503 fail-fast)
                    │  [queue]                                              (BLOCKING for new request only)
srv-sched (one thread per worker) — one iteration:
  ┌ ADMIT_PREFILL  qwen_tts_generate(prefill_only)  ← 28 layers, bf16/AMX matmat
  │                 all established slots STALLED; ~73 ms flat with SL-1     [COMPUTE+BLOCKING, deadline-sensitive]
  │ ADMIT_INSTALL  memcpy KV into slot                                       [serial]
  ├ codec head GEMM (all slots, 1 dispatch)                                  [COMPUTE, batch ✓]
  ├ SAMPLE_SLOT ×B  (serial, pool parked)                                    [serial, ownership: loop thread]
  ├ qwen_batch_cp_predict (1 dispatch, 16 stages × 5 layers, ~700 barriers)  [COMPUTE, batch ✓, RENDEZVOUS-heavy]
  │   B=1 → solo path ≈330 matvec dispatches; B≥3 → AMX int8, B=2 → VNNI
  ├ RECORD_FRAME_AND_EMBED ×B (serial)                                       [serial]
  ├ gang decode: leader pending≥4, joiners pending≥2
  │   qwen_speech_decoder_decode_streaming_batch  D(B) ≈ 30 + 29·B ms (6T)   [COMPUTE, batch ✓, BLOCKING all slots,
  │   nit==1 → per-slot fallback; int8 non-AMX → per-item fallback             DEADLINE-SENSITIVE: this is the gap]
  │   └ per item sink→on_chunk → send_pcm_chunk: 3 blocking write()          [BLOCKING on slow client, serial]
  └ qwen_batch_talker_step_ragged (1 dispatch, ~225 barriers; B=1 solo ≈112) [COMPUTE at DRAM roof, batch ✓]
     pos++, next iteration
```

Marks: **compute** = head, CP, decoder, Talker, prefill. **blocking** = prefill
(stalls slots), decoder (stalls slots), PCM write (stalls loop on slow client),
`next_job` only when `n_active==0`. **rendezvous** = one `qwen_parallel` per batched
call plus ~925 in-region barriers per frame (Talker ~225 + CP ~700), ~41–102
call-site rendezvous per decoder call (exact count UNKNOWN). **ownership** = one
process, one loop thread, one pool; per-slot state (KV, CP state, decoder stream
state, trailing text) lives in the slot; nothing migrates. **batch opportunities** =
head, CP, Talker (all slots every iteration), decoder (slots at the same boundary;
ragged). **deadline-sensitive** = the decoder burst and the admission prefill, because
they are the only intervals longer than one frame period during which no stream
produces audio.

# 6. Stage-pressure census

Goal: for each client-observed gap ≥ 250 ms, name the server phase(s) that overlapped
it and whether pool workers were runnable-but-parked, the DRAM traffic was near roof,
or the phase was a serial loop-thread window. This separates ENGINE from PHYSICAL
with one diagnostic generation.

What exists today (Section 9 of the tooling audit): `[ITER]` has a monotonic stamp,
`n_active` and cumulative `pf_*` buckets per worker but no per-iteration phase
durations and no slot id; `[DECODE]` gives per-call duration with a stamp;
`[SDPHASE]` gives per-call phase ms without a stamp; `[TTFA2]` fires once per slot;
the cost map gives inclusive totals per thread; the client wave already stamps every
receive mark in a clock domain that the F2 header aligns with the server.

Minimum additions (diagnostic-only, default-off, for Codex):

1. `[ITER] v=3`: one line per iteration with monotonic start, `n_active`, and the
   duration of this iteration's admit / head / sample / CP / embed / decode / write /
   Talker phases (8 `clock_gettime` calls per iteration ≈ negligible), plus decode
   group size, frames per item, and whether the decode was ragged or per-item fallback.
2. Per slot per iteration, appended to the same line: `frames_emitted`, `decpos`,
   `pending`, and `lead_ms` computed exactly as `sink_step_allowed` computes it today.
3. Per phase: number of `qwen_parallel` dispatches and the caller's wait-at-completion
   time (both already counted by the cost map regions `RT_POOL_DISPATCH` /
   `RT_POOL_WAIT`; the change is to bucket them per phase per iteration instead of
   per process).
4. External, non-intrusive, Linux only: `perf stat -e cache-misses,cycles -I 100 -p
   <worker pids>` during the diagnostic arm, as done on c8a (M4). This is the only
   DRAM-pressure signal available without new code; label it DIAGNOSTIC.
5. Offline join script: for every client gap ≥ 250 ms, list the `[ITER] v=3` records
   overlapping `[t_gap_start, t_gap_end]`, sum phase durations, and report the
   dominant phase, the pool wait share and the `perf` interval bandwidth. Output one
   table per (model, C): gap count, dominant phase histogram, median pool-idle share
   during gaps, median DRAM GB/s during gaps.

Run matrix (three waves each, short and mixed bank, diagnostic arm labelled
NON-QUALIFYING): 1.7B C1/C2/C3 and 0.6B C1/C2/C3/C4 on the same 8-core host, plus one
two-minute closed loop at 1.7B C3 to catch the drifting windows.

Decision rule: gaps dominated by the decoder or serial phases with pool idle share
> 30% and DRAM < 60% of roof ⇒ ENGINE; gaps dominated by Talker/CP phases at > 85%
of roof with pool idle < 10% ⇒ PHYSICAL; anything else ⇒ MIXED with the measured
split. This can be built from current counters plus item 1; nothing requires a
scheduler change.

# 7. Audio-deadline scheduling review

Priority-not-gating semantics: lead orders and sizes work; it never parks a stream.
The LS-1 gate parked 95.8% of checks, cut cores 7.3 → 5.2 and worsened STREAM p95;
that outcome is expected from any rule that withholds ready work while the loop is
otherwise idle.

Existing state that already approximates the signals: `audio_ready_samples` and
`first_audio_ready_us` per job (`qwen_tts_server.c:1895–1902`, used by
`sink_step_allowed`), `chframes`/`decpos`/`pending`/`target` per slot in the loop,
the gang leader/joiner rule (`g_gang_lead=4`, `g_gang_min=2`), the per-slot decoder
target ladder (1 → 2 → 4 → q), the step mask machinery (three opt-in masks), and the
`[DECODE]` duration history.

Missing state: `audio_lead_ms` and a `next_deadline` field on the slot (today the
lead is recomputed only inside the opt-in gate and is not visible to the gang decision
or the target ladder); an estimate of `D(B)` and `T_step(B)` from the last calls; a
socket backlog signal (only `write_complete` at the first write exists).

Safe scheduling boundaries in the current architecture: only the iteration boundary
and the choices inside it. The four decisions the loop already makes are: which slots
step (mask), which slots join the decoder gang and with how many frames, each slot's
decoder target, and whether to admit now (`next_job` non-blocking when
`n_active>0`). There is no intra-call preemption anywhere: Talker, CP, decoder and
prefill are complete calls; q1/q2/q4 are smaller complete calls (AR-2, Codex audit).

A priority realization without rewrite would therefore be: (a) make the slot with the
smallest lead the gang leader and give it its full target while joiners contribute
what they have; (b) size each slot's decoder target from its lead (q2 when lead <
D(B)+2·T_step, q4 otherwise, q8 only above a generous lead) — this is LS-2 in the
plan; (c) defer `ADMIT_PREFILL` by one iteration when any established slot's lead is
below `D(B) + T_step + prefill_estimate` (this gates a not-yet-playing request, which
is admissible, and bounds TTFA by one iteration); (d) never mask a stream from
stepping. Each is an iteration-boundary change that touches no kernel and no state
ownership.

What it cannot do: it cannot shrink `D(B)`, so it redistributes gap risk rather than
creating capacity. Expected effect: better max-gap p95 and stall@250 at the first
failing point, STREAM p50 unchanged. Falsifier: at 1.7B C3 and 0.6B C4, max-gap p95
and stall@250 improve while STREAM p95 stays within ±0.02 and core-equivalents stay
within ±0.3; if STREAM p95 regresses or cores drop, the realization is parking work.

Talker/CP progression cannot expose finer boundaries than a frame without changing
arithmetic order (the region runs 28 layers under one dispatch); the frame boundary is
the right quantum. q4 decoder calls are a suitable quantum only because the intercept
(23–28 ms) is paid once per call; q2 pays it twice as often and q1 was rejected.

# 8. The "2 cores per stream" idea

Static pinning (cores 0–1 for A, 2–3 for B) is wrong on this workload: a 2-thread
chain cannot execute the Talker weight pass in a frame period on any measured host
(Emerald Rapids: 2 threads ≈ 28 GB/s → 50 ms for 1.42 GB; Zen 5: 2 threads ≈ 85
GB/s → 17 ms, plus CP and decoder). The correct reading of "16 cores / 2 per stream
≈ 8 streams" is a capacity budget: ≈150 core-ms per slot-frame on the 8-core AMX host
(74 ms × 8 cores / 4 slots), ≈300 core-ms per slot-frame on the old c8a stack (76 ms ×
8 cores / 2 slots per worker). At the AMX efficiency, 16 cores × 80 ms / 150 ≈ 8.5
slot-frames per frame period; at the old c8a efficiency ≈ 4.3. So the budget exists
only if Turin approaches the AMX-host core efficiency, which the per-item VNNI decoder
fallback and the missing fused residual make unlikely for 1.7B.

The dynamic version the question describes (request state independent of core
ownership, narrow high-priority CP, opportunistically wide Talker, bounded decoder
progression, shared pool, per-stream credits) maps onto the current engine as
follows:

- Request state is already core-independent inside a worker (slots, not threads);
  across workers it is process-owned and cannot move (prefork).
- "Opportunistically wide Talker" exists: the batched region uses all slots and all
  pool threads.
- "Narrow CP" does not exist and would be a regression: CP is batched across slots
  and barrier-bound; narrowing it lengthens the 16-stage chain.
- "Bounded decoder progression on a sub-team while the loop continues" does not
  exist. The two tested overlaps failed for construction reasons, not for the
  mechanism: the private team oversubscribed (21 threads on 8 cores), and the
  same-pool consumer serialized on `submit_mtx` with `group=1` and forced
  `dec_batch=0`. A partitioned pool (k step threads + (N−k) decoder threads, decoder
  call issued at the chunk boundary and collected at the next boundary, no
  oversubscription) has not been tested. It is the one ownership change that attacks
  the dominant term (`D(B)` leaves the serial chain: iteration → max(4·T_step, D)
  instead of 4·T_step + D).
- Its cost is the Talker bandwidth with fewer step threads: on Emerald Rapids 5
  threads pull ≈70 GB/s (Talker pass 13 → 20 ms), on Zen 5 4 threads already pull
  ≈100 GB/s. The partition is therefore plausible on Turin and doubtful on the 8-core
  Intel host, which is exactly why the next host should measure the two curves
  `T_step(B, k)` and `D(B, N−k)` before any implementation (Section 14, item 1).

Is prefork fundamentally incompatible? No: prefork is orthogonal (it duplicates whole
chains); the partition lives inside one worker. What is incompatible is a second
submitter on the same single-slot pool, which is what both failed experiments were.

# 9. 16-core thought experiment

Assumptions to verify on the box: 16 physical Zen 5 cores, SMT absent or off, two
8-core CCDs each with its own 32 MiB L3 (AWS `c8a.4xlarge` measured 2 × 32 MiB),
one NUMA node, Triad ≈ 103–106 GB/s host with the knee at 4 threads and ≈44 GB/s per
core. A GCP `c4d` shape may differ (SMT on by default, CCD layout); the fingerprint
decides.

**1x16.** One chain, B up to C in one call, 16-thread barriers (~925 per frame),
one decoder burst `D(B)` for all slots, weights read once. Measured on c8a (old
stack): C1 .523 (best single-stream point), C4 1.10. Prediction: same shape with the
current stack; single-stream latency winner, not a capacity topology. Do not SOAK it.

**2x8.** Two chains, one per CCD, each with a private L3 that does not hold the
1.7B CP working set (≈112 MB per the doctor's config-derived figure; the c8a note's
"~60 MB for 0.6B" is a stale box_info estimate and the two figures need reconciling
from `--caps`), weights read twice per frame period (≈2 × 1.42 GB / 80 ms ≈ 36
GB/s of the ≈105 GB/s roof for the Talker alone). Measured old stack: C1 .710,
C2 .721, C4 .952 (max .999). Chain model per worker (PREDICTED): B1 ≈ 55 ms, marginal
≈ 15–19 ms → each worker sustains B2 at ≈ .9 → host C4 at the edge, C3 full-envelope.
SL-1 and q4/fail-fast improve startup and cadence but not the mean; the VNNI per-item
decoder fallback removes ragged batching, so `D(B)` becomes `B × D(1)` instead of
`30 + 29·B`, which is worse per slot. Expected 1.7B outcome: C3 GOOD, C4 screen.

**2x8, 0.6B (PREDICTED).** B1 chain ≈ 37 ms (Talker pass ≈ 9 ms on one CCD), marginal
≈ 15 → each worker at B3 ≈ 67 ms → host C6 near .85, C5 comfortable. This is the
first economically interesting point the campaign can reach.

**4x4, 0.6B only (PREDICTED, gated).** Four chains of 4 threads; on Zen 5 each pulls
≈50–100 GB/s. B1 chain ≈ 45–60 ms → each worker at B2 ≈ 60–75 ms → host C6–C8. Test
only if the synthetic chain (Section 10) shows a 4-thread 0.6B B1 chain ≤ 50 ms;
otherwise it is the `4x4` c8a failure again (1.7B B1 at 4 threads was 94 ms).

**Request ownership and pool.** Prefork parent returns at 2xN (parent cap, `--max-
queue 0` fail-fast, F2 timeline, LS-4 page). Pool spin 4096 is the x86 reference,
not a Turin value; `QWEN_POOL_SPIN=65536` was neutral in wave and +11% p95 in soak
on c8a. Keep the reference value unless a bounded A/B on the new host says otherwise.

**Bandwidth and cache.** The per-CCD L3 decides whether the CP is L3- or
DRAM-resident per worker; if the 1.7B CP does not fit, every frame streams ≈1.3 GB
per worker from DRAM in addition to the Talker pass, and two workers demand ≈70
GB/s continuously. That is the single most likely PHYSICAL limiter on a 2-CCD Zen 5
host for 1.7B, and it does not apply to the Emerald Rapids host with its 260 MiB
L3. The 0.6B has the same CP, so this test is about L3, not model size.

**Expected scaling.** Relative to the 8-core AMX host: 1.7B ≈ 1.5× (C2 → C3, C4
screen), 0.6B ≈ 2× (C3 → C6). If the measured result is ≈1.3× for both, the shared
DRAM/L3 roof dominates and the engine-wall hypothesis loses weight; if 0.6B reaches
C6 and 1.7B stays at C3, the model-invariant per-slot term is confirmed as the
chain-shaped limiter.

# 10. Next AMD/Turin VNNI campaign (minimal)

Prerequisites before renting: (1) set `QWEN_TTS_STREAM_LAYOUT=1` in `vnni-product`
(it is pinned to 0 today while `amx-product` pins 1; SL-1 is common runtime code and
the long-prefill confounder must not be reintroduced into the comparison); (2)
confirm the doctor's CP working-set figure; (3) attach hourly prices for both hosts
to the campaign manifest.

Sequence with STOP/GO rules:

1. **Fingerprint (≈2 min).** `make cpu-check`, `make bench-fingerprint`, SMT off,
   physical core count, CCD/L3 domains, NUMA, governor/quota. GO only if 16 physical
   cores are online and the L3 domains are known.
2. **Roofs (≈3 min).** `make roofs` for masks host, CCD0, CCD1, and one 4-core
   quarter: read/copy/triad curves, knee. Record per-CCD read roof; it is the Talker
   term of the chain model.
3. **Build and dispatch (≈5 min).** Clean `SIMD=avx512vnni` (and the `avx512bf16`
   twin if the profile permits), `--caps`, `--self-test` native and fallback,
   `tools/serving_profile.py preflight` for `vnni-product` and `common-control`.
   STOP if any requested path resolves to a fallback not declared VALID.
4. **Synthetic chain (≈2 min, needs no model, see Section 11).** `--matmat-bench`
   B=1/2/4 at `-j` 4/8/16, `make roof-matvec` at 4/8/16 threads, and one CLI run per
   model with a short text at `-j 4/8/16` to read `[DECODE]` and `[ITER]` for B=1.
   Use these to fill the chain model per topology. GO to `4x4` only if the 0.6B
   4-thread iteration is ≤ 50 ms.
5. **Topology screen (WAVE, short bank, 3 waves).** 1.7B: `2x8` and `1x16` at
   C1/C2/C4. 0.6B: `2x8` at C1/C3/C4 and, if step 4 permitted, `4x4` at C4. STOP
   further work on any topology whose C4 STREAM p95 > 1.0. Pick one topology per model.
6. **Concurrency screen (WAVE, mixed bank, 3 waves).** 1.7B: C1/C2/C3/C4, then
   C5/C6 only if C4 is GOOD (p95 ≤ .90, prebuffer p95 ≤ 300 ms, stall@250 = 0).
   0.6B: C1/C3/C4, then C5/C6/C8 while the previous point remains GOOD. Add the
   `2+1`/`3+1` long-arrival probe at the candidate point.
7. **SOAK (5 min) only at the single best economic point per model** (Section 11
   decides which point). No SOAK on screen-only points.
8. **Diagnostic arm (optional, ≈15 min).** Section 6 census at the first failing
   point for each model, plus `perf stat -I 100` on the workers.

Total wall budget ≈ 3–4 hours including two SOAKs. STOP conditions: dispatch
contradiction, dirty tree, coalesced-read share above a few percent in a KPI arm,
or a topology whose C1 STREAM p50 exceeds .75 for 1.7B (the chain is too slow for
the exercise to matter).

# 11. Economic model

`cost_per_GOOD_stream_hour = hourly_cost / highest_full_envelope_GOOD`. Report also
`highest_screen_GOOD` and the accepted-rate transition from the Poisson probes as
context, never as the denominator.

Grounded prices in the repository (AWS on-demand list, `us-east-1`, captured
2026-09-03, `docs/reference-aws-c8a-16c-vnni.md` §5): `c8a.4xlarge` $0.862/h,
`c8i.4xlarge` $0.7497/h. GCP `c4-highcpu-16` and `c4d` prices: UNKNOWN; attach them
to the campaign manifest with a capture date.

| host | model | full-envelope GOOD | $/h | $/GOOD stream-hour |
|---|---|---:|---:|---:|
| 8-core AMX (`c4-highcpu-16`) | 1.7B | C2 | UNKNOWN (P8) | P8 / 2 |
| 8-core AMX | 0.6B | C3 | P8 | P8 / 3 |
| Turin 16c at c8a list price | 1.7B | C3 (predicted) | $0.862 | $0.287 |
| Turin 16c | 1.7B | C4 (if screen becomes full) | $0.862 | $0.216 |
| Turin 16c | 0.6B | C5 / C6 / C8 (predicted C6) | $0.862 | $0.172 / $0.144 / $0.108 |

Break-even against the unknown P8: Turin wins 1.7B at C3 if P8 > $0.575/h, at C4 if
P8 > $0.431/h; Turin wins 0.6B at C6 if P8 > $0.431/h, at C4 if P8 > $0.647/h. The
campaign therefore does not need C8; it needs a trustworthy C3/C4 (1.7B) and C5/C6
(0.6B) plus one price.

Data still missing for a CTO-grade statement: both hourly prices with dates; a
five-minute SOAK at the chosen point per model on Turin; the accepted-rate transition
at that point (short Poisson probes as on the 8-core host); and an explicit statement
of the quality lane (per-item VNNI decoder vs AMX Design-D is a product difference,
not a parity result, per QL-2a).

# 12. Fast box oracle and Doctor v2

**Why Doctor v1 ranked 1x8 > 2x4 correctly but missed C4.** Its frame cost is
`talker_bytes/gemv_gbs × b_scale + cp_bytes/l3_gbs × b_scale + (25·√(6/K) +
B·q·9.5·(6/K))/q`, with `b_scale = 1 + 0.10·(B−1)` (`tools/doctor.py:526–544`). The
Talker term is the only one that depends on the worker's measured bandwidth, and it
is large enough (13 ms at 8 threads vs ≈26 ms at 4 threads on this host) to order the
topologies correctly. Evaluated for 1x8, B=4, q=4 it gives ≈62 ms → rho .77; measured
was 74 ms → .925. The missing 12 ms and the missing tail are: (a) `b_scale` charges
1.1 ms per extra slot for Talker+CP where the measured per-slot step cost is ≈5–6 ms
(serial per-slot sections, per-slot heads, gather/scatter); (b) it uses a mean ρ,
while the client metric at q4 is `4·T_step + D(B)` per delivered chunk and the p95
catches the burst; (c) it has no term for admission prefill, barriers (~925/frame),
sampling, PCM writes or pool wait; (d) the decoder per-item slope is calibrated at 6
threads and scaled by 6/K as if it parallelized perfectly; (e) the non-AMX decoder
factor 1.5 is a labelled guess. The tool is honest about all of this (`doctor.py:966`
"cannot see admission/prefill coupling").

**What can be predicted in 10–60 s without the model** (Doctor v2, justified):

- Per-mask roofs and knee (already).
- `T_step(B, k)` and `D(B, k)` for B ∈ {1,2,3,4}, k ∈ {4, 8, 16}: run the real
  batched Talker region, the CP frame region and one ragged decoder call on random
  weights of the 1.7B and 0.6B shapes through the existing kernels and pool (the
  kernels are model-independent; `roof_matvec_int8.c` already allocates 28 distinct
  Talker matrices). This measures barriers, region scaling and the decoder glue on the
  actual host instead of scaling constants from another host.
- From those, the chain model per topology: `gap(B) = q·T_step(B,k) + D(B,k)`;
  predicted GOOD concurrency per worker is the largest B with `gap(B) ≤ 0.9 × q × 80
  ms`; host capacity = W × that B. Also whether the host is compute- or
  bandwidth-limited per stage (compare `T_step` against bytes/roof) and whether
  batching pays (`D(2)` vs `2·D(1)`, `T_step(2)` vs `2·T_step(1)`).
- A "obviously bad box" verdict: 1.7B B1 chain > 72 ms at the widest sensible worker,
  as on the c3d slice (C1 STREAM 1.05–1.12).

**What still requires the real model WAVE/SOAK:** admission coupling and TTFA
tails, phase alignment of decoder bursts across slots, KV-length effects on per-slot
attention, allocator and page-cache behaviour, noisy-neighbour and thermal drift
across SOAK windows, fail-fast/overload behaviour, and audio quality per ISA. The
oracle should output a concurrency range to screen, not a GOOD point.

# 13. Evidence matrix of the last two days

| experiment | hypothesis | mechanism | observed | why | local to AMX 8/12-core? | closed on 16c VNNI? | reopen condition |
|---|---|---|---|---|---|---|---|
| q1/q2/q4/q8 | smaller complete calls improve cadence | fewer frames per burst, intercept paid more often | q1 C4 p95 1.005; q2 .892/.914; q4 .862/.868 best Pareto; q8 best RTF, worst cadence | intercept 23–28 ms per call; burst size vs frequency | no (structural) | yes: q4 anchor | census shows intercept < 10 ms on the new host (then q2) |
| hard lead/credit gate (LS-1) | withholding steps above 250 ms lead frees cadence | parking | 95.8% checks parked, cores 7.3→5.2, STREAM p95 .838→.986 | parked work is lost; D(B) unchanged | no | yes as a gate | never as a gate; only as priority/sizing (Section 7) |
| prefill helper + LOW | off-thread one-shot prefill protects streams | second submitter, cloned ctx | TTFA p95 435→2379 ms, STREAM unchanged/worse | still one complete prefill; queueing hides it | no | yes | request-owned resumable prefill state exists (PF-1) |
| SL-1 known-text layout | prefill length-independent | official dual-track schedule | C1 long TTFA 329→65, C3 long 855→158, injection gap 513→245–289 | removes positions from prefill | no (class A) | keep ON; pin in VNNI/Arm profiles | ICL/clone quality gate still open |
| async output isolation | slow client must not block the loop | bounded queue + writer thread | KPI-neutral (C4 p95 .821→.830), isolation proven | inference was never blocked by normal readers | no | keep default-off, available | a slow-client incident, or the census attributes gaps to write() |
| same-pool decoder consumer | decoder overlaps Talker/CP | second submitter on one pool | C4 p95 .847→1.296, TTFA +952 ms, csw ×2.9, group=1 | `submit_mtx` serialization, no ragged batching | partly: pool design is common | yes for this realization | a partitioned pool with no oversubscription and preserved ragged batching (Section 14 item 1) |
| ragged decoder batching (threshold 2; scratch reuse; claim-first) | one call for co-ready slots | column concatenation | threshold 2 retained; scratch reuse rejected; claim-first neutral | shared weight pass, fewer calls | AMX-only implementation (Design-D); VNNI/Arm fall back per item | reopen as a portable question | if per-item VNNI `B·D(1)` exceeds `D(B)` by > 30% on Turin, a portable ragged path is the largest decoder lever |
| fused residual | remove one residual pass | Design-D epilogue | C4 p95 .831→.788, stall@250 17→0%, SOAK p95 .893 | critical-path work removed | AMX-only | keep in AMX lane; unavailable on VNNI | a VNNI equivalent only if the decoder is the census-dominant phase on Turin |
| fail-fast (`--max-queue 0`) | reject at parent instead of backlog | listener stays polled | 3.6–4.9 s hidden waits → immediate 503 | makes overload visible | Linux prefork (returns at 2xN) | keep | — |
| cap3 | third slot per worker adds capacity | wider B, more admission | C5 p95 .969, fifth proxy 1.028, stall@250 13% | +29 ms per slot per burst | no | yes | only after a census shows `D(3)` fits with margin |
| global/cross-worker batching (F3) | natural B≥3 cohorts exist | offline coincidence | 2.7% at ±1 ms, 11.7% at ±8 ms | chains desynchronize by design | yes (two workers) | yes | never without a deliberate wait policy; not before C6 is otherwise reached |
| LS-4 utilization-aware admission | recent 37 ms iterations mean headroom | transient third slot | fifth interactive; established p95 .985–1.004, stall@250 50% | iteration p50 is not the burst | no | yes as this predicate | a lead-based predicate on the established slots (Section 7c) |
| common-control / product profiles | fair cross-ISA lanes | strict resolved dispatch | operational; no measurement | — | — | required | pin SL-1 consistently |
| low-N M split / 1x12 cap4 | wider AMX worksets | more tiles / one process | rejected (p95 .971→1.106; 1.125/1.277) | more tasks, same serial chain | yes | yes | — |
| transport header truth (MT-4) | TTFB is a real event | header at admission, `TCP_NODELAY` | proven | — | no | keep | — |

# 14. Highest-value improvements IF the next host repeats the wall

Ranked; at most one of these should be started, and only after the Section 6 census
on the new host names the dominant phase.

1. **Partitioned pool: asynchronous decoder on a reserved sub-team inside a worker.**
   Mechanism: issue the ragged decoder call at the chunk boundary on `N−k` reserved
   threads; the loop continues head/sample/CP/Talker on `k` threads and collects PCM at
   the next boundary. Iteration becomes `max(4·T_step(B,k), D(B,N−k))` instead of the
   sum. Expected benefit: at the 12-core numbers, C4 per-worker gap 306 → ≈ 235 ms
   (STREAM .96 → ≈.73) if `T_step(B,5)` stays ≤ 55 ms. Difficulty: high (scratch and
   decoder-state ownership per slot, no nested dispatch, completion handshake).
   Correctness risk: low for audio (per-slot decoder state is untouched; output order
   preserved), medium for lifecycle (cancel/finalize while a decode is in flight).
   Evidence needed: measured `T_step(B,k)` and `D(B,N−k)` on the host (Section 10 step
   4). Smallest falsifier: a CLI/batch harness that runs the two calls concurrently on
   a fixed 5+3 (or 8+8 on Turin) split at B=2–4 and reports iteration wall; kill if the
   concurrent iteration is not ≥ 20% shorter than the serial one.
2. **Lead-sized decoder quantum and lead-ordered gang (LS-2, priority semantics).**
   Mechanism: Section 7 (a)–(c). Benefit: max-gap p95 and stall@250 at the first
   failing point; no mean gain. Difficulty: low. Risk: low. Evidence: census showing
   gaps coincide with decoder bursts of slots that had lead to spare. Falsifier:
   Section 7's rule.
3. **Remove per-slot serial windows from the loop thread.** Mechanism: run sampling,
   `RECORD_FRAME_AND_EMBED`, VQ scalar loops and gather/scatter inside the region
   (one slot per thread) and enable the bounded async writer. Benefit: ≈2–5 ms per
   slot per iteration (bounded by the measured ≈5–6 ms per-slot step marginal).
   Difficulty: low–medium; bit-identical is achievable for everything except the RNG
   ownership of sampling (per-slot RNG already exists). Risk: low. Evidence: census
   pool-idle share during step phases. Falsifier: per-slot step marginal drops below
   3 ms at B=3 in `[ITER] v=3`.
4. **Decoder intercept/glue reduction (bounded strip executor or fewer, larger
   pool submissions).** Mechanism: cut the ≈25–30 ms per-call intercept and the ~41–102
   rendezvous. Benefit: ≈6–7 ms per frame at q4 regardless of B, and it makes q2
   affordable. Difficulty: high. Risk: numerical (any fusion changes accumulation
   order; separate quality qualification). Evidence: census showing the intercept, not
   the slope, dominates gaps. Falsifier: `[SDPHASE]` intercept fit < 15 ms with
   unchanged slope.
5. **Narrow-worker topology for 0.6B on high-per-core-bandwidth hosts (`4x4`).**
   Not a code change. Mechanism: more independent chains. Benefit: up to C8 for 0.6B
   if a 4-thread chain fits. Difficulty: none. Risk: none. Evidence: Section 10 step
   4 chain measurement. Falsifier: 0.6B `4x4` C4 STREAM p95 > 2x8's.

Not on the list, deliberately: AMX kernel work, W4/BF16 serving, global batching,
cooperative prefill (SL-1 removed its known-text motivation; it returns only for
ICL/clone lanes), and any cap/threshold sweep.

# 15. What NOT to do next

- Another AMX kernel, tile-gate or `NCHUNK` cycle on any host: the product point at
  C2 does not even reach the AMX Talker/CP gate, and the decoder's wall is glue.
- Any `1x16` SOAK, or `4x4` for 1.7B.
- cap3, LS-4 threshold sweeps, helper/LOW variants, same-pool consumer variants,
  q1, `QWEN_POOL_SPIN` sweeps as a first move.
- Treating doctor rho, average core-equivalents, or STREAM_RTF p50 as capacity.
- Reusing any c8a/Scaleway/c3d number as a current control (different model
  identity `1.7b-base`, pre-SL-1, q8, no fused residual, dirty trees).
- Comparing AMX Design-D ragged decode against VNNI per-item decode as "ISA parity".
- Inventing prices; and starting a serving run on a host whose SL-1 pin, dispatch
  preflight or SMT state is unresolved.
- A global scheduler rewrite before the Section 6 census exists on two hosts.

# 16. Codex handoff

**A. Frozen.** Serving semantics: cap2/q4 (1.7B), cap3/q4 (0.6B), `--max-queue 0`
fail-fast, synchronous output, prefix cache, ragged threshold 2, engine-owned pool,
SL-1 known-text ON in every lane that runs the known-text bank, Design-D + fused
residual in the AMX lane only, per-item INT8 decoder declared VALID FALLBACK in the
VNNI lane. Playback definitions, banks, seeds, gates and the WAVE/SOAK lifecycle. No
runtime, kernel, scheduler or default change before the Turin screen is complete.

**B. Measure on the next host** (in this order): identity and CCD/L3 domains; per-mask
roofs (host, each CCD, one 4-core quarter); clean VNNI build, caps, self-test,
strict `vnni-product` and `common-control` preflight; the synthetic chain
`T_step(B,k)`/`D(B,k)` for both models; model load and RSS; prices for both hosts.

**C. Topology and concurrency screen.** 1.7B: `2x8` and `1x16` at C1/C2/C4, then the
winner at C3, C5/C6 only if C4 is GOOD. 0.6B: `2x8` at C1/C3/C4 (+ `4x4` at C4 only if
the 4-thread chain ≤ 50 ms), then C5/C6/C8 while the previous point is GOOD. Add the
`2+1`/`3+1` long-arrival probe at the candidate point. One 5-minute SOAK per model at
the best economic point only.

**D. Diagnostics to enable** (diagnostic arms only, never in KPI arms): `[ITER] v=3`
per-iteration phase durations with per-slot lead/pending (Section 6 items 1–3),
`[DECODE]`, `[TTFA2]`, F2 timeline with the client clock header, and external `perf
stat -I 100` on the worker PIDs. Ship the offline gap-attribution join with the
campaign.

**E. STOP.** Dispatch contradiction or dirty tree; a topology whose 1.7B C1 STREAM p50
> .75; C4 STREAM p95 > 1.0 on both 1.7B topologies (no further 1.7B work on that host
beyond recording C3); coalesced reads above a few percent in a KPI arm; any survivor
process between arms.

**F. When an architecture change becomes justified.** Only if, on the second host,
the census attributes the majority of ≥250 ms gaps at the first failing point to the
decoder burst or to serial loop-thread windows while pool idle share exceeds ~30% and
DRAM traffic stays below ~60% of the per-mask roof. Then start item 1 of Section 14
as a bounded falsifier on the host where `T_step(B,k)` at the reduced k still fits
the frame period; start item 2 immediately if gaps coincide with high-lead slots
being decoded ahead of low-lead ones. If instead the census shows the Talker/CP
phases at the roof with the pool busy, the wall is physical on that host: choose the
model and topology by price, and stop architecture work.

Verdict for this review: **INCONCLUSIVE on PHYSICAL vs ENGINE by design, ranked
MIXED > ENGINE > PHYSICAL; PROMOTE the census and the Turin screen as the next two
actions; KEEP the frozen reference; REJECT further AMX and admission-threshold work.**
