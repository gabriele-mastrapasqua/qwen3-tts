# Task · AMX-3/AMX-7 cross-request workset and low-N decomposition

## Question

Can the current C4 server turn already-independent decoder work into a larger
AMX Design-D matrix workload, or is the remaining low-panel work better split
over the decoder output-row axis?

## Known facts

- Canonical source was clean at `c72e6c6` before the exploratory candidate.
- The reference host was the one-socket, SMT-off Xeon 8581C with CPUs `0-11`
  online; all server runs below used the same 1.7B model and `2x6` control
  unless explicitly stated.
- With `QWEN_SD_RAG_MIN_PANELS=2`, the best existing C4 screen is still
  marginal (`STREAM_RTF` p50/p95 about `0.900/0.954`).
- The decoder already has persistent Design-D INT8 B packs and a ragged batch
  path; no new precision or kernel change was part of this investigation.

## Unknowns

- Whether a process-wide scheduler can gather work across the two prefork
  workers without an intentional wait or a materially different ownership
  model.
- Whether a larger single-process batch pays for itself on the production
  mixed workload; the bounded probe below used the short class only.

## Files/functions inspected

- `qwen_tts.c`: `qwen_tts_serve_continuous`, inline `db_items` selection,
  `dec_worker_main` and its opt-in decoder thread.
- `qwen_tts_server.c`: prefork dispatch and per-worker batch ownership.
- `qwen_tts_speech_decoder.c`: `sd_stream_batch_body`, `rag_conv1d_amx`,
  `rag_recompute`, panel construction, Design-D B-pack use and tail updates.
- `qwen_tts_thread.c`: engine-pool reentrancy and nested-dispatch behavior.

## Evidence

### Safe aggregation axes

| axis | classification | reason |
|---|---|---|
| ready items in one `db_items` call | `INDEPENDENT_NOW` | Each item has its own codes/state; `sd_stream_batch_body` concatenates columns into `fr`, keeps `fr.off[]/len[]`, uses one weight object and scatters audio by item. |
| N panels inside one ragged call | `INDEPENDENT_NOW` | Panels have disjoint output columns and share the same prepared B representation; the engine pool already claims them independently. |
| different decoder layers for one item | `DEPENDENT_SEQUENTIAL` | The next layer consumes the previous layer output. |
| later chunks of the same stream | `DEPENDENT_SEQUENTIAL` | KV/latent/tail state must be updated in order. |
| equal-shape work in different prefork workers | `SHARED_WEIGHT_ONLY` | The workers are separate processes with separate scheduler state; there is no shared `db_items`/output ownership to gather. |
| delaying jobs until a larger group appears | `SPECULATIVE_ALGORITHM_CHANGE` | It introduces a queueing/cadence policy and can change TTFA and playback regularity. |

The strongest safe mapping is therefore already present inside one decoder
call: request A contributes its contiguous `[fr.off[0], fr.off[0]+fr.len[0])`
columns, request B contributes the next range, and `rag_conv1d_amx` runs the
same persistent B tiles over the concatenated logical N. Bias is added once
after the full workset, while causal tails are saved from each request's own
offset and length. No request-local state crosses the scatter boundary.

The missing larger workset is across prefork processes, not inside the decoder
mathematics. With `2x6` and batch cap `2`, each worker can see at most two
active streams. Raising the cap alone does not make a C4 workload enter one
worker: the parent load-balancer assigns connections to the least-loaded
worker. A global gather would require changing process/scheduler ownership or
using an asynchronous queue with a latency policy.

### Exploratory low-N M split

An opt-in candidate split caller-serial ragged panels with `M >= 192` into
disjoint 96-row tasks. Every task reused the already-quantised activation panel
and the corresponding offset into the persistent Design-D B pack; fallback used
the same original weights and output subrange. It was gated off by default and
was not committed.

The real server diagnostic, short class, threshold `2`, chunk `32`, batch `2`,
`2x6`, showed the path was executable: 39 `M=768, n_panels=1` calls, 312 row
tasks, 312 AMX tasks and zero fallback tasks. This proves the mapping and AMX
counter, but the affected family is only the single-panel `M=768` work.

The targeted non-intrusive short C4 A/B rejected it:

| arm | requests | STREAM_RTF p50/p95 | TTFA p50/p95 ms | prebuffer p50/p95 s | stall p50/p95 s |
|---|---:|---:|---:|---:|---:|
| control, M split off | 58 | `0.8850/0.9707` | `126.8/224.6` | `0.3786/1.1609` | `0.2489/0.8799` |
| M split on | 55 | `0.9194/1.1056` | `131.1/194.2` | `0.4966/1.3211` | `0.3343/0.9887` |

The lower TTFA p95 did not compensate for worse streaming and cadence. A
mixed-bank C4 screen also did not show a trustworthy attributable win: control
`0.9079/1.0086` versus candidate `0.9129/0.9799` over 34 samples per arm, and
the candidate had not exposed counters in that run. It is insufficient for
promotion, especially after the controlled short-class regression.

### Larger logical workset probe

A diagnostic single-process `1x12`, batch cap `4` run did form real larger
worksets: the decoder log saw `items=2/3/4`, with 456 `items=4` records in the
short diagnostic and the same persistent Design-D AMX path. This is genuine
cross-request aggregation, not merely pool parallelism.

The non-intrusive short probe was nevertheless poor: 44 samples,
`STREAM_RTF` p50/p95 `1.125/1.277` (window p95 `1.197, 1.277, 1.154`), TTFA
p50/p95 `168.1/290.1 ms`, zero errors/rejects/timeouts. Statistics-enabled
diagnostics were worse still and are not used as performance evidence.

### Validation and controls

- Local native BLAS build, self-test and flag registry passed for the candidate.
- Remote clean AMX build at `c72e6c6` passed caps and self-test; host remained
  SMT off with CPUs `0-11` online.
- Decoder batch parity's existing first and second patterns both reported the
  same non-zero audio differences with M split on and off; the candidate did
  not change those values. The test target remains red because its pre-existing
  GEMV/GEMM comparison reports a worst absolute difference around `0.04`.
- All exploratory server runs completed with zero errors, queue rejects and
  request timeouts. Their zero-buffer prebuffer/stall fields remain client
  diagnostics, not direct server starvation counters.

## Conclusion

The current implementation already performs the safe cross-request fusion
available within a worker. The desired larger `N` across the two `2x6`
prefork workers cannot be added as a local decoder slice without changing
scheduler/process ownership or introducing a deliberate gather delay. The
only bounded local fallback tested, low-N M decomposition, is rejected by a
controlled streaming A/B and must remain absent from the canonical tree.

The Xeon is still not shown to be the bottleneck; the evidence points to the
serving ownership/arrival geometry. C4 remains `NOT QUALIFIED`.

## Next action

Do not try another isolated decoder knob. The next legitimate experiment needs
one explicit serving architecture choice: either a shared process-level decoder
gather that defines its TTFA/cadence bound, or a topology that gives one worker
the whole C4 workset. That is a new scheduler/ownership change, not a safe
kernel-local patch. Keep the low-N split rejected and preserve threshold `2`
as the current best control.
