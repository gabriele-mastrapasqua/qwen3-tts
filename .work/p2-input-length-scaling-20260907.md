# Task · Question

Measure startup and admission scaling with input length on the q8 decoder control and
the current P2 decoder slices, then identify whether long input requires a later bounded
prefill design.  This is evidence only: no text segmentation or model-semantic change
was made.

## Known facts

- The measured runtime source is the clean `cb4be8b` code commit; `e00346f` is a
  documentation-only descendant of that runtime state.
- The reference host was a single-socket, single-NUMA GCP c4-standard-24 with an Intel
  Xeon Platinum 8581C, CPUs 0-11 online and SMT off.
- Both arms used the 1.7B INT8 model, Design-D INT8 decoder AMX, 2x6 prefork, engine
  decoder pool, batch cap 2, q8, ragged threshold 2, and no intrusive profiler.
- Control disabled the three experimental decoder slices.  P2 enabled the warm strip,
  direct ragged transposed-convolution, and direct ragged depthwise-convolution slices.
- The lifecycle traces define queue wait as `admitted - enqueued` and prefill wall as
  `prefill_done - prefill_start`.  The `[REQ] tokens` field is total prompt/KV length,
  not raw text-token count.

## Unknowns

- This probe does not implement incremental prefill and therefore cannot establish the
  eventual quality or memory cost of a bounded initial text window.
- The delayed-admission probe records client playback metrics but does not persist the
  complete receive-mark timeline; its `coalesced_reads` count is not a citation-grade
  share denominator.  The matched length bank runs do report receive fidelity.
- A dedicated long-request admission run with many established streams would be needed
  to characterize the production tail beyond this bounded four-request probe.

## Files/functions inspected

`qwen_tts.c` (`qwen_tts_generate`, `qwen_tts_generate_batch_multi`, `ADMIT_PREFILL`,
`ADMIT_INSTALL`, batch generation state), `qwen_tts_talker.c` (`qwen_talker_prefill`),
`qwen_tts_speech_decoder.c` (stream state, causal tails, decoder slices),
`tests/soak_client.py`, `tests/playback_sim.py`, and the existing server lifecycle traces.

## Evidence

### Length bank and receive fidelity

The stationary bank contained four whole inputs; no input was split:

| class | chars | prompt tokens | produced audio |
|---|---:|---:|---:|
| short | 23 | 16 | 1.44 s |
| five_s | 70 | 27 | 5.36 s |
| ten_s | 143 | 45 | 10.16 s |
| long | 470 | 112 | 30.40 s |

Matched C1/C3 wave runs had zero errors, rejects, and timeouts in every cell.  The
client-observed coalesced-read share was the same for control and P2: 16.7% (short),
8.3% (five_s), 5.0% (ten_s), and 2.0% (long).  These cadence values remain upper
bounds on server-side lateness under the receive-mark contract.

### C1 startup scaling

Values below are representative C1 results from the lifecycle-traced runs; prefill is
the full Talker prefill wall after admission.

| class | control prefill / TTFA ms | P2 prefill / TTFA ms | admission wait |
|---|---:|---:|---:|
| short | 45.9 / 80.0 | 46.2 / 80.4 | ~0 ms |
| five_s | 108.8 / 144.3 | 108.5 / 143.3 | ~0 ms |
| ten_s | 162.5 / 198.6 | 161.7 / 195.4 | ~0 ms |
| long | 394.4 / 431.7 | 397.3 / 433.2 | ~0 ms |

The C1 request is not queued behind another request, so TTFA tracks prefill plus a
roughly stable first-frame/audio term.  From 16 to 112 prompt tokens, prefill grows by
about 348-351 ms; the P2 decoder slices do not change this term.

### C3 same-wave admission interference

For each C3 wave, the first admitted request had negligible queue wait and the third
request was delayed behind earlier inline prefill.  The delayed request's lifecycle
values were:

| class | control delayed wait / TTFA-after-admit ms | P2 delayed wait / TTFA-after-admit ms |
|---|---:|---:|
| short | 54.2 / 102.1 | 54.8 / 102.9 |
| five_s | 119.2 / 168.5 | 117.9 / 168.2 |
| ten_s | 186.3 / 228.8 | 187.9 / 229.1 |
| long | 425.9 / 480.1 | 422.1 / 480.6 |

The delayed wait is approximately the preceding request's full prefill wall, not a
decoder first-step cost.  C3 aggregate TTFA p95 was approximately 158/907 ms for
control (short/long) and 162/904 ms for P2.  The long request therefore makes the
inline admission coupling visible even though the decoder P2 slices are active.

### Dedicated delayed-long probe

A separate bounded server probe started three concurrent `ten_s` streams, then sent a
whole `long` request at 700 ms.  Both arms completed with zero errors/rejects/timeouts.
The client-observed results were:

| arm / stream | TTFA ms | STREAM_RTF | prebuffer ms | safe start ms | max gap ms | stall max ms | stalls @250 / @500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control established (range) | 363-407 | .791-.866 | 310-587 | 717-950 | 525-642 | 130-429 | 1-2 / 0 |
| control admitted long | 536 | .715 | 354 | 890 | 578 | 181 | 1 / 0 |
| P2 established (range) | 260-448 | .912-.956 | 1013-1078 | 1339-1461 | 984-1130 | 460-664 | 2-4 / 1-3 |
| P2 admitted long | 501 | .734 | 937 | 1437 | 1130 | 490 | 3 / 2 |

This four-request diagnostic is not a serving A/B qualification and the P2 arm is
noisier/worse in this single sample; it does not overturn the matched-bank result that
P2 is a decoder-side optimization.  It does establish that admission of a long prefill
can overlap the playback-safety budget of established streams.  The dedicated probe
returned one fast read per request; because it did not retain total chunk counts, no
coalesced percentage is claimed for this table.

### State-boundary audit

- **Text/KV:** `qwen_tts_generate` tokenizes and embeds the whole input, computes one
  `prefill_len`, and calls `qwen_talker_prefill` for the full request.  The existing
  prefix cache only reuses a matching prefix in the same context; server admission
  explicitly resets `prev_prefill_len` and performs a full prefill.  KV and final hidden
  state are copied to the server batch slot only after prefill completes.
- **Talker:** the resumable state currently available after prefill is `kv_cache_*`,
  `kv_len`, and `dec_x`/the final hidden state.  There is no request-owned prefill
  cursor/state that lets admission yield halfway through a long text and resume through
  the current server handoff.
- **Code Predictor:** installed slots own `pos`, `prev_tok`, `nprev`, `code0`, and the
  generated code/frame counters.  These are generation state after admission, not a
  partial-prefill interface.
- **Codec/history:** each streaming slot owns `qwen_sd_stream_state_t`, code history,
  decoder position, causal tails/carries, and accumulated audio.  The decoder slices
  retain causal state across decode calls, so this boundary is suitable for future
  bounded decode work; it does not make Talker prefill incremental by itself.

## Conclusion

**P3/PREFILL requirement confirmed.**  TTFA scales materially with total input length
because full Talker prefill is completed before generation.  Under C3, an arriving
request can wait roughly 54/119/186/422-426 ms for short/five_s/ten_s/long before its
own prefill starts.  P2 direct decoder work leaves this prefill/admission term unchanged.
The first playable audio of a future professional server should depend on a bounded
initial text window, with request-local text/KV/Talker state that can resume without
recomputing or resetting.  This is a P3 architecture requirement, not an instruction to
add naive segmentation now.

The first P2 control-plane conclusion is unchanged: continue measuring decoder fixed
cost only where a bounded implementation slice has a concrete target; do not attribute
the long-input startup term to AMX decoder work.

## Next action

Return to the next bounded P2 decoder fixed-cost target.  Keep incremental/chunked
prefill as a documented P3/PREFILL task and do not implement it during this decoder
slice cycle.
