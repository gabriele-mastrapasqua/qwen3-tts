# P3 prefill-helper falsifier — 2026-09-07

## Task · Question

Test whether the existing cloned-context prefill helper, with a bounded LOW-priority
window, protects the current streaming server enough to justify using it as the P3
admission solution. This is not a resumable prefill implementation: it still computes
one complete prompt and copies the completed state into a batch slot.

## Known facts

- Runtime source was clean commit `04f00d3`; the transferred archive had no `.git`.
- Binary SHA-256 prefix was `8358ca9c520e0986`.
- Host was c4-standard-24, Xeon Platinum 8581C, one socket/NUMA, 12 physical cores,
  SMT off, CPUs `0-11`, 2x6 prefork.
- Both arms used the 1.7B INT8 model, Design-D decoder, engine pool, batch cap 2,
  q8, ragged threshold 2, synchronous output, the same 21-text bank, seed base 9200,
  and two true simultaneous waves at C3 and C4.
- The control was the matched synchronous-writer wave. The treatment changed only
  `QWEN_PREFILL_HELPER=1`, `QWEN_PREFILL_LOW_MS=150` and `QWEN_QUEUE_PREFILL=1`.

## Unknowns

- This wave does not isolate a complete admission event against already-playing
  streams; it is a bounded serving screen, not a long qualification.
- It does not test a helper without LOW priority or a different queue cap. Those are
  knob variants, not evidence for the missing request-owned resumable state.
- Client playback marks remain subject to the documented receive-fidelity contract.

## Files/functions inspected

`qwen_tts.c` (`prefill_helper_main`, `prefill_q_t`, `ADMIT_INSTALL`),
`qwen_tts_talker.c` (`qwen_talker_prefill`), `qwen_tts_server.c`,
`tests/serve_parallel_wave.py`, and the existing input-length/admission evidence.

## Evidence

The treatment resolved the requested helper and low-priority flags in the server logs;
the helper added one thread per worker (22 versus 21 in the control). No errors,
rejects, timeouts, failed enqueues or surviving processes were observed.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p95 ms | safe start p95 ms | max gap p95 ms | stall@250/@500 | coalesced reads | mean cores |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| synchronous control | 3 | 218/435 | .590/.775 | .605/.883 | 321 | 754 | 524 | 33%/0% | 6.1% | 6.6 |
| helper + LOW 150 ms | 3 | 173/2379 | .616/.770 | .679/.794 | 281 | 2659 | 583 | 17%/0% | 6.0% | 6.1 |
| synchronous control | 4 | 424/503 | .790/.821 | .817/.975 | 367 | 753 | 574 | 25%/0% | 5.1% | 7.1 |
| helper + LOW 150 ms | 4 | 600/2459 | .794/.883 | .808/.997 | 427 | 2749 | 649 | 50%/0% | 5.0% | 7.1 |

The helper treatment did not provide a server-level streaming gain. C3 STREAM_RTF p95
was effectively unchanged (`.770` versus `.775`), and C4 worsened (`.883` versus
`.821`). The small prebuffer movement did not compensate for the admission/TTFA tail:
TTFA p95 increased from `435` to `2379` ms at C3 and from `503` to `2459` ms at C4.
The server remained error-free, but this is not an acceptable first-play envelope.

## Conclusion

**REJECT as the current P3 serving policy.** The existing helper's serialized full
prefill plus bounded queue/LOW scheduling can delay first audio by seconds without
improving steady-state RTF. This does not reject PF-1 itself; it rejects treating a
cloned-context one-shot helper as equivalent to a request-owned resumable prefill.

PF-1 remains a separate architecture task: a fixed prompt must retain its input cursor,
per-layer/intermediate state and completed KV in request-owned storage so the scheduler
can yield at safe boundaries without restarting or recomputing. Live text, text
segmentation and appending after generation remain out of scope.

## Next action

Do not run more helper/LOW knob variants in this cycle. Keep the helper default-off and
proceed only with a bounded implementation of true fixed-prompt state if its ownership
and safe boundary can be made explicit; otherwise leave PF-1 open for a focused design
checkpoint. Preserve the existing q8/threshold2 control.
