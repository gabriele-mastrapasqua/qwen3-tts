# Task · Question

Can the ragged decoder gather one activation row and quantise it immediately, avoiding
the temporary FP32 `[N][K]` panel while preserving the Design-D INT8 path and server
output exactly?

## Known facts

The control path materialises a ragged FP32 im2col panel, then quantises the complete
panel before AMX.  The candidate `aa90518` added default-off `QWEN_SD_DIRECT_QUANT`:
one worker-local FP32 row is gathered and quantised at a time; the established full
panel is reconstructed only if AMX rejects and the existing BLAS fallback is needed.

The candidate was reverted by `4089bbd` after serving A/B evidence.  The production
control therefore remains unchanged.

## Unknowns

The experiment does not establish whether a batch-aware gather/quantise kernel could
recover the lost panel-level SIMD locality.  It only evaluates the one-row helper.
The server WAVE is a bounded screening run, not a C4 SOAK or qualification.

## Files/functions inspected

`qwen_tts_speech_decoder.c` (`sd_rag_gather_row`, `sd_rag_panel_worker`,
`rag_conv1d_amx`), `qwen_tts_kernels.c` (`qwen_int8_quant_rows`),
`docs/feature-flags.md`, and `tests/serve_parallel_wave.py`.

## Evidence

### Local gates

The candidate built cleanly on the local native BLAS target.  `--self-test` passed with
zero failures, the flag registry reported `194` read and declared flags, and
`git diff --check` passed.  The AMX server build was from the clean `aa90518` source
fingerprint; startup and self-test reported AMX INT8/BF16 capability.

### Server path proof

The candidate ran on the single-socket Xeon Platinum 8581C reference host with twelve
physical CPUs online, SMT off, 2x6 prefork, engine decoder pool, batch cap 2, q8,
ragged threshold 2, Design-D INT8 and fused residual enabled.  The diagnostic log
reported `direct_quant=1`, `amx=panels` and `fallback=0` for representative ragged
shapes, including 240 panels at M=96, 20 at M=384, 80 at M=192 and 4 at M=768.
Thus the candidate exercised the intended direct-quant AMX path in the actual server.

Representative diagnostic preparation costs for one worker were approximately:

| M | total columns | direct prepare time |
|---:|---:|---:|
| 96 | 30,720 | 18.7–21.3 ms |
| 192 | 10,240 | 11.0–13.1 ms |
| 384 | 2,560 | 5.7–7.1 ms |
| 768 | 512 | 2.6–3.1 ms |

The direct path reported zero FP32 panel build bytes, but the one-row quantiser call
repeated the gather/quantisation boundary for every column.

### C4 WAVE A/B

Both arms used the same clean binary, model, q8/Design-D configuration, 2x6 SMT-off
topology, batch cap 2, engine pool, and three synchronized waves of four requests.
No profiler or census was enabled in the KPI arms.  Receive coalescing was low and
similar: 5.1% control and 4.7% direct.  All 12 requests per arm completed with zero
errors, rejects and timeouts.

| arm | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p50/p95 ms | safe start p50/p95 ms | max gap p95 ms | stall @250 / @500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control (`DIRECT_QUANT=0`) | 430/542 | 0.755/0.776 | 0.790/0.954 | 249/275 | 706/786 | 517 | 0% / 0% |
| direct (`DIRECT_QUANT=1`) | 420/533 | 0.758/0.954 | 0.816/0.979 | 306/1018 | 757/1219 | 1151 | 33% / 17% |

The direct arm's C4 STREAM_RTF p95 and playback tail are materially worse despite a
similar effective batch (`1.93` versus `1.97`) and CPU use (`7.3` versus `7.4` cores).

### Audio quality gate

A separate four-request C4 quality run used the same seed and wrote WAVs.  All four
control/feature WAV pairs were byte-identical, with matching lengths and SHA-256.
The implementation therefore preserved the quantisation arithmetic and ragged output
ownership on the tested paths.

## Conclusion

**REJECT as a serving optimization; preserve the evidence, revert the code.**  The
candidate is mathematically and server-path correct and removes the large temporary
FP32 panel on successful AMX execution, but one-row gather plus one-row quantisation
loses the locality/vectorization benefit of the panel.  It increases preparation and
produces worse C4 cadence in the controlled WAVE.  `aa90518` is retained in history
for provenance and `4089bbd` restores the clean control.

## Next action

Do not retry this exact one-row design or start another isolated INT8 quantisation knob.
If preparation remains a P2 target later, evaluate only a batch-aware gather/quantise
kernel with a measured reason to expect panel-level SIMD locality.  Continue the P2
checkpoint with the already validated default-off slices and move to the next
architecture milestone only after the fixed-cost evidence is recorded.
