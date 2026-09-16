# C12-WIN checkpoint — Turin VNNI ladder closed

Task · C12-WIN discriminating ladder

Question · Which bounded mechanism can move the frozen Turin VNNI product toward
`STREAM_RTF p95 <= 0.90` without giving back first-play or playback safety?

Known facts · The reference is `turin-c8a-32c-vnni-product`: 4x8, cap 4, q4,
DL-2 elastic decoder lane, DL-4 `RES1_V2`, per-item INT8 VNNI decoder, native
BF16 prefill, synchronous output, and fail-fast admission. The control sanity
wave was inside the established range (`STREAM_RTF p50/p95=.772/.841`;
TTFA p95=272 ms; prebuffer p95=256 ms; safe-start p95=469 ms; zero errors,
rejects and stalls). The 10-minute control soak was pooled `.854/.923`, with
short `.867/.966` and conversational `.856/.919`.

Unknowns · The exact hardware cause of the CP overlap tax is not isolated into
cache, DRAM, or a specific CP section. The `1,4` first-ramp alternative was not
run. No new C12 serving point was qualified by this ladder.

Files/functions inspected · `qwen_tts.c`, `qwen_tts_speech_decoder.c`,
`qwen_tts_kernels.c`, `qwen_tts_kernels.h`, the Turin profile, the decoder
quantum benchmark, the C12 WAVE/SOAK harnesses, and the linked experiment notes:
`.work/c12-win-step1-3-20260909.md`,
`.work/c12-win-bf16-preup-20260909.md`,
`.work/c12-win-convt-one-gemm-20260909.md`, and
`.work/c12-win-glue-vnni-20260909.md`.

## Results

* `QWEN_PREFILL_HELPER=1`: **NO-GO**. Pooled STREAM p95 moved `.923 -> .915`,
  but TTFA p95 moved `172 -> 683 ms`, safe-start `417 -> 922 ms`, and
  stall@250 appeared. The short-class improvement alone was not a serving win.
* Fixed-B3 diagnostic: **CP overlap confirmed**. Decoder overlap accounted for
  54.9% of active-B3 wall; CP median grew `22.2 -> 36.5 ms` while the decoder
  was in flight. Decoder unit wall was 49.2 ms by cost map. The broad inflation
  is measured; the cache/DRAM mechanism and overlap-conditioned CP subregions
  remain unknown.
* `QWEN_POOL_SPIN`: **CLOSED / NO-GO**. 4096 and 16384 did not beat the frozen
  65536 control beyond run noise.
* `QWEN_SD_BF16_PREUP=1`: **NO-GO**. The bounded server screen moved pooled
  STREAM p95 `.842 -> .825`, but the same-generation paired audio gate failed
  (`mel_corr=.97890 < .98`); it remains default-off.
* ConvT one-GEMM: **REJECTED / REVERTED**. Exact parity passed, but the expanded
  f32 panel was 14–32% slower across B1–B4 and chunks 1–8.
* Allocation-only glue: **REJECTED / REVERTED**. No useful q4 B1–B4 microbench
  movement; no server A/B was justified.
* VNNI RES1_V2 split-input: **REJECTED / REVERTED**. Continuation parity was
  exact (`max_abs=0`), but final timings were approximately +0.3–2.0% at q4
  and +4.7–5.9% at q8 across B1–B4.
* First-ramp `1,4`: **CANCELLED / NOT RUN**. The temporary knob and runtime
  code were removed; there is no performance or quality claim for it.

## Conclusion

The ladder did not produce a qualified C12 win. The current Turin VNNI product
profile remains the control; no unpromoted runtime experiment is active in the
local tree. The measured remaining mechanism is a decoder-in-flight CP tax,
not pool spin or the tested preparation/layout variants. The deferred
DECODER-XISA item remains documentation-only and was not opened by this work.

## Next action

Pause the paid-host campaign. Reopen C12 only under a new, explicitly scoped
experiment with a fresh control and quality/cadence gates. Do not infer a win
from any screen-only movement recorded here.
