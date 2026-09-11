# Task

ARM-LINUX-V2 Item 1 and Item 3 on `feature/arm-parity-vnni`: wire the prepared
KleidiAI BF16 pre-transformer consumer, implement the DL-4 multi-slot kernel and
lane cohort, then validate the Arm and x86 parity contracts.

# Question

Does the implementation select the intended Arm leaves safely, preserve the
single-slot numerical contract, and have enough evidence to change a product
profile or default?

# Known facts

- The BF16 KAI prepared-state API is keyed by the original persistent f32 weight
  pointer. The decoder registers all input/output/QKV/O/FFN projections after
  allocation and unregisters them before freeing the BF16 copies.
- Full, streaming, and ragged pre-transformer call sites now pass the original
  weight key. KAI is attempted first; the existing f32/packed fallback remains
  available.
- The multi-slot API supports S=2..3, equal-length cohorts, rectangular in/out
  shapes, causal tails, residuals, and a strided ragged global workset. The lane
  queues one cohort leader and keeps one bounded mailbox unit per slot.

# Unknowns

- No 16-core Axion production qualification or 21-text BF16 quality campaign has
  been run for the new paths.
- The standard local mel comparison helper could not import librosa because the
  installed numba cache has no source locator. The reported BF16 mel values below
  are an independent NumPy/scipy log-mel proxy, not a production qualification.
- x86 VNNI/AMX execution was not available on the Arm box; x86 source coverage is
  compile-only plus the existing local dispatched self-test on Apple Arm.

# Files/functions inspected

- `qwen_tts_speech_decoder.c`: `sd_bf16_preup_prepare`,
  `sd_bf16_preup_matmat`, `rag_conv1d`, and the full/streaming/ragged forwards.
- `qwen_tts_kleidi.c/.h`: BF16 prepared-state lookup, registration, and owner
  scoped unregister.
- `qwen_tts_kernels.c/.h`: VNNI and Arm SDOT multi-slot leaves, compact/strided
  APIs, and self-test oracles.
- `qwen_tts.c`: lane cohort admission, worker completion, busy accounting, and
  mismatch flush.
- `qwen_tts_dispatch.c`, `tools/serving_profile.py`, and
  `configs/perf/{arm-product,axion-16c-ttfa}.json`.

# Evidence

- `make blas` passed locally. `make check-isa` passed both the Arm i8mm/BF16
  and x86 AVX512/VNNI/BF16/AMX compile profiles after making AMX tile-zero
  indices compile-time constants. Flag registry and flag parity pass.
- Local `./qwen_tts --self-test` passed all cases. The new exact oracles passed:
  compact S=2 square and rectangular, production strided S=2 square and
  rectangular, and compact S=3.
- A native Neoverse-V2 build passed the same self-test, including all new
  compact/strided/S=3 oracles. Dispatch with INT8 + RES1_V2 + lane + KAI BF16
  showed `decoder.pre_up_bf16=ON`, `decoder.multislot=ON`, KAI BF16, and Arm
  SMMLA/DOTPROD gates active. Full 0.6B and 1.7B BF16 ON runs registered all
  prepared rows and produced valid 2-second WAVs.
- Lane streaming smoke with two concurrent requests completed with zero curl
  errors, `dec_group_max=2`, and zero mailbox overruns. The three-slot smoke
  completed with `dec_group_max=3` and zero mailbox overruns.
- Short WAVE A/B on the Arm box, same build/settings and 2-second output:

  | model | cohort | multi ON | multi OFF | result |
  |---|---:|---:|---:|---|
  | 0.6B | 2 | 3.273 s | 3.183 s | ON 2.8% slower |
  | 1.7B | 2 | 4.785 s | 4.648 s | ON 2.9% slower |
  | 0.6B | 3 | 3.749 s | 3.613 s | ON 3.8% slower |
  | 1.7B | 3 | 5.278 s | 5.132 s | ON 2.8% slower |

  All runs returned valid audio with equal byte counts and durations. The
  comparison is a short WAVE screen, not a production capacity qualification;
  multi-slot remains default-off.
- BF16 full-path A/B generated valid, equal-duration WAVs but the independent
  log-mel proxy was 0.911 (0.6B) and 0.823 (1.7B) versus the f32 control. BF16
  remains default-off and is not represented as a qualified product behavior.

# Conclusion

The code paths are implemented and structurally verified. Item 3 has an exact
kernel/ABI oracle and a working lane cohort, but its measured short-run speed is
negative, so it is opt-in only. Item 1 is wired and KAI-active on Arm, but the
quality screen is a no-go for promotion; the f32 control remains the product
configuration.

# Next action

Keep the new flags explicit in the Arm profiles, with RES1_V2 available and
BF16/multi-slot off. A future clean-tree 16-core campaign may requalify the lane,
multi-slot, and BF16 separately; do not quote the short WAVE numbers as SOAK or
production capacity.
