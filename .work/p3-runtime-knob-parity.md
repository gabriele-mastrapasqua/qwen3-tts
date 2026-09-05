Task: P3.3a P3.4 P3.5 (knob parity) — also closes what P3.1/P3.2 established
Question: which runtime knobs exist per backend, where each is read, which are pthread-only, and what the docs miss?
Written: 2026-09-05 from the getenv census (152 names read; registry 179/179 after 4e8d447; docs/feature-flags.md 91 names). Line numbers = tree at 620279c..4e8d447.

KleidiAI (Arm Linux, built only when `KAI_HAS_I8MM` = `__ARM_FEATURE_MATMUL_INT8`; bf16 family when `__ARM_FEATURE_BF16`; M1 = DOTPROD only → no KleidiAI at all)
| knob | read at | default | x86 VNNI/BF16 equivalent | AMX equivalent |
|---|---|---|---|---|
| `QWEN_NO_KLEIDI` | kleidi.c `qwen_kleidi_enabled` ~92 | on | `QWEN_NO_VNNI` (kills int8+q4 GEMV and GEMM) | `QWEN_NO_AMX` |
| `QWEN_NO_KAI_I8` / `QWEN_NO_KAI_BF16` | ~378 / ~722 | on | `QWEN_NO_VNNI` / `QWEN_NO_BF16DOT` + `QWEN_NO_BF16_MATMUL` | `QWEN_NO_AMX_INT8` / `QWEN_NO_AMX_BF16` (+`QWEN_NO_AMX_Q4`) |
| `QWEN_KAI_OPS` (per-op routing incl. prefill via `qwen_kleidi_prefill_enabled` ~189) | `kai_ops_parse` ~140 | all on | `QWEN_PREFILL_MATMAT`, `QWEN_NO_VNNI_QKV`, `QWEN_NO_X86_QKV` | `QWEN_AMX_PREPACK_KINDS` |
| `QWEN_KAI_NCHUNK` (bf16 GEMM n-subtile) | `kai_nchunk` ~176 | 384 | `QWEN_VNNI_NCHUNK`, `QWEN_AVX512_NCHUNK`, `QWEN_X86_NCHUNK` (`qwen_x86_nchunk` ~2398, align 4 for B<=4 else 2) | `QWEN_AMX_NCHUNK` (align 16) |
| `QWEN_KAI_LHS=sym` | `kai_lhs_sym_mode` ~498 | asym | `QWEN_NO_VNNI_ACT_QUANT` (u8 activation trick, `qwen_vnni_uact_enabled`) | same code as VNNI |
| `QWEN_KAI_QKV_FUSED` | `kai_qkv_fused` ~577 | on | `QWEN_NO_VNNI_QKV` (`qwen_vnni_qkv_disabled`, GEMV) / `QWEN_NO_X86_QKV` (`qwen_x86_qkv_disabled`, matmat int8+bf16) | `QWEN_AMX_INT8_QKV_MIN_B` (`qwen_amx_int8_qkv_allowed` ~4771) |
| `QWEN_KAI_REPEAT` (diag) | ~865 | off | `QWEN_VNNI_PHASE_TIMING` (diag) | — |
| `QWEN_KLEIDI_MIN_B` (gate row, 1..64) | `g_mm_gate[KLEIDI_Q4]` | 1 | `QWEN_VNNI_MIN_B` (2..16), `QWEN_BF16_MATMUL_MIN_B` (1..16), `QWEN_AVX2MM_MIN_B` | `QWEN_AMX_MIN_B` 4, `QWEN_AMX_INT8_MIN_B`, `QWEN_AMX_BF16_MIN_B`, `QWEN_AMX_MIN_ROWS` 32, `QWEN_AMX_{INT8,BF16,Q4}_MIN_COLS` (64/32/32) |
| packed RHS at registration (`qwen_kleidi_register_*_fam`, talker.c `qwen_kleidi_prepack` ~1054), `QWEN_NO_Q8REPACK` (q8_0 GGUF) | kleidi.c, q8repack.c | on | `QWEN_VNNI_PREPACK` (+ row-sum cache, `QWEN_NO_VNNI_ROWSUM`) | `QWEN_AMX_PREPACK`, `QWEN_AMX_PERSIST_CFG`, `QWEN_AMX_B32` (prototype, no caller) |

Arm in-house kernels: `QWEN_NO_SDOT` (int8/q4 GEMV + conv), `QWEN_NO_SMMLA` / `QWEN_SMMLA_MIN_B` (gate rows INT8_SMMLA, Q4_SMMLA; `apple_off`=1 → on Apple only with `QWEN_APPLE_MMLA=1`), `QWEN_INT8_SDOT_MM` (opt-in on_env, SDOT loop over B) / `QWEN_INT8_SDOT_MIN_B`, `QWEN_NO_BFMMLA` / `QWEN_BFMMLA_MIN_B` (2..64, `apple_off`), `QWEN_ARM_BFDOT=1` (opt-in bf16 GEMV, Linux bf16 hosts; `qwen_arm_bfdot_on`).
x86 tiling knobs with no Arm counterpart: `QWEN_NO_VNNI_TILE`, `QWEN_VNNI_TILE_N8`, `QWEN_VNNI_TILE_M4N2`, `QWEN_VNNI_GEMV_MR`, `QWEN_Q4_VNNI_V3` / `QWEN_Q4_VNNI_V4` (`qwen_q4_vnni_variant`; v3 default), `QWEN_NO_AVX2MM`.

Pool / scheduler (P3.3a)
- pthread (Linux, macOS with `QWEN_FORCE_PTHREAD`): `QWEN_POOL_SPIN` default 4096 spins before condvar; `QWEN_POOL_NARROW`; `QWEN_POOL_HI_WINDOW_US` 200; `QWEN_PREFILL_LOW_MS` 0 (needs `QWEN_PREFILL_HELPER=1`, `qwen_parallel_is_reentrant`); `QWEN_POOL_STATS` compile-time. Thread names `qwen-pool-N`.
- GCD: no spin/narrow/priority; `qwen_pool_spin_value()` = -1, `qwen_pool_narrow_value()` = -1 (thread.c ~559-564); `dispatch_apply` on `QOS_CLASS_USER_INITIATED`.
- Windows: own pool, same stubs as GCD for active/priority/team.
- Measured facts already in tracked docs/memory: c8a 2x8 C4: `QWEN_POOL_SPIN=65536` +11% STREAM p95 in the soak (wave said neutral); Axion profile carries 65536 (`configs/perf/axion-16c-ttfa.json`), x86 profiles 4096. Not a scheduler priority; a spinning worker steals a core. What is NOT known: whether the pthread pool's wake path (condvar broadcast, `submit_mtx`) costs the same on Neoverse vs Zen; needs the static read of `qwen_parallel` submit/wait (thread.c ~430-545) before any A/B (P3.3b).

Decoder / server knobs (P3.4)
- `QWEN_SD_INT8`: `sd_int8_enabled` (speech_decoder.c ~185): `#if __AVX512VNNI__` default ON (unless `=0`), else opt-in `=1`; kernels exist for DOTPROD and VNNI only (kernels.c `sd_tile_*` ~8577-8752). `QWEN_SD_INT8_BLK` block size. dispatch.c reports "opt-in on this ISA (measured slower on the first frame elsewhere)".
- `QWEN_SD_THREADS` (= `-j`), `QWEN_SD_POOL` (server default `qwen`), `QWEN_BLAS_OWN` (server default 1, Linux only effective), `QWEN_SD_SGEMM_CENSUS`, `QWEN_SD_SCRATCH_STATS`, `QWEN_SD_WINDOWED`, `QWEN_SD_PHASE`, `QWEN_SD_DEBUG`.
- `QWEN_DECODER_BATCH`: server sets 1 at start unless env says 0 (dispatch.c); Axion profile 1, x86 profiles 0 (provisional).
- `QWEN_CP_PREFILL2`: default ON only `#if __AVX512VNNI__` (dispatch.c ~183-191), opt-in elsewhere.
- `QWEN_CP_REGION`, `QWEN_TK_REGION`, `QWEN_CP_BATCH_HEAD`: on, VNNI only (see .work/p2-cross-backend-runtime.md).
- Prefill: `QWEN_PREFILL_MATMAT` (resolver `qwen_prefill_matmat_resolved`, talker.c ~1321, fixed 6ce84e4: explicit 1 with no native unit → f32/SGEMM fallback + gate refusal), `QWEN_PREFILL_QUANT` (Base models only), `QWEN_PREFILL_HELPER`, `QWEN_PREFILL_LOW_MS`; working-tree only: `QWEN_PREFILL_ROWPACK`, `QWEN_PREFILL_QKV_SHARE`, `QWEN_PREFILL_INT8MM`, `QWEN_PREFILL_CHUNK`.

Docs gaps (P3.5) — in code, absent from docs/feature-flags.md and relevant to an operator: `QWEN_AMX_PREPACK`, `QWEN_AMX_PREPACK_KINDS`, `QWEN_AMX_PERSIST_CFG`, `QWEN_AMX_B32`, `QWEN_NO_VNNI_QKV`, `QWEN_NO_X86_QKV`, `QWEN_Q4_VNNI_V3`/`V4`, `QWEN_VNNI_TILE_N8`, `QWEN_NO_SIN_POLY`, `QWEN_SD_INT8_BLK`, `QWEN_SD_WINDOWED`, `QWEN_THREADS_TALKER`/`QWEN_THREADS_DECODER`, `QWEN_DEC_FIRSTCHUNK_GROUP`, `QWEN_CUDA_*`, `QWEN_METAL_*`. Diagnostics not worth documenting: `*_DEBUG`, `*_JSON`, `QWEN_DUMP_*`, `QWEN_TUNE_*`, `QWEN_VNNI_PHASE_TIMING`, `QWEN_KAI_REPEAT`. Stale-looking in docs vs code: none found; the 19 doc-only names (`QWEN_*_MIN_B`, `QWEN_*_NCHUNK`, `QWEN_NO_AMX_*`, `QWEN_NO_SMMLA`, `QWEN_VNNI_GEMV_MR` …) are read through `qwen_mm_env_int` / the gate table, not a literal `getenv`, so a getenv grep misses them by design.

Files/functions inspected: qwen_tts_kleidi.c (enable predicates, ops parse, nchunk, lhs mode, qkv fused), qwen_tts_kernels.c (`g_mm_gate[]`, `qwen_mm_use`, `qwen_mm_specific_minb_env`, `qwen_x86_nchunk`, `g_qwen_reported_flags[]`, `qwen_sd_int8_available`), qwen_tts_speech_decoder.c (`sd_int8_enabled`), qwen_tts_dispatch.c (feature rows), qwen_tts_thread.c (spin/narrow/priority, GCD/Windows stubs), qwen_tts_talker.c (prefill predicate, prepack), configs/perf/*.json (axion vs x86 values), docs/feature-flags.md, tools/check_flag_registry.py.

Next action: P3.3a = read thread.c submit/wait path and write the pthread-vs-GCD contract into .work (or P2.6), then decide whether a 0/4096/65536 A/B is even needed; P3.4 = one first-frame measurement of `QWEN_SD_INT8=1` on an Arm box; P3.5 = add the rows above to docs/feature-flags.md (docs only).
