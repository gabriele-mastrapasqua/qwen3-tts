# Backend matrix — merged from each build's own `--dispatch-map`

Three states, all reported by the engine: **IMPLEMENTED** compiled in · **SELECTABLE** this host can run it · **EFFECTIVE** it is what resolves now. `—` means not compiled into that build. Regenerate with `tools/backend_matrix.py`.

**Scope: CPU dispatch only.** The columns are CPU build profiles and the rows are the features
`--dispatch-map` resolves on them. The GPU backends are selected with `--backend metal|cuda`
rather than resolved by ISA dispatch, so they are not rows here; see
[`cuda-performance.md`](cuda-performance.md) and [`hardware-testing.md`](hardware-testing.md).

| feature | apple_m1 (native) | x86_avx2 (portable) | x86_avx512vnni (avx512vnni) | x86_amx (amx) |
|---|---|---|---|---|
| `cp.precision` | follows | follows | follows | follows |
| `cp.prefill2` | SELECTABLE | SELECTABLE | SELECTABLE | SELECTABLE |
| `decoder.batch` | SELECTABLE | SELECTABLE | SELECTABLE | SELECTABLE |
| `decoder.int8` | SELECTABLE | SELECTABLE | EFFECTIVE | EFFECTIVE |
| `decoder.pool` | private | private | private | private |
| `matmat.bf16.family` | see reason | see reason | see reason | see reason |
| `matmat.int8.batch_ceiling` | 0 | 16 | 16 | 16 |
| `matmat.int8.family` | see reason | see reason | see reason | see reason |
| `matmat.q4.family` | see reason | see reason | see reason | see reason |
| `matvec.bf16.dpbf16` | — | — | — | EFFECTIVE |
| `matvec.int8.native` | EFFECTIVE | — | EFFECTIVE | EFFECTIVE |
| `matvec.int8.sdot` | EFFECTIVE | — | — | — |
| `matvec.int8.vnni` | — | — | EFFECTIVE | EFFECTIVE |
| `matvec.q4.native` | EFFECTIVE | — | EFFECTIVE | EFFECTIVE |
| `matvec.q4.vnni_variant` | — | — | SELECTABLE | SELECTABLE |
| `pool.concurrent_submit` | EFFECTIVE | EFFECTIVE | EFFECTIVE | EFFECTIVE |
| `pool.narrow` | — | OFF | OFF | OFF |
| `pool.nested_dispatch` | EFFECTIVE | OFF | OFF | OFF |
| `pool.spin` | n/a | 4096 | 4096 | 4096 |
| `pool.submit_priority` | OFF | EFFECTIVE | EFFECTIVE | EFFECTIVE |
| `pool.threads` | 4 | 4 | 4 | 4 |
| `prepack.amx` | — | — | — | SELECTABLE |
| `prepack.vnni` | — | — | SELECTABLE | SELECTABLE |
| `q8repack.neon` | EFFECTIVE | — | — | — |
| `region.cp` | see reason | see reason | see reason | see reason |
| `region.cp_batch_head` | see reason | see reason | see reason | see reason |
| `region.cp_frame` | see reason | see reason | see reason | see reason |
| `region.int8_runner` | OFF | OFF | EFFECTIVE | EFFECTIVE |
| `region.talker` | see reason | see reason | see reason | see reason |
| `region.team` | 0 | 4 | 4 | 4 |
| `talker.prefill.f32_blas_fallback` | EFFECTIVE | EFFECTIVE | EFFECTIVE | SELECTABLE |
| `talker.prefill.matmat_bf16` | — | — | — | EFFECTIVE |
| `talker.prefix_cache` | EFFECTIVE | EFFECTIVE | EFFECTIVE | EFFECTIVE |
| `gate.bf16.amx` | — | — | — | EFFECTIVE |
| `gate.bf16.avx512` | — | — | — | EFFECTIVE |
| `gate.int8.amx` | — | — | — | EFFECTIVE |
| `gate.int8.avx2` | — | EFFECTIVE | EFFECTIVE | EFFECTIVE |
| `gate.int8.sdot_mm` | SELECTABLE | — | — | — |
| `gate.int8.vnni` | — | — | EFFECTIVE | EFFECTIVE |
| `gate.q4.amx` | — | — | — | EFFECTIVE |
| `gate.q4.avx2` | — | EFFECTIVE | EFFECTIVE | EFFECTIVE |
| `gate.q4.vnni` | — | — | EFFECTIVE | EFFECTIVE |

## Legacy runtime classes not in the merged host table

The table above is merged from captured per-host dispatch maps and does not yet include a live
AVX-512F-without-VNNI or dotprod-only Linux server map. The source-derived routing is:

| class | B=1 | B>1 | prefill / decoder | qualification |
|---|---|---|---|---|
| `x86_avx512f_no_vnni` | AVX2/FMA GEMV by default; AVX2 and AVX-512BW signed-dot candidates are opt-in | AVX2 INT8/Q4 matmat gates | FP32/BLAS prefill; AVX2 signed-widening decoder INT8 is available with `QWEN_SD_INT8=1 QWEN_SD_RES1_V2=1` and remains default-off | AVX2 leaf parity passed under Rosetta and AVX-512BW compiles; no native no-VNNI server or capacity qualification |
| `x86_avx2` | AVX2/FMA GEMV by default; signed-dot INT8/Q4 candidates are opt-in | AVX2 INT8/Q4 matmat gates | FP32/BLAS prefill; direct AVX2 decoder INT8 requires both flags and remains default-off | Direct-v2 leaf parity passed under Rosetta; no native Linux complete-call performance qualification (the older Milan decoder candidate A/B was slower) |
| `arm_dotprod` without i8mm (for example, Neoverse N1) | SDOT INT8/Q4 GEMV; optional `QWEN_KAI_DOTPROD_GEMV=1` B=1 Q4/INT8 candidate | INT8 SDOT matmat is opt-in; fused Q4 SDOT matmat is opt-in with `QWEN_Q4_SDOT_MM=1`; default Q4 remains B x SDOT GEMV. INT8 otherwise uses fixed-B f32-accum matmat on the normal B>1 path (unless matvec is forced). Full KleidiAI GEMM/regions remain unavailable without i8mm | FP32/BLAS prefill; decoder INT8 and direct DL-4 are available behind opt-in policy gates | M1 microbench rejects INT8 SDOT matmat promotion; fused Q4 parity is covered but performance remains unmeasured. The KAI dotprod candidate has local pack/kernel parity but needs Linux complete-call qualification. N1 has older single-stream results; current v2 stream qualification is open. V1 is i8mm-capable ([Arm reference](https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/neoverse-v1-platform-a-new-performance-tier-for-arm)) and has older SMMLA server tests, but no v2 stream qualification. |

The v2 streaming engine calls the same projection dispatch APIs as other CPU entry points. In a
server run, `QWEN_SERVE_PROFILE=1` prints the resolved map in that process's startup log; a warm
kernel census is still needed to confirm the leaf used by each operation. Source audit and exact
open measurements: [legacy CPU/v2 audit](../.work/legacy-cpu-v2-audit-20260916.md).

## Present on one family only

- `matvec.bf16.dpbf16` — on x86 only
- `matvec.int8.sdot` — on ARM only
- `matvec.int8.vnni` — on x86 only
- `matvec.q4.vnni_variant` — on x86 only
- `pool.narrow` — on x86 only
- `prepack.amx` — on x86 only
- `prepack.vnni` — on x86 only
- `q8repack.neon` — on ARM only
- `talker.prefill.matmat_bf16` — on x86 only
- `gate.bf16.amx` — on x86 only
- `gate.bf16.avx512` — on x86 only
- `gate.int8.amx` — on x86 only
- `gate.int8.avx2` — on x86 only
- `gate.int8.sdot_mm` — on ARM only
- `gate.int8.vnni` — on x86 only
- `gate.q4.amx` — on x86 only
- `gate.q4.avx2` — on x86 only
- `gate.q4.vnni` — on x86 only

## PARITY-3 — ARM/KleidiAI as the oracle

Not "what might x86 be missing" but: take the mature Arm backend feature by feature and locate
the real x86 equivalent. Status is one of EQUIVALENT · DELIBERATELY DIFFERENT · MISSING ·
HARDWARE-BLOCKED · N/A. Cells above come from each build's own dispatch map; the rows below are
the semantic features that are not single dispatch rows.

| ARM/KleidiAI feature | x86 status | evidence / note |
|---|---|---|
| native INT8 GEMV | EQUIVALENT on VNNI, **MISSING** on AVX2 and AVX-512F | `matvec.int8.native` above: `—` on portable. Kernel work, not a gate — P3.6, deferred to the profiler phase |
| native Q4 GEMV | EQUIVALENT on VNNI, **MISSING** on AVX2/AVX-512F | `matvec.q4.native` |
| INT8 GEMM | EQUIVALENT | `gate.int8.avx2` / `gate.int8.vnni` / `gate.int8.amx` |
| Q4 GEMM | EQUIVALENT | q4 gates present on every x86 family |
| fused QKV | EQUIVALENT | Arm `kai_i8_qkv_task`; x86 `qwen_matmat_int8_qkv` and the in-region `qwen_region_i8_run_qkv` |
| shared QKV activation pack | EQUIVALENT | one `qXt`/`sx` and one AMX `pXt` serve Q, K and V; it is why the fused gate is judged on `q+2kv` |
| packed RHS | DELIBERATELY DIFFERENT | KleidiAI packs by construction; x86 prepack is opt-in and MEASURED not to pay by default (X86-3: +4.3 GB, +10.8% TTFA at C=6 for <=7% RTF) |
| persistent packed-RHS lifetime | EQUIVALENT where enabled | built in the parent before fork, inherited by prefork workers through copy-on-write |
| persistent regions | **HARDWARE-BLOCKED on Arm** | `region.int8_runner` is EFFECTIVE on VNNI/AMX and OFF on Arm: the interface exists (`qwen_kleidi_i8_region_usable/_prep/_run`), the region body needs a second gather shape and an Arm i8mm box (P2.7) |
| direct source-row quantization | follows the region | x86-only today because the regions are; unblocks with P2.7 |
| activation-pack reuse across workers | MISSING both sides | each thread packs the same activation; ~5-10% of an AMX projection, worthless at this concurrency (X86-5) |
| small-B strategy / GEMV-vs-GEMM crossover | EQUIVALENT, measured | INT8 AMX now gated on rows-per-thread and B>=3 (X86-2) |
| row/block scheduling | EQUIVALENT | both split output rows across the pool; the decoder conv splits columns and now sizes the panel from the work (X86-4b) |
| pool persistence and spin | DELIBERATELY DIFFERENT | pthread spins (`QWEN_POOL_SPIN`), GCD has no spin loop and now SAYS so via `qwen_pool_flag_inert()` |
| decoder INT8 conv | DELIBERATELY DIFFERENT | kernels exist for Arm dotprod, x86 AVX-512 VNNI and an AVX2 signed-widening direct-v2 leaf. VNNI is default-on; AVX2 and Arm remain policy-gated |
| snake / vectorized activation | EQUIVALENT | one `qwen_snake_activation` for every backend |
| quantization SIMD | EQUIVALENT, contracts differ per platform | NEON and AVX-512 both vectorised; ARM rounds half-to-even, x86 half-away, each internally consistent since P3.11 |
| gather/scatter avoidance | EQUIVALENT at B=1 | both take the per-slot GEMV branch with no staging matrix |
| scratch reuse | EQUIVALENT | grow-once TLS scratch on the common path (P2.4) |
| batch-aware dispatch | EQUIVALENT | one gate table; `matmat.int8.batch_ceiling` is reported and warned about at server start |
| capability reporting / fallback observability | EQUIVALENT | `--dispatch-map`, `--effective-config`, `matvec.*.native`, `matmat.*.family`, `region.*` |
