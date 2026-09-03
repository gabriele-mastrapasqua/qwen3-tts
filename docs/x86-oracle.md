# x86 kernel oracle: oneDNN on a VNNI host

This is a bounded reference experiment, not a runtime dependency and not an engine
integration proposal. It asks how far the current x86 INT8 matrix path is from a mature CPU
backend on the same machine and the same Qwen-shaped cells.

## Setup

- AMD EPYC 9555P, 16 physical cores, one NUMA node, AVX-512 VNNI and BF16, no AMX.
- oneDNN 3.9.1, eight OpenMP threads, implementation reported as
  `brg_matmul:avx512_core_vnni`.
- INT8 cells from the Code Predictor: QKV, gate-up, down and lm-head.
- Warmed repeated calls at `B=1,2,4,8`.

oneDNN reorders the RHS once into `BA16a64b4a` and receives already-quantized INT8
activations. The engine numbers include its normal activation quantization, row-sum handling
and call wrapper, so the table is a diagnostic comparison, not an apples-to-apples product
benchmark.

## ARM design oracle: what transfers structurally

The mature ARM paths are useful as a dataflow oracle, not as an intrinsic porting template.
They solve the same small-`B` problem with layout, reuse and separate GEMV/GEMM dispatches:

| concern | ARM implementation | closest AVX-512/VNNI choice |
|---|---|---|
| `B=1` GEMV | SDOT fuses four output rows (then a two-row cleanup) while reusing one activation vector across the row loads | Keep the VNNI GEMV multi-row strategy (`MR=2`, with `MR=4` still experimental); tune it against complete GEMV cost, not just the dot instruction |
| `B=2` matmat | Generic SMMLA uses a compact two-output-row by two-batch cross-product tile; it does not assume a deep `M=8` tile | The opt-in `QWEN_VNNI_TILE_M4N2=1` candidate removes inactive `N=4` accumulators from the current `M4xN4` helper. It is a structural experiment, not a new default |
| larger `B` | SMMLA grows to an `M4xN4` tile; KleidiAI selects a separate GEMM kernel with library-defined `MR/KR/SR` | Keep `M4xN4`/secondary `N8` shape classes separate from the B2 route; do not infer the best `M` from ARM's `SMMLA` register geometry |
| RHS lifetime | KleidiAI packs the RHS once at registration/load time | The VNNI prepack experiment is the analogous mechanism. It changes weight layout but did not yet produce a stable server win, so it remains opt-in |
| activation lifetime | KleidiAI quantizes and packs the LHS per call, choosing different pack geometry for GEMV and GEMM | `qXt` is already built per call for VNNI. A future improvement must show that a new layout reduces the complete quantize+compute+store path |
| row reuse | GEMV reuses each loaded activation vector for several output rows; SMMLA reuses paired activation vectors across paired weight rows | VNNI `vpdpbusd` should keep activation loads shared across rows. The B2 candidate follows this principle without translating `SMMLA` literally |
| QKV reuse | KleidiAI packs one LHS and schedules the Q/K/V output tiles from that shared input | The x86 fused QKV wrapper already shares `qXt` and partitions the combined output; changing this needs route and end-to-end evidence |
| threads/barriers | SMMLA partitions output rows and aligns boundaries to its row tile. KleidiAI partitions output-N tiles and falls back to one thread when the matrix is too small | The x86 VNNI task partitions output rows and calls the persistent pool once per projection. The pool join is a per-call barrier; a B2 kernel must not add another synchronization point |

The dispatch rule is therefore the important transfer: ARM uses GEMV at `B=1`, a compact
batched path beginning at `B=2`, and a different packed GEMM path for larger batches. The
AWS C4 census has now observed actual `B=2` VNNI calls in Talker, Code Predictor and speech
decoder during a true-wave `C=4`; this makes a B2 candidate relevant to serving, while the
measured effective batch remains a separate runtime field and is never inferred from client
concurrency.

The ARM inspection does not support claiming that `M8xN2` is automatically better. The
current candidate is deliberately `M4xN2`; an `M8xN2` implementation would require a new
assembly/spill check and a separate A/B. In particular, the generic ARM B2 fallback is
closer to `M2xN2`, while the mature KleidiAI advantage comes from its complete packed-RHS,
packed-LHS and output-tile contract rather than from one intrinsic's nominal tile size.

## ARM-guided B2 candidate: screening result

`QWEN_VNNI_TILE_M4N2=1` is retained as an opt-in candidate on the AWS VNNI host. It was
tested on the OSS 1.7B base model with the same 2x8 server configuration used for the
control. This is a screening experiment on a dirty tree, not a production qualification.

| path | M4xN4 control | M4xN2 candidate | interpretation |
|---|---:|---:|---|
| Talker QKV, `B=2`, 8 pinned threads | 16.34 us | 16.05 us | about 1.8% local improvement |
| Code Predictor QKV, `B=2`, 8 pinned threads | 9.04 us | 8.52 us | about 5.8% local improvement |
| server C=1 TTFA p50 | 99 ms | 99 ms | no regression |
| server C=4 TTFA p50 | 187 ms | 189 ms | neutral within screening variance |
| server C=4 TTFA p95 | 246 ms | 233 ms | lower tail in this campaign |
| server C=4 total RTF | 1.027 | 1.027 | no end-to-end improvement established |

The microbenchmark used five separate processes per arm, 64 warmed samples, and one
8-thread CPU range per process. The server screen used three OFF/ON rounds, three true-wave
waves per round, the short diverse bank, explicit runtime flags, and no errors or rejects.
The candidate route was independently proven with `QWEN_SHAPE_CENSUS=1`: B=2 calls reached
the M4N2 entry in Talker, Code Predictor and speech decoder. A separate parity control also
covered row counts (`rows=1,3,5,17,517`) and K tails (`K=65,127,1056`) at B=2; the integer
path was exact in those cases. That verifies dispatcher output correctness, but does not prove
that M4N2 itself handles every partial or misaligned worker range: ranges such as `[1,5)` and
`[3,7)` with output canaries remain a targeted control. Because the server **total RTF** did not
move and the p95 campaign is still a screening sample, the flag remains off by default.

The server rows above report `total RTF`, not the frozen per-request `STREAM_RTF` KPI. They
support total-RTF neutrality only; no production streaming claim is made from this screening.

The server screening artifact is deliberately not a qualification baseline:

```text
model=qwen3-tts-1.7b-base (OSS)
benchmark_family=TRUE_SIMULTANEOUS_WAVE  workload_class=PARALLEL_SHORT_DIVERSE
harness=serve_parallel_wave.py  topology=2x8  prefork_workers=2
threads_per_worker=8  batch_size=8  concurrency=1,4  waves=3
source_commit=UNKNOWN  dirty=yes  source_diff_sha256=NOT_RECORDED
binary_sha256=e592b5512bc480c01e40b02736f02d20c347755bce2159162cb64f07969a0443
effective_batch_at_C4=2.981  worker_B=1.51,1.47
runtime_profile=OPENBLAS_THREAD_TIMEOUT=1,QWEN_CP_PREFILL2=1,
  QWEN_DECODER_BATCH=0,QWEN_POOL_SPIN=4096,QWEN_PREFILL_MATMAT=1,
  QWEN_PREFIX_CACHE=1,QWEN_STREAM_DECODE_CHUNK=8,
  QWEN_STREAM_DECODE_CHUNK_BUSY=0,QWEN_VNNI_GEMV_MR=2,
  QWEN_VNNI_TILE_N8=0,QWEN_NO_VNNI_TILE=0,
  QWEN_VNNI_TILE_M4N2=(0 for control | 1 for candidate)
all other declared runtime flags=(default)
```

Since the tree was dirty and its source-diff hash was not captured, this
screening cannot replace a clean, fully identified baseline.  The measured
effective batch is included to show that `C=4` is not synonymous with
`engine_batch=4`.

The next B2 experiment should target the larger dataflow gap (packed/block-reused RHS or
activation/epilogue cost) rather than widening the row tile again without evidence.

## Warm kernel cells

Values are nanoseconds per call; lower is better.

| component / shape | B | engine | oneDNN |
|---|---:|---:|---:|
| CP QKV, 4096×1024 | 1 | 6,105 | 6,784 |
|  | 2 | 8,721 | 8,010 |
|  | 4 | 11,884 | 8,585 |
|  | 8 | 29,621 | 12,482 |
| CP gate-up, 6144×1024 | 1 | 8,818 | 9,220 |
|  | 2 | 11,957 | 12,955 |
|  | 4 | 15,802 | 12,260 |
|  | 8 | 41,437 | 35,153 |
| CP down, 1024×3072 | 1 | 4,596 | 4,283 |
|  | 2 | 9,801 | 4,989 |
|  | 4 | 15,410 | 7,849 |
|  | 8 | 30,140 | 10,633 |
| CP lm-head, 2048×1024 | 1 | 3,313 | 3,464 |
|  | 2 | 5,392 | 4,751 |
|  | 4 | 7,843 | 5,368 |
|  | 8 | 15,162 | 6,455 |

## AWS B=2 decomposition (diagnostic, not a qualification)

The same CP-down shape was measured on the AWS non-AMX VNNI host after the
`QWEN_VNNI_TILE_M4N2` candidate had been exercised by the server.  This is a
kernel-shaped diagnostic, not a serving result: it does not replace the
AWS C1/C4 true-wave or realistic-workload measurements.

Host identity recorded by the run:

```text
CPU: AMD EPYC 9R45
logical CPUs: 16  physical cores: 16  threads/core: 1  NUMA nodes: 1
ISA build: AVX512 VNNI + AVX512 BF16 (SIMD profile: avx512bf16)
candidate binary SHA256: e592b5512bc480c01e40b02736f02d20c347755bce2159162cb64f07969a0443
```

Shape: CP `down`, `K=1024`, `N=3072`, `B=2`, eight worker threads, warm
measurement.  `kernel_ns` includes the engine's per-column activation
quantization, VNNI compute, worker dispatch and the engine epilogue; it is not
raw `vpdpbusd` time.  The harness's separate `lhspack_ns=0` means that no
separate LHS-pack phase was exposed, not that activation preparation was free.

| M4N2 candidate | gather ns | LHS pack ns | complete engine call ns | output ns | synthetic drop-in ns | GMAC/s | relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 1,639 | 0 | 10,161 | 931 | 12,731 | 619.2 | 5.579e-3 vs F32; 4.050e-3 vs Qwen |
| on | 1,585 | 0 | 10,046 | 709 | 12,340 | 626.2 | 5.579e-3 vs F32; 4.050e-3 vs Qwen |

This single bounded pair is a candidate signal, not a promoted end-to-end
claim: the complete engine call improves by about 1.1% and the synthetic
gather+call+scatter path by about 3.1%.  The candidate remains opt-in until a
replicated server campaign shows a stable C1/C4 benefit.

## Bounded external-oracle survey

This is a source/design survey, not a benchmark result.  No external runtime
was installed on the AWS host and no Qwen3-TTS quality or serving support is
implied.  The purpose is to identify transferable mechanisms before writing
another custom kernel.

The source revisions were recorded before this survey was written (2026-09-03):

```text
oneDNN v3.9.1       80a3a8e745d2f0186e674b0af9332fd6e074c94f
KTKernel main       31985f40bcc40da08107efdb1f81bf88cb38c6b2
OpenVINO master     9b1d5c9494e838d42b5ed90d662c8ce84e5742f8
AutoRound/ARK main  b48c9db6744e66546e8d9e80715dc6a2d3f76501
```

| oracle | optimization level | applicable Qwen path | transferable idea | expected ceiling | experiment needed |
|---|---|---|---|---|---|
| [oneDNN BRGeMM](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_brgemm.html) | packed RHS, B/K/N blocking, integer accumulation and post-ops | CP/Talker/decoder batched matmul, especially B=2..8 | Pack the RHS in the kernel-required layout for its lifetime; block B/K/N together; keep post-ops close to the accumulator; align per-thread scratch to avoid sharing | Large host-local backend gap observed on Scaleway; transferable ceiling is unknown, and only the measured matmul share can move the server | Finish the CP-down B=2 complete-path split, then test one candidate blocking change on the exact server shape |
| [KTKernel CPU backends](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md) | ISA/backend selection plus CPU execution policy; the public material does not establish a Qwen3-TTS-specific dense VNNI result | non-AMX AVX512/VNNI B=1 and small-B paths, and larger B where reuse exists | Keep low-intensity AVX512 and AMX as separate decisions; inspect any tile-aware/aligned weight transform; use physical-core/NUMA-aware pools and work stealing where the implementation supports it | A design oracle for B=1/B=2 dataflow and large-B packing, with no transferable speedup established yet | Read the native source/layout and benchmark one matching shape only if the dependency is already available; do not install a runtime for this survey |
| KTKernel oneDNN backend | backend selection around INT8 BRGEMM | non-AMX AVX512 B>1 | It provides a concrete native-vs-oneDNN comparison point and makes backend choice explicit instead of assuming that every B>1 wants the same kernel | Oracle only until same-host numbers exist | Compare on a host with oneDNN already present; do not transfer Scaleway measurements to AWS |
| [OpenVINO CPU](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.rst) plus its [CPU graph](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/graph.h) | graph compilation, low-precision transforms, conditional layout/reorder handling, threading and perf counters | repeated QKV/MLP linear paths and static representative shapes | Inspect whether matching descriptors make a reorder unnecessary; cache compiled low-precision transforms; use physical-core/NUMA-aligned streams; inspect the selected primitive rather than inferring it from the model | Potentially larger than a kernel-only gain if transforms/reorders disappear, but unquantified for Qwen3-TTS and not established by the cited docs alone | Build or inspect one toy graph with the CP-down shape, record primitive/reorder counters, and compare complete transformed-path cost |
| [AutoRound](https://github.com/intel/auto-round) | offline quantization policy | model-level or component-level weight policy, not a runtime kernel | Use per-layer/component schemes and calibration matched to deployment; protect sensitive components such as output heads when evidence requires it | Unknown: fewer bytes could help bandwidth, but quality and Qwen3-TTS import compatibility are unproven | Separate OSS-only quantization/quality study; no precision change in this branch and no serving claim |
| [AutoRound Kernel (ARK)](https://github.com/intel/auto-round/blob/main/auto_round_extension/ark/README.md) | packed weight-only linear operators; CPU INT8 compute can include dynamic activation quantization and floating-point dequantization | a future model-representation experiment for selected Qwen linear components, not the current shipped INT8 path | Inspect the complete repack → quantize activation → low-bit GEMM → scale/dequant dataflow; compare its packed-weight lifetime and `woqgemm_s8` contract, without assuming its layout maps to this engine | Unknown: it could reduce weight traffic, but it changes the quantization contract and is validated for Intel CPUs, not Qwen3-TTS | Read-only source audit first; later, reproduce one compatible linear shape and run OSS audio/quality checks before considering any precision change |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | W8A8/GPTQ/SmoothQuant policy recipes | only if Qwen3-TTS tensors and quality path are supported | Compare channelwise weights, dynamic per-token activation quantization and selective exclusion of sensitive layers as policy ideas | Unknown and possibly offset by activation-quant overhead; not a kernel conclusion | First prove model support, calibration behavior and audio quality on OSS; defer all model-precision changes |

### Source-level details behind the matrix

These are implementation observations from the public sources, not measurements
on the AWS host:

- **KTKernel:** the public API can load pre-quantized CPU weights or quantize
  from tensors, pre-allocate buffers for explicit batch sizes, select physical
  CPU threads and a NUMA-pool count, and choose `KT_INT8_VNNI_BACKEND=auto|onednn|native`.
  Its worker-pool source binds each worker to a single logical CPU in that
  core's cpuset, claims work through an atomic task index, waits for all tasks
  at the call boundary, and spins for roughly 50 ms before falling back to a
  condition variable. These are concrete scheduling/layout observations, not
  evidence that its dense
  kernels match Qwen3-TTS.
- **oneDNN BRGeMM:** the example divides K into blocks, accumulates into the
  same C buffer, and uses a final one-call kernel for post-ops and scales. Its
  pack transform makes a special low-precision/VNNI B layout and the packed
  buffer can be retained across executions. This supports testing a lifetime
  and blocking contract; it does not identify which part caused the
  cross-machine cell gap.
- **OpenVINO CPU:** the graph API inserts a reorder when descriptors or
  in-place constraints require it, and can mark a reorder as optimized/no-op.
  That is a precise layout-propagation idea to inspect, not proof that a
  Qwen3-TTS graph would eliminate its quantization or copies.
- **ARK:** the CPU API exposes repacked weight blobs, `woqgemm`, and
  `woqgemm_s8`; its documentation says CPU INT8 compute may include dynamic
  activation quantization and floating-point dequantization. This is a
  representation/dataflow oracle only: it is weight-only quantization, not a
  drop-in replacement for the current W8A8 engine path.

### What the survey says about the batching hierarchy

The external systems do not justify one global `B` threshold.  The next
experiments must preserve these separate regimes:

| regime | Qwen question | current implication |
|---|---|---|
| `B=1` | Is complete GEMV cost dominated by activation quantization, weight traffic, dot product or epilogue? | Keep the competitive VNNI GEMV path; do not send it through BRGeMM merely because AMX/VNNI matmul exists. |
| `B=2` | Can two activation rows reuse each RHS load without paying disproportionate setup? | Keep `QWEN_VNNI_TILE_M4N2=1` as the opt-in candidate and compare complete warm paths, not only the dot kernel. |
| `B=3..4` | Does the current M4N4 path have the same reuse/blocking gap as the oneDNN cells? | Treat it as its own micro-GEMM/BRGeMM class; do not extrapolate from B=2. |
| `B=5..8` | Does blocked-B/N8 amortize RHS and epilogue cost under real worker assignment? | N8 is secondary and must be checked against C1/C4/C8, because the server's effective B is not the client C. |
| large `B` / AMX | Can AMX keep tiles full while reducing DRAM traffic? | Separate regime and separate qualification; AWS VNNI observations do not prove an AMX result. |

The common structural ideas worth testing are therefore: persistent
weight-side preparation, per-call activation preparation only when necessary,
reuse of one RHS tile across several activation rows, explicit layout
propagation, and topology-aware worker ownership.  The correct transfer is the
dataflow, not a literal port of ARM `SMMLA` or an external runtime.

## Ranked gap list

The ranking is intentionally conditional.  The existing `33% matmat / 36% GEMV /
19% solo` census is an operation-share census, not a time-share or byte-share
census, so it cannot by itself establish which path deserves the next rewrite.

| rank | gap or hypothesis | current evidence | status | cheapest discriminator |
|---|---|---|---|---|
| A | B=1 GEMV/single-slot dataflow and weight traffic | B=1 is close to oneDNN in the tested cells, but the comparison includes different activation-preparation contracts; server census says GEMV is substantial | Highest potential, conditional on wall-time/bytes | Per-request C4 timing split plus effective GEMV GB/s, activation-quantization time and bytes read |
| A | B=2/B>1 RHS reuse and blocking | oneDNN is much faster in several B>1 cells, but the AWS M4N2 complete-path delta is only about 1.1% on CP down | Worth one bounded experiment only if B=2 has material wall-time share | Attribute one real C4 shape, then compare one blocked/persistent-RHS variant end to end |
| B | `QWEN_VNNI_TILE_M4N2` geometry | parity and route proof pass; local CP-down signal is small and server RTF is unchanged in the screening campaign | Retain opt-in; not default | Replicated C1/C4 campaign after a dataflow change, not another tile-width sweep |
| B | N8/wider small-B tiles | Some isolated B=8 cells have room, but effective B depends on worker assignment and C1 can regress | Secondary, local-to-component until census supports it | Measure B=4/8 wall-time share and C1/C4/C8 together |
| C | Raw B=1 VNNI dot kernel is broadly deficient | Tested Scaleway cells do not show a large raw-call deficit, although they are not same-host complete-path comparisons | No quick GEMV-kernel rewrite justified yet | Same-host oracle only if the external backend is already available |
| D | Integrating oneDNN, KTKernel, OpenVINO or changing model precision | Outside the engine scope; no Qwen3-TTS quality/import evidence | Not applicable to this branch | None in the current session; keep these as design or quantization oracles |

This keeps the next decision tied to the full serving profile: B1 latency,
B2 small GEMM, B3/4, B5..8, and large-B/AMX remain separate regimes.  A local
win in one regime is not promoted if it worsens C1, C4 or C8, and a result from
Scaleway is never used as an AWS A/B without reproducing it on AWS.

Primary design references used for this survey include the [KTKernel CPU
README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md),
the [KTKernel AMX design notes](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md),
the [KTKernel worker-pool implementation](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.cpp),
the [OpenVINO CPU graph](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/graph.h),
the [llm-compressor W8A8 example](https://docs.vllm.ai/projects/llm-compressor/en/stable/examples/quantization_w8a8_int8/),
and the [llm-compressor AutoRound example](https://docs.vllm.ai/projects/llm-compressor/en/stable/examples/autoround/).

## What this answers

At `B=1`, the measured engine call is within roughly ten percent of oneDNN on these cells,
with oneDNN ahead on CP down. This is not an apples-to-apples GEMV proof: the engine timing
includes its dynamic activation quantization, row-sum handling and wrapper, while oneDNN
receives pre-quantized activations. It does not support a broad 20–40% deficit in the complete
measured cell, but it does not isolate the raw dot kernel or the full serving path either.

At `B>1`, oneDNN is often faster, especially on CP down and the larger QKV cells. Its packed
RHS layout and blocked `B` path are candidate explanations to test, not established causes of
the gap. It is not yet evidence that the same gain transfers to the server, because the engine
also pays dynamic activation quantization and the server's scheduling/worker costs.

The server census explains why this oracle is not sufficient by itself: only about 33% of the
counted work is INT8 matrix-matrix, while about 36% is INT8 GEMV and 19% is solo/single-slot
work. A large batched-matmul win cannot move all of C=4 by the same percentage.

## Decision

The oneDNN result justifies keeping the packed VNNI layout as an opt-in experiment and studying
its `BA16a64b4a`-style blocking. It does not justify replacing the engine or adding oneDNN as a
runtime dependency. The parent-side CP prepack candidate preserved audio exactly but did not
improve the measured C=4 server objective, so it remains outside the Scaleway deployment
profile.

The bounded source survey is complete without installing another runtime.  Before choosing B=2
as the next implementation priority, measure a time-weighted, per-request C4 decomposition of
quantization, B=2/B>2 matmul, GEMV, solo work, epilogue and scheduling.  If B=2 has material
wall-time share, transfer one relevant RHS blocking/reuse idea and re-run the C1/C4 smoke;
otherwise return to B=1 GEMV and single-slot weight traffic.  No result from KTKernel or
OpenVINO is claimed here, and neither is a runtime dependency.

See [`reference-scaleway-16c-vnni.md`](reference-scaleway-16c-vnni.md) for the server tables and
[`x86-optimization.md`](x86-optimization.md) for the dispatch and profile guidance.
