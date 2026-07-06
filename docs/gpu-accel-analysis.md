# Optional GPU acceleration — backends, dependency weight, realistic gains (2026-07-02)

*Analysis only. Ground rule: the project is and stays **CPU-first, minimal, plain-Makefile**. GPU
backends live on experimental branches, are opt-in build targets, and must not change `make blas` at all.
TODOs in `plan_v4.md` §E4.*

---

## 0. The seam we already have

Everything hot funnels through `qwen_matvec_{bf16,int8,q4_0}[_qkv]` / `qwen_matmat_*` in
`qwen_tts_kernels.c`; BLAS is a 2-line Makefile block (`-framework Accelerate` / `-lopenblas`). That is
the integration pattern to copy. **But the offload seam must NOT be per-matvec**: per-op upload/sync
ping-pong (µs per dispatch) kills a matvec-bound decode. The llama.cpp lesson is unambiguous — **weights
+ KV resident on the device; transfer only token IDs in and logits/frame-codes out; encode a whole step
as one command buffer / CUDA graph.**

Proposed seam (one level above the kernels):

```
qwen_tts_backend.h    — tiny vtable: init/free, talker_step, cp_forward (16 passes fused),
                        decoder_forward, capability query. Default impl = current CPU code (zero change).
qwen_tts_metal.m      — built only by `make metal` (clang, -framework Metal -framework Foundation)
qwen_tts_metal_kernels.metal — matvec_bf16/int8, matmat, rmsnorm, rope, attn, conv1d (runtime-compiled)
qwen_tts_cuda.c       — cuBLAS-first backend (gcc-compilable; optional dlopen of libcublas)
```

Offload unit = whole subgraph: (a) Talker step + CP 16-pass block (sampling stays CPU — logits are tiny),
(b) speech-decoder ConvNet, (c) batched prefill/matmat for the server. Orchestration/tokenizer/WAV stay CPU.

## 1. Apple Metal (direct) — first backend, zero dependencies

- **Zero added deps**: one `.m` TU compiled with clang (`-fobjc-arc`) + `-framework Metal -framework
  Foundation`. Same weight class as the existing Accelerate link. gcc-11 can't build ObjC → compile just
  that TU with clang (fallback already exists in the Makefile).
- **Shaders**: ship a `.metal` source in-repo, compile at runtime via `newLibraryWithSource:` (llama.cpp's
  `ggml-metal.m` pattern; embed a metallib later only if it ships).
- **Zero-copy weights**: `newBufferWithBytesNoCopy` accepts our mmapped safetensors region (mmap is
  page-aligned; wrap the whole file as ONE MTLBuffer, round length up to 16 KB pages, address tensors by
  offset, `MTLResourceStorageModeShared`). Exactly llama.cpp's GGUF mapping. Apple-Silicon-only (on
  Intel+dGPU this becomes a PCIe trap — fine to declare unsupported).
- **MPS/MPSMatrix**: tuned GEMM without writing shaders, but fp16/fp32 only (no bf16) and poor for the
  matvec decode path; llama.cpp deliberately does NOT use MPS. Use hand-written matvec kernels; MPS at
  most for prefill GEMM experiments.
- **Honest expectations on M1 (the key point)**: CPU and the 8-core GPU share the same **68 GB/s** —
  single-stream decode is bandwidth-bound, so **GPU matvec ≈ CPU matvec** (llama.cpp M-series data: TG
  scales with bandwidth, not GPU cores; our own past Metal experiment measured 1.3× *slower*). Where the
  GPU genuinely wins on M1:
  1. **Prefill / batched matmat** (compute-bound → GPU PP ≫ CPU PP) — server `--batch-size` and TTFA.
  2. **The ConvNet speech decoder** (compute-heavy convs, GPU-friendly) — and offloading it *frees CPU
     DRAM bandwidth* for Talker/CP.
  3. **CPU/GPU overlap**: decoder on GPU while CPU runs the next frame's Talker/CP — a real pipeline win
     specific to our architecture.
  Realistic M1 numbers: GPU-matvec alone ~1.0–1.2× (possibly negative vs int8 CPU); decoder-offload +
  overlap ~**1.2–1.5× end-to-end**; batched server **2–4×**. Frame the branch goal accordingly.

## 2. Apple MLX — skip

MLX is a C++ array framework; the C binding **mlx-c** is real but young, and using it means vendoring
**libmlx + libmlxc, CMake, C++17, its own shader build** into a plain-Makefile C project. The belief "MLX
lets you write tensors in its format without adding dependencies" conflates *no third-party runtime deps
beyond macOS* (true — it's Metal underneath) with *no build dependency* (false). For surgically offloading
3 op types, direct Metal is strictly leaner. Steal kernel ideas from MLX (MIT) if useful.

## 3. CUDA — biggest absolute win, cuBLAS-first

- **Minimal path needs NO nvcc**: cuBLAS/cuBLASLt are host-side C APIs → a plain-C TU compiled with gcc +
  `-lcublas -lcudart`. `cublasGemmEx`/Lt do **BF16 in/out, fp32 accumulate** directly. nvcc only the day
  we write custom `.cu` kernels.
- **Binary stays dep-free via dlopen**: `dlopen("libcublas.so")`+`dlsym` a handful of symbols inside
  `qwen_tts_cuda.c` (ggml/llamafile's `GGML_BACKEND_DL` pattern) → the same binary runs on GPU-less boxes
  and prints "CUDA backend unavailable". `make blas` untouched; `make cuda` adds one TU.
- **Arch support**: CUDA 13 dropped Maxwell/Pascal/Volta → **Turing (sm_75) is the floor** (Pascal needs
  CUDA ≤12.x). bf16 native from **Ampere (sm_80)**; on Turing fall back to fp16-convert or int8. Blackwell
  is two families: sm_100/103 (B100/B200) vs **sm_120 (RTX 5090, RTX PRO 6000) / sm_121 (a ~270 GB/s unified-memory NVIDIA part,
  aarch64 Grace)**. cuBLAS-only integration dodges all gencode pain (NVIDIA ships the kernels).
- **Expected wins** (1.7B bf16 ≈ 3.4 GB/step, decode bandwidth-bound):
  - **RTX 5090 / PRO 6000** (~1.79 TB/s): ~26× M1 bandwidth; watch kernel-launch overhead (16 sequential
    CP passes/frame = hundreds of tiny launches → CUDA Graphs). Realistic end-to-end **5–15×** vs M1 CPU
    ⇒ RTF ~0.1–0.3 and huge batched-server headroom.
  - **a ~270 GB/s unified-memory NVIDIA part** (273 GB/s LPDDR5x, cache-coherent unified memory): only ~4× M1 bandwidth —
    llama.cpp there confirms dense decode is bandwidth-capped. Expect **~3–4× single-stream** (RTF
    ~0.3–0.5 bf16), excellent prefill/batch. Unified memory ⇒ the mmap-and-share trick works
    (`cudaHostRegister`/HMM), echoing the Metal story.

## 4. AMD ROCm/HIP and Vulkan — later, on demand

- **ROCm/HIP**: multi-GB install, narrow official GPU list, version-fragile. HIP is a ~mechanical port of
  the CUDA TU → do it only *after* CUDA exists, never first. Phoronix (2025-11) even shows RADV Vulkan
  beating ROCm 7.1 on several llama.cpp TG tests.
- **Vulkan**: broadest coverage (AMD/NVIDIA/Intel, no vendor stack — just driver + loader), and llama.cpp's
  backend is now competitive. Build-time-only shader toolchain (glslc → SPIR-V committable to the repo, end
  users need only libvulkan). BUT it's by far the most code (descriptor/pipeline plumbing, ~weeks). Only if
  cross-vendor demand materializes.

## 5. Decision table

| Backend | Added deps | Build impact | Effort | Single-stream gain | Notes |
|---|---|---|---|---|---|
| **Metal direct** | none (macOS SDK) | +1 `.m` +1 `.metal`, 2 Makefile lines | M (4–8 d; +2–3 d decoder) | M1 ~1.0–1.2× (honest) | decoder-offload+overlap 1.2–1.5× e2e; batch 2–4×; forces the resident-weights architecture every backend reuses |
| MLX / mlx-c | libmlx+libmlxc, CMake, C++17 | heavy | M–L | same silicon ceiling | **skip** |
| **CUDA cuBLAS-first** | toolkit on the GPU box only; dlopen ⇒ binary dep-free | +1 `.c`, gcc-compilable | S–M (2–4 d prefill/batch) → M–L (8–15 d full resident decode + CUDA Graphs) | 5090: 5–15×; ~270 GB/s-class: 3–4× | Turing+ (CUDA 13); bf16 sm_80+ |
| ROCm/HIP | multi-GB stack | hipify of CUDA TU | S after CUDA | ≈CUDA-class on supported HW | port only |
| Vulkan | libvulkan; glslc build-time (SPIR-V committable) | most code | L (2–4 wk) | between ROCm and CUDA | on demand |

**Recommended order**: (1) backend seam + **Metal** on an experimental branch — developable on the dev M1,
zero deps, realistic goals = decoder-offload/overlap/batch/prefill; (2) **CUDA cuBLAS-first with dlopen**
— smallest code for the largest numbers, validated on rented boxes per the existing `bench-matrix`
workflow; (3) HIP port if an AMD box appears; (4) Vulkan only on demonstrated demand.

Key references: llama.cpp `ggml-metal.m` + Metal backend internals (deepwiki.com/ggml-org/llama.cpp),
apple/metal-cpp, ml-explore/mlx-c, llama.cpp M-series bench (discussion #4167), CUDA 13 release notes,
unified-memory llama.cpp (discussion #16578), llama.cpp Vulkan (discussion #10879), Phoronix ROCm-vs-Vulkan
(2025-11), machinethink.net MPS matmul.
