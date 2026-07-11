# Hardware testing guide — SIMD extensions, where to rent, what to benchmark

This is the playbook for validating qwen-tts on real CPUs: **which boxes to rent**, **what
SIMD each one has**, **how to check the extension actually fires**, and **what to benchmark**
(RTF in single / batch / streaming / server). It generalizes the Turin/EPYC validation
("did AVX-512/VNNI fire?") to every platform we care about.

> **Why this matters.** Single-stream RTF on this engine is largely **memory/cache-bound** with
> a compute floor set by the SIMD path. Different CPUs move the floor: M1 is compute-bound with
> high per-core bandwidth (so int4/heavy-unpack loses, bf16+batch wins ~1.7×); server x86 is
> throughput-oriented (lower per-core bandwidth → batching + int8/int4 + VNNI/AMX is the play).
> The only way to know is to run the checks below on the actual silicon.

---

## Make targets / commands quick reference

| command | what it does |
|---|---|
| `./qwen_tts --caps` | runtime SIMD-extension detection (ARM dotprod/bf16/i8mm/SVE/SME, x86 AVX-512/VNNI/BF16/AMX) + the `lever:` line — "does it fire?" |
| `./qwen_tts --self-test` | cross-ISA kernel correctness oracle (bf16/int8/int4 matmul + matmat twins vs f32 ref). Run twice: native, then `QWEN_NO_SDOT=1 QWEN_NO_VNNI=1` for the fallback |
| `make matmat-bench` | batched matmat twins (`qwen_matmat_{bf16,int8,q4_0}`) vs B×matvec, per precision/threads (no model) |
| `make bench-matrix` | **the per-box report**: caps + self-test (native+fallback) + matmat-bench + RTF matrix single/batch × bf16/int8/int4 |
| `make bench-matrix-full` | same + streaming + **server request-batching throughput** |
| `make bench-server` | **server request-batching THROUGHPUT** alone (M concurrent clients vs single-stream, per precision) — the vLLM-style lever |
| `make test-serve-all` | server **correctness** gate (incl. batching: per-request mel 1.0, continuous admission, streaming compose) |
| `make check-isa` | compile-check the newer-ISA kernel paths (BFMMLA/SMMLA/SME ; VNNI/BF16/AMX) on the dev box, before the hardware exists |

Scripts (copy onto any rented box): `tests/bench_matrix.sh <model> [--full]`,
`tests/serve_batch_bench.sh <model> [port] [batch_N] [clients_M] [threads]`.

### The 2-command rented-box workflow

```bash
make blas && ./qwen_tts --caps && ./qwen_tts --self-test   # build + does-it-fire + correct?
make bench-matrix-full                                       # ALL numbers: RTF single/batch/stream + server-batching throughput
```
Optionally `make test-serve-all` once for the server correctness gate. Paste the
`bench-matrix-full` block + `--caps` into §5 below. Quiet machine only.

---

## 0. The 3-command check (run first on any new box)

```bash
make blas                      # build (macOS/Linux ARM: -march=native; x86: see §4)
./qwen_tts --caps              # what the CPU HAS at runtime + what the binary was built for
./qwen_tts --self-test         # cross-ISA kernel correctness (bf16/int8/int4 matmul vs f32 ref)
QWEN_NO_SDOT=1 QWEN_NO_VNNI=1 ./qwen_tts --self-test   # same, scalar/fallback path
```

- **`--caps`** is the "does it fire?" check. It prints the **runtime** ISA (e.g. `NEON dotprod/SDOT
  bf16/BFDOT i8mm/SMMLA` on an M2+, or `sse2 avx avx2 fma avx512f avx512vnni` on Zen4+) *independent
  of how the binary was compiled*, plus a `lever:` line naming the throughput path. A gap between
  "compiled" and "runtime" is the SIGILL trap (built past what the CPU has).
- **`--self-test`** proves the dispatched kernels are numerically correct on THIS ISA (immune to the
  greedy audio trajectory-fork that makes cross-ISA mel-corr a false alarm). Run it twice (native +
  fallback) to A/B the dispatch. PASS = the SIMD path is safe to trust.
- **`--matmat-bench`** (next section) times the batched matmat twins per precision.

If `--caps` shows an extension your build didn't compile in, rebuild with the right `-march`
(§4) to actually use it.

---

## 1. What each platform has (ARM)

Apple does **not** expose SVE for general use; its matmul lever is **bf16 (BFDOT/BFMMLA)** + **i8mm
(SMMLA)** from M2 on, and **SME/SME2** from M4 on. Server ARM (Neoverse) adds **SVE/SVE2**.

| CPU | Arch | dotprod (SDOT) | bf16 matmul | i8mm (SMMLA) | SVE / SVE2 | SME | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Apple M1** (Pro/Max/Ultra) | Armv8.5 | ✅ | ❌ | ❌ | ❌ | ❌ | our dev box; bf16 decode is **scalar** |
| **Apple M2** | Armv8.6 | ✅ | ✅ | ✅ | ❌ | ❌ | first Apple bf16+i8mm → native matmul lever |
| **Apple M3** | Armv8.6 | ✅ | ✅ | ✅ | ❌ | ❌ | same CPU SIMD as M2 |
| **Apple M4** | Armv9.2 | ✅ | ✅ | ✅ | ❌¹ | ✅ | adds **SME/SME2** (matrix engine) |
| **Apple M5** | Armv9.x | ✅ | ✅ | ✅ | ❌¹ | ✅ | newest; SME2 + GPU neural accel (verify w/ `--caps`) |
| **AWS Graviton2** (Neoverse N1) | Armv8.2 | ✅ | ❌ | ❌ | ❌ | ❌ | dotprod only, like M1-class |
| **AWS Graviton3** (Neoverse V1) | Armv8.4+ | ✅ | ✅ | ✅ | ✅ (256b) | ❌ | bf16+i8mm+SVE — strong ARM server |
| **AWS Graviton4** (Neoverse V2) | Armv9.0 | ✅ | ✅ | ✅ | ✅ SVE2 | ❌ | newest AWS ARM |
| **NVIDIA Grace** (Neoverse V2) | Armv9.0 | ✅ | ✅ | ✅ | ✅ SVE2 | ❌ | GH200; very high bandwidth |
| **Ampere Altra** (Neoverse N1) | Armv8.2 | ✅ | ❌ | ❌ | ❌ | ❌ | dotprod only (Oracle/Hetzner/Scaleway) |

¹ Apple implements SME's streaming-SVE internally but does not expose classic SVE to user code; treat
**SME** as the M4/M5 matrix lever. Always confirm with `--caps` on the actual chip.

**Reading it:** dotprod-only chips (M1, Graviton2, Altra) run the bf16 matmat with a **scalar decode**
(today's code) — that's the M1 story. **bf16+i8mm chips (M2+, Graviton3+, Grace)** can run a *native*
bf16/i8 matmul twin (PLAN 21.3b, not built yet) → the biggest ARM lever for batching/quant.

---

## 2. What each platform has (x86)

The matmul levers are **AVX-512** (wide FMA), **VNNI** (`vpdpbusd` int8 dot), **AVX-512-BF16**
(`vdpbf16ps`), and **AMX** (Intel tile matrix, int8/bf16) — biggest first for matmul.

| CPU | AVX2 | AVX-512F/BW | VNNI | BF16 | AMX | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **AMD Zen3** (Ryzen 5000, 6800H) | ✅ | ❌ | ❌ | ❌ | ❌ | our mini-PC; int8 via widen+FMA |
| **AMD Zen4** (Ryzen 7000, EPYC Genoa) | ✅ | ✅ (2×256) | ✅ | ✅ | ❌ | first AMD AVX-512+VNNI+bf16 |
| **AMD Zen5** (Ryzen 9000, EPYC Turin) | ✅ | ✅ (full 512) | ✅ | ✅ | ❌ | our Scaleway 9555P; best AMD |
| **Intel Ice Lake** (Xeon Gen3) | ✅ | ✅ | ✅ | ❌ | ❌ | AVX-512+VNNI |
| **Intel Sapphire Rapids** (Xeon Gen4) | ✅ | ✅ | ✅ | ✅ | ✅ | **AMX** int8/bf16 matrix — the matmul lever |
| **Intel Granite/Emerald Rapids** | ✅ | ✅ | ✅ | ✅ | ✅ | improved AMX |

**Reading it:** VNNI is what makes int8 fast (native `--self-test` exercises `_mm512_dpbusd`). AMX
(Sapphire+) is a dedicated matrix unit — the strongest x86 target for a batched int8/bf16 GEMM (not
wired yet; a future twin). Zen3 (AVX2-only) is where **int4 + batching** already paid (RTF 2.81→2.02).

---

## 3. Where to rent (cheapest path to each lever)

| Want to test | Box | Provider (examples) | Rough cost |
|---|---|---|---|
| **Apple M2/M4 (bf16+i8mm, SME)** | Mac mini M4 | **buy** (~€700) — cheapest; or MacStadium | one-off / hourly |
| Apple M1/M2 in cloud | EC2 `mac2`, `mac2-m2` | AWS, Scaleway Apple silicon | $$ (dedicated host, 24h min) |
| **ARM server bf16+i8mm+SVE** | Graviton3 `c7g` | AWS | ¢/hr, on-demand |
| ARM server SVE2 (newest) | Graviton4 `c8g` | AWS | ¢/hr |
| ARM dotprod-only (M1-like server) | Ampere Altra | **Hetzner Cloud CAX**, Oracle A1 (free tier!), Scaleway | cheap / free |
| **x86 AVX-512+VNNI+BF16** | Ryzen 7950X (Zen4) | **Hetzner AX102** dedicated | ~€100/mo |
| x86 Zen5 (Turin) | EPYC 9005 | Scaleway, latitude.sh, OVH | ¢/hr |
| x86 AVX2-only (no AVX-512) | Ryzen 6800H | a local LAN mini-PC | owned |
| **Intel AMX** (int8/bf16 matrix) | Sapphire Rapids `c7i` | AWS, GCP `c3` | ¢/hr |
| NVIDIA Grace (ARM V2, high BW) | GH200 | Lambda, CoreWeave | $/hr |

**Tip:** the single cheapest way to cover the *biggest untested ARM lever* (bf16+i8mm, and SME) is a
**Mac mini M4** — owned, no clock, native `-march=native`, and it's a realistic target device.

---

## 4. Build notes per platform

- **macOS / Linux ARM (single-vendor host):** `make blas` uses `-march=native` — picks up dotprod/
  bf16/i8mm automatically. Verify with `--caps`.
- **x86 you control (own/dedicated):** `make blas` (`-march=native`) is fine.
- **x86 cloud / portable binary:** do **NOT** ship `-march=native` (locks codegen to the build host →
  SIGILL elsewhere). Build for a baseline + check at runtime: e.g. `make blas SIMD=avx2` for a broad
  binary, or `SIMD=avx512` only when you know every target has it. `--caps` will WARN if the binary
  was built past the CPU.
- **Force the fallback** to measure the scalar floor or dodge a missing ISA: `make blas SIMD=scalar`,
  or at runtime `QWEN_NO_SDOT=1 QWEN_NO_VNNI=1`.

---

## 5. The benchmark matrix (fill this in per box)

Run all four delivery modes with **identical explicit params** (per CLAUDE.md testing rules), at the
default thread count and `-j1`, for bf16 / int8 / int4. Text = one paragraph (~6–8 sentences) so
`--batch` has something to split.

### ✅ Measured: Mac mini **M2 Pro** (16-core GPU, 200 GB/s, macOS 26.3, rented) — 2026-07-06

First GPU-inclusive rental. `bootstrap_m2.sh` (curl one-liner) → native `make metal CC=clang` (Xcode
preinstalled) → `bench_m2.sh`. RTF = audio_s / gen_s, seed 42, ryan/Italian, fused Metal pipeline
(resident Talker + device-frame CP). CPU column is the **native M2** path (`i8mm/SMMLA + bf16/BFDOT`
fired — confirmed via `--caps`).

| Model | Backend | bf16 | int8 | int4 |
|---|---|---|---|---|
| **0.6B** | **Metal** | 0.53 | **0.39** ⭐ | 0.44 |
| 0.6B | CPU (M2) | 0.89 | 0.62 | 1.09 |
| **1.7B** | **Metal** | 0.79 | 0.53 | **0.50** ⭐ |
| 1.7B | CPU (M2) | 1.33 | 0.79 | 1.47–1.85 |

**Server mode** (Metal fused, 0.6B int8, `--serve`): **sequential steady-state RTF ~0.37** (no overhead vs
CLI); **streaming TTFA = 269 ms to first audio** (chunk = 10 frames), stream RTF 0.36.

**⭐ Batched Metal server (throughput, `QWEN_METAL_BATCH=1 --batch-size N`, 0.6B, temp0, measured 2026-07-06):**

| N concurrent | wall | per-req RTF | **batching speedup** (N·wall₁/wallₙ) |
|---|---|---|---|
| 1 | 19.5s | 1.01 | baseline (B=4 server does B× work even for 1 req) |
| 2 | 21.9s | 1.13 | **1.78×** |
| 4 | 27.7s | 0.99 | **2.81×** |

4 requests served in 27.7s vs ~78s serial → **2.81× throughput**; streaming **TTFA 314 ms**. batch==single
audio bit-identical (mv_b float4). Trade-off: per-request latency rises to RTF ~1 (each step does B-slot work)
but aggregate multi-user throughput scales ~2.8× at B=4 — the vLLM-style lever, now working on Metal.
(CUDA batch = 3.35×; CPU + CUDA + Metal now all batch.)

**MMA experiment (opt-in `QWEN_METAL_BATCH_MMA`, measured 2026-07-06 — REJECTED at B≤8):** replacing the
simdgroup-per-row float4 `mv_b` with a simdgroup_float8x8 MMA matmat gave a better *scaling ratio* (3.27× vs
2.81× at B=4) but a **2× worse baseline** (per-req RTF 1.88 vs 0.99, wall4 34s vs 20s) → net LOSS in absolute
throughput. Reason: at B=4 the 8×8 MMA tile is half-empty (4 of 8 B-cols) and the kernel uses only 32
threads/threadgroup → GPU underutilized for a matvec. MMA wins only at large B (≥16, compute-bound; cf. the
matmat selftest 4.61× at B=32) — not our batch sizes. **float4 mv_b stays the default** (bit-identical + faster
here). MMA kept opt-in (may pay off at large B on M4). Kernel-level lever measured, like ICB.

**Findings:** (1) **Metal M2 beats native CPU M2 ~1.5–2×** everywhere. (2) Best = **0.6B Metal int8 = 0.39**
(M1 floor was 0.60 → **1.54×**); **1.7B now sub-realtime** (0.50). (3) On M2 **int8 > int4 for Metal**
(bandwidth-rich → nibble-unpack of int4 not worth it; int8 is the M-series GPU sweet spot). **int4 on
CPU is a trap** (1.09→1.85 — nibble-unpack dominates). Use int8 on Apple Silicon. (4) 1.7B **int4 hurts
quality** (gen ran long, 10s audio vs ~6s) → prefer 1.7B int8. (5) Bandwidth is 2.9× M1 but RTF only
1.54× → **now dispatch/compute-bound, not bandwidth-bound.** `QWEN_METAL_PROFILE`: Talker step = encode
0.96ms (12%) + GPU commit+wait 7.31ms (88%); **CP dominates** (1017ms vs Talker 603ms). So the remaining
lever is **cutting GPU dispatch/work in the CP** (fusion / fewer weight re-reads), **not ICB** (ICB only
removes the 12% encode → ~6–8% overall, not worth the epic — measured, not assumed).

```bash
TXT="<a ~7-sentence paragraph>"
SEED=42; V=ryan; L=Italian; D=qwen3-tts-0.6b

# (a) single — whole text, one synthesis
for P in "" "--int8" "--int4"; do
  /usr/bin/time -p ./qwen_tts -d $D $P -T 0 --seed $SEED -s $V -l $L -o /tmp/s.wav --silent --text "$TXT"
done

# (b) batch — long-form, chunks stepped together (the weight-stationary path)
for P in "" "--int8" "--int4"; do
  /usr/bin/time -p ./qwen_tts -d $D $P --batch --batch-words 14 -T 0 --seed $SEED -s $V -l $L -o /tmp/b.wav --silent --text "$TXT"
done

# (c) streaming — TTFA + RTF
./qwen_tts -d $D --int8 --stream -T 0 --seed $SEED -s $V -l $L -o /tmp/st.wav --text "$TXT"

# (d) server — warm steady-state RTF
./qwen_tts -d $D --int8 --serve 8000 &
for i in 1 2 3; do timeout 60 curl -s localhost:8000/v1/tts \
  -d "{\"text\":\"$TXT\",\"seed\":$SEED,\"speaker\":\"$V\",\"language\":\"$L\"}" -o /tmp/sv$i.wav; done
pkill -9 -f "qwen_tts.*--serve"

# (e) server REQUEST-BATCHING throughput — N users stepped together vs single-stream
#     (the vLLM-style lever; speedup ~N on a bandwidth-bound box, ~1 on bandwidth-rich M1)
make bench-server          # or: bash tests/serve_batch_bench.sh $D 8900 4 4 4

# kernel-level: batched matmat twins vs B*matvec, per precision (no model needed)
make matmat-bench
```

RTF = wall-seconds / audio-seconds (lower = faster; <1.0 = sub-realtime). Use **RTF, not wall-clock**,
to compare (chunked synthesis emits slightly more audio). For correctness across ISA, compare
`--self-test` PASS + audio **mel-corr** (≥0.98), never md5 (cross-ISA fp-order differs benignly).

**Server request-batching throughput** (`make bench-server`) — the key metric for PRODUCT 2. `speedup`
= (M × single_wall) / burst_wall for M concurrent clients at `--batch-size N`. On M1 (bandwidth-rich,
bf16) ≈ 1 (no win — one synthesis already saturates DRAM); the real ~N× lands on bandwidth-bound x86
EPYC / Sapphire (AVX-512/VNNI) and is the reason PRODUCT 2 must be measured on rented silicon. Fill in:

| box | batch_N | clients_M | bf16 speedup | int8 speedup | int4 speedup | notes |
|---|---|---|---|---|---|---|
| M1 8-core (ref) | 4 | 4 | ~1.0 | ~1.0 | ~1.0 | bandwidth-bound; correctness-only |
| _Zen5 Turin_ | 4 | 4 | | | | int8/int4 throughput target |
| _Graviton3/Grace_ | 4 | 4 | | | | bf16+i8mm+SVE |


RTF = wall-seconds / audio-seconds (lower = faster; <1.0 = sub-realtime). Use **RTF, not wall-clock**,
to compare (chunked synthesis emits slightly more audio). For correctness across ISA, compare
`--self-test` PASS + audio **mel-corr** (≥0.98), never md5 (cross-ISA fp-order differs benignly).

| box | mode | bf16 RTF | int8 RTF | int4 RTF | TTFA (stream) | notes |
|---|---|---|---|---|---|---|
| M1 8-core (ref) | single | 1.48 | 0.89 | ~1.3 | — | dev box |
| M1 8-core (ref) | batch | **0.78** | 0.82 | slower | — | bf16+batch = 1.74× |
| **M1 8-core** (post-PR#17, 0.6B) | single file `-j4` | — | 0.69 | **0.52** | — | 2026-07-10; conv-exact + float32x4-q4-acc; int4 fastest on M1 (unchanged) |
| **Neoverse-N1** (Ampere Altra Max, 4c, 0.6B) | single stream24 `-j4` | — | — | **1.49** | ~910 ms | 2026-07-10 post-PR#17; `main` was 1.98 → **−25%** (conv-exact + snake-thread + BLAS-lever) |
| _Neoverse-N1_ | + `QWEN_SD_INT8=1` (opt-in) | — | — | **1.28** | ~950 ms | conv int8 decoder (−18%); noise ~−65 dBFS, ear-OK on emotion/clone |
| _Neoverse-N1_ | single file int4 `-j4` | — | 1.58 | **1.49** | — | after the decoder work int4 ≈ int8 e2e (collo spostato dal Talker) |
| _M2/M4_ | single | | | | | bf16+i8mm should lift int8/int4 |
| _M2/M4_ | batch | | | | | native matmul twin candidate |
| _Zen4 (AVX-512+VNNI)_ | single | | | | | VNNI int8 |
| _Zen4_ | batch | | | | | int4+batch expected sweet spot |
| **Zen5 Turin** (EPYC 9555P, 4vCPU, 1.7B) | single -j4 | 2.27 | **2.03** | 2.41 | int8 1.08s / int4 1.25s / bf16 1.68s | 2026-07-09 (v2 q4-VNNI); int8 won, int4 trailed |
| **Zen5 Turin** (EPYC 9555P, 0.6B, ms/frame) | q4-VNNI **v3** vs v2 vs int8 | — | Tlk 25.3 / CP 69.6 | **Tlk 22.9 / CP 63.5** | — | 2026-07-10 **q4-VNNI v3: the q4 KERNEL now beats v2 (−9%) and int8 per-frame** — the v2 compute-bound finding is fixed. (Wall RTF int8 still leads 0.93 vs int4 1.01: int4/int8 fork trajectory, 95 vs 147 frames, so wall isn't cross-quant comparable — the per-frame kernel is.) `SIMD=avx512vnni`, `QWEN_Q4_VNNI_V3=0`→v2. Compare ms/frame not RTF (int4/int8 fork trajectory: 95 vs 147 frames) |
| _Zen5 Turin_ | batch (matmat, B=8) | 11.9× | **3.0×** | 1.6× | — | kernel-level batching speedup (int8/q4 now VNNI, `--matmat-bench`); e2e server-batching = WIP (§5.note) |
| _Sapphire (AMX)_ | batch | | | | | AMX int8 GEMM (future twin) |

**⭐ Full RTF+TTFA — EPYC 9555P Zen5, 4 vCPU, `-j4`, `SIMD=avx512vnni` (2026-07-10, post-PR#17, min of 3):**

| model | config | RTF | TTFA |
|---|---|---|---|
| 0.6B | file **int8** | **0.95** | 1073 ms |
| 0.6B | file int4 (v3) | 1.01 | 1142 ms |
| 0.6B | stream int4 (chunk24) | 0.99 | **575 ms** |
| 0.6B | file bf16 | 1.08 | 1219 ms |
| 1.7B | file **int8** | **1.17** | 1705 ms |
| 1.7B | file int4 (v3) | 1.30 | 1865 ms |
| 1.7B | stream int4 (chunk24) | 1.28 | 1145 ms |
| 1.7B | file bf16 | 1.44 | 2069 ms |

Headline: **int8 is the x86 wall-clock winner** (0.6B sub-RT 0.95). q4-VNNI v3 made the int4 KERNEL faster
per-frame than int8 (row above), but int4/int8 fork the greedy trajectory so wall RTF isn't cross-quant
comparable.

**⭐ Re-validation round (2026-07-11, same instance type, `perf/rental-prep` branch):**
- ⚠️ **Build gotcha (bit us again):** `make blas` on x86 defaults to PORTABLE AVX2 — VNNI needs
  **`make blas SIMD=avx512vnni`**. `--caps` says which one you got ("int8 dot: VNNI" vs "widen->FMA");
  READ IT before trusting any number.
- **Batched-server stall fix VALIDATED**: B=4 server, sequential single requests = **8–9 s each**
  (the historical intermittent bug was 262 s; `submit_mtx` + inactive-slot compaction now on).
- **fp16 q4 scale (18 B/block)**: `--self-test` PASS on real VNNI; `matmat_q4 vs matvec_q4 L2 = 0`
  (bit-identical). int8 0.96 vs int4 1.05 wall — x86 story unchanged, int8 stays the pick.
- **1.7B config note: pure `--int8` (1.22) beats `--quant-mixed` (1.30) on x86** — the int4 Talker is
  SLOWER than int8 on VNNI (43.2 vs 35.5 ms/f). quant-mixed is an Apple-silicon (M1) config;
  **on x86 use `--int8`**.
- **`QWEN_CP_PREFILL2=1` is a measured x86 WIN**: CP 49.6/50.7 → **46.8/46.6 ms/f (−6-8%)**, RTF −3%
  (the B=2 matmat rides the VNNI GEMM; on M1 the same lever is neutral — SDOT-seq is already optimal).
- **`QWEN_BLAS_GEN_THREADS` sweep (4 vCPU)**: optimum = **1** (RTF 0.95 vs 0.99 at the nt−1 default of
  3; =4 catastrophic 1.46, oversubscription). Per-box knob — sweep it on every new box (N1 file-mode
  optimum was 2).

**⭐ Graviton3 (AWS c7g.2xlarge, Neoverse-V1, 8 vCPU, `-j4`, 2026-07-11) — first server-ARM with
i8mm/bf16; the MMLA twins' first silicon:**

| what | result |
|---|---|
| `--self-test` | **SMMLA int8 matmat L2 = 0.00e+00 (bit-identical to B× SDOT matvec)**; BFMMLA 3.4e-03 (expected bf16-act signature) — both PASS first run |
| matmat-bench int8 (B=8) | old twin **0.32-0.38×** (batching LOST vs seq) → **SMMLA 2.03-2.10×** (batch 0.90→0.16 ms — ~6× better) |
| matmat-bench bf16 (B=8) | old 1.01-1.20× → **BFMMLA 1.38-1.58×** |
| matmat-bench int4 (B=8) | 0.29× with the old scalar batch → **1.55-1.63× with the q4-SMMLA twin** (same-session follow-up; self-test L2 ~7e-8). On M1-class (no i8mm) the batch now falls back to B× SDOT matvecs: 0.43-0.65× → 0.95-1.13× (loss gone) |
| single-stream RTF | 0.6B int8 **0.66** (beats EPYC Turin 0.96!) · int4 0.73 · bf16 1.11 · **1.7B int8 0.95 — sub-realtime 1.7B on an ARM server CPU** |
| batched server B=4 (e2e A/B) | 4 concurrent: **17 s with MMLA vs 21 s without = −19% wall** (aggregate RTF 0.84) |
| batched server B=4 **int4** (e2e) | aggregate RTF **0.94** with the q4-SMMLA twin — int4 batched serving viable on ARM |
| `QWEN_BLAS_GEN_THREADS` | optimum = 3 (= the nt−1 default) on 8 vCPU — opposite of the 4-vCPU EPYC; the knob is genuinely per-box |

Build: plain `make blas` (Linux ARM uses `-march=native` → i8mm/bf16 auto-enabled; `--caps` must say
"SMMLA ACTIVE"/"BFMMLA ACTIVE"). Kill-switches for A/B: `QWEN_NO_SMMLA=1` / `QWEN_NO_BFMMLA=1`.

**⭐ Apple M4 (Scaleway M4-S Mac mini, 10-core, 16 GB, `-j4`, 2026-07-11) — first M4 + SME silicon:**

| what | result |
|---|---|
| `--caps` | i8mm + bf16 + **SME/SME2 detected at runtime** (first SME silicon; no SME kernels yet) — MMLA twins ACTIVE |
| `--self-test` | ALL PASS (SMMLA int8 L2=0, q4-SMMLA 7.6e-08, BFMMLA 3.4e-03 — same signatures as Graviton3) |
| matmat-bench (B=8) | **per-box heterogeneity vs Graviton**: q4-SMMLA **1.50-1.84× WIN** · int8-SMMLA 0.61-0.91× (loses — M4's SDOT matvec + bandwidth too strong) · BFMMLA 0.72-0.91× (loses — transpose overhead doesn't pay on bandwidth-rich cores) → per-platform gating is a real question for the merge |
| CPU single-stream | 0.6B int8 **0.44** · **int4 0.32** ⚡ · 1.7B quant-mixed **0.57** — the M4 CPU alone nearly matches the A100 GPU numbers |
| **Metal** | 0.6B int8 0.38 · **int4 0.28 — new all-device record** (M2 Pro was 0.39) · 1.7B int4 **0.41** (M2 Pro: 0.50). **int4 > int8 on M4 Metal even with the SCALAR q4 shader** — the M4 GPU reversed the M2-era ordering by itself |
| q4-vec shader verdict | `QWEN_METAL_Q4_VEC=1` = **NEUTRAL on M4** (0.28 vs 0.28) and a regression on M1 → the vectorized twin is NOT the win anywhere tested; keep opt-in or drop at merge |

**⭐ Full RTF+TTFA — Neoverse-N1 (Ampere Altra Max, 4 vCPU, `-j4`, 2026-07-10, post-PR#17, min of 3):**

| model | config | RTF | TTFA |
|---|---|---|---|
| 0.6B | file int8 | 1.60 | 1842 ms |
| 0.6B | file int4 | 1.60 | 1883 ms |
| 0.6B | stream int4 (chunk24) | **1.49** | 908 ms |
| 0.6B | stream int4 + `QWEN_SD_INT8=1` | **1.28** | 932 ms |
| 1.7B | file int8 | 2.04 | 3116 ms |
| 1.7B | stream int4 + `QWEN_SD_INT8=1` | **1.80** | 1968 ms |

Decoder-bound box: exact-streaming conv + threaded snake + int8 decoder conv land hardest here (`main`
0.6B stream int4 1.98 → 1.49, −25%; +conv-int8 1.28). Opposite of x86, **int4 ≈ int8 e2e** after the
decoder work (the bottleneck moved off the Talker).
| _Graviton3_ | batch | | | | | bf16+i8mm+SVE |

### §5.note — EPYC 9555P Zen5 run (2026-07-09) — what DIFFERS from M1
Real AVX-512-VNNI box (Scaleway STANDARD3-X4C-16G, `avx512_vnni`+`avx512_bf16`). Findings that
diverge from the M1 dev box — record these so we don't assume M1 behavior carries over:
1. **C7 q4-VNNI is CORRECT but NOT faster.** `--self-test` PASS on real VNNI (numeric correctness proven —
   the thing M1 can't check). But single-stream int4-VNNI **RTF 2.76 vs int8 2.01 at -j1 (same 44 frames)
   = ~37% SLOWER** — the *opposite* of M1, where int4-SDOT beats int8. Even the legacy f32-dequant q4 (2.31)
   beats int4-VNNI. Cause: the v1 kernel is correctness-first (per-block 2× `_mm512_reduce`, 32-wide block
   zero-extended into a 512-bit `dpbusd` = half the lane width wasted) → compute overhead eats the
   half-the-bytes bandwidth win. The plan_v4 C7 throughput TODOs (2-blocks-per-512b full-width, drop the
   per-block reduce, 2-row fusion) are **REQUIRED, not optional**, before int4 can win on x86.
2. **int4 < int8 on Zen5 single-stream** (int8 is the fastest quant here), vs **int4 > int8 on M1**. Do NOT
   port the M1 "int4 is the fast default" conclusion to x86 until C7 is optimized + re-measured.
3. **Batched matmat twins now use VNNI (FIXED) — but the e2e `--batch-size` server has a rare intermittent
   stall (open bug).** FIXED: `int8_matmat`/`q4_matmat` used to dequant int8→f32 (no VNNI); added
   `int8_matmat_vnni_slice` + `q4_matmat_vnni_slice` (weight-stationary `vpdpbusd`) → **`--matmat-bench`:
   batched int8 3.0× / q4 1.6× over B×matvec, bit-exact** (self-test L2_rel 0.00) = the real x86 server-
   throughput kernel win. STILL OPEN: the e2e `--batch-size` server is USUALLY fine (batch-1 int8 ~9s, 14
   consecutive good runs) but RARELY stalls to ~200-260s on a fresh request — genuinely intermittent, NOT
   root-caused. It is NOT the pool submission race (on Linux the batched path is single-threaded: A1 helper
   gated off, decode inline) and NOT over-generation (identical output size). Needs **ThreadSanitizer / a
   `perf` capture of a slow run** (a proper source debug), not more black-box timing — do that on a future
   box run. Meanwhile CLI and **plain `--serve`** (VNNI, fully reliable) are the fast paths; the kernel-level
   batching win is proven, only the e2e throughput has this rare stall. (`submit_mtx` in the Linux pool =
   a defensive fix for concurrent `--workers` submission, kept; it wasn't the cause of this stall.)
4. **int4 audio trajectory diverges from M1 (benign):** `--int4 --emotion joy` over-drove into a 108-frame
   ramble on x86 (vs a clean 36-frame render on M1, same seed/text) — greedy-argmax flips on the different
   x86 fp path. Not a C7 bug (self-test clean); use quant-mixed (int8 CP stabilizes) or bf16 for emotion on
   x86-int4. Plain int4/int8/quant-mixed audio = ear-clean.

---

## 6. What we expect to learn (hypotheses to confirm)

- **M2/M3/M4 (bf16+i8mm):** a native bf16/i8 matmul twin should make int8/int4 + batching clearly
  beat M1, since the scalar bf16 decode (M1's bottleneck) goes away. M4's **SME** could host a real
  matrix-engine GEMM. → confirm with `--matmat-bench` once the native twin lands.
- **Zen4/5 (AVX-512+VNNI):** more memory-bound per core than M1 → **batching pays more than 1.74×**,
  and **int4 + batching** is the predicted sweet spot (Zen3 already showed int4 wins there). int8-VNNI
  + batch should be excellent on Zen5.
- **Sapphire Rapids (AMX):** the strongest x86 matmul lever for a batched int8/bf16 GEMM (future).
- **int8 on M1 is already SDOT-fast (RTF 0.89)** → batch is ~break-even *there*; on bandwidth-bound
  server x86 the same int8+batch should win. The **int8-SDOT batched twin** (integer-dot instead of
  f32-accum) is the open optimization that would also tip int8+batch positive on M1.

Keep this file updated as boxes are tested — paste `--caps` output + the RTF matrix row per box.

---

## 7. Per-ISA optimization roadmap (the "flows" — fill in on real hardware)

All the batched-matmul levers plug into the **same three twin families** in `qwen_tts_kernels.c`,
which are already split into compile-time-B fixed kernels (a clean home — the ISA branch goes
*inside* each `*_matmat_b<BV>` body, guarded by the feature macro, with the current scalar/NEON path
as the `#else`):

- bf16: `bf16_matmat_b1..b8,b16`
- int8: `int8_matmat_b2..b16`
- int4: `q4_matmat_b2..b16`

Each newer-ISA path is **guarded by its feature macro** so M1 keeps compiling the scalar path
untouched; `make check-isa` (below) cross-compiles the guarded paths so syntax is verified *now*,
on M1, before any hardware exists. Build it, measure with `make matmat-bench` / `bench-matrix` on
the target, keep the scalar `#else` as the always-correct fallback + the `--self-test` oracle.

| # | ISA / instruction | where (function) | what it does | restructuring | expected lever | validate on |
|---|---|---|---|---|---|---|
| 1 | **ARM bf16 BFMMLA** `vbfmmlaq_f32` | `bf16_matmat_b*` | 2×4·4×2 bf16→2×2 f32 tile in one op — kills the scalar bf16 decode (M1's bottleneck) | tile 2 rows × 2 B-cols; pack weight+X as bf16 2×4 | **large** (removes decode) | M2/M3/M4, Graviton3+ |
| 2 | **ARM i8mm SMMLA** `vmmlaq_s32` | `int8_matmat_b*` | 2×8·8×2 int8→2×2 int32 — native int8 matmul | quantize X per-column to int8; same 2×2 tile | **large** for int8+batch | M2+, Graviton3+ |
| 3 | **ARM SME/SME2** (ZA tiles) | new `*_matmat_sme` | outer-product matrix engine — whole GEMM in tiles | streaming mode; tile config; biggest rewrite | **largest, research** | M4/M5, (no server ARM yet) |
| 4 | **x86 AVX-512-VNNI** `_mm512_dpbusd_epi32` | `int8_matmat_b*` | int8 dot accumulate (already used in `int8_matvec_vnni`) | per-column int8 X; 16-wide B tiles | **large** for int8+batch | Zen4/5, Ice Lake+ |
| 5 | **x86 AVX-512-BF16** `_mm512_dpbf16_ps` | `bf16_matmat_b*` | fuse bf16 decode+FMA | 16-wide; bf16-pack X | medium (bf16 stays compute-bound on AVX-512) | Zen4/5, Cooper Lake+ |
| 6 | **Intel AMX** TMUL tiles | new `*_matmat_amx` | dedicated int8/bf16 matrix unit — strongest x86 GEMM | tile config (TILECFG), 16×64 int8 tiles, load/TMUL/store | **largest x86** | Sapphire Rapids+ |
| 7 | **int8-SDOT/int-accum twin** (M1-testable) | `int8_matmat_b*` | integer-accumulate (int8 W × int8 X → int32) instead of f32-accum | quantize X per-column to int8 | tips int8+batch positive (the matmat-bench TODO) | **M1 now**, all |

**Sequencing.** #7 is the only one fully implementable+measurable on M1 today (uses NEON int / SDOT,
no new ISA) — do it first if int8+batch break-even on M1 bugs us. #1–#2 (ARM bf16/i8mm) want an M2+
(a Mac mini M4 covers both + SME). #4 (VNNI) wants Zen4/5. #6 (AMX) wants Sapphire. Each is a localized
fill-in once the box is in hand; the dispatch skeleton + `make check-isa` keep them compile-clean
meanwhile. Always re-run `--self-test` (the cross-ISA correctness oracle) + ear/mel-corr after.
