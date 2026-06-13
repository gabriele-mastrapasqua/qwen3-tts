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
| **NVIDIA Grace** (Neoverse V2) | Armv9.0 | ✅ | ✅ | ✅ | ✅ SVE2 | ❌ | DGX/GH200; very high bandwidth |
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
| _M2/M4_ | single | | | | | bf16+i8mm should lift int8/int4 |
| _M2/M4_ | batch | | | | | native matmul twin candidate |
| _Zen4 (AVX-512+VNNI)_ | single | | | | | VNNI int8 |
| _Zen4_ | batch | | | | | int4+batch expected sweet spot |
| _Zen5 Turin_ | single | | | | | |
| _Zen5 Turin_ | batch | | | | | throughput target |
| _Sapphire (AMX)_ | batch | | | | | AMX int8 GEMM (future twin) |
| _Graviton3_ | batch | | | | | bf16+i8mm+SVE |

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
