# Building

## Dependencies

Only a C compiler and BLAS are required:
- **macOS**: Accelerate framework (ships with Xcode)
- **Linux**: OpenBLAS

LZ4 compression (for `.qvoice` voice files) is embedded in the repo — no separate install needed.

### Optional: Python deps for the golden-reference test

The engine itself needs **no Python runtime**. Only the `make test-golden`
correctness check (and `make golden-update`) compares audio output against
committed reference WAVs, which requires Python with `librosa`:

```bash
pip install librosa numpy soundfile
```

On recent Ubuntu/WSL2 the system pip is locked, so either use a virtualenv
or install with `python3 -m pip install --break-system-packages librosa numpy soundfile`.
`make` calls the system `python3`, so make sure that interpreter can
`import librosa` (if you use a venv, run `make test-golden` with it activated).
Without these packages `make test-golden` simply prints `SKIP`.

## macOS

```bash
make blas    # Uses Accelerate framework
```

## Linux

```bash
# Install OpenBLAS
sudo apt install libopenblas-dev    # Ubuntu/Debian
sudo dnf install openblas-devel     # Fedora/RHEL

make blas
```

## Windows (WSL2) — Beta

WSL2 runs a real Linux kernel, so the build is identical to native Linux.
Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu if you haven't already.

> **Validated** on a Ryzen 7 6800H mini PC under WSL2 (Ubuntu): builds clean with
> the portable `-mavx2` baseline, `--caps` reports `AVX2 (2-row fused, FMA)` + pthread
> pool, and output is coherent (golden-reference correctness confirmed by ear — see the
> cross-ISA note under Testing). See "Performance notes (x86 / WSL2)" below before benchmarking.

```bash
# In a WSL2 terminal (Ubuntu)
sudo apt update && sudo apt install build-essential libopenblas-dev

git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git
cd qwen3-tts
make blas

# Download a model and generate speech
./download_model.sh --model small
./qwen_tts -d qwen3-tts-0.6b --text "Hello from Windows!" -o hello.wav
```

### Playing audio from WSL2

```bash
# Option 1: Open with Windows media player (works out of the box)
powershell.exe Start-Process "$(wslpath -w hello.wav)"

# Option 2: Use aplay if PulseAudio/PipeWire is configured in WSL2
aplay hello.wav

# Option 3: Copy to Windows and play manually
cp hello.wav /mnt/c/Users/$USER/Desktop/
```

### Performance notes (x86 / WSL2)

Measured on a **Ryzen 7 6800H** (8C/16T, Zen3+, dual-channel DDR5-4800), WSL2/Ubuntu,
0.6B model, deterministic run (`-j1 --temperature 0 --seed 42`). Read these before
drawing perf conclusions on x86 — the bottleneck is **not** what you'd expect:

- **The hot path is memory-bandwidth-bound, not compute-bound.** The Code Predictor
  re-reads its weights 16× per frame, so it waits on DRAM, not the ALU. Consequences:
  - **AVX2 buys only ~6% over the scalar build** (`SIMD=scalar`). AVX2 *is* compiled and
    used (`./qwen_tts --caps`), it just can't speed up memory traffic. Don't expect a
    SIMD miracle on x86 — that's the workload, not a bug.
  - **4 threads is the sweet spot; more regresses.** Going 4→8 threads made the Code
    Predictor *slower* (140 → 156 ms/f) because the threads saturate and then contend for
    the same memory bus. The default pool size (4) is correct — do **not** pass `-j8`.

- **Quantization is the real lever on x86 — and `--int4` is the biggest.** Fewer weight
  bytes = less DRAM traffic, which is exactly what a memory-bound CPU needs:
  - `--int8` (Talker + Code Predictor): Code Predictor −24% vs bf16, RTF 3.91 → 3.29 at `-j1`.
  - `--int4` (Q4_0, Talker + Code Predictor): at **4 threads** it beats int8 by a wide margin
    — RTF 2.81 → **2.02** (−28%), Code Predictor 128.8 → 106.1 ms/f, Talker 65.8 → 37.5 ms/f.
    Quality is intact by ear (a slight timbre shift, no artefacts).
  - **Catch: int4 only wins *multi-threaded*.** At `-j1` int4 ties int8 (the q4 nibble-unpack
    cost cancels the byte saving while a single core can't saturate the memory bus). The win
    appears at 4 threads, when the bus *is* saturated and int4's half-bytes relieve it. This
    is the opposite of an Apple M1, where int4 is *slower* than bf16 (bandwidth-rich, so the
    unpack overhead dominates) — **int4 is an x86/memory-starved-CPU lever, not a universal default.**

- **4 threads is the sweet spot for every precision — including int4.** 8 threads regresses
  (bus contention) even with int4's lighter traffic. Do not pass `-j8`.

- **Set the Windows power plan to "High performance"/"Ultimate", not "Balanced".**
  On Balanced the CPU sat at ~2566 MHz; on High performance it holds its ~3194 MHz base
  clock under load (`powercfg /getactivescheme` to check, `powercfg /setactive <GUID>` to
  change). The RTF gain is modest *because the workload is memory-bound*, but there is no
  reason to leave performance on the table.

- **Best measured config: High performance + `--int4` + 4 threads → RTF ~2.02** for 0.6B
  (15.5 s of audio in ~31 s on a Ryzen 7 6800H) — nearly 2× faster than the original
  AVX2 bf16 build (~3.9) and approaching an Apple M1 (~1.3–1.4). The remaining gap is the
  memory subsystem: the M1's unified memory + large system-level cache handle the Code
  Predictor's 16×-per-frame weight re-read better than a mini-PC's DDR5 at comparable peak GB/s.

### AVX-512 / VNNI (Zen4+/Intel)

On a CPU with AVX-512-VNNI, build the VNNI int8 path:

```bash
make blas SIMD=avx512vnni     # needs avx512f/bw/vl + avx512_vnni in /proc/cpuinfo
./qwen_tts --caps             # should report: int8 dot: VNNI _mm512_dpbusd_epi32 (native)
./qwen_tts --self-test        # kernel numeric correctness, ISA-independent (no model needed)
```

Validated on an **AMD EPYC 9555P** (Zen5 "Turin", full-width 512-bit AVX-512). The VNNI int8
kernel is a real **~1.85× win at equal core count** (scalar-bf16 `-j1` RTF 3.04 → VNNI-int8 `-j1`
1.64). Caveat: this was a 4-vCPU **VM**, where the hypervisor scatters vCPUs across CCDs and
limits thread scaling (`-j1` beat `-j4`) — threading scales on **bare metal**. The full
cross-device table (M1 / 6800H / 9555P) and the reproducible A/B harness are in
[docs/performance.md](performance.md) → run `bash tests/x86_bench.sh` on your own box.

## Other Build Targets

```bash
make debug      # Debug build with AddressSanitizer
make clean      # Clean build artifacts
make info       # Show build configuration
```

## Testing

```bash
make test-small       # Run 0.6B tests (English, Italian, multiple speakers)
make test-large       # Run 1.7B tests (config check, English, Italian, instruct styles)
make test-large-int8  # Run 1.7B INT8 quantization tests (Italian + English, seed 42)
make test-large-int4  # Run 1.7B INT4 quantization tests (Italian + English, seed 42)
make test-large-quant # Run all 1.7B quantization tests (INT8 + INT4)
make test-regression  # Cross-model regression checks (safetensors, config parsing)
make test-golden      # Correctness safety net: mel-corr (>=0.99) + duration vs committed reference WAVs
                      #   -> requires Python + librosa (see Dependencies); SKIPs if missing.
                      #   Also the cross-ISA check (ARM vs AVX2/x86 won't be bit-identical, must stay ~0.99+).
make golden-update    # Regenerate the reference WAVs after an INTENDED output change (needs librosa)
make test-caps        # Assert ./qwen_tts --caps matches the arch (NEON on arm64, AVX2/scalar on x86)
make test-errors      # Bad invocations fail cleanly (no model needed, fast)
make test-all         # Run everything (0.6B + 1.7B + regression + errors + caps + golden)
make test-serve          # HTTP server integration test (health, speakers, TTS)
make test-serve-bench    # Server benchmark: 2 runs, same seed, verify bit-identical output
make test-serve-openai   # OpenAI-compatible /v1/audio/speech endpoint test
make test-serve-parallel # 2 concurrent requests, verify both complete
make test-serve-all      # Run all server tests
make serve               # Start HTTP server on port 8080
make bench               # RTF benchmark (short+long, normal+stream, both models)
make bench-full          # Full benchmark (+ server, instruct, INT8, .qvoice)
```
