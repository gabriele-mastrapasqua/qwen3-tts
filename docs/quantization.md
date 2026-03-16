# Weight Quantization

The `--int8` and `--int4` flags quantize Talker and Code Predictor weights at load time,
reducing memory usage and (for INT8) improving speed on the 1.7B model.

These flags have **no meaningful effect on the 0.6B model** — matrices are too small to
be bandwidth-bound, so quantization adds overhead without benefit.

## INT8 (Recommended for 1.7B)

```bash
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int8 -o hello.wav
```

- 15% Talker speedup (reduced memory bandwidth)
- Good audio quality — minimal perceptual difference from BF16
- Halves Talker RAM usage (2.8 GB → 1.4 GB)
- Works with all features: server, streaming, custom voices, instruct

## INT4 (Experimental)

```bash
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int4 -o hello.wav
```

- Q4_0 format (4-bit with per-block scale factors)
- Smallest memory footprint (0.7 GB Talker RAM)
- Slightly **slower** than BF16 due to nibble unpacking overhead
- Audio quality may degrade on some inputs

## Comparison

**1.7B, Italian, seed=42, Apple M1 16 GB, 4 threads:**

| Config | Talker ms/f | Total time | RTF | Talker RAM |
|--------|-------------|------------|-----|------------|
| BF16 (default) | ~80 ms/f | ~13s | ~4.3 | 2.8 GB (mmap) |
| **INT8 (recommended)** | **~67 ms/f** | **~11s** | **~3.6** | **1.4 GB** |
| INT4 (experimental) | ~83 ms/f | ~14s | ~4.5 | 0.7 GB |

## Recommendation

Use `--int8` for the 1.7B model. It gives 15% Talker speedup with good audio quality.

INT4 saves memory but is slightly *slower* due to nibble unpacking overhead.
For maximum speed, use the 0.6B model (RTF ~1.3–1.7 vs 3.6 for 1.7B INT8).

On systems with 16+ GB free RAM, expected performance is better than shown above
(our test machine had high system memory pressure from other applications).
Projected RTF with free RAM: **0.6B ~1.3, 1.7B BF16 ~3.0, 1.7B INT8 ~2.5**.

## Testing

```bash
make test-large-int8  # 1.7B INT8 tests (Italian + English, seed 42)
make test-large-int4  # 1.7B INT4 tests (Italian + English, seed 42)
make test-large-quant # All 1.7B quantization tests (INT8 + INT4)
```
