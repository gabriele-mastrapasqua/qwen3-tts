# X86 INT8 dataflow study

This note records the x86 activation-dataflow experiment landed after the
VNNI/AMX and oneDNN/ARK oracle work. Measurements are screening evidence, not
a production qualification.

## Motivation

The earlier estimate that Talker was close to its local DRAM read roof was not
proof that the dataflow was optimal. The denominator described the current
kernel's useful model reads; it did not charge or expose avoidable temporary
writes, scale passes, repacks, gathers, or cache traffic around that kernel.

The relevant path was:

```text
source activations -> gather/materialization -> absmax + dynamic q8 quantization
-> VNNI/AMX preparation -> matmul -> scale/output -> scatter
```

ARM/KleidiAI and oneDNN made this worth checking: a mature x86 kernel may be
fast enough that activation preparation, rather than integer dot-product work,
becomes visible at small batch sizes.

## 1.7B oracle results

### oneDNN W8A8

The oneDNN oracle selected `brg_matmul:avx10_1_512_amx` on Sapphire Rapids. Its
activation-preparation share of `prep + compute` was material even at B1 and
usually dominant at B4:

| Projection | B1 | B2 | B4 |
|---|---:|---:|---:|
| CP Down | 44.2% | 57.6% | 72.5% |
| Talker QKV | 15.7% | 26.7% | 42.4% |
| Talker Gate/Up | 6.9% | 12.5% | 22.3% |
| Talker Down | 28.4% | 42.8% | 61.6% |

oneDNN choosing an AMX BRGEMM implementation at low B is an important oracle
observation, but it does not prove that the serving dispatcher should change:
its thread, layout, quantization, and end-to-end scheduling contract differs
from the engine's.

### ARK W4A16

ARK/BestLA W4A16 kept activations in BF16/FP32 and removed dynamic INT8
activation quantization, but its packed weight-only path was frequently slower
than oneDNN W8A8 on the real 1.7B projection shapes. Removing A8 preparation
alone therefore did not justify changing the engine's INT8 format.

## Implemented optimization

The source change originated in commit `4964190` and was cherry-picked onto
the current runtime history as `d847d9c`:

- CP and Talker regions quantize each contiguous source row directly.
- The old `[cols][B]` FP32 `Xt` materialization is no longer written and read
  before q8 preparation.
- `qwen_region_i8_quant_row` uses the existing quantizer contract: same
  absmax, scale, rounding, clamp, zero handling, and integer arithmetic.
- The parity test compares q8 bytes and scales before the matmul; the final
  output remained bit-identical.

The isolated B1/B2/B4 screen measured roughly **68–85% lower activation
preparation cost**. Representative complete-projection changes were:

| Projection | B2 | B4 |
|---|---:|---:|
| CP Down | about −4% | about −9% |
| Talker Down | about −2% | about −5% |
| Talker Gate/Up | about −0.3% | about −0.6% |

Gate/Up gains are smaller because the weight compute dominates its preparation
cost. A short engine wave moved positively at C4 (STREAM_RTF about 1.89→1.68,
total RTF about 2.10→1.87, with no errors); the single C1 wave was
noise-sized. This is a WAVE screen, not a production SOAK qualification.

## 0.6B implications

The 0.6B checkpoint was **not present on the GCP machine**. Dimensions below
come from the official [Qwen3-TTS 0.6B Base configuration](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/config.json)
and repository evidence. oneDNN and ARK used synthetic weights with these real
shapes, so this is a shape/kernel/dataflow oracle, not a full 0.6B serving
benchmark.

Weights are written as output rows × input columns:

| Component | 1.7B | 0.6B |
|---|---:|---:|
| Talker QKV | 4096×2048 | 4096×1024 |
| Talker O | 2048×2048 | 1024×2048 |
| Talker Gate/Up | 12288×2048 | 6144×1024 |
| Talker Down | 2048×6144 | 1024×3072 |
| CP QKV | 4096×1024 | 4096×1024 |
| CP O | 1024×2048 | 1024×2048 |
| CP Gate/Up | 6144×1024 | 6144×1024 |
| CP Down | 1024×3072 | 1024×3072 |
| MTP | 1024×2048 | 1024×1024 |
| lm_head | 2048×1024 | 2048×1024 |

The Talker hidden/intermediate sizes halve from 2048/6144 to 1024/3072;
Talker has 28 layers and CP has 5. CP projection shapes are unchanged.

All tested dimensions are aligned to the current blocking: no obvious tail or
alignment pathology appeared. Current dispatch is VNNI GEMV at B1, VNNI matmat
at B2, and AMX INT8 at B4. Persistent RHS packing is dimensionally eligible,
but remains controlled by the current prepack policy/flags.

The strongest result is that preparation becomes proportionally more important
on the smaller Talker:

| Projection | 1.7B prep share B1/B2/B4 | 0.6B prep share B1/B2/B4 |
|---|---:|---:|
| Talker QKV | 15.7 / 26.7 / 42.4% | 15.3 / 27.8 / 40.4% |
| Talker Gate/Up | 6.9 / 12.5 / 22.3% | 12.3 / 23.3 / 35.7% |
| Talker Down | 28.4 / 42.8 / 61.6% | 52.8 / 70.3 / 79.5% |

CP differences are primarily run variation because its important shapes are the
same in both models. The generic source-row optimization is therefore at least
as relevant to 0.6B, without a model-specific fork.

### 0.6B ARK check

The synthetic 0.6B-shape ARK probe used 12 threads, persistent W4 packing
(1,769,600 bytes for 1024×3072), BF16 input → FP32 compute, and no dynamic
activation quantization. Steady total latency was 22.0/26.4 µs for CP Down
B1/B2 and 19.6/27.7 µs for Talker Down B1/B2; BF16 cast preparation was about
3.2–4.4 µs. One-time packing was about 1.2–2.8 ms and must be amortized.

The result is mixed: CP B1 can lose while Talker Down is competitive or wins in
the micro-oracle. It is not evidence for a general W4A16 migration or a
0.6B-specific weight-format branch.

## Follow-up conclusion

The next useful questions are generic x86 questions: low-B AMX/VNNI crossover,
actual persistent-RHS consumption, remaining activation gather/pack/scatter
passes, and AMX activation-pack reuse. A real 0.6B checkpoint is still needed
for server-level TTFA and STREAM_RTF qualification.
