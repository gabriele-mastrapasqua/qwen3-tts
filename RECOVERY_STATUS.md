# Qwen3-TTS C Implementation - Recovery Status

## What Happened

The original C implementation files were accidentally overwritten when copying the official `qwen_tts/` Python module from the Qwen3-TTS repository.

## Recovery Status

### ✅ Rebuilt Files

1. **qwen_tts.h** - Main header with:
   - All configuration structs
   - Talker/CP/Decoder layer weight structs
   - Speech decoder state
   - Main context structure
   - API function declarations

2. **qwen_tts.c** - Main pipeline with:
   - Config loading from JSON
   - Model loading (talker, CP, speech decoder)
   - bf16 helpers
   - Language/speaker mapping

3. **qwen_tts_talker.c** - Talker LLM with:
   - **CORRECTED: Interleaved RoPE** (was NeoX split-half)
   - Weight loading
   - Single-token step (stub)
   - Prefill (stub)

4. **qwen_tts_kernels.h / qwen_tts_kernels.c** - Kernel functions with:
   - RMSNorm
   - Linear/MatVec
   - **CORRECTED: Interleaved RoPE apply**
   - Causal attention
   - Element-wise ops (SiLU, add, mul)
   - bf16 rounding

5. **qwen_tts_code_predictor.c** - Stub
6. **qwen_tts_speech_decoder.c** - Stub
7. **qwen_tts_audio.c** - WAV writer
8. **qwen_tts_sampling.c** - Stub
9. **qwen_tts_tokenizer.c/h** - Stubs
10. **qwen_tts_safetensors.c/h** - Stubs
11. **main.c** - CLI entry point
12. **Makefile** - Build system

### ✅ Build Status

```bash
make
# Successfully builds qwen_tts_bin
```

### ⚠️ Missing Implementations (Stubs)

The following components need full implementation:

1. **qwen_talker_prefill()** - Multi-token prefill with interleaved RoPE
2. **qwen_talker_step()** - Complete single-token forward pass
3. **qwen_cp_load()** - Code Predictor weight loading
4. **qwen_cp_predict()** - Code Predictor 15-pass generation
5. **qwen_speech_decoder_load()** - Speech decoder weight loading
6. **qwen_speech_decoder_decode()** - Full decoder forward pass:
   - VQ dequantization (16 codebooks)
   - Pre-conv (512→1024, k=3)
   - Pre-transformer (8 layers with sliding window attention)
   - ConvNeXt upsample (2 blocks, 2x each)
   - Initial conv (1024→1536, k=7)
   - 4 decoder upsample blocks (rates: 8,5,4,3)
   - Final conv (96→1, k=7)
   - Snake activation

7. **qwen_tts_generate()** - Main generation loop:
   - Prompt construction with THINK tokens
   - Autoregressive generation
   - Code Predictor integration
   - Speech decoder call

### ✅ Key Corrections Applied

1. **RoPE: Interleaved (NOT NeoX)**
   - Original implementation used NeoX split-half RoPE
   - Corrected to interleaved RoPE (consecutive pairs)
   - Applied to: Talker, Code Predictor, Speech Decoder

2. **Codec Prefix: THINK tokens**
   - Format: `[THINK/NO_THINK, THINK_BOS, language (opt), THINK_EOS, speaker, PAD, BOS]`
   - Required for proper EOS generation

3. **Architecture Parameters (verified against official config)**
   - Talker: hidden=1024, layers=28, heads=16/8, head_dim=128
   - CP: hidden=1024, layers=5, heads=16/8, head_dim=128
   - Decoder: hidden=512, layers=8, heads=16, head_dim=64, rope_theta=10000
   - Codebooks: 16 × 2048 entries × 256 dim
   - Upsample: [8, 5, 4, 3] rates, [2, 2] ConvNeXt ratios

## Next Steps

To complete the implementation:

1. **Implement full talker prefill/step** - Copy from qwen-asr structure, adapt for GQA + Q/K RMSNorm
2. **Implement speech decoder** - Most complex part, needs:
   - VQ dequant with EMA correction
   - Pre-transformer with sliding window attention (window=72)
   - ConvNeXt blocks with Snake activation
   - Decoder upsample blocks with dilated residual convs
3. **Implement generation loop** - Prompt construction, sampling, EOS detection
4. **Test against Python reference** - Compare intermediate tensors at each stage

## Files to Reference

- **MODEL.md** - Architecture documentation (mostly accurate)
- **ARCHITECTURE_COMPARISON.md** - Detailed comparison with official config
- **qwen-asr/** - Similar structure for reference (RMSNorm, attention, etc.)
- **qwen_tts/** (Python module) - Official implementation for algorithm reference

## Build & Test

```bash
# Build
make clean && make

# Test (once fully implemented)
./qwen_tts_bin -d qwen3-tts-0.6b --text "Hello world" -o test.wav --language English
```
