# Qwen3-TTS C Implementation - STATUS REPORT

**Date:** March 3, 2026  
**Build Status:** ✅ SUCCESS (with warnings)  
**End-to-End Inference:** ❌ NOT WORKING (critical gaps)

---

## 🎯 Executive Summary

### What Works ✅
- **Build system**: Compiles successfully with Accelerate (macOS) / OpenBLAS (Linux)
- **Model loading**: Weights loaded via memory-mapped safetensors
- **Speech decoder**: Full 9-step decoder implemented and complete
- **Sampling**: Temperature, top-k, top-p, repetition penalty all working
- **Audio output**: WAV writer (24kHz, 16-bit PCM, mono) functional
- **Architecture**: All dimensions verified correct against official config

### Critical Issues ❌
1. **BLAS Memory Corruption**: All BLAS calls disabled in `qwen_tts_talker.c`
   - Lines: 406-407, 467-468, 512-513
   - Comment: "DISABLED: BLAS causes memory corruption"
   - Impact: Prefill runs on slow manual loops instead of optimized BLAS
   
2. **BPE Tokenizer Missing**: Only character-level placeholder implemented
   - Real BPE requires `vocab.json` + `merges.txt` parsing
   - GPT-2 style byte-level encoding not implemented
   
3. **Talker Prefill/Step Incomplete**: Uses manual loops, not production-ready
   - Missing proper KV cache management
   - Debug `fprintf` statements throughout code
   
4. **Code Predictor Simplified**: Only argmax sampling
   - Missing temperature/top-k/top-p for codebooks 1-15

---

## 📊 Comparison: C Implementation vs vLLM/Transformers

### vLLM Omni (github.com/vllm-project/vllm-omni)

| Feature | vLLM Omni | Our C Implementation |
|---------|-----------|---------------------|
| Backend | CUDA (GPU) | CPU-only (BLAS) |
| Dependencies | PyTorch, CUDA | BLAS only |
| KV Cache | PagedAttention | Manual allocation |
| Precision | FP16/BF16 | BF16 weights, F32 compute |
| Streaming | ✅ 97ms latency | ⚠️ Theoretical |
| RTF | ~0.313 (GPU) | TBD (BLAS disabled) |

### HuggingFace Transformers (Official)

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", 
    trust_remote_code=True
)
audio = model.generate("Hello world")
```

**Our Implementation Alignment:**
- ✅ Same architecture parameters (verified)
- ✅ Same special token IDs (verified)
- ✅ Same prompt format (ChatML + codec prefix)
- ✅ Same codec generation schedule
- ❌ BPE tokenizer not implemented
- ⚠️ Sampling parameters match but not validated

---

## 🔍 Critical Bugs to Fix

### 1. BLAS Memory Corruption in Talker Prefill

**File:** `qwen_tts_talker.c`  
**Lines:** 406-407, 467-468, 512-513

**Current code:**
```c
#if 0 && defined(USE_BLAS)  /* DISABLED: BLAS causes memory corruption */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, q_dim, h, 1.0f,
                pref_x_norm, h,
                wq_f32, h, 0.0f,
                pref_q, q_dim);
#else
    // Manual loop fallback
    for (int t = 0; t < seq_len; t++) {
        for (int r = 0; r < q_dim; r++) {
            float sum = 0;
            for (int c = 0; c < h; c++) {
                sum += pref_x_norm[t * h + c] * wq_f32[r * h + c];
            }
            pref_q[t * q_dim + r] = sum;
        }
    }
#endif
```

**Fix required:**
- Debug BLAS call parameters (matrix dimensions, strides)
- Check memory alignment for bf16→f32 converted weights
- Verify row-major vs column-major layout
- Test with AddressSanitizer to find corruption source

### 2. Missing BPE Tokenizer

**File:** `qwen_tts_tokenizer.c`  
**Current:** Character-level stub (NOT functional for real inference)

**Required implementation:**
```c
// 1. Load vocab.json (GPT-2 byte-level unicode mapping)
// 2. Load merges.txt (BPE merge rules)
// 3. Encode text → bytes → apply merges → token IDs
```

**Reference:** Qwen2.5 tokenizer or GPT-2 tokenizer implementation

---

## 📁 File Status Summary

| File | Status | Completeness |
|------|--------|--------------|
| qwen_tts.h | ✅ | 100% |
| qwen_tts.c | ⚠️ | 90% (tokenizer stub) |
| qwen_tts_talker.c | ❌ | 60% (BLAS broken) |
| qwen_tts_code_predictor.c | ⚠️ | 70% (argmax only) |
| qwen_tts_speech_decoder.c | ✅ | 100% |
| qwen_tts_kernels.c | ✅ | 100% |
| qwen_tts_sampling.c | ✅ | 100% |
| qwen_tts_tokenizer.c | ❌ | 10% (stub) |
| qwen_tts_safetensors.c | ⚠️ | 80% (multi-shard stub) |
| qwen_tts_audio.c | ✅ | 100% |
| main.c | ✅ | 100% |
| Makefile | ✅ | 100% |

---

## ✅ Architecture Verification

All parameters verified against `config.json` and official model:

### Talker (0.6B)
- hidden_size: 1024 ✅
- text_hidden_size: 2048 ✅
- num_layers: 28 ✅
- num_heads: 16, KV heads: 8 (GQA 2:1) ✅
- head_dim: 128 (q_dim=2048) ✅
- intermediate_size: 3072 ✅
- RoPE: **Interleaved** (NOT NeoX), theta=1e6 ✅
- Q/K RMSNorm: per-head, eps=1e-6 ✅

### Code Predictor
- hidden_size: 1024 ✅
- num_layers: 5 ✅
- lm_head groups: 15 (codebooks 1-15) ✅

### Speech Decoder
- hidden_size: 512 ✅
- pre_transformer_layers: 8 ✅
- sliding_window: 72 ✅
- codebooks: 16 × 2048 × 256 ✅
- upsample_rates: [8, 5, 4, 3] = 480× ✅
- ConvNeXt_ratio: [2, 2] = 4× ✅
- Total: 1920× (12.5 Hz → 24 kHz) ✅

---

## 🧪 Test Infrastructure

### Python Tests (Working)
- `test_full_decoder.py` - 9-step decoder reference ✅
- `test_hf_direct.py` - Direct HuggingFace test ✅
- `test_codec_head.py` - Codec head validation ✅
- `validate.py` - C vs Python comparison suite ✅

### C Tests
- `test_decoder_standalone.c` - Decoder test ✅

### Test Audio Files (Generated)
- 28+ WAV files in root directory
- Various tests with different configurations

---

## 📋 Action Items (Priority Order)

### P0 - Critical (Blocks End-to-End)
1. **Fix BLAS memory corruption** in `qwen_tts_talker.c`
   - Debug sgemm parameters
   - Check memory alignment
   - Test with AddressSanitizer
   
2. **Implement BPE tokenizer**
   - Load `vocab.json` + `merges.txt`
   - Implement GPT-2 byte-level encoding
   - Apply BPE merges

### P1 - High Priority
3. **Clean up Talker prefill/step**
   - Remove debug `fprintf` statements
   - Implement proper KV cache growth
   - Test with real model

4. **Add Code Predictor sampling**
   - Temperature/top-k/top-p support
   - Proper KV cache per codebook

### P2 - Medium Priority
5. **End-to-end validation**
   - Generate audio with C implementation
   - Compare vs Python reference
   - Measure RTF and quality

6. **Multi-shard safetensors support**
   - Required for 1.7B model
   - Implement shard loading logic

### P3 - Nice to Have
7. **Streaming inference**
   - Chunk-by-chunk generation
   - Callback-based audio output

8. **Performance optimization**
   - Benchmark BLAS vs manual loops
   - Profile hotspots
   - Optimize memory access patterns

---

## 🚀 How to Test (Once Fixed)

```bash
# Build
make clean && make

# Test with 0.6B model
./qwen_tts_bin \
    --model-dir qwen3-tts-0.6b \
    --text "Hello world" \
    --output test.wav \
    --speaker ryan \
    --language English \
    --temperature 0.9 \
    --top-k 50 \
    --top-p 1.0

# Expected output:
# - test.wav (24kHz, mono, ~1-2 seconds)
# - Console output with generation stats
```

---

## 📚 References

- **MODEL.md**: Architecture documentation
- **ARCHITECTURE_COMPARISON.md**: Official vs C comparison
- **PLAN.md**: Implementation plan (Italian)
- **AGENT.md**: Developer guide
- **qwen-asr**: Reference implementation (similar architecture)
- **vLLM Omni**: https://github.com/vllm-project/vllm-omni
- **Official Qwen3-TTS**: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
