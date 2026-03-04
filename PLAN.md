# Qwen3-TTS — Next Steps & Performance Plan

## Current Status (Fully Working)

- Full pipeline verified: prompt → Talker → Code Predictor → Speech Decoder → WAV
- Correlation 0.999996 with Python reference (fully greedy)
- ~0.7x realtime on Apple Silicon M-series (4-thread dispatch_apply + BLAS)
- Bottleneck: Code Predictor at 55% of total time, near memory bandwidth ceiling

---

## TODO 1: Adopt qwen-asr Kernel Optimizations (Quick Wins) — DONE ✓

qwen-asr ha diverse ottimizzazioni nei kernel che noi non usiamo ancora.
Sono tutte provate in produzione e portabili.

### 1.1 Multi-Row BF16 Matvec — DONE ✓
qwen-asr processa **2 righe in parallelo** su NEON (4 su AVX-512), usando
8 accumulatori per riga. Noi processiamo 1 riga per thread call.
Il vantaggio è che il vettore x viene letto **una volta** e riusato per più righe,
dimezzando i load dalla memoria.

- **File**: `qwen_tts_kernels_neon.c`, `qwen_tts_kernels_avx.c`
- **Expected speedup**: 20-40% su matvec (che è il bottleneck)
- **Rischio**: Basso, è code collaudato da qwen-asr

### 1.2 Unified QKV Dispatch — DONE ✓
qwen-asr usa un singolo `parallel_for` che computa Q, K, V in un solo dispatch.
Noi facciamo 3 dispatch_apply separati (uno per Q, uno per K, uno per V).
Questo spreca overhead di sincronizzazione thread (3x barrier).

- **File**: `qwen_tts_talker.c`, `qwen_tts_code_predictor.c`
- **Expected speedup**: 5-15% per layer (elimina 2 barriere/layer)

### 1.3 Fused Gate+Up Weight Layout — DONE ✓
qwen-asr interleave le righe di gate e up_proj durante il loading, così un
singolo matvec produce entrambi i risultati. Noi facciamo 2 matvec separati.
Il vettore x viene letto 1 volta invece di 2.

- **File**: `qwen_tts_talker.c` (loading + forward), `qwen_tts_kernels.c`
- **Expected speedup**: ~30% sul FFN (che è ~40% del per-layer time)
- **Nota**: Già avevamo `gate_up_fused_bf16` nella struct ma non è chiaro se usato

### 1.4 BF16 Persistent Cache (for Prefill)
qwen-asr ha un cache opzionale per weight matrices bf16→f32 convertite, utile
durante il prefill dove la stessa matrice è usata su più token. Controllato via
`QWEN_BF16_CACHE_MB` env var.

- **Expected speedup**: 10-20% su prefill (evita riconversioni bf16→f32)
- **Rischio**: Basso, è opt-in

### 1.5 Pthread Pool (Portabilità)
Noi usiamo `dispatch_apply` (Apple-only). qwen-asr usa un thread pool con
pthread + condition variables, funziona ovunque. Valutare migrazione per
supporto Linux.

---

## TODO 2: INT4 Weight Quantization — ABANDONED ✗

### Risultato: NON FUNZIONA per il modello 0.6B

**Testato con Q4_0** (32 pesi/blocco, 1 scale fp16, kernel NEON con
unpack int4→int8→int16→int32→f32 + FMA):

| Metric | BF16 | Q4_0 | Delta |
|--------|------|------|-------|
| CP ms/frame | 57.1 | 68.6 | **+20% SLOWER** |
| Talker ms/frame | 27.2 | 27.9 | ~same |
| Audio quality | Perfect | **Degraded** | EOS shifted 44→146 frames |

### Perché non funziona

1. **Non siamo bandwidth-bound**: Con hidden=1024 le matrici sono piccole
   (largest: gate_up_fused 6144×1024 = 12MB bf16). Il matvec completa prima
   di saturare la bandwidth DRAM (~60-100 GB/s su Apple Silicon).

2. **Overhead di unpacking domina**: La catena int4→int8→int16→int32→f32
   richiede ~8 NEON istruzioni per 32 pesi, vs bf16→f32 che è un semplice
   shift left (quasi gratis). Per matrici piccole il compute extra > bandwidth saved.

3. **Perdita di precisione inaccettabile**: Q4_0 symmetric quantization
   introduce troppo rumore nei logits, causando EOS mancati e audio degradato.

### Dove potrebbe funzionare

- Modello 1.7B (hidden=2048, matrici 4x più grandi → più bandwidth-bound)
- Con Q4×Q8 integer dot products (vdotq_s32 SDOT) invece di float FMA
- Con Q6_K o Q5_1 (meno perdita di precisione)

**Codice Q4 rimosso** — non committato.

---

## TODO 3: Apple Silicon GPU Offloading (Metal, Optional)

### Opzione `make metal`
Metal compute shaders per accelerare il matvec bandwidth-bound.
Apple Silicon ha GPU integrato con accesso alla stessa DRAM (zero-copy).

- **Target**: Talker decode + CP forward
- **Approach**: Metal compute shaders o MPS
- **Expected speedup**: 3-5x (GPU bandwidth 200-400 GB/s vs CPU ~60-100 GB/s)
- **Con INT4 + Metal**: potenzialmente 5-10x vs attuale

```makefile
metal: CFLAGS += -DUSE_METAL
metal: LDLIBS += -framework Metal -framework MetalPerformanceShaders -framework Foundation
metal: SRCS += qwen_tts_metal.m
metal: blas
```

**Nota**: INT4 + Metal è il setup usato da llama.cpp per i benchmark più veloci
su Apple Silicon. I kernel Metal di llama.cpp supportano nativamente Q4_0.

---

## TODO 4: Build System Improvements (Done ✓)

- ✓ `make help` as default target
- ✓ Test targets: `test-en`, `test-it-ryan`, `test-it-vivian`, `test-all`
- ✓ Explicit header dependencies
- ✓ Linux OpenBLAS flags fix
- ✓ `-DDEBUG` in debug builds
- ✓ `CFLAGS_BASE` / `UNAME_S` variables

---

## Available Speakers & Languages

### Speakers (--speaker / -s)
| Name | ID | Notes |
|------|------|-------|
| ryan | 3061 | Default, native English male |
| serena | 3066 | Native Chinese female |
| vivian | 3065 | Female |
| uncle_fu | 3010 | Male |
| aiden | 2861 | Male |
| ono_anna | 2873 | Native Japanese female |
| sohee | 2864 | Native Korean female |
| eric | 2875 | Male |
| dylan | 2878 | Male |

### Languages (--language / -l)
Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

---

## Priority & Implementation Order

| # | Task | Impact | Status |
|---|------|--------|--------|
| 1 | **Multi-row matvec** | ~5% decode | ✓ Done |
| 2 | **Unified QKV dispatch** | ~5% decode | ✓ Done |
| 3 | **Fused gate+up** | ~4% Talker | ✓ Done |
| 4 | **INT4 quantization** | Slower + bad quality | ✗ Abandoned |
| 5 | **BF16 persistent cache** | 10-20% prefill | TODO |
| 6 | **Metal GPU offload** | 3-5x decode | TODO |
| 7 | Streaming output | UX | TODO |

Steps 1-3 done. INT4 non funziona per 0.6B.
Next viable perf gain: Metal GPU offload (TODO 3) o BF16 cache (TODO 1.4).
