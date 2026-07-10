# PR #17 — review, port selettivo, e i bug che ha fatto emergere

> **Stato: numeri M1 + Neoverse-N1 REALI (2026-07-10).** Restano ⏳ solo le voci non ancora portate
> (snake poly, spin-pool fixato, conv int8) e il profiling. La risposta all'autore si dà dopo quelle.

PR: `ARM perf: SDOT Q4_0 matvec, spin-pool, int8 decoder convs, exact streaming conv state`
Autore: **TrinityTF (Agu Toomas Pihelgas)**, contributor esterno. Qualità alta: verifica md5
bit-exact su più pattern di chunk, analisi SNR delle block-scale, risultati negativi documentati.
Target: **4-core Oracle A1 / Neoverse-N1**. Dichiarato: 0.6B stream RTF **2.21 → 1.34**, file 1.59 → 1.21.
`mergeable: CONFLICTING` (base `a4f6337`; su `main` sono poi atterrati B1, C7, `submit_mtx`, D2 —
negli stessi tre file).

**Metodo: nessun merge.** Ogni commit valutato da solo, portato a mano solo se regge la misura.
Branch: `perf/pr17-review` (5 commit).

---

## 1. Verdetto per commit

### 1.1 `cae8fff` — server `chunk_frames` come param JSON → **PRESO il param, RIFIUTATO il default**

Loro alzano il default 10 → 50. Quel default esisteva per **ammortizzare il re-decode del contesto
conv** su meno confini di chunk. Col conv esatto (§1.4) quel costo **non esiste più**, quindi 50
peggiora soltanto la granularità dello stream. Preso il parametro (utile: il client sceglie il
compromesso latenza/throughput), default lasciato a 10.

Verificato: nessun `chunk_frames` == `chunk_frames:10` (stesso md5, retrocompatibile); audio
invariante a chunk 2/10/100 (`mel_corr 1.00000`) — proprietà che **prima del conv esatto non avevamo**.

### 1.2 `91a09f1` — spin-then-sleep pool → **RIFIUTATO il diff. L'idea si riprende, il codice no.**

Tre motivi, in ordine di gravità:

1. **Cancella il nostro `submit_mtx`** (commit `d2b5df2`). La loro `qwen_parallel` riscrive esattamente
   il corpo dove vive il lock che serializza i submitter concorrenti. Un rebase cieco reintroduce
   l'hang intermittente del batched-server su Linux.

2. **Lost-wakeup reale.** Il dispatcher pubblica `generation` con `memory_order_release` e poi legge
   `sleeping` **fuori dal mutex**, saltando la `broadcast` se nessuno dorme. È il litmus store-buffer
   (Dekker): perché l'esito `(sleeping=0, generation=old)` sia vietato servono **due** store seq_cst.
   Su ARM64 la coppia `stlr`→`ldar` ordina lo store-load e il bug **non si manifesta** — ecco perché
   il loro box non l'ha mai visto. **Su x86 lo store resta nel store buffer** oltre la load: il
   dispatcher salta la broadcast mentre il worker si addormenta, `completed` non raggiunge mai
   `nworkers`, deadlock. E x86 è proprio dove già inseguiamo un hang intermittente del batched-server.
   Fix necessario: publish seq_cst + `atomic_thread_fence(seq_cst)` prima di leggere `sleeping`.

3. **Su macOS non gira.** `qwen_parallel` è GCD (`dispatch_apply`, `qwen_tts_thread.c:19`); il pool
   POSIX che ottimizzano è compilato solo fuori da Apple. **Non misurabile su M1** → ⏳ box ARM-Linux.

Nota: anche il loro `QWEN_POOL_SPIN` non è un env var in questo commit (è un `#define`); l'override
via env arriva solo in `a471840`.

### 1.3 `10fa776` — NEON SDOT q4_0 → **PRESA 1 idea su 2**

Il kernel è il nostro **B1** re-inventato in parallelo (`ee0df43` + `882a504`), e senza il gemello
x86 VNNI (`5095e31`/`9a53787`/`fc49db3`) che noi abbiamo. Ma contiene due miglioramenti veri:

- ✅ **Accumulo `float32x4` (PRESO).** Il nostro loop faceva un `vaddvq_s32` (riduzione cross-lane) **+**
  una FMA scalare **ogni 32 pesi**, entrambi sulla catena di dipendenze. Accumulare `scale*lanes` in un
  `float32x4` e ridurre una volta per riga è la stessa somma, riassociata. **Indipendente dal layout**,
  raggio d'azione zero. → commit `e6208f2`.

- ❌ **Packing nibble deinterleaved (RIFIUTATO per ora).** Toglie la `vzipq_u8` (ARM) e la `unpacklo/hi`
  (x86). Ma è un **cambio di formato che tocca 13 siti**, inclusi `qwen_tts_metal.m` e
  `qwen_tts_cuda_talker.cu`, che la loro base non conosceva: mergiarlo lascerebbe i decoder GPU a
  leggere il layout vecchio su byte nuovi → **audio GPU sbagliato, in silenzio**. E non risolve il vero
  collo di bottiglia x86 (il q4-VNNI è compute-bound sulla reduce per blocco + datapath 512-bit
  mezzo vuoto — vedi `project_x86_epyc_vnni_validation`). Da rivalutare **dopo** aver misurato se la
  `vzipq` è ancora sul percorso critico una volta applicato l'accumulo.

**Misura M1** (0.6B, `--int4 -j4`, A/B interleaved ×5, talker+CP, mediana):
`4478 ms → 4179 ms` (**−6.7%**); minimi `4343 → 4035`. int8/bf16 invariati (controllo). `--self-test` PASS.

> **Perché il gate qui non è mel-corr.** L'int4 **forka traiettoria** a qualunque riassociazione fp:
> già oggi su `main`, SDOT vs `QWEN_NO_SDOT` dà `mel_corr 0.906` su una clip di due parole. Il gate
> corretto è `--self-test` (rel_L2 ~4e-3) **+ ascolto**.

### 1.4 `a471840` — decoder 2.5× → **PRESO 1/3 (C). A e B rimandati.**

Tre sotto-modifiche indipendenti, impacchettate insieme.

**(A) snake NEON polinomiale — ⏳ RIMANDATO (non misurabile su M1).**
Su M1 `snake_activation` prende il ramo `#if defined(__APPLE__) && defined(USE_BLAS)` → `vvsinf` di
Accelerate, già vettoriale. Il ramo `__ARM_NEON` che loro ottimizzano (oggi: `sinf()` scalare per
elemento) **non viene compilato su Apple**. Il loro 1209 → 90 ms è un guadagno Linux-ARM reale, ma
qui è dead code. Additivo e a rischio zero per M1/x86. → validare a orecchio su box ARM-Linux.

**(B) conv int8 SDOT nel decoder (`QWEN_SD_INT8=1`) — ⏳ RIMANDATO, opt-in, non attivare.**
Quantizza **entrambi** gli operandi con block-scale per-64 (Q8_0-style). SNR full-clip 37.7 dB ma
**26 dB nel segmento peggiore da 4k**: basso per l'audio, e il rischio cade esattamente sui gioielli
del progetto (emotion + voice clone). Da ear-validare su HW vero prima di considerarlo. Loro stessi
documentano un risultato negativo onesto: int8 ConvTranspose è **più lento** dell'sgemm fp32 (K piccolo).

**(C) conv streaming esatto — ✅ PRESO. È il vero premio.** → commit `fff169e`.
Lo streaming ridecodava `conv_rf=20` frame di contesto **per chunk**, buttandone l'audio: **3× il lavoro
conv a chunk=10**, 1.8× a chunk=24. Portando lo stato causale (tail di `pad_left` per ogni conv1d,
carry overlap-add di `(kernel-stride)` per ogni ConvTranspose) si consumano **solo i frame nuovi**, e
l'output chunked **coincide con il one-shot**.

Portato a mano, non mergiato — la loro base non conosceva né D2 né il refactor per-slot:
1. **Bug loro:** leggono il latent a `(latent_frames - new_frames)` **senza `- latent_base`** → indice
   sbagliato dopo la prima compattazione D2. Silenzioso sulle clip corte. Corretto.
2. Stato spostato nel `qwen_sd_stream_state_t` posseduto dal chiamante (non `ctx->sd_stream`), così il
   batching per-slot del server mantiene B stati indipendenti.
3. `convnext_mlp` estratto: one-shot e streaming condividono la coda del blocco **per costruzione**.
4. Decoder CUDA-resident → fallback al windowed (non porta stato). `QWEN_SD_WINDOWED=1` ripristina il vecchio.

---

## 2. I bug che il port ha fatto emergere — **due erano nostri**

### 2.1 `causal_conv_transpose1d`: il guard che rendeva impossibile il carry (nostro, latente)

```c
if (out_pos < full_len - trim_right && out_pos < out_len)   // prima
if (out_pos < out_len)                                      // dopo
```
Il trim a destra **è già espresso da `out_len`**: i chiamanti one-shot passano
`out_len = in_len*stride = full_len - trim_right`, quindi sono **bit-identici**. Ma il path streaming
ha bisogno proprio di quelle colonne di coda come carry, e il guard interno le azzerava sempre.

Senza il fix il carry era tutto zeri. **Non l'ha trovato la review, l'ha trovato il test**:
chunked-vs-one-shot fermo a `mel_corr 0.9954` a chunk=2 — *peggio* del path windowed. Un port che
"sembrava funzionare".

### 2.2 Makefile: dipendenze header a mano → `.o` stantii → **corruzione heap silenziosa** (nostro, grave)

`qwen_tts_compose.o`, `qwen_tts_audio.o`, `qwen_tts_emotion.o`, `qwen_tts_speech_encoder.o` includono
`qwen_tts.h` ma **non avevano una regola che lo nominasse**. Allargare `qwen_sd_stream_state_t`
(embedded per valore in `qwen_tts_ctx_t`) li ha lasciati compilati sul **layout vecchio**: scrivevano
attraverso campi a offset sbagliati.

**Sintomo:** `SIGABRT` dentro `realloc()` in `qwen_tts_generate`, sul **secondo span** di un compose
inline-emotion (`[joy] … [sad] …`), su 0.6B **e** 1.7B. Nessun messaggio di malloc. Il crash era
lontanissimo dalla causa, e **si riproduceva anche con `QWEN_SD_WINDOWED=1`** — che è precisamente ciò
che ha scagionato il codice nuovo.

**Diagnosi che ha funzionato:** bisect sui binari → notare che crasha anche il path vecchio →
confrontare i timestamp dei `.o` con l'header modificato. (ASan: inutilizzabile, timeout a 10 min.)

**Fix:** `-MMD -MP` + `-include $(OBJS:.o=.d)` (commit `0d7246b`). Lo **stesso buco** era aperto su
`qwen_tts_metal.m` e `qwen_tts_cuda_{talker,decoder}.cu`, che includono `qwen_tts.h` e incorporano
`qwen_tts_ctx_t`: un `make metal`/`make cuda` incrementale sarebbe stato **silenziosamente corrotto**
(commit `73b8576`).

> Conseguenza onesta: **un giro di benchmark è stato buttato** perché preso con un binario
> parzialmente stantio. Regola: `make clean` prima di credere a un bug **o a una misura**.

---

## 3. Numeri — Apple M1 (8-core, 16 GB)

Build pulite di entrambi i binari. `-j4`, `--seed 42 -s ryan -l Italian`, stesso testo.
Protocollo: coppie BEFORE/AFTER **adiacenti**, 5 ripetizioni, statistica = **min**
(la mediana era contaminata: load average fino a 13.8 su 8 core).

| modello | config | RTF prima | RTF dopo | Δ RTF |
|---|---|---|---|---|
| 0.6B | file bf16 | 1.22 | 1.08 | **−11.5%** |
| 0.6B | file int8 | 0.84 | 0.69 | **−17.9%** |
| 0.6B | file int4 | 0.65 | **0.52** | **−20.0%** |
| 0.6B | stream int4 (chunk 24) | 0.67 | **0.56** | **−16.4%** |
| 1.7B | file bf16 | 1.89 | 1.65 | **−12.7%** |
| 1.7B | file int8 | 1.56 | 1.54 | −1.3% |
| 1.7B | file int4 | 1.13 | **0.87** | **−23.0%** |
| 1.7B | stream int4 (chunk 24) | 0.91 | 0.87 | −4.4% |

**TTFA: invariato.** Tutti i Δ misurati (−14% … +18%) cadono **sotto la dispersione** dei 5 run
(spread 100–3600 ms). Rumore, non segnale: non rivendicati.

**Perché int8 guadagna poco (e cosa implica per le altre box).** Il conv esatto taglia il decoder del
~40–50% in *tutte* le precisioni, ma il decoder gira **in parallelo** alla generazione. In int8/bf16 la
generazione è più lenta e resta lei il percorso critico → il guadagno del decoder è nascosto. In int4
la generazione è veloce, il decoder emerge, e si vede tutto.
→ **Su hardware più lento della M1 il Δ end-to-end sarà più piccolo di quanto suggerisca `decoder_ms`.**

### Gate superati (M1)
- `--self-test` PASS (prima e dopo) · `make test-golden` PASS (0.6B en/it/int8, 1.7B en)
- `make test-batch` PASS · `make test-serve-repro` PASS (3 richieste bit-identiche)
- **chunked == one-shot**: `mel_corr 1.00000` a chunk **2 / 7 / 24 / 150** (prima: 0.9954 / 0.99918 /
  0.99968). Deviazione ≤ 2 LSB su <0.1% dei campioni: Accelerate `sgemm` riassocia in base alla forma,
  quindi il *byte-identical* dichiarato dalla PR **non è raggiungibile su Accelerate** — ma la
  matematica è esatta. One-shot **bit-identico** a `main`.
- **leaks**: 137 leak / 420096 B, **identici a `main`**, nessuna traccia dei nuovi `cs_*` (debito preesistente).
- **1.7B + emotion** (`--emotion sad|joy`): stream esatto == file == windowed, `mel_corr 1.00000`.
- **inline `[joy]`/`[sad]`** e **`[laugh]`**: OK (prima del fix Makefile: SIGABRT).
- Ascolto: `samples/tests/2026-07-10_pr17-review/` (17 file, README con cosa ascoltare per ognuno).

---

## 4. Validazione ARM-Linux — **FATTA** (Ampere Altra Max, Neoverse-N1, 4 core, Ubuntu 26.04)

Stessa µarch dell'Oracle A1 della PR. `--caps`: `NEON (2-row fused)` · `SDOT vdotq_s32 (native)` ·
**`pthread pool (4 threads)`** (→ qui il pool POSIX esiste davvero) · `OpenBLAS` · nessun bf16/i8mm/SVE.
Build con **gcc 15.2 reale** (su Mac `gcc` è clang: il fix `-MMD -MP` è quindi validato su entrambi).
`--self-test` PASS su entrambi i binari.

⚠️ Baseline = **il nostro `main`**, non il loro. 0.6B, `--seed 42 -s ryan -l Italian -j4`, 3 rip, min.

| config | RTF prima | RTF dopo | Δ RTF | decoder prima | decoder dopo | Δ dec | TTFA prima | TTFA dopo |
|---|---|---|---|---|---|---|---|---|
| file `--int4` (chunk 10) | 2.70 | **1.59** | **−41.1%** | 19184 ms | 8300 ms | −56.7% | 1832 ms | 1851 ms |
| file `--int8` | 2.56 | **1.58** | **−38.3%** | 15035 ms | 6339 ms | −57.8% | 1852 ms | 1855 ms |
| stream `--int4` chunk 24 | 1.96 | **1.48** | **−24.5%** | 11821 ms | 7602 ms | −35.7% | 920 ms | **994 ms** |
| stream `--int4` chunk 150 | 1.44 | 1.44 | **±0%** | 5786 ms | 5650 ms | −2.4% | 934 ms | 945 ms |

**Il chunk 150 che non guadagna nulla è la conferma del meccanismo**, non un'anomalia: a chunk 150 il
re-decode `conv_rf=20` pesa 13%; a chunk 10 (default file mode) pesa **3×**. Teoria e misura coincidono.

### TTFA streaming: regressione trovata qui, e poi **chiusa** (vedi §5.7)
Con il solo conv esatto, N1 mostrava TTFA 920 → 994 ms (su M1 cadeva nel rumore). Causa: il path esatto
anteponeva le code **anche al primo chunk**, quando sono ancora tutte zero — e un prepend di zeri è
identico allo zero-pad causale che `causal_conv1d` già applica. Lavoro puro, buttato.
Fix `cs_warm` (`2142bc9`). **A/B appaiato `main` vs HEAD ×5: nessuna regressione residua** (TTFA mediana
948 vs 962 ms, a favore di HEAD). ⚠️ Il +8% qui sotto è pre-fix e proveniva da run non appaiati.

### Confronto col claim della PR
| | loro (dichiarato) | noi (misurato, stessa µarch) |
|---|---|---|
| stream chunk 24 | 2.21 → **1.34** | 1.96 → **1.48** |
| file chunk 150 | 1.59 → **1.21** | 1.44 → 1.44 |

Il nostro **prima** è già migliore del loro (`main` ha D2 e altro che la loro base non aveva). Il nostro
**dopo** è più lento del loro **esattamente dei pezzi che non abbiamo portato**: snake poly + conv int8
+ spin-pool. Il conto torna, e dice quanto valgono quei pezzi su questa box: **~0.14 RTF sullo stream**.

### 4.1 snake poly (commit `5f6e654`) — misurata su N1: **corretta, ma il guadagno è modesto**

Correttezza (stesso binario, `QWEN_NO_SIN_POLY=1` come A/B): **SNR 101.3 dB**, max **1 LSB**, 0.04% dei
campioni diversi. Il kernel è giusto.

Velocità (0.6B, `--int4 -j4`, coppie adiacenti ×3, min):

| config | libm `sinf` | poly NEON | Δ decoder |
|---|---|---|---|
| file `--int4` | RTF 1.58, dec 8367 ms | RTF 1.57, dec 8163 ms | **−4%** |
| stream chunk 24 | RTF 1.52, dec 7856 ms | RTF 1.51, dec 7479 ms | **−3%** |

**Non è il 3.5× che la PR dichiara per il polinomio** (1209 → 341 ms). Motivo: da noi la snake non è
mai stata il collo di bottiglia — vedi il profilo qui sotto. Il commit resta (è corretto, gratis, e
toglie 200-380 ms), ma **non è il pezzo che vale**.

### 4.2 `perf record` — dove va DAVVERO il tempo su N1 (0.6B, `--int4 -j4`)

| simbolo | % |
|---|---|
| `sgemm_kernel_NEOVERSEN1` (OpenBLAS: conv del decoder) | **30.9%** |
| `q4_0_matvec_sdot` (Talker + CP) | **28.8%** |
| `__schedule` + `el0_svc` + `__sched_yield` + `do_sched_yield` (**kernel: scheduling/futex/yield**) | **~21%** |
| `conv_decoder_forward_streaming` | 2.7% |
| `causal_conv1d_blas` | 1.7% |
| snake | **non compare** (< 1.5%) |

**Il ~21% nello scheduler è il vero mostro, e nessuno l'aveva visto.** `__sched_yield` è OpenBLAS che
fa spin nelle sue barriere; `__schedule` è il futex del nostro pool. Cioè: **il nostro pool a 4 thread e
OpenBLAS a 4 thread si contendono 4 core** (oversubscription da parallelismo annidato).

→ Questo **valida l'idea** del commit spin-pool della PR (che avevamo rifiutato per il lost-wakeup e per
`submit_mtx`): il costo che attaccava è reale e grosso. Ma la prima mossa non è nemmeno lo spin-pool:
è capire quanti thread deve avere OpenBLAS quando il nostro pool ne ha già 4. ⏳ (misura in corso)

⏳ Restano: spin-pool **dopo** il fix (`submit_mtx` + fence seq_cst), `QWEN_SD_INT8=1` **con ascolto** su
emotion/clone, e il ramo `__AVX2__` della snake (stesso problema, box x86).

## 5. ⚠️ Il "debito delle trascendenti": tesi **in gran parte FALSIFICATA** sulla box

**Cosa avevo dedotto dal grep** (e che è finito, sbagliato, in una prima versione di questo doc e nel TODO):
usiamo Accelerate vForce in due soli punti (`vvexpf` in `qwen_swiglu_inplace:2941`, `vvsinf` in
`qwen_snake_activation:3046`), entrambi col ramo Apple per primo → quindi ARM-Linux e x86 pagherebbero
`expf()`/`sinf()` **scalari, per elemento**, e servirebbe un kernel trascendente vettoriale multi-ISA.

**Cosa dice la box.** `nm -D` sul binario compilato lì:
```
_ZGVnN4v_sinf@GLIBC_2.38   _ZGVnN4v_expf@GLIBC_2.38
_ZGVnN4v_erff@GLIBC_2.40   _ZGVnN4v_tanhf@GLIBC_2.40   _ZGVnN2v_expf@GLIBC_2.39
```
**gcc, con `-ffast-math`, auto-vettorizza già quei loop chiamando `libmvec` di glibc.** Non c'è nessun
`sinf` scalare per elemento: il compilatore lo sostituisce con la variante vettoriale a 4 lane. Coerente
col profilo, dove **`sinf` non compare nemmeno sopra l'1%**.

Conseguenze:
- **La snake poly rende poco** (−3/−4%, §4.1): non batte lo scalare, batte `_ZGVnN4v_sinf` — e solo perché
  calcola `sin²` senza gestire il segno né i quadranti. Il commit resta (corretto, gratis), ma **non è il pezzo che vale**.
- **La tesi "la expf della SwiGLU è un buco"** è morta due volte: prima per volume (40.6M vs 278.6M
  chiamate, ~1.3% di talker+CP), poi perché `_ZGVnN4v_expf` è già lì.
- **Un kernel trascendente multi-ISA scritto a mano NON è prioritario** su gcc+glibc≥2.38. Resterebbe utile
  solo dove libmvec non c'è: **clang** (non emette chiamate libmvec), **musl**, glibc vecchie, e
  **senza `-ffast-math`**. Da verificare sulla box x86 prima di scriverne una riga.

> Tre previsioni fatte a tavolino, tre smentite dall'hardware in un giorno: il peso della `expf`; il
> "TTFA invariato" (su N1 è una regressione reale dell'8%); e il buco delle trascendenti. Il pattern:
> **su questo progetto il grep propone, l'hardware dispone.**

## 5.1 Quello che il profilo dice DAVVERO di fare (N1)

| simbolo | % | azione |
|---|---|---|
| `sgemm_kernel_NEOVERSEN1` | 30.9% | conv del decoder via OpenBLAS — vedi oversubscription |
| `q4_0_matvec_sdot` | 28.8% | già ottimizzato (B1 + accumulo float32x4) |
| kernel scheduling/futex/yield | **~21%** | ⬅ **il vero mostro** |

**Oversubscription: il nostro pool (4 thread) e OpenBLAS (4 thread) su 4 core.** Sweep di
`OPENBLAS_NUM_THREADS` (0.6B `--int4 -j4`, 3 rip, min):

| | OB=1 | OB=2 | OB=3 | OB=4 (default) |
|---|---|---|---|---|
| file `--int4` | 2.14 | **1.48** | 1.52 | 1.58 |
| stream chunk 24 | 2.22 | 1.53 | **1.46** | 1.49 |

Reale ma **più modesto** del 21% suggerito: −6% in file, −2% in stream (una parte di quel 21% è il futex
del *nostro* pool, non la contesa con OpenBLAS). E **l'ottimo cambia col modo** (2 in file, 3 in stream) →
è una manopola da tarare per fase, non una costante. Il prefill usa BLAS pesantemente e vorrebbe più thread.

→ **Questo, non la snake, è ciò che il commit spin-pool della PR sfiorava.** La sua diagnosi (~7300 futex
per frame) era giusta; la cura (spin nel *nostro* pool) è la **seconda** mossa. La prima è non oversubscribere.

## 5.2 Buchi ISA verificati (inventario, non ipotesi)

`qwen_tts_kernels.c`: **40 funzioni con SIMD** — 870 intrinsics NEON, 202 AVX2, 70 AVX-512, 25 VNNI.
Matvec/matmat (bf16/int8/q4), rms_norm, attention, quantizzazione, SDOT, VNNI: **tutto coperto**.
`int8_matvec_fused` e `q4_0_matvec_inner` sembrano "senza AVX-512" ma sono **fallback**: su box VNNI il
dispatch prende `int8_matvec_vnni` / `q4_0_matvec_vnni` (verificato). Buchi veri che restano:

| dove | cosa | note |
|---|---|---|
| x86 | `qwen_causal_attention*` e `qwen_rms_norm` sono **AVX2, mai AVX-512** | metà larghezza su EPYC/SPR — da misurare |
| x86 | **q4-VNNI compute-bound** (int4 2.76 vs int8 2.01 su EPYC) | il blocco x86 maggiore; serve throughput-packing, **non** il packing della PR |
| x86 | AMX / AVX512-BF16 mai usati | Track C, serve Sapphire Rapids |
| ARM server | niente SVE / i8mm / BFMMLA | Track C, serve Graviton3/4 (questa Ampere non li ha) |
| ARM-Linux | spin-pool | **dopo** il fix; e dopo aver tarato OpenBLAS |
| ARM-Linux | conv int8 del decoder | opt-in, solo dopo ascolto su emotion/clone |

## 5.3 Il SUO albero contro il NOSTRO, sulla stessa box (N1) — chi arriva dove

Compilati entrambi sulla N1, stesso testo/seed/flag (0.6B, `--int4 -j4`, min di 3).

| build | stream chunk 24 | file chunk 150 | decoder (stream) |
|---|---|---|---|
| `a4f6337` — **la SUA base** | **3.45** | 2.85 | 15295 ms |
| il **nostro `main`** (stessa data della sua PR) | **1.96** | 1.44 | 11821 ms |
| **il suo albero completo** (4 commit) | **1.41** | 1.31 | 7735 ms |
| il suo albero **+ `QWEN_SD_INT8=1`** | **1.15** | **1.11** | 5112 ms |
| **il nostro HEAD** (branch review) | 1.49 | 1.44 | 7476 ms |
| il nostro HEAD **+ `OPENBLAS_NUM_THREADS=3`** | **1.46** | — | 7366 ms |

### Tre cose che questo tavolo dice, e che nessuna delle due parti sapeva

1. **Il claim `2.21 → 1.34` è vero ma non trasferibile.** La sua base `a4f6337` fa **3.45** dove il nostro
   `main` fa **1.96**: tra la sua base e il nostro main c'è **B1 (SDOT-q4)**, che lui ha **re-inventato**
   dentro la PR. Buona parte del suo Δ è un guadagno che il nostro albero aveva già preso per altra strada.
   Il suo contributo **netto sopra il nostro main** è molto più piccolo del Δ che dichiara.

2. **Il pezzo grosso non è quello che pensava nessuno dei due.** Non lo spin-pool, non la snake, non il
   packing: è la **conv int8 del decoder**. `1.41 → 1.15` in stream (**−18%**), `1.31 → 1.11` in file,
   decoder 7735 → 5112 ms. È **l'unica cosa che avevo messo da parte per la qualità** (26 dB worst-segment).
   → La decisione è **d'orecchio, non di benchmark**: audio in `samples/tests/2026-07-10_pr17-int8-conv/`.

3. **Lo spin-pool vale poco, isolato.** Sul suo stesso binario, `QWEN_POOL_SPIN=0` vs default:
   **1.43 → 1.39** (~3%). La sua *diagnosi* (~7300 futex/frame) era giusta e preziosa — il profilo la
   conferma col 21% nello scheduler — ma la sua *cura* cattura solo una fetta. La fetta più grossa la
   prende `OPENBLAS_NUM_THREADS` (che lui non tocca): il nostro HEAD passa da 1.53-1.58 a **1.46** con OB=3.
   ⚠️ E solo in certi modi: in stream OB=2 **peggiora** (TTFA 943 → 1300). Manopola per fase, non costante.

**Divario residuo noi↔lui:** ourHEAD+OB=3 **1.46** vs hisPR **1.39** ≈ 5%, attribuibile a spin-pool (~3%)
+ il suo packing deinterleaved. Nulla che giustifichi il rischio dei 13 siti CUDA/Metal (§1.3).

## 5.4 Il port della conv int8 nel NOSTRO albero — validato su N1 (commit `68bddbf`)

Portato dopo l'ok all'ascolto. Tre scostamenti deliberati dal suo codice:
- **Threading:** lui crea *sempre* un pool privato (il pool POSIX non è rientrante ed è posseduto dal thread
  di generazione; col nostro `submit_mtx` un secondo submitter lo bloccherebbe). Vero su Linux, **inutile su
  macOS** (GCD rientrante) → usiamo `qwen_parallel` se `qwen_parallel_is_reentrant()`, il suo pool altrimenti.
- **ConvTranspose int8 NON portata:** lui stesso l'ha misurata più lenta dell'sgemm fp32 e la spedisce spenta.
- **Stesso scoping** (`in_ch == out_ch && in_ch <= 768`): le conv nel dominio latente restano fp32.

`QWEN_SD_INT8=1`, **mai default**: scambia qualità con velocità, la scelta resta di chi usa il motore.

### Il percorso completo su N1 (0.6B, `--int4 -j4`, stream chunk 24, min di 3)

| build | RTF | decoder | TTFA |
|---|---|---|---|
| la sua base `a4f6337` | 3.45 | 15295 ms | 1137 ms |
| **il nostro `main`** | 1.96 | 11821 ms | 920 ms |
| + accumulo float32x4 + conv esatto (`e6208f2`,`fff169e`) | 1.51 | 7545 ms | 975 ms |
| + snake poly (`5f6e654`) | ~1.48 | 7479 ms | — |
| **+ conv int8** (`68bddbf`, `QWEN_SD_INT8=1`) | **1.29** | 5159 ms | 1020 ms |
| il suo albero + conv int8 (riferimento) | 1.19 | — | — |

File mode `--int4`: **1.65 → 1.39** con la conv int8 (decoder 8577 → 6528 ms).

**Divario residuo noi↔lui ≈ 8%.** Attribuibile a: spin-pool (~3%, misurato isolando `QWEN_POOL_SPIN=0`),
il suo **snake threading** (lui distribuisce le righe della snake sul pool del decoder — non portato),
e il packing deinterleaved. Nessuno dei tre giustifica, oggi, il rischio dei 13 siti CUDA/Metal.

### Qualità: `--emotion` VERO + voice clone (M1, nostro albero)
M1 ha dotprod, quindi la conv int8 gira anche qui: la qualità si valida in locale, senza la box.

| caso | SNR full | rumore RMS | rumore di picco |
|---|---|---|---|
| `--emotion sad` (1.7B, THE recipe) | 37.4 dB | −65.7 dBFS | −43.0 dBFS |
| `--emotion joy` | 37.3 dB | **−55.1 dBFS** | −34.5 dBFS |
| voice clone (`--ref-audio`, 0.6B-base) | 43.3 dB | −67.6 dBFS | −47.3 dBFS |

> `joy` ha rumore assoluto 10 dB più alto di `sad` **ma lo stesso SNR**: il rumore di quantizzazione
> **segue il livello del segnale**, e joy è semplicemente più energico. Non è una fragilità dell'emozione.
> ⚠️ Costo TTFA: misurato **prima** del fix `cs_warm`. Dopo (§5.7), l'A/B appaiato `main` vs HEAD non
> mostra regressione. Il costo della conv int8 sul TTFA va rimisurato appaiato — non è ancora fatto.

Gate su M1: default OFF **bit-identico** al HEAD precedente · `--self-test` PASS · `make test-golden` PASS ·
leaks invariati (137 / 420096 B, come `main`). Audio: `samples/tests/2026-07-10_pr17-int8-conv/audio_our_port/`.

## 5.5 Debito aperto (nostro, creato oggi)

- ~~TTFA in streaming~~ **CHIUSO** (§5.7): `cs_warm` (`2142bc9`); A/B appaiato `main` vs HEAD senza regressione.
  Resta l'inefficienza (una `aligned_malloc` per conv per chunk) ma **senza costo osservabile** → bassa priorità.
- **Costo TTFA della conv int8** da rimisurare in coppie adiacenti (finora solo run non appaiati).
- **`OPENBLAS_NUM_THREADS` non è tarato**: default = ncpu ⇒ oversubscription col nostro pool. Ottimo
  dipendente dal modo (file 2, stream 3). Serve una manopola per fase, non una costante.
- **snake threading** e **spin-pool fixato** (publish seq_cst + fence, sopra `submit_mtx`): ~8% residuo.

## 5.6 `chunk_frames`: il default 50 avrebbe **peggiorato M1 del 18%**

Sweep sul codice finale, 0.6B `--int4 -j4`, min di 3. **Il TTFA è piatto** a ogni chunk (il ramp del primo
chunk lo fissa a 2 frame): la granularità si paga solo in RTF.

| chunk | RTF su **N1** | RTF su **M1** | audio/chunk |
|---|---|---|---|
| 2 | 2.11 | — | 0.2 s |
| 7 | 1.68 | — | 0.6 s |
| **10 (default)** | 1.60 | **0.56** | 0.8 s |
| 24 | 1.51 | 0.57 | 1.9 s |
| 50 (il loro default) | 1.49 | 0.61 | 4.0 s |
| 150 | **1.44** | **0.66** | 12.0 s |

**Direzioni opposte.** La causa, misurata su M1:

| chunk | decoder totale | **drain dopo la generazione** | RTF |
|---|---|---|---|
| 10 | 2929 ms | **134 ms** | 0.56 |
| 24 | 2569 ms | 398 ms | 0.58 |
| 50 | 2823 ms | 856 ms | 0.66 |
| 150 | 1867 ms | **1710 ms** | 0.69 |

Il decoder *totale* scende sempre, anche su M1. Ma il **drain** — l'ultimo chunk, decodificato dopo che la
generazione è finita, quindi senza nulla dietro cui nascondersi — esplode, e finisce tutto sul wall clock.

> **Regola: il chunk ottimale dipende da chi è il collo di bottiglia.** Decoder-bound (N1, generazione lenta)
> → chunk grandi vincono, riducono il lavoro totale. Generation-bound (M1, generazione veloce) → chunk grandi
> perdono, il drain non si nasconde più.

**Conseguenze:**
1. **Default 10 confermato.** È l'ottimo su M1 e costa il 6% su N1. Il loro 50 sarebbe stato **−18% su Apple
   Silicon**, la piattaforma principale del progetto. Avevano misurato su una macchina sola, e su quella
   avevano ragione.
2. **Il `chunk_frames` per-richiesta è la cosa giusta**: non esiste un numero buono per tutti, esiste una
   manopola. "Prendi il parametro, rifiuta il default" (§1.1) si rivela corretto per una ragione che nessuna
   delle due parti aveva previsto.
3. ⚠️ **Da documentare per gli utenti del parametro:** sotto i ~7 frame la curva punisce (chunk 2 → RTF 2.11
   su N1, contro 1.51 a chunk 24). Il parametro invita a scendere; il costo non è ovvio.

## 5.7 Il fix del TTFA (`2142bc9`) — **la regressione è chiusa**

A/B appaiato del fix (N1, coppie adiacenti ×3):
- **chunk=2:** pre-fix 970/980/1008 → post-fix 949/943/955. Range **disgiunti**: −30 ms reali (−3%).
- **chunk=24:** pre 982/991/945 → post 970/936/959. Sovrapposti (il primo chunk è 2 frame in entrambi).

E soprattutto, **`main` vs HEAD appaiati ×5** (stream chunk 24), che è il confronto che conta:

| | TTFA mediana | TTFA min | RTF |
|---|---|---|---|
| `main` | 962 ms | 944 ms | 1.98 |
| **HEAD** | **948 ms** | **921 ms** | **1.52** |

**Nessuna regressione residua**: HEAD è uguale o meglio di `main` sul TTFA, con −23% di RTF.

> ⚠️ **Correzione.** Il "+8% di TTFA" riportato in §4 e §5.4 veniva da run presi in momenti diversi
> (`main` misurato in una sessione, HEAD in un'altra). Misurato in **coppie adiacenti**, dopo `cs_warm`,
> non esiste. È la stessa lezione della mediana contaminata dal load: **su questa macchina solo l'A/B
> appaiato è affidabile.** Il debito delle allocazioni per-chunk (§5.5) resta, ma non ha un costo osservabile.

## 5.8 `OPENBLAS_NUM_THREADS`: nessuno lo impostava, e l'ottimo è un compromesso RTF↔TTFA

Il motore non chiamava mai `openblas_set_num_threads`: OpenBLAS prendeva il suo default (**un thread per
core**), ignorando `-j`. Su un server 64-core con `-j4` erano **4 thread nostri contro 64 suoi**. Legato al
budget di `-j` (`b1b0ab2` + `4f6302c`, weak symbol → no-op su Accelerate).

Ma è **il pavimento, non la taratura**. Sweep su N1 (`-j4`, 4 core, 0.6B int4):

| BLAS threads | RTF file | **TTFA file** | RTF stream |
|---|---|---|---|
| 2 | **1.48** | **2409 ms** | 1.52 |
| 3 | 1.51 | 1966 ms | **1.47** |
| 4 (= `-j`) | 1.59 | **1845 ms** | 1.54 |

Abbassare i thread BLAS **migliora l'RTF e peggiora il TTFA del 30%**: il **prefill** è tutto BLAS e NON ha
il decoder concorrente, quindi vuole tutti i core; la **generazione** ha il decoder concorrente, quindi ne
vuole meno. → serve una manopola **per fase**, non per processo. E `OB=1` è disastroso (RTF 2.14): una
partizione rigida `nostri + BLAS = core` sarebbe sbagliata — una lieve oversubscription vince, perché il
nostro pool ha sezioni seriali in cui i core devono poter andare al decoder.

## 5.9 Dopo il nostro lavoro, int4 NON batte più int8 su N1

| | int8 | int4 |
|---|---|---|
| 0.6B file | 1.60 | 1.60 |
| 1.7B file | **2.04** | 2.09 |

La PR dichiarava int4 > int8 su N1, e **sul suo albero era vero**: lì il decoder era lentissimo e il Talker
dominava. Ora che il decoder è sceso del 57%, **il collo si è spostato**: il vantaggio dell'int4 sul Talker
non è più sul percorso critico, mentre il decoder è identico nei due casi.
→ La regola "int4 batte int8 su ARM" ([[project_quant_ladder_findings]]) resta vera **per il Talker isolato**,
non end-to-end. Ottimizzare una parte riordina la classifica di tutte le altre.

## 5.10 Snake threading + leva BLAS per fase (`8ea26fc`) — le due leve del profilo

Nessuna delle due è un kernel: vengono dal `perf` (§5.1), che metteva il 21% nello scheduler e la snake
sotto l'1%. A/B **appaiato** su N1 (0.6B `--int4 -j4`, coppie adiacenti):

| config | PREV | NOW | Δ RTF |
|---|---|---|---|
| stream chunk 24 | 1.53 | **1.46** | −5% |
| file mode | 1.62 | **1.49** | **−8%** |
| stream + conv int8 | 1.29 | 1.28 | ~0 |
| file TTFA | 1944 ms | **1871 ms** | migliora |

Su M1 (decoder non è il collo): decoder 2440 → 2318 ms (−5%), **RTF invariato** — il guadagno resta nascosto
dietro la generazione, come sempre.

**Cosa insegnano questi numeri:**
- **File mode guadagna più dello stream (−8%)**: lì il prefill è una fetta maggiore del wall, e la leva BLAS
  gli tiene tutti i thread. Il decoder scende (8503 → 7885 ms) grazie alla snake threaded.
- **Con la conv int8 il guadagno sparisce**: la conv int8 ha già dimezzato il decoder, quindi la snake non è
  più il collo. Ennesima conferma: **ottimizzare una parte sposta il collo, e le due ottimizzazioni non si
  sommano** — si sovrappongono sullo stesso collo.
- **Il TTFA non regredisce** (anzi migliora in file): la leva è a **due valori** — prefill largo, generazione
  stretta — proprio per non pagare il time-to-first-audio, che un `OPENBLAS_NUM_THREADS=2` piatto avrebbe
  peggiorato del 30% (§5.8).

**Implementazione:** il dispatcher `sd_pool_run` è uscito dalla guardia `__ARM_FEATURE_DOTPROD` (la conv int8
ce l'aveva confinato) perché la snake lo usa su ogni ISA; il suo path GCD non usa più un globale. Snake
seriale sotto `QWEN_SNAKE_MIN_WORK` (le snake dei primi stadi ConvNeXt sono piccole). La leva BLAS: prefill
tiene `-j`, la generazione scende a `-j−1` quando parte il thread del decoder, risale al join. **Non** una
partizione rigida `nostri+BLAS=core`: `OB=1` dà RTF 2.14, la lieve oversubscription vince.

Gate: one-shot bit-identico, chunked==one-shot (mel_corr 1.00000), `--self-test` + `make test-golden` PASS.

## 5.11 Riepilogo — il branch finale contro `main`, su due architetture

| | M1 (RTF) | N1 (RTF) |
|---|---|---|
| 0.6B file int4 — `main` → HEAD | 0.65 → **0.52** | 2.70 → **1.49** |
| 0.6B stream int4 — `main` → HEAD | 0.67 → **0.56** | 1.98 → **1.46** |
| 0.6B stream int4 **+ conv int8** | — | → **1.28** |
| 1.7B file int4 — `main` → HEAD | 1.13 → **0.87** | — |

TTFA senza regressione (A/B appaiato, §5.7). 18 commit. Aperto: spin-pool fixato (~3%), ramo `__AVX2__`
della snake su x86, AVX-512 per attention/rms_norm su x86.

## 6. Da dire all'autore (dopo §4 e §5)

- Credito pieno: l'analisi è di qualità, i risultati negativi documentati, il conv esatto è il pezzo
  che ci mancava (era il nostro Track D **D3**, che avevamo rinviato come effort L).
- Perché non mergiamo il diff: `submit_mtx`, il lost-wakeup x86, e i 13 siti del layout q4 (CUDA/Metal).
- I due bug trovati nel suo codice: `latent_base`, e lo stato non per-slot.
- Il bug trovato nel **nostro** (il guard della ConvTranspose) grazie al suo lavoro.
- I numeri reali su N1, non i nostri su M1.
