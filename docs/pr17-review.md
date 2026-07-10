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

### ⚠️ Regressione REALE trovata solo qui: TTFA streaming **+8%** (920 → 994 ms)
Su M1 era dentro il rumore. Su N1 i tre range sono **disgiunti** (920/964/968 vs 994/1006/1007) → segnale.
Causa probabile: il path esatto fa `cs_ensure_alloc` + una `aligned_malloc` di `ext`/`full` **per ogni conv,
per ogni chunk**, e il primo chunk è di soli 2 frame → overhead relativamente grande.
→ **TODO: pre-allocare gli scratch per-stream una volta sola** invece che per chunk. Non bloccante
(74 ms su ~950), ma va chiuso prima di dichiarare il lavoro finito.

### Confronto col claim della PR
| | loro (dichiarato) | noi (misurato, stessa µarch) |
|---|---|---|
| stream chunk 24 | 2.21 → **1.34** | 1.96 → **1.48** |
| file chunk 150 | 1.59 → **1.21** | 1.44 → 1.44 |

Il nostro **prima** è già migliore del loro (`main` ha D2 e altro che la loro base non aveva). Il nostro
**dopo** è più lento del loro **esattamente dei pezzi che non abbiamo portato**: snake poly + conv int8
+ spin-pool. Il conto torna, e dice quanto valgono quei pezzi su questa box: **~0.14 RTF sullo stream**.

⏳ Restano da misurare qui: snake poly (ramo `__ARM_NEON`), spin-pool **dopo** il fix (`submit_mtx` +
fence seq_cst), `QWEN_SD_INT8=1` **con ascolto** su emotion/clone, e un `perf record` per ordinare il resto.

## 5. ⏳ Il lavoro derivato che la PR ha rivelato (e che la PR NON copre)

La PR ha trovato **un** ramo che M1 non compila. Non è un caso isolato: è un **pattern strutturale**.
M1 è la dev box, quindi ogni ramo `#if` che M1 non prende non viene mai profilato né curato.

Usiamo Accelerate vForce in **esattamente due punti**, ed entrambi mettono il ramo Apple **per primo**:

| sito | M1 (`__APPLE__ && USE_BLAS`) | ARM-Linux **e** x86 | frequenza |
|---|---|---|---|
| `qwen_swiglu_inplace` (`qwen_tts_kernels.c:2941`) | `vvexpf` vettoriale | **`expf()` scalare, per elemento** | FFN, **ogni layer, ogni token** |
| `qwen_snake_activation` (`:3046`) | `vvsinf` vettoriale | **`sinf()` scalare** — la PR fixa **solo** `__ARM_NEON` | decoder conv |

Volume dello scalare fuori da Apple:
- Talker 0.6B: 28 × 3072 = **86k `expf()`/token** · Talker 1.7B: 28 × 6144 = **172k/token**
- CP: 5 × 3072 = 15k/pass × **15 pass/frame** = **~230k `expf()`/frame**

> **⚠️ Ipotesi FALSIFICATA dalla misura (2026-07-10).** Avevo ipotizzato, dal solo conteggio delle
> chiamate, che la `expf` della SwiGLU costasse **più** della snake. **È falso.** Micro-benchmark su M1
> (`scratchpad/expbench.c`, exp poly NEON scritta apposta, err. rel. max 3.3e-6):
>
> | | chiamate / clip (1.7B, 8.1s) | costo scalare |
> |---|---|---|
> | `expf` (SwiGLU: Talker + CP) | 40.6M | ~96 ms → **1.3% di talker+CP** |
> | `sinf` (snake: 4 upsample × (1 + 3×2) + finale) | **278.6M — 6.9×** | è qui che stanno i 1209 ms della PR |
>
> **La PR ha puntato il bersaglio giusto.** La snake domina; la SwiGLU è marginale.
> Il micro-benchmark misura comunque il *potenziale del kernel*: `expf` scalare 2.37 ns/elem →
> **NEON poly 0.41 ns/elem (5.8×)**, e persino **più veloce di `vvexpf` di Accelerate** (0.53 ns/elem).
> Quindi il kernel vettoriale resta la cosa giusta da scrivere — ma la priorità è **sin**, non **exp**,
> e il guadagno atteso su swiglu è ~1%, non la doppia cifra che avevo lasciato intendere.

Il ramo `__AVX2__` della snake (`:3089`) chiama anch'esso `sinf()` scalare → **stesso fix, guadagno x86
che la PR non copre**.

Inoltre `qwen_causal_attention` (`:2580`) usa `expf()` scalare nel softmax **su tutte le piattaforme,
M1 compresa**: win universale, più piccolo (O(seq_len)/layer).

→ **Deliverable: un kernel trascendente vettoriale multi-ISA** (`qwen_vec_sinf` **prima**, poi
`qwen_vec_expf`; NEON + AVX2/AVX-512 + scalare) dietro il dispatch attuale, con Apple che resta su
vForce (o vi passa sopra: la poly NEON batte `vvexpf` in micro-bench — da verificare a livello di engine).
Ordine dettato dalla misura: **sin (278M chiamate) ≫ exp (40M) > softmax**.

> **Nota sull'inventario SIMD (verificato, 2026-07-10):** l'algebra lineare è coperta ovunque —
> `qwen_tts_kernels.c` ha **40 funzioni con SIMD**, 870 intrinsics NEON, 202 AVX2, 70 AVX-512, 25 VNNI
> (matvec/matmat bf16-int8-q4, rms_norm, attention, quantizzazione, SDOT, VNNI). **Non manca nulla lì.**
> Il buco è **esclusivamente** sulle trascendenti: `qwen_swiglu_inplace` è l'unica funzione con *solo*
> vForce(Apple) e nessun ramo NEON/AVX2; la snake ha rami NEON/AVX2 che però vettorizzano solo load/store
> e chiamano `sinf()` **scalare per lane**; il softmax dell'attention usa `expf()` scalare ovunque.

**Prerequisito, non extra:** `--self-test` oggi copre i matvec, **non** copre snake/swiglu/softmax.
Estenderlo prima di toccarli.

---

## 6. Da dire all'autore (dopo §4 e §5)

- Credito pieno: l'analisi è di qualità, i risultati negativi documentati, il conv esatto è il pezzo
  che ci mancava (era il nostro Track D **D3**, che avevamo rinviato come effort L).
- Perché non mergiamo il diff: `submit_mtx`, il lost-wakeup x86, e i 13 siti del layout q4 (CUDA/Metal).
- I due bug trovati nel suo codice: `latent_base`, e lo stato non per-slot.
- Il bug trovato nel **nostro** (il guard della ConvTranspose) grazie al suo lavoro.
- I numeri reali su N1, non i nostri su M1.
