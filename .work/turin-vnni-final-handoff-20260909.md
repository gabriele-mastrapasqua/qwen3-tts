# Turin VNNI product point — final handoff (2026-09-09)

**Task.** Qualify the DL-2 elastic decoder lane + DL-4 direct conv architecture on the
32-core Turin host (AWS c8a.8xlarge), freeze it as one committed profile, and leave a
handoff another engineer or agent can resume from without rereading the campaign.
**Question.** Does the frozen `turin-c8a-32c-vnni-product` profile hold C12 (and C16) under
the project's serving contract, and is the RES1_V2 numerical change quality-safe?
**Known facts.** The host screen of 2026-09-10 (one wave, shell overrides): C12 STREAM p95
0.80-0.81, C16 0.88-0.92, stall@250 0; RES1_V2 res1 1.72x, decoder unit 64 -> 49.8 ms,
overlap share 48 -> 39 %; the inline engine on the same host tops out near C8.
**Unknowns at start.** V2 audio quality beyond a 0.99978 waveform correlation; C12/C16 under
SOAK, open arrival, overload and long-request interference; whether the product profile
(spin 65536, bf16 prefill) reproduces the screen taken on a 16-core profile.

Operational document: sections are meant to be copied from, not read in order.

## 1. Current winning architecture (DL-2 + DL-4)

* Host: 4 CCX x 8 Zen 5 cores, 32 MiB L3 per CCX. Four prefork workers, one per CCX
  (`--prefork 4 --prefork-threads 8`), each holding up to 4 streams (`--batch-size 4`),
  fail-fast admission (`--max-queue 0`), INT8 VNNI Talker/CP in one batched region per
  frame, native AVX-512 BF16 prefill, per-item INT8 VNNI speech decoder, decode quantum 4.
* DL-1/DL-2 decoder lane (`QWEN_SD_LANE_SPLIT=4 QWEN_SD_LANE_ELASTIC=1`): the last 4 cpus of
  the worker's CCX form a private pinned decoder team. The frame loop enqueues one bounded
  decoder unit per slot (mailbox: at most one unit in flight per slot, lead <= 1 quantum) and
  never runs the decoder inline. Elastic: the engine pool keeps all 8 cpus and is narrowed to
  the 4 step cpus only while a decoder unit is in flight (`qwen_pool_set_width`), so the
  Talker/CP step pays the 4-thread width only during real overlap.
* DL-4 (`QWEN_SD_RES1_V2=1`): the residual convs (k=7, dilation 1/3/9, and the 1x1 res2)
  run a direct dilated int8 VNNI kernel (`qwen_conv1d_int8_v2`): activations quantised per
  position (u8 = q+128, one scale per position), weights per (channel, tap), a 4x4 register
  tile walking the 7 taps, weights read once per time block. No im2col panel.
* Why it wins: the inline decoder sat on the critical path of every frame; the lane removes
  that serialisation; the lane's cost is a roughly fixed interference tax on the step while
  a decoder unit runs; V2 shortens the unit, so fewer step iterations overlap.

## 2. Exact product configuration (copy-pastable)

Profile of record: `configs/perf/turin-c8a-32c-vnni-product.json` (control arm for A/Bs:
`turin-c8a-32c-vnni-control`, differs in `QWEN_SD_RES1_V2=0` only).

```bash
make clean && make blas                       # SIMD=avx512bf16 auto-selected on Zen 5
./qwen_tts --self-test                        # 20 conv1d_int8_v2 cases among the rest
taskset -c 0-7 python3 tools/serving_profile.py preflight turin-c8a-32c-vnni-product \
    --binary ./qwen_tts --out preflight.json  # errors must be [], decoder_lane/res1_v2 ACTIVE
python3 tools/perf_profile.py command turin-c8a-32c-vnni-product --model qwen3-tts-1.7b --port 9500
# -> env $(python3 tools/perf_profile.py server-env turin-c8a-32c-vnni-product | tr ',' ' ') \
#    ./qwen_tts -d qwen3-tts-1.7b --int8 --serve 9500 --batch-size 4 --prefork 4 --prefork-threads 8 \
#    --max-queue 0 --queue-timeout-ms 0 --max-request-seconds 60
```

The variables that make this profile different from `vnni-bf16-product`:
`QWEN_SD_LANE_SPLIT=4 QWEN_SD_LANE_ELASTIC=1 QWEN_SD_RES1_V2=1 QWEN_DECODER_BATCH=0
QWEN_STREAM_DECODE_CHUNK=4 QWEN_VNNI_GEMV_MR=2 QWEN_POOL_SPIN=65536`; the rest is the
VNNI+bf16 backend contract (INT8 VNNI Talker/CP, per-item INT8 VNNI decoder, native
AVX-512 BF16 prefill, `QWEN_TTS_STREAM_LAYOUT=1`, `QWEN_SERVER_ASYNC_OUTPUT=0`,
`OPENBLAS_THREAD_TIMEOUT=1`, OPENBLAS/OMP thread counts absent). Topology: 4 prefork
workers pinned to cpus 0-7 / 8-15 / 16-23 / 24-31; the lane reserves the last 4 cpus of
each CCX for the decoder team while a unit is in flight.

Phase A record (2026-09-09 10:56 UTC): c8a.8xlarge `i-09dbe01f7cf251a3a` us-east-1a,
AMD EPYC 9R45, 32 cores / 32 threads, 1 NUMA node, L3 128 MiB in 4 instances, kernel
7.0.0-1006-aws, gcc 15.2.0, OpenBLAS apt; membw Triad 189.5 GB/s at 32 threads (Copy
169.3), knee at 16 threads; earlier doctor on the same class: read 231 GB/s host, 55.8
GB/s one CCX. Binaries: a45e74a4 (94be84d, gates), eeb2255f (1acf02e, preflight),
b0bfe82f (28d6436, every Phase B/C number). Model fingerprint 787679a2 (qwen3-tts-1.7b,
`model.safetensors` sha256 38b1d597...). Spot price at launch ~0.68 $/h (campaign
record), on-demand list 1.724 $/h.

## 3. Qualified capacities

### C12 — QUALIFIED for the mandatory contract, preferred STREAM gate with one exception

Waves (temperature 0, ryan, 3 waves x 12, `tests/serve_parallel_wave.py`, client-observed
playback):

| class | TTFB p95 | TTFA p50/p95 | STREAM p50/p95 | TOTAL p95 | prebuffer p50/p95 | safe-start p95 | max gap p95 | stall @250/@500 | rej/err |
|---|---|---|---|---|---|---|---|---|---|
| short | 102 | 238/270 | 0.798/**0.848** | 0.961 | 175/260 | 467 | 315 | 0/0 | 0/0 |
| medium | 104 | 238/269 | 0.783/**0.820** | 0.866 | 173/256 | 467 | 313 | 0/0 | 0/0 |
| long | 104 | 240/274 | 0.801/**0.816** | 0.823 | 175/258 | 468 | 316 | 0/0 | 0/0 |
| mixed | 104 | 238/270 | 0.787/**0.830** | 0.939 | 174/259 | 467 | 314 | 0/0 | 0/0 |
| long+short (interference) | 103 | 238/271 | 0.793/**0.848** | 0.968 | 177/259 | 466 | 315 | 0/0 | 0/0 |

Short requests inside the long+short wave: STREAM p95 0.854 vs 0.848 alone; the long
ones 0.796 — no measurable interference from long requests on established short streams.
Throughput: short 4.88 req/s, medium 2.03, long 0.48; ~28 core-equivalents busy.

SOAK 30 min closed-loop C12, temperature 0.9, whole bank (2205 completed, 0 errors, 0
rejects, 0 timeouts; `RESOURCE STABILITY PASS`, `LATENCY KPI PASS`, memory growth -0.1 %):

| slice | n | TTFA p50/p95 | STREAM p50/p95 | TOTAL p95 | prebuffer p95 | safe-start p95 | stall @250/@500 |
|---|---|---|---|---|---|---|---|
| all | 2209 | 138/170 | 0.838/**0.912** | 0.941 | 262 | 408 | 0/0 % |
| short | 444 | 138/177 | 0.860/**0.959** | 0.999 | 267 | 422 | 0/0 |
| conversational | 451 | 140/169 | 0.838/**0.914** | 0.927 | 271 | 413 | 0/0 |
| medium | 427 | 137/170 | 0.833/0.885 | 0.900 | 251 | 395 | 0/0 |
| long | 436 | 137/168 | 0.830/0.880 | 0.882 | 245 | 385 | 0/0 |
| italian | 451 | 138/168 | 0.837/0.888 | 0.900 | 267 | 408 | 1/0 |
| windows 0-30 min (5 min each) | 364-372 | 166-172 p95 | 0.898-0.917 p95 | 0.932-0.954 | 257-264 | 399-412 | <=1/0 |

Poisson open arrival (mixed bank, 120 requests, no client gate): 1.5 req/s -> TTFA p95
172 ms, Q 11.4, 1 reject; 2.5 req/s -> TTFA p95 175 ms, Q 12.3, 28 rejects at 16 in
flight (24 clean 503s, 4 TCP resets — see PLAN). Overload waves: C20 accepts 32 / rejects
8, C24 accepts 32 / rejects 16 (the per-worker cap of 4 rejects when its own worker is
full, so the host rejects more than C-16); accepted streams STREAM p95 0.930/0.954,
prebuffer p95 366, stall@250 0. No hidden queue: TTFB p95 217 ms even at C24.

Diagnostic (STAGE trace, C12 mixed): overlap share 29.8 %; CP 21.5-22.9 ms per iteration
outside overlap vs 35.2-35.6 inside; Talker 32-35 ms either way; decoder unit on the lane
mean 50.9 / p95 53.7 ms; mailbox overruns 0; decoder team 0.55-2.1 core-eq.

Verdict: **mandatory contract PASS in every class** (STREAM < 1, prebuffer <= 500 ms,
safe-start <= 1 s, stall@500 0, no inference errors/timeouts). **Preferred STREAM p95 <=
0.90**: PASS in every wave class and in the medium/long/italian soak classes; **FAIL for
short (0.959) and conversational (0.914) under closed-loop soak** — twelve permanently busy
slots at temperature 0.9 cost the short clips ~0.1 of STREAM over the wave figure (their
fixed per-request cost is a larger share of a 2-3 s clip). The screen figure 0.80-0.81
was taken at spin 4096 on the 16-core profile; the frozen profile (spin 65536) measures
0.82-0.85 in waves. C12 is the recommended production point; a strict 0.90 SLA on short
clips wants C10 (not measured here).

### C16 — NOT RUN

The spot host was reclaimed at 12:25:55 GMT ("instance-terminated-no-capacity") after the
C12 soak and before Phase D started. The screen of 2026-09-10 (C16 STREAM p95 0.919 short
/ 0.881 long, prebuffer 360, stall@250 0) stands as the only C16 evidence and is SCREEN
grade. Commands in section 8; the runner treats C16 exactly like C12.

## 4. Quality status of RES1_V2 (Phase B, 2026-09-09, revision 28d6436)

Paired bank, two committed profiles that differ in one line (`turin-c8a-32c-vnni-control`
= panel conv, `turin-c8a-32c-vnni-product` = V2), 4x8 cap 4 at C=4 (lane active on every
CCX), temperature 0, seed 42+idx, `tests/serve_parallel_wave.py --save-audio`:

| run | model | voice / language | classes | pairs |
|---|---|---|---|---|
| en17b | 1.7B | ryan / English | short, medium, long, conversational | 20 |
| it17b | 1.7B | ryan / Italian | italian | 4 |
| ser17b | 1.7B | serena / English | short, medium | 12 |
| en06b | 0.6B | ryan / English | short, medium | 12 |
| it06b | 0.6B | ryan / Italian | italian | 4 |

Automated results over the 52 pairs (`~/bench/phaseB/pairs.json` on the box, copied to
`samples/tests/2026-09-09_turin-qualification/phaseB_pairs/` on the Mac):

* valid PCM in every file; duration identical to the sample in every pair (V2 only changes
  the decoder, the Talker codes are the same);
* waveform correlation >= 0.9995, log-mel correlation min 0.9948 / typical 0.997-0.998
  (the golden gate of `tests/compare_audio.py` is 0.98);
* ASR CER (qwen3-asr-0.6b) identical between arms: mean 0.0500 vs 0.0503, no pair worse
  by more than 0.05; the high Italian CERs (0.38-0.79) are the same in both arms and are
  the ASR on short Italian lines, not the decoder;
* `tools/wav_qc.py` flags (clipping, DC, holes, clicks) identical between arms.
* `--self-test` gains 20 `conv1d_int8_v2` cases: own-quantisation integer reference to
  2e-7, f32 causal conv within 5.5e-3 (quantisation error), streaming continuation exact.

**PROVEN**: no measurable structural or intelligibility regression; numerics are within
the decoder's own quantisation error. **NOT PROVEN**: an ear verdict — the automated
gates cannot hear timbre; the listening set is on the Mac (README lists the lowest
mel-corr pairs first). Until a human listens, V2 quality is **PASS (automated) / pending
ear**, and the product profile stays `provisional`.

## 5. Causal performance model (measured, single CCX unless stated)

* Lane law, inline engine: `T(B) = 40 + 13.5*B ms` per frame iteration (1.7B); the decoder
  was 9.7 ms/slot of that (72 %), Talker +1.6, CP +1.8, serial 0.05.
* DL-1 static 4+4: iteration wall p95 192 -> 77 ms at B4, stall@250 100 -> 0 %, but the
  4-thread step inflated Talker+CP +27-29 % (STREAM p95 B4 0.997, gate 0.92 missed).
* DL-2 elastic: Talker+CP 69.5 vs 69.8 ms static with the full team 69 % of the time — the
  static partition is NOT the tax. The step slows ~2x only while a decoder unit runs.
* DL-3 falsifiers: CP 23.5 ms with no overlap, 35-38 ms in overlap, whatever the decoder
  does (direct ConvT/dwconv/input paths, sub-quantum units, NTA prefetch, hot lane workers,
  panel sizing). The tax is ~+20 ms per overlapped iteration; only the overlap share moves
  the mean.
* DL-4: res1 13.4 -> 7.8 ms (8T), decoder unit on the 4-thread lane 64 -> 49.8 ms, overlap
  share 48 -> 39 %, lane B4 long 0.869 / fixed 0.918; CP in overlap unchanged (34-35 ms).
* Host: two weight streams on one CCX halve each other; one CCX saturates at 2-4 VNNI GEMV
  threads (~55 GB/s); 16-thread teams across two CCX thrash (40 GB/s cache-resident);
  hence 4x8, never 2x16 or 1x32 for the 1.7B.

## 6. Rejected experiments (do not repeat)

| experiment | result | lesson |
|---|---|---|
| 2 step + 6 decoder cores | NO-GO on paper and in the per-slot region sections (+17 ms/slot at 2 threads) | the step side needs >= 4 threads |
| static 4+4 lane (DL-1) | wall p95 192 -> 77, stall@250 -> 0, STREAM B4 0.997 | architecture right, allocation not enough |
| 5+3 split | fixed B4 1.113, long 1.015 | decoder needs 4 cores at B4 q4 |
| 6+2 split | fixed B4 1.364, long 1.263 | same, worse |
| elastic 4+4 (DL-2) | 0.987 vs 0.997 static, Talker+CP unchanged | partition tax is not the mechanism; kept as the lane of record |
| direct ConvT / dwconv / input (DL-3) | CP in overlap unchanged | the tax is not the decoder's op mix |
| sub-quantum 1/2-frame units | worse (73 % in flight, mailbox wait 10 ms/frame) | 4x the calls saturate the lane |
| q8 units | long B4 0.864 but prebuffer 806 ms, stall@250 100 % | cadence law; inadmissible |
| NTA weight prefetch | no change | not a cache-pollution story that a hint fixes |
| hot lane workers | no change | wake latency is not the tax |
| panel-size fix | no change | panel shape is not the tax |
| RES1_V2 (DL-4) | unit 64 -> 49.8 ms, overlap 48 -> 39 %, host C12/C16 | the lever is decoder residency, not the tax per iteration |
| 1x32 / 2x16 for the 1.7B | 1x32 cap 8 STREAM 1.3-1.5; 2x16 C10 over realtime | cross-CCX pools thrash; one worker per CCX |

## 7. Remaining headroom (evidence-backed only)

* The decoder unit on the 4-thread lane is 50.9 ms mean / 53.7 p95 at C12 (kernel gate
  target <= 45-48 ms was narrowly missed); every ms off the unit lowers the overlap share
  (29.8 % at C12 mixed) and the fixed ~+13 ms CP tax per overlapped iteration follows it.
* Short clips under closed-loop soak sit at STREAM p95 0.959: the fixed per-request cost
  (admission + first-chunk ramp 1,2,4) is the lever for the short class, not the frame law.
* C16: screen only; the short-class tail margin is smaller than C12's (0.919 vs 0.848 in
  the screen vs wave).
* Fail-fast boundary: 4 of 28 rejects at a full host surfaced as a TCP reset instead of a
  503 (client-visible, no audio impact); the rejection is per worker, so a host with free
  slots elsewhere still rejects (8 rejects at C20 with 16 slots).
* Cross-ISA: `qwen_conv1d_int8_v2` and the decoder lane exist for x86 AVX-512 VNNI + Linux
  only (stubs elsewhere); AMX/Arm equivalents are neither implemented nor qualified.

## 8. Exact next commands (on a fresh c8a.8xlarge, Ubuntu 26.04)

```bash
sudo apt-get install -y build-essential git libopenblas-dev python3-numpy python3-pip ffmpeg
git clone --branch feature/x86-amx-vnni-oss https://github.com/gabriele-mastrapasqua/qwen3-tts.git qwen-tts
cd qwen-tts && bash download_model.sh --model large && make blas -j32
./qwen_tts --caps && ./qwen_tts --self-test && make cpu-check && make bench-fingerprint
taskset -c 0-7 python3 tools/serving_profile.py preflight turin-c8a-32c-vnni-product --binary ./qwen_tts --out preflight.json
# quality A/B (paired, two committed profiles), then pair analysis (librosa + qwen-asr):
pip install --break-system-packages librosa soundfile; git clone https://github.com/antirez/qwen-asr ~/qwen-asr && (cd ~/qwen-asr && make blas && bash download_model.sh --model small)
for P in control product; do python3 tests/serve_parallel_wave.py --model qwen3-tts-1.7b --bin ./qwen_tts \
  --profile turin-c8a-32c-vnni-$P --topo 4x8 --batch-cap 4 --conc 4 --waves 5 --seed 42 --precision int8 \
  --speaker ryan --language English --text-file tests/load_texts_en.txt --classes short,medium,long,conversational \
  --label q_$P --out /tmp/q/$P --port 9500 --no-crosscheck --save-audio /tmp/q/wav/$P; done
# pairs: tests/compare_audio.py REF V2 per file, tools/wav_qc.py, qwen_asr -d ~/qwen-asr/qwen3-asr-0.6b -i WAV
# C12 / C16 qualification (waves, quality wave, STAGE diag, interference, overload, Poisson, 30-min SOAK):
for C in 12 16; do python3 tests/serve_parallel_wave.py --model qwen3-tts-1.7b --bin ./qwen_tts --profile turin-c8a-32c-vnni-product \
  --topo 4x8 --batch-cap 4 --conc $C --waves 3 --seed 42 --precision int8 --speaker ryan --language English \
  --text-file tests/load_texts_en.txt --classes short --label short_c$C --out /tmp/qual/c$C/short --port 9500 --no-crosscheck; done
#   repeat with --classes medium | long | short,medium,long,conversational | long,short; overload: --conc 20,24;
#   diagnostic arm: --server-env QWEN_STAGE_TRACE=1,QWEN_SD_PHASE=1 ([STAGE] cp_ms by overlap=, [SDPHASE] total= at frames=4)
python3 tests/serve_soak.py --model qwen3-tts-1.7b --bin ./qwen_tts --profile turin-c8a-32c-vnni-product --port 9500 \
  --bank tests/load_texts_en.txt --speaker ryan --language English --temperature 0.9 --precision int8 \
  --concurrency 12 --minutes 30 --window-s 300 --min-per-class 5 --min-per-class-p95 15 --out /tmp/qual/c12/soak
# Poisson: start the profile command by hand (see section 2), then
python3 tests/load_test.py --url http://127.0.0.1:9500 --speaker ryan --language English --text-file tests/load_texts_en.txt \
  --classes short,medium,long,conversational --concurrency 12 --arrival poisson --rate 1.5 --requests 120 --seed 42 --json poisson.json --csv poisson.csv
# kernel microbench / census: QWEN_SD_PHASE=1 QWEN_SD_RES1_V2={0,1} on a 1x8@0-7 wave -> [SDUP] res1= / [SDPHASE] total=
```
The scripts used on 2026-09-09 (`phaseB_run.sh`, `phaseB_analyze.py`, `phaseC_run.sh`,
`phaseC_report.py`) are in the session scratchpad and reproduced by the commands above;
their raw outputs are in `.work/evidence/turin-qualification-20260909/` (local only).

## 9. Git state

Branch `feature/x86-amx-vnni-oss`, pushed to origin. Commits of this sprint on top of
94be84d (DL-4 kernel): 1acf02e (frozen profile + verifiable lane/V2 contracts + V2
self-test), 7c2490f (self-test window fix), 28d6436 (control profile) — every Phase B/C
number was produced by a clean checkout of 28d6436 — and the closing docs commit. Worktree
clean after the closing commit. Listening set on the Mac:
`samples/tests/2026-09-09_turin-qualification/` (gitignored).

## 10. Findings to carry (also in PLAN)

* Fail-fast boundary: 4/28 rejects at a full host answered with a TCP reset, not a 503.
* Rejection is per worker (cap 4), not per host: C20 -> 8 rejects with 16 slots.
* The strict preflight refuses a `--server-env` override that changes a parity value —
  by design; an A/B arm is a committed profile (`turin-c8a-32c-vnni-control`).
* `--dispatch-map` now prepares the lane on the inherited mask so `decoder.lane` resolves in
  a probe; the prefork worker's own `[DISPATCH]` block remains the engagement proof.
