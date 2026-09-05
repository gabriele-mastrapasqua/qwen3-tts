# P1.4 AMX runtime checkpoint — GCP C4

Status: **DONE** (2026-09-05)

Owner: Codex, live GCP C4-class box (24 vCPU / 12 physical cores, SMT on
during these runs). The evidence was collected in the
dedicated worktree and from the dirty source snapshot `620279c-dirty`.

This checkpoint is read-only with respect to the engine: no runtime, kernel,
dispatch, or shared API was changed while closing P1.4. The three pre-existing
dirty source files remain dirty and are not part of this closure.

## Evidence scope

- AMX build: `SIMD=amx`; the binary hash and source fingerprint are recorded
  with the run artifacts (private).
- `--caps` and the default dispatch map prove AVX-512 VNNI/BF16 plus AMX INT8
  and BF16 readiness, including XCOMP permission. Native self-test and the
  fallback self-test both passed.
- Profile basis was
  `configs/perf/gcp-c4-standard-24-vnni-ttfa.json`; the AMX binary was run
  with the same profile environment. No AMX threshold override was used for
  the default evidence.
- Default C1/C4 WAVE: `2x6`, three waves, census enabled, zero errors and
  zero rejects. C4 effective batch was `2.55` aggregate, with worker batches
  `1.20` and `1.35` in the census run (`2.538`, `1.19`, `1.34` in the clean
  run).
- The existing C8 soak census is used only as already-collected reachability
  evidence. No new soak or experiment was run for this closure.

Primary artifacts: `--caps`, the dispatch map, the census report and the
per-worker census of the default C1/C4 run, plus the existing C8 soak census
summary. They stay under the untracked private evidence area (`.work/evidence/`,
gitignored) because they carry machine identifiers, binary hashes and raw run
paths; this note is the public-safe summary of what they show.

## The `suspicious=1` C4 finding

The default census reports one finding:

```text
[SUSPICIOUS] gate.int8.amx: resolved ON, executed calls = 0
```

This is not a concrete operation row. It is a synthetic selected-vs-executed
join finding produced by `tools/census_report.py`; consequently it has no
operation, shape, or AMX leaf of its own. The exact observed C4 rows that
explain it are:

- CP `matmat_int8`: `B=2`, shapes `6144x1024`, `1024x3072`, `2048x1024`,
  `1024x2048`, and `1024x1024`; kernel marker `int8 VNNI vpdpbusd`.
- CP `matvec_int8`/QKV: `B=1`; shapes `1024/2048/6144 x 1024/2048/3072`
  and `4096x1024`; leaf `vnni`.
- Talker `matvec_int8`/QKV: `B=1`; shapes `12288x2048`, `2048x2048`,
  `2048x6144`, and `4096x2048`; leaf `vnni`.

The default AMX INT8 gate is `B>=4`, `B<=16`, `rows>=32`, `cols>=64`.
The C4 workers never present `B>=4`, so zero AMX INT8 calls is the expected
dispatch, not an unexpected runtime dispatch. The raw census also has
`UNKNOWN=0` and fallback calls `=0`.

The classification issue is in the reporter join: its AMX predicate at
`tools/census_report.py:171` only tests `B>=2`, while the runtime AMX gate is
`B>=4` (and also shape-gated). Lines 183–186 then call an enabled-but-unused
gate suspicious without considering that threshold. This is a reporter false
positive for this workload; the census data itself is consistent. The tool is
left untouched in this read-only checkpoint.

## Compact leaf table

Counts below are call counts. In the current census format AMX and VNNI
matmat rows carry the kernel marker in `kernels` and may have an empty
`leaves` array; the AMX/VNNI columns therefore mean the resolved kernel class,
not an invented leaf name. `fallback=0` means no generic/twin/scalar fallback;
BLAS is listed separately as an allowed decoder path.

### Production-like C4 (per-worker census, the C4-only worker)

| stage / op | observed B / shape | AMX calls | VNNI calls | fallback calls | other native / allowed |
|---|---|---:|---:|---:|---|
| Talker BF16 matmat / prefill | `B={7,8,9,13,16}`; `1024x2048`, `12288x2048`, `2048x2048`, `2048x6144` | 1176 | 0 | 0 | AVX-512 BF16: 171 calls at `B=2` |
| Talker INT8 matvec | `B=1`; `12288x2048`, `2048x2048`, `2048x6144` | 0 | 3360 | 0 | — |
| Talker INT8 QKV matvec | `B=1`; `4096x2048` | 0 | 1120 | 0 | — |
| CP INT8 matmat | `B=2`; `6144x1024`, `1024x3072`, `2048x1024`, `1024x2048`, `1024x1024` | 0 | 3246 | 0 | — |
| CP INT8 matvec | `B=1`; `1024x2048`, `1024x3072`, `2048x1024`, `6144x1024` | 0 | 9640 | 0 | — |
| CP INT8 QKV matvec | `B=1`; `4096x1024` | 0 | 2800 | 0 | — |
| CP argmax head wrapper | `B=1`; `2048x1024`; 600 wrapper calls | 0 | underlying VNNI matvec (included above) | 0 | wrapper has no own leaf row |
| Decoder INT8 conv | time buckets `B=32..32768`, decoder shape set in the artifact | 0 | 1056 | 0 | — |
| Decoder BF16 matmat | `B=2`; `3072x2048` | 0 | 0 | 0 | AVX-512 BF16: 66 |
| Decoder SGEMM | decoder time buckets / model shapes | 0 | 0 | 0 | BLAS: 2024 (not a fallback) |

### Existing high-C reachability census (C8, already collected)

This is not a new qualification run. It proves the other side of the gate:

| stage / op | observed B / shape | AMX calls | VNNI calls | fallback calls |
|---|---|---:|---:|---:|
| Talker INT8 matmat | `B=4,5`; Talker projection shapes and QKV `4096x2048` | 30072 | 0 in matmat rows | 0 |
| Talker INT8 QKV matmat | `B=4,5`; `4096x2048` | 9968 | 0 | 0 |
| CP INT8 matmat | `B=4,5` AMX; `B=2,3` VNNI; CP projection shapes | 85920 | 5672 | 0 |
| CP INT8 QKV matmat | `B=4,5`; `4096x1024` | 28480 | 0 | 0 |
| Talker BF16 matmat / prefill | `B=7..16` AMX; `B=1` AVX-512 BF16 | 11818 | 0 | 0 |
| Decoder BF16 matmat | `B=4,5` AMX; `B=2,3` AVX-512 BF16; `3072x2048` | 357 | 0 | 0 |

The C8 census had `frames_single=52`, `frames_batched=493`, no dropped ops,
no UNKNOWN rows, and no fallback rows. It shows the within-run mode boundary:
CP `B=2..3` remains VNNI while CP/Talker `B=4..5` reaches AMX.

## Answers to the P1.4 runtime questions

1. **Actual AMX activation.** AMX BF16 is real in Talker prefill on the
   production-like C4 WAVE. AMX INT8 is real on this binary/host, but only in
   the already-collected higher-C census at Talker/CP `B=4,5`. The C8 rows
   include both Talker and CP AMX INT8 matmat/QKV calls.

2. **VNNI vs AMX dispatch.** C4 decode is VNNI: Talker GEMV/QKV, CP GEMV/QKV,
   CP batched projections at `B=2`, and decoder INT8 conv. The C8 census shows
   VNNI at `B=1..3` and AMX at `B=4..5`. Prefill is native BF16 AMX at its
   larger token batches; small BF16 matmat/GEMV shapes use AVX-512 BF16.

3. **Region behavior.** At C4 the measured per-worker batch is below the AMX
   INT8 threshold, so the VNNI in-region-compatible path remains active. At
   higher C, the existing census shows the expected flip: VNNI rows at B2/B3,
   AMX dispatched rows at B4/B5. The artifacts do not expose a dedicated
   region-counter field, so the region statement is the runtime leaf result
   joined with the existing `qwen_i8mm_usable()` gate, not a new counter.

4. **Batched CP head.** CP projections batch at B2 on C4 and at B4/B5 on the
   existing C8 census. The argmax head remains a per-slot `B=1` wrapper whose
   underlying matvec is VNNI; no batched AMX argmax-head leaf was observed.

5. **AMX BF16 allocation path.** This remains an implementation finding, not
   a new allocation benchmark: `qwen_matmat_bf16()` still does `malloc(Xb)`
   in the AMX branch (`qwen_tts_kernels.c:3159`), and
   `qwen_matmat_bf16_qkv()` does the same (`:5020`). The packed activation uses
   `mm_scratch_packb()` after that allocation. This is P2.4 follow-up; no
   source change was made here.

6. **Same-box VNNI/AMX comparison.** The C4 default vs `QWEN_AMX_MIN_B=17`
   pair is not an isolated INT8 AMX A/B: default C4 never executes AMX INT8,
   and the global `QWEN_AMX_MIN_B=17` override also disables BF16 AMX. The
   existing C8 census has AMX activation but no matched VNNI control at the
   same high-C workload. Therefore no isolated AMX performance benefit outside
   prefill is measured yet.

## Final verdict and follow-up ownership

P1.4 is **DONE**: the runtime question is answered for the current GCP C4
box and topology. In production-like `2x6/C4`, AMX contributes to native BF16
prefill; steady Talker/CP decode and decoder INT8 work are VNNI/BLAS, because
the effective per-worker batch never reaches AMX INT8 `B>=4`. AMX INT8 is
compiled, ready, and reachable at higher C, as proven by the existing C8
census, but its isolated performance value is not yet established.

Implementation work is deliberately separate and remains follow-up only:

- P2.1/P2.2: decide whether to add AMX-tile in-region runners or keep the
  current threshold/dispatch split; do not infer the answer from the C4
  `suspicious=1` reporter false positive.
- P2.3: batched CP-head implementation/acceptance (argmax parity).
- P2.4: remove the per-call BF16 AMX/QKV `malloc(Xb)` allocations.
- P3: complete feature/knob parity and any later controlled A/B measurements.

Deferred TODO, after the current parity tasks: review the CPU AMX attention
example at <https://leeroopedia.com/index.php/Implementation:Vllm_project_Vllm_CPU_Attn_AMX>
for reusable ideas. It is not evidence for this checkpoint and was not used to
change code.
