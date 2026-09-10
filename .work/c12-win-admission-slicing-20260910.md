# C12-WIN-10 — Admission slicing: implementation record

**Task.** Implement `.work/c12-win-admission-slicing-implementation.md` on the frozen Turin
architecture, default off, and leave the repository ready for an x86 measurement session.
**Question.** Does the sliced admission compute the same request as the monolithic one, and
can a partially prefilled request be dropped without breaking the worker?
**Known facts.** Inline prefill runs inside the frame loop's admission block and stalls
every active slot for the whole prompt; the LOW-priority helper arm moved the closed-loop
short class 0.966 -> 0.915 while raising TTFA p95 172 -> 683 ms, which established the
mechanism and disqualified that implementation. **Unknowns.** Every number: nothing in this
document was measured on x86, and no server A/B or soak has been run.

## 1. What was built

`QWEN_PREFILL_SLICE=N` (default unset = 0 = today's monolithic path, byte for byte).

Three pieces, in the order the spec names them.

**The kernel boundary.** `qwen_causal_attention_bf16kv` was single-row and serial, because
until now only the decode step attended over the bf16 cache. It is now a head-ranged body
(`..._heads`, called with `[0, n_heads)` it is the previous function verbatim) plus a block
form `qwen_causal_attention_bf16kv_prefill` that partitions heads over the pool exactly the
way `qwen_causal_attention_prefill` partitions the f32 one. The head split is a partition of
the output rows, so the result does not depend on the thread count. Self-test: four
`attn_bf16kv` cases (multi-row, grouped-query, single-row, zero offset) assert the block
form is **bit-identical** to the per-row decode form.

**The resume boundary.** `qwen_talker_prefill_plan(ctx, seq_len, &pos0)` runs once per
request: grows the KV cache, sizes the staging buffers, materialises a prefix-cache hit into
the bf16 cache for every layer, and sets `ctx->kv_len = pos0`.
`qwen_talker_prefill_range(ctx, embeds, seq_len, pos0, t0, t1)` then runs all 28 layers over
the new-token rows `[t0, t1)` only. `qwen_talker_prefill` is untouched and remains the
control path.

Why the boundary is valid: a token's residual stream never crosses a slice — each token
walks all 28 layers inside its own slice — and a slice reads earlier tokens only through
`ctx->kv_cache_{k,v}`, which the slices before it have already filled for every layer. The
slice writes its own K/V into the cache before its own attention, so causal masking
(`k_end = q_offset + i + 1`) covers the intra-slice dependency.

**The scheduling hook.** In the batched frame loop's admission block: a pending admission
owns the block until it completes — one slice per iteration while other slots stream, the
whole remainder in one visit when the worker is idle (nobody to protect, and cold-start TTFA
must not grow). At most one pending admission per worker; no new queue, no change to
`--max-queue`, cap, quantum, ramp, lane, transport or fail-fast admission. The prompt is
built by the existing builder through a deferred mode (`ctx->prefill_defer`) that hands the
embeddings back instead of running the Talker; text semantics, token order and RoPE angles
are unchanged.

## 2. Two deviations from the spec, both reported rather than hidden

**(a) A prompt that would POPULATE the prefix cache runs monolithically, once.** Filling a
prefix slot needs the f32 K/V of every layer to stay alive to the end of the prompt, which
is exactly the state slicing must not carry (spec section 5 forbids keeping it: 69 MB per
300 tokens). The spec is silent on filling — it only describes the hit. `..._plan` returns 1
for that case and the loop runs `qwen_talker_prefill` on the already-built prompt: the same
computation, the same result, the inline path. It happens a handful of times per worker
lifetime, at cold start, and is counted as `inline_requests` in `[ADMSLICE]`. The steady
state is fully sliced.

**(b) The M1 late admission is skipped while a slice is pending.** M1 is a second,
monolithic admission site at the end of the iteration. Its `qwen_talker_prefill` would
overwrite `ctx->kv_cache` rows `[0, seq_len)` for every layer plus `dec_x`,
`bg_text_content_len` and the trailing text — i.e. it would destroy the pending request.
The guard is `!adm.active`, one condition, and it is a correctness requirement, not a policy
change: with no pending slice M1 behaves exactly as today.

**A test-only knob, not a product arm.** A NEGATIVE `QWEN_PREFILL_SLICE` means `|N|` and
slices even on an idle worker. Without it the boundary is unreachable with one client and
one slot (`n_active == 0` takes the whole prompt in one visit), and "resume equals
uninterrupted" would only be testable under a race. Product arms use a positive value. The
dispatch row labels a negative value as a parity-test setting.

## 3. Local correctness — what was run and what it proves

Two harnesses. The STATE oracle is the one that decides; the audio harness exists because
the spec asks for it, and its verdict is a spec question, not an implementation one.

### 3.1 Why audio cannot decide this

A WAV is a function of INTEGER codes. So a correct slicing and a slicing that never ran
both produce a byte-identical file — the first two runs of the audio harness "passed"
everything while the treatment had never executed, because `--batch-size 1` routes to the
NON-batched server where the admission block does not exist. Conversely one flipped argmax
produces a completely different, equally valid utterance. Both failure modes are invisible
to a mel-corr gate. Hence `--prefill-slice-check` and the `[ADMSLICE]` marker.

### 3.2 `--prefill-slice-check <S>` — the state oracle

One process, no server, no sampling: build the prompt, run `qwen_talker_prefill`, snapshot
`dec_x` + the whole KV cache; then run `plan` + `range` slices and compare.

| comparison | dec_x max abs / max ref | K rows differing | V rows differing |
|---|---|---|---|
| slice 2, 3, 4, 5, 7, 16, 32, 48 **vs unsliced** | **0.000e+00** | **0** | **0** |
| sliced vs MONOLITHIC, 38-position prompt | 7.96e-04 | 783/1064 | 783/1064 |
| sliced vs MONOLITHIC, 107-position prompt | 1.686e-03 | 2646/2996 | 2646/2996 |
| slice 1 vs unsliced (M=1 matvec path) | 1.37e-03 | 779 | 783 |

`kv_len` equals the monolithic value in every arm.

**Resume is bit-exact.** Pausing between any pair of token ranges leaves exactly the state
an uninterrupted run leaves — user requirements 2, 3, 4 and 5, proven at zero, not at a
tolerance.

**One real defect was found and fixed by this oracle.** A slice of exactly ONE token takes
the M=1 matvec path inside the shared prefill projection kernels, whose accumulation order
differs from the matmat path; the prefill then depended on where the boundaries fell
(S=2 and S=4 on a 29-token prompt differed from S=one; S=3, 8, 16 did not — the tell was
that failure tracked "some slice has 1 row", not the slice size). `qwen_prefill_slice_next`
now absorbs a 1-token remainder into the current slice, so a slice is at most `S+1` tokens
and never one. The rule is exported and used by BOTH the serving loop and the oracle, so
the gate exercises the shipped rule. After the fix every partition is exact.

`QWEN_PREFILL_SLICE=1` still takes the M=1 path by definition; the oracle labels it
EXPECTED-DIFFERENT rather than counting it as a defect. The minimum useful slice is 2.

### 3.3 `tests/prefill_slice_parity.py` — the server harness

Mac, `qwen3-tts-0.6b`, batch-size 2 (mandatory: 1 is a different server), temperature 0,
seed 42, four texts.

| gate | result |
|---|---|
| S — state oracle at slice 2/3/16/48 | PASS |
| 0 — the sliced arms really sliced, the control really did not (`[ADMSLICE]` marker) | PASS |
| A — sliced-many vs sliced-one, byte-identical audio, all four texts | PASS |
| B — sliced vs monolithic, spec 9.2 mel-corr >= 0.99 | **SPEC CONFLICT** (see 3.4) |
| C — disconnect mid-admission, treatment vs control | INCONCLUSIVE (PLAN TQ-2) |
| D — pending admission beside an established stream: sliced under load, both requests complete, no error, length drift 5.9 % / 14.4 % | PASS |
| E — an unusable flag value stops the server | PASS |

Gate C is inconclusive because the post-cancel request is refused with 503 on the CONTROL
arm too: that is the open fail-fast defect (PLAN TQ-2), it reproduces without this change,
and it therefore cannot be a verdict on this change. It stays open.

Gate D is deliberately NOT an exact oracle: at two active slots the engine is not
batch-invariant, so two runs of the control against each other already score mel-corr ~0.4.
Claiming a parity gate there would be claiming a property the engine does not have.

### 3.4 THE OPEN CONFLICT — spec section 5 versus spec section 9.2

Section 8 predicted the deviation ("attention over earlier tokens uses bf16 K/V instead of
f32 -- NOT EXACT; bounded") and section 9.2 set the gate at mel-corr >= 0.99 with identical
codes for >= 9/10 texts. **That gate is unreachable for ANY token-outer sliced prefill.**
The measured state deviation is ~8e-4 (38 positions) to ~1.7e-3 (107 positions) — below one
bf16 ulp (3.91e-3), i.e. genuinely at the KV quantisation floor — but at temperature 0 that
is enough to flip one sampled code, and from there the utterance differs entirely.

Exactness would require the f32 K/V of all 28 layers to stay alive across slices, because
slices are token-outer and layer-inner: at slice 2, layer 5 needs layer 5's K/V of slice-1
tokens, which slice 1 computed and discarded. Section 5 forbids exactly that (69 MB per 300
tokens). The two sections are incompatible and the measurement says which one falls.

Note the asymmetry this exposes in the existing engine: during a monolithic prefill a
prompt token attends over f32 K/V, but every decode step afterwards attends over the bf16
cache. The sliced path is the self-consistent one; it is the control that is special.

**This is escalated, not resolved.** The options, none of them taken here:
1. Accept that enabling the flag changes which valid utterance each request produces, and
   qualify it the way RES1_V2 was qualified — a paired listening/CER bank — instead of a
   mel-corr identity gate.
2. Make the monolithic prefill attend over the bf16 cache too. Sliced would then equal
   monolithic exactly, at the cost of changing the product's audio once, deliberately.
3. Drop the mechanism.

No product profile names the flag, and the default is off, so nothing is shipped either way.

## 4. What has NOT been done

* No x86 execution of any kind. No AWS box was touched.
* No `admit_ms` diagnostic, no server A/B, no soak — spec sections 10 and 11 are open.
* The frozen product profile is unchanged and does not name `QWEN_PREFILL_SLICE`.
* Spec 11 step B (weight-only bf16) not started, by instruction.

## 5. Exact next commands on the Turin host

The order is the specs' own, and the first x86 gate belongs to WIN-12, not to this item.

```
# PHASE 0  baseline
make clean && make blas SIMD=avx512bf16 -j$(nproc)
./qwen_tts --self-test                       # first REAL x86 gate for QWEN_SD_GLUE
python3 tools/serving_profile.py preflight --profile configs/perf/turin-c8a-32c-vnni-product.json

# PHASE 1  WIN-12 microbench      GO >= 3 ms and identical WAV sha256
taskset -c 4-7 ./tests/decode_quantum_bench ...   # matrix from spec section 10
#   control QWEN_SD_GLUE=0   treatment QWEN_SD_GLUE=1

# PHASE 2  WIN-11A microbench     GO >= 2 ms, parity <= 1e-5
#   control QWEN_SD_CONVT_STACK=0   treatment QWEN_SD_CONVT_STACK=1

# PHASE 3-4  only for a microbench winner: single-CCX STAGE diagnostic, short server A/B,
#            then the 10-minute C12 soak.

# PHASE 5  WIN-10
python3 tests/prefill_slice_parity.py --model qwen3-tts-1.7b --slice 48
#   then the DIAGNOSTIC of spec section 10: 1x8@0-7, cap 3, QWEN_STAGE_TRACE=1, 3 clients,
#   3 minutes, arms unset vs QWEN_PREFILL_SLICE=48, metric = admit_ms distribution.
#   GO to the A/B only if admit_ms p95 <= 30 ms (control 60-240) and total prefill work is
#   within +15 % of inline.
```

One mechanism at a time. A failed gate is an IMPLEMENTATION NO-GO for that implementation,
never a verdict on the mechanism.

## 6. Status

Implementation gates: all PASS. Spec 9.2 gate: CONFLICT, escalated in 3.4. x86: NOT RUN.
