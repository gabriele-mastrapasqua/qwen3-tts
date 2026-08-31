# Engineering Method — MACHINE-FIRST / MEASURE-FIRST

> **Do not optimize code. Optimize work.**
> **Do not trust expected behavior. Observe the machine.**
> **Do not celebrate a faster component until the system is faster.**
> **Every unexplained millisecond is an engineering question.**

This is a permanent project rule, alongside the benchmark, provenance, correctness and
qualification rules. It applies to **all architectures and all future optimization work** —
Arm, x86, GPU, or whatever comes next. It is not an Axion rule and not a TTFA rule.

Understand the machine. Understand the data. Shape the work for the machine. Measure
reality. Never trust intuition when it can be measured.

*The lineage is Michael Abrash's* Graphics Programming Black Book *— the machine is the
authority, the profiler outranks the intuition, and the largest wins come from changing the
shape of the work rather than the instructions that do it. What follows is that stance
applied to this engine, with the mistakes this project actually made written into it.*

A note on what this is not: it is not an aesthetic of low-level programming. Hand-written
SIMD is not the point, and preferring it for its own sake would be the opposite of the
lesson. The rule exists to make us **more suspicious, especially of results we like**.

---

## 1. Never trust your own performance claims

Every performance conclusion is a hypothesis until independently supported.

This project has already produced each of these failures:

- two unrelated benchmark numbers combined into a causal story that was never measured;
- a harness that made INT8 look slower because its preparation work was serialized while
  the other arm's was not;
- the conclusion reversing once that harness was fixed;
- and a corrected result that still carried a threading confound until a third arm was added.

**Never promote an interpretation because the numbers look convincing.** Before accepting a
result, actively try to prove yourself wrong.

## 2. Account for the time

For every hot path, keep asking: **why do these milliseconds have to exist?**

Decompose wall time until the dominant terms correspond to actual machine work: useful
arithmetic, memory traffic, packing and unpacking, layout conversion, quantization and
dequantization, copies, synchronization, barriers, scheduling, queue residence, allocation,
cache and TLB effects, thread creation and wakeup, kernel or dispatch overhead, redundant
computation.

A label — `prefill`, `decode`, `GEMM`, `queue`, `conv_up` — **is not a causal explanation**.
It is another box to open. Keep decomposing while a large unexplained term remains.

## 3. Optimize the shape of work before the instruction

Before optimizing a kernel, ask whether the program is presenting the right problem to the
machine. Look for structural waste first: N calls that could be one; serial work that can
safely become parallel; small irregular operations that can become one regular dense
operation; A→B→A layout conversions; repeated packing; unnecessary copies; redundant
computation; poor locality; bad traversal order; pathological dimensions; batching
accidentally disabled by another feature; excessive synchronization; work done before it is
needed; state recomputed instead of preserved.

A faster SIMD/INT8/BF16/AMX/SVE/CUDA kernel is irrelevant if selecting it forces the
surrounding program into a worse shape. **Optimize the program, not the benchmarked
instruction.**

## 4. Data shape is part of the algorithm

Layout, batching, traversal order, working-set size, tiling, alignment and temporal reuse are
first-class algorithmic decisions, not implementation details.

Reason explicitly about the target machine. On CPU: cache hierarchy, cache lines, TLB,
memory bandwidth, SIMD and matrix ISA, register pressure, core topology, synchronization,
NUMA. On GPU: memory hierarchy, coalescing, occupancy, warp/wave execution, shared memory,
register pressure, tensor units, launch and dispatch, fusion, persistent work.

These details change between machines. **The method does not.** Do not hard-code today's
assumptions into the philosophy — tomorrow the best implementation may be on entirely
different silicon.

## 5. Every optimization needs a cost model

Before writing optimization code, state: current cost · suspected cause · proposed
transformation · maximum plausible saving · new work introduced · correctness and quality
risk · the smallest experiment capable of killing the idea.

If the maximum plausible saving is too small to matter to the product objective, **do not
optimize it**. Prefer killing weak ideas cheaply.

## 6. Measure the complete transformation

Never claim a program optimization from a kernel-only benchmark when the transformation
requires additional work.

INT8 timing must include everything production needs — packing, quantization, scales, copies,
GEMM, conversion, post-op — not the INT8 GEMM alone. Batching must include packing,
compaction and synchronization. Measure from the earliest operation the optimization
introduces to the point where both paths have equivalent outputs.

## 7. Always build the control that can embarrass the hypothesis

When an experiment changes several dimensions, add the controls that isolate them.

Do not compare only `serial FP prep + FP GEMM` against `parallel INT8 prep + INT8 GEMM`.
Also run `parallel FP prep + FP GEMM` — otherwise an INT8 result may be a threading result.

Before every experiment ask: **what control would make my preferred explanation look stupid
if it were wrong?** Run it when practical.

## 8. Harness validity precedes performance

A result is invalid if the harness materially changes the execution model under evaluation.

Verify before trusting: real production shapes · real thread topology · persistent pool vs
per-call thread creation · which component owns which threads (BLAS included) · real layouts
· real batching and group size · real ISA path · real model/checkpoint · first-chunk vs
steady-state semantics · equivalent work on both sides.

If the harness differs, document the difference and **do not extrapolate through it as fact**.
Never call a measured value a floor, ceiling, lower bound or upper bound unless the argument
establishing that bound is actually proven.

## 9. Source tells you what may happen; runtime tells you what did

Static inspection is evidence about implementation and control flow. It does not establish
runtime dispatch when runtime configuration, compile-time ISA selection, environment
variables, dynamic batching or fallback paths are involved.

For performance-critical dispatch, **prove the runtime path**. Do not infer a historical
benchmark's configuration from today's source tree.

## 10. Correlation is not causation

Queue residence rising does not prove the queue is the bottleneck. A faster kernel coinciding
with lower TTFA does not prove the kernel caused the improvement. An INT8 configuration being
slower does not prove INT8 arithmetic is slower.

Trace the causal chain until the mechanism is supported, and use the words deliberately:
**measured · observed · statically established · inferred · hypothesis · unresolved**.

## 11. Simple wins are first-class wins

When performance is comparable, prefer: deleting work over accelerating work; one good
execution path over many feature combinations; existing primitives over new infrastructure;
static or simple dispatch over configuration state explosion; small understandable patches
over elaborate frameworks; transformations that preserve numerical behaviour over
approximate ones.

Complexity carries a permanent maintenance and performance cost. **A ten-line change removing
15 ms is better engineering than a thousand-line subsystem removing 17 ms**, all else equal.

## 12. Do not confuse generality with quality

General-purpose implementations must serve workloads and machines this engine may not need.
llama.cpp, KleidiAI, BLAS libraries and compiler output are **inputs and references, not
authorities** — never assume upstream is optimal for our exact shapes.

Equally, never assume our implementation is better because it is specialized. Measure both on
the actual workload. Specialization earns its place only when knowledge of our workload
removes work or presents a better shape to the machine.

## 13. Optimize product metrics, not microbenchmarks

Microbenchmarks answer mechanism questions. They do not establish product wins.

`microbenchmark → component → critical path → server → concurrency → sustained workload →
quality/correctness`

**Never skip a level in the claim.** "Kernel 2x faster" is a kernel result; it is not
"TTFA 2x faster".

## 14. Correctness is an invariant

Performance never excuses wrong audio, changed language, broken cancellation,
nondeterministic corruption, invalid accounting, unsustainable STREAM_RTF, hidden overload,
or misleading metrics.

Approximate numerical paths require their own quality gates. Prefer exact transformations
first when their expected gain is competitive.

## 15. No large experiment without a kill-first

Inspect → derive a cost model → identify the dominant term → build the cheapest falsification
→ validate the harness → run the smallest useful measurement → only then escalate.

Do not spend twenty minutes proving what a twenty-second component measurement could reject.

## 16. The agent is not an authority

You are an engineering assistant, not a source of truth. Do not grant yourself permission to
turn an inference into a fact.

When evidence conflicts with an earlier conclusion, **withdraw it immediately**. Never defend
a previous answer for consistency. If uncertain, say so. If a decision rests on an assumption
you cannot establish, stop and surface it.

For architectural decisions, surprising results, ambiguous causality, destructive changes,
quality/performance tradeoffs, or anything that would materially redirect the project:
**present the evidence and ask before proceeding.** Independent adversarial review of your
reasoning by another model or tool is desirable engineering practice, not friction.

## 17. Keep an engineering ledger

Every meaningful performance investigation records: **FACT** (measured or statically
established) · **HYPOTHESIS** (the believed mechanism) · **TEST** (smallest falsification) ·
**RESULT** (raw outcome with provenance) · **DECISION** (pursue / reject / unresolved) ·
**CLAIM SCOPE** (microbench / component / server / workload / architecture).

Rejected ideas are evidence. Do not silently rediscover and re-run them.
Keep it in the repository, next to the benchmarks it explains.

## 18. The final machine-first question

Before proposing any optimization, answer explicitly:

**What work is the machine doing that it should not have to do?**
**Is the useful work presented in the shape this machine executes best?**

Only then: **can I make the instructions faster?**

---

This rule supersedes convenience, intuition, framework convention, and agent confidence.

**Think first. Measure second. Change third. Measure again.**
