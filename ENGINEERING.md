# Engineering Rules

Normative for every coding agent and every human working in this repository.
`CLAUDE.md` and `AGENTS.md` only point here. Keep this file under ~150 lines.

The principle: the agent must not remember the project; the repository must make
forgetting impossible. `PLAN.md` says what remains, this file says how we work,
`.work/` addenda hold the details of one task, and the program itself proves which
path it is running.

## 1. Work from a small plan

Before non-trivial work read `PLAN.md` (tracked, next to this file).
It is a checklist, not a research document or a diary.

- Keep it below ~150 lines. If it is longer, shorten it before adding anything.
- `- [ ]` tasks with an id (`P1.4`). One task = one independently verifiable outcome.
- Mark `[x]` only after validation.
- No investigations, no transcripts, no rewritten history in the plan.
- A task that needs substantial analysis links one addendum:
  `- [ ] P1.4 Audit AMX B>=4 runtime path — detail: .work/p1-4-amx-runtime.md`
- Load an addendum only when working on its task. Do not load unrelated addenda.

## 2. Separate plan from evidence

Long notes and hypotheses live under `.work/` as reviewed public-safe `*.md` addenda;
the profiler dumps, transcripts and run artefacts they cite stay in the untracked
private areas beside them.
Every addendum starts with: Task · Question · Known facts · Unknowns ·
Files/functions inspected · Evidence · Conclusion · Next action.
An addendum never becomes a second global plan.

Privacy split. The public-safe control plane is TRACKED: `ENGINEERING.md`, `CLAUDE.md`,
`AGENTS.md`, `PLAN.md`, the reviewed `.work/*.md` addenda, stable docs and code. Untracked
and private: the raw evidence those notes summarise (`.work/evidence/`, `.work/private/`,
`private/`), profiler and debug dumps, benchmark artifact trees, `plan_*.md` scratch notes.

Tracking the control plane is deliberate: untracked `PLAN.md`/`.work/` state diverges
between worktrees, and that caused real coordination failures. Worktrees now share both
through normal git history.

Never `git add .work/` wholesale — track one reviewed file at a time. Before staging any
control-plane file, read the STAGED content and reject: credentials and tokens, private
hostnames or addresses, SSH commands and key paths, absolute personal paths, customer or
commercial material, raw profiler dumps, raw benchmark artifact paths, machine
identifiers. A tracked note SUMMARISES private evidence; it never reproduces it.

## 2b. Plan integrity

`PLAN.md` never contains dangling references. Every `detail: .work/<file>.md`, every
tracked doc it names, and every evidence file behind a `[x]` task MUST exist before the
plan update is reported as complete. Before finalising any plan edit run
`python3 tools/check_plan.py`; it fails on a missing `.work/` or repository path, a
duplicate task id, a `[x]` task pointing to a missing file, or an addendum naming a task
the plan does not have. On failure: do not claim completion, create the real file or
remove the reference, report the failure. Never create placeholder links or empty
addenda for work that has not been written.

Worktrees share the control plane through git. A task finished in another worktree still
ends with an explicit statement of what changed — PLAN task and status, which addenda —
but reconciliation is a merge of tracked files, not a re-typing of prose, and the private
evidence behind an addendum does not travel with it.

## 3. Code is the source of truth

Docs, feature tables and old benchmark pages may be stale. For implementation or
audit questions: inspect the current code, then git history when needed, then docs
as secondary context. Never claim backend support from documentation alone.

## 4. Backend claims require a matrix

Never call a change "cross-platform", "common", "ARM", "x86" or "AMX" without
reading its gates: `#if`, runtime predicates, `qwen_mm_use`, fallbacks.
Classify runtime/kernel work as
**A** common runtime (shared automatically) ·
**B** common design, backend-specific implementation ·
**C** backend/ISA-specific.
Backends: x86 AVX2 · x86 AVX-512F · x86 AVX-512 VNNI/BF16 · x86 AMX ·
Arm NEON/DOTPROD/i8mm/SVE with KleidiAI · Apple Arm (GCD pool, Accelerate) · CUDA · Metal.
Reference: `docs/cross-backend-audit-2026-09-05.md`.

## 5. A benchmark is invalid until dispatch is proven

CPU/server performance work MUST follow `docs/BENCHMARKING.md`. If a canonical benchmark
tool, profile or procedure changes, the same change updates the runbook. Do not select
benchmark commands from memory or old docs.

Before every performance run record, from the process being measured and after its
environment is applied: git commit and dirty state, source fingerprint and binary
hash, compiler SIMD target, CPU model and features, profile name, exact command,
exact environment, worker topology and CPU masks, and the resolved backend, GEMV,
GEMM/matmat, prefill and decoder paths with the relevant flags.

Every benchmark directory carries one machine-readable `run_manifest.json`, produced
from the actual serving configuration after the environment is applied (fields in
`docs/BENCHMARKING.md` §4 and §6). A dispatch file generated independently before the
env is never authoritative evidence.

Never infer the active kernel from a profile filename, a requested env flag or the
compile target name. The strongest evidence is in-process: the server's own resolved
table at startup and the kernel census after warm-up, compared against an
expected/allowed/forbidden manifest per operation and profile (a native-BF16 prefill
profile forbids the f32/generic fallback; the decoder's serial OpenBLAS SGEMM may still
be allowed; `auto` reports what it resolved to and does not fail for differing from
another backend). A side tool that re-runs the predicates is second-best.

An explicit request is a request, not a capability. If an explicitly requested path
differs from the resolved path, or a forbidden leaf ran: stop, do not run the
benchmark. A fallback run is never reported as a measurement of the requested path.

## 6. Fallbacks must be visible

Every important optimized operation exposes which implementation ran, at least:
native optimized kernel · generic project kernel · external BLAS/Accelerate ·
scalar/f32 fallback · unsupported. A silent fallback in a qualification run is a failure.
Capability declarations (`g_mm_gate[]`, `qwen_mmk_compiled/supported`, the
`*_available()` predicates) are the single source the dispatcher and the reports read;
never keep a second copy of them.

## 7. Never benchmark a contradiction

Examples: AVX-512 BF16 requested on a binary compiled without `-mavx512bf16`; AMX on a
topology whose B never reaches the AMX gate; KleidiAI claimed on a build without i8mm;
server batching parity shown with the CLI paragraph splitter. Whenever a contradiction
can be detected mechanically, add a preflight gate that refuses the run.

## 8. Runtime changes need a backend impact review

For changes to allocations/scratch, thread pools, scheduler, dispatch/barriers, BLAS
ownership, batching or stream lifecycle, state before completion which backends
actually receive the change (rule 4). Common-looking code is not common behaviour.

## 9. Performance reports use explicit terminology

WAVE (screening) · SOAK (the production gate) · POISSON (overload) · DIAGNOSTIC
(profiling, never a number). Never a bare "RTF". Report separately: STREAM_RTF
p50/p95, TOTAL_RTF when relevant, TTFA p50/p95, request count, errors/rejects/timeouts,
wall duration, artifact path.

## 10. Run lifecycle is strict

A persistent server is never inside a bare `wait`. For every automated run: save the
exact client PIDs and wait only for them, bounded timeouts, confirm the counters
advance and the CPU load is real shortly after launch, terminate samplers and server
explicitly, report survivors=0. Print the artifact path and the phase at every step.

## 11. No opportunistic scope expansion

While executing task X do not start task Y, do not change numerical precision, do not
change production defaults, do not build a new profiler framework, do not refactor
unrelated code. Record discoveries as new `PLAN.md` tasks and continue X.

## 12. Numerical changes are separate work

An optimization that changes arithmetic or output (BF16 to INT8, a different
accumulation order, an approximation or fusion that reorders floating point) is not a
structural optimization. It needs its own quality qualification and is never promoted
on performance numbers alone. Production defaults change only by explicit decision.

## 13. What gets committed is what was built, and what qualifies is committed

Canonical SOAK evidence comes from a CLEAN COMMITTED TREE: build that exact commit.
A dirty-tree binary serves development, WAVE and DIAGNOSTIC only and is labelled
NON-QUALIFYING. An uncommitted candidate that needs canonical qualification gets an
isolated candidate commit or worktree first, and that exact tree is built. Both the
GCP dispatch incident and the broken-HEAD discovery were evidence produced by an
artifact different from the source state we believed we had.

Selective staging (`git add -p`, patched copies) is verified by building an isolated
checkout of the staged/committed tree, not the working tree, on every touched
platform that is available; platforms that were not available are listed under
WHAT REMAINS UNKNOWN, never reported as tested. Commit messages: English, imperative,
what and why, no tool attribution. `plan_*.md`, `private/` and the raw evidence areas
(`.work/evidence/`, `.work/private/`) are never committed; the public-safe control plane
(`PLAN.md`, reviewed `.work/*.md`) is.

## 13b. Repository integrity

Tracked canonical tools and scripts never depend on untracked local files. Every
repo-local runtime dependency of a tracked script is itself tracked, produced by a
documented build step, or explicitly optional; a fresh clone never silently depends on
one developer's working tree. `python3 tools/check_repo_integrity.py` fails on a tracked
script that references a file present here but not in git. Three failure classes, all
seen in practice: PLAN integrity (no dangling references, §2b), repository integrity
(this rule), benchmark integrity (no qualifying run without proven commit, binary, env
and dispatch, §5 and §13).

## 14. Completion rule

A task is complete only when the final report contains:
WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS UNKNOWN ·
VERDICT: PROMOTE / KEEP / INCONCLUSIVE / REJECT. Unknowns are stated, not omitted.
