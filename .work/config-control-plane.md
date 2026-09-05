# Server config control plane — why, and what "done" means

Addendum to PLAN task **P1-CONFIG**. Written 2026-09-05, from the tree.

## The evidence that this is an abstraction problem, not a documentation problem

The correct configuration for the GCP box already existed, versioned, in
`configs/perf/gcp-c4-standard-24-vnni-ttfa.json`: `prefork_workers: 2`,
`threads_per_worker: 6`, affinity `0-5` / `6-11`, and an explicit warning not to use the 24
logical CPUs without re-measuring SMT. Three server campaigns were nevertheless run at 2x8,
from memory, with both workers landing on the same twelve physical cores. **The JSON was right
and the experiment was wrong**, which is the proof that the next step is not better docs: it is
removing the normal path that can ignore them.

Measured duplication in the current profiles (8 carry an `environment` block):

- **22 of 28** distinct env keys appear in more than one profile;
- `OPENBLAS_NUM_THREADS`, `OPENBLAS_THREAD_TIMEOUT`, `QWEN_PREFIX_CACHE` in **8**;
- `QWEN_PREFILL_MATMAT`, `QWEN_POOL_SPIN`, `QWEN_DECODER_BATCH` in **7**;
- `QWEN_POOL_SPIN` is `4096` in six x86 profiles and `65536` in the Arm one.

So PLAN P5.0 — "set `QWEN_POOL_SPIN=65536` as the x86 server default" — currently means editing
six files and hoping none is missed. A common engine default has no owner.

Worse, one 200-line object holds four different kinds of statement at once: runtime
configuration, experimental candidates ("candidate only", `QWEN_VNNI_PREPACK: null`), benchmark
history ("previous campaign used..."), and open TODOs ("re-measure on the corrected build").
An agent or a human reading it reconstructs a *probable* configuration from half-sentences.

## Target layering — a value has exactly one owner

    common engine defaults
      -> Linux CPU server defaults
        -> backend-family policy      (ARM/KleidiAI | x86 VNNI | x86 AMX)
          -> host topology            (AWS c8a | GCP c4 | Axion | ...)
            -> experiment override    (ONLY the variable being A/B tested)

A host profile writes a value only when it has a MEASURED reason to diverge, and says why.
Everything parity proves to be common engine semantics — BLAS ownership, decoder batching,
prefill mode, persistent regions, prefix cache, pool spin, scheduler policy, decoder chunking,
AMX/VNNI crossover policy, prepack policy — belongs at the common or backend-family level.

## One canonical resolver/launcher, and no way around it

Today every harness is trusted to interpret the profile itself. Instead one launcher must:
load the inheritance chain, resolve defaults, verify the REAL host against the resolved
profile (CPU, physical cores, SMT state, online CPUs, NUMA), verify build/ISA, compute and
apply the affinity masks, prepare the environment, start the server, run the workload, and
write the resolved configuration into the artifact. Other scripts stop knowing what
`QWEN_POOL_SPIN` is: they receive a configured endpoint. A qualification run started by hand
around the resolver must FAIL, not start with whatever defaults it finds.

## Every run materialises a small, verifiable resolved object

    host:   cpu, physical_cores, smt, online_cpus, numa
    server: workers, threads_per_worker, masks[]
    effective: pool_spin, blas_threads, decoder_batch, prefill, prefill_chunk, ...
    config_hash, binary_hash, model_hash

Requested vs effective must both appear, with the reason when they differ
(`foo.requested=1 foo.effective=0 foo.reason=unsupported_on_avx512vnni`). Two results are
comparable only when those hashes are.

## Namespace separation

`effective configuration` · `experimental candidates` · `historical provenance` ·
`qualification evidence` are four different things. The runtime reads only the first.

## Definition of done

- no duplicated semantic runtime default across host profiles;
- one inheritance/resolution mechanism;
- one canonical launcher used by every benchmark suite;
- qualification scripts cannot bypass profile resolution;
- effective config emitted for every run;
- actual host topology verified against the resolved profile, mismatch is fatal;
- every runtime flag has one owner and one default;
- experiment overrides contain only the variables intentionally changed;
- resolved config + binary + model hashes stored with the results;
- changing one common x86 default propagates to every x86 host profile.

## Sequencing

PARITY-1/2/3 first: server defaults cannot be made sane before the backend matrix says which
features are genuinely equivalent, and which knob owns each. Then P0-PROFILER. Then this.
The one piece pulled forward is the EFFECTIVE-CONFIG DUMP, which lives in PARITY-2 because the
profiler needs it: without it the profiler can report 50 ms in AMX while an env everyone
believed active was silently ignored.
