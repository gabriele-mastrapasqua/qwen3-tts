# Claude Code — read this first

`ENGINEERING.md` in this directory is normative: work from `PLAN.md` (local), keep
evidence in `.work/`, trust the code over the docs, classify every runtime change per
backend, prove the dispatch before any benchmark, never commit `plan_*.md` / `.work/`.

Build: `make blas` (SIMD auto-detected; `make blas SIMD=avx512bf16|amx|portable` on x86).
Gates: `./qwen_tts --caps`, `--self-test`, `--dispatch-map`; `make test-golden`.
Do not touch AWS/GCP boxes or start runs longer than two minutes without the lifecycle
rules of `ENGINEERING.md` §10.
