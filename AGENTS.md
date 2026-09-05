# Agents — read this first

`ENGINEERING.md` in this directory is normative for every coding agent (Codex, Claude,
others): small `PLAN.md` (local, untracked), evidence in `.work/`, code over docs,
backend matrix for every runtime claim, dispatch proven before any benchmark, strict run
lifecycle, explicit completion report. Never commit `plan_*.md`, `.work/` or `private/`.

Build: `make blas`. Gates: `./qwen_tts --caps`, `--self-test`, `--dispatch-map`,
`make test-golden`.
