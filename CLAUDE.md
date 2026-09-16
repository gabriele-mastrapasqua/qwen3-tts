# Claude Code — repository entrypoint

`ENGINEERING.md` is normative for this repository. Read it first, then use the concise
`PLAN.md` and only the `.work/*.md` addendum linked by the task. The addenda are the
reviewed detail/control plane; raw run artifacts remain private as specified there.

Use the backend matrix and dispatch/effective-config gates from `ENGINEERING.md` for
every runtime or benchmark claim. Build with `make blas` (or an explicit `SIMD=...` on
x86) and run the applicable `--caps`, `--self-test`, `--dispatch-map`, and golden gates.
