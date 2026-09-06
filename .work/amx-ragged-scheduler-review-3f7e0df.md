# AMX ragged scheduler review — 3f7e0df

Task · Preserve the independent review state for the ragged decoder AMX panel
scheduler before server qualification.

Question · Is commit `3f7e0dfa5832216e6c26a5b66f57e6f6a6c0bf67` safe to qualify on
the streaming server?

Known facts · The independent read-only review verdict is **SAFE TO CONTINUE
QUALIFICATION**. The reviewed implementation is the parallel ragged-panel
scheduler at that exact code commit.

Unknowns · Sustainable C4 tail latency over a realistic multi-wave run; the
effect of decode-chunk size; and whether further matrix aggregation is needed
after qualification.

Files/functions inspected · Decoder ragged panel dispatch and join/fallback
logic; decoder-pool wrapper; existing serial/parallel server output checks.

Evidence · The review verified disjoint panel output ownership, per-worker
scratch/TLS accumulator semantics, join-before-bias/tail update, exact
recompute fallback, no nested engine-pool BLAS dispatch, and unchanged
portable/VNNI/Arm paths. Serial and parallel server WAVs were already
byte-identical.

Conclusion · **SAFE TO CONTINUE QUALIFICATION**.

LOW / PERF (deferred) · A worker can allocate scratch after the atomic panel
cursor is effectively drained. Do not apply only an `atomic_load` early return:
if optimized later, use a claim-first scheme that claims real work before large
scratch allocation and preserves allocation-failure full-recompute semantics.

LOW / DEFERRED / PRE-EXISTING · The private decoder pool has no complete
after-fork reset if it had been started before fork. The current production
path does not appear to make this reachable. If fixed later, reinitialize the
pool synchronization state (mutex/condition/team state), not only counters.

Targeted future tests (non-blocking) · Forced fragmented-ragged parity;
allocation-failure recompute; BF16 serial/parallel parity; private-pool
variant; forced AMX-reject parallel fallback.

Next action · Qualify the clean committed Design D INT8 server at C4 on the
canonical 2x6 SMT-off host, then run the bounded decode-chunk sweep only if C4
has useful steady-state margin.
