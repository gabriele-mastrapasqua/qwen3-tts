# Task · P2 checkpoint

## Question

Close the bounded small-quantum decoder work with an evidence-backed status of SQ-1,
SQ-2 and SQ-3, without claiming that the full end-to-end strip executor has landed.

## Known facts

- Checkpoint source is clean `aa4e93a` on `feature/x86-amx-vnni-oss`.
- The serving reference is Qwen3-TTS 1.7B INT8 on the single-socket Xeon Platinum
  8581C host, twelve physical CPUs online, SMT off, 2x6 prefork, engine decoder pool,
  batch cap 2, q8 and ragged threshold 2.
- Persistent Design-D INT8 B packs and the ragged decoder path are already trusted
  serving infrastructure.  Diagnostics show real AMX execution for the tested eligible
  decoder panels, not merely an AMX capability claim.

## Unknowns

- No retained clean whole-request weighted census supports a current percentage for
  all four SQ-3 quantities across VQ, transformer, ConvNeXt, decoder and output work.
- A complete strip executor that keeps the whole residual/upsample chain in a bounded
  cache-local strip is not implemented.
- A batch-aware gather/quantise kernel might differ from the rejected one-row design,
  but there is no evidence to justify implementing it before architecture review.

## Files/functions inspected

`qwen_tts_speech_decoder.c` (`cs_conv1d_amx_range`, `cs_convt_direct`, `rag_conv1d_amx`,
`sd_rag_panel_worker`, `conv_decoder_forward_streaming`), `qwen_tts_kernels.c`
(`sd_gemm_panel_amx_d`, `qwen_int8_quant_rows`), `tests/serve_parallel_wave.py`,
`PLAN.md`, and the linked P2/P1/AMX evidence addenda.

## Evidence

### Status

| area | status | current meaning |
|---|---|---|
| Design-D INT8 persistent B pack + ragged server AMX | PROMOTED / active reference | Real `TDPBSSD`, persistent packs, ragged batching, zero fallback in tested eligible panels |
| SQ-1 warm range slice / direct INT8 A preparation | PROMOTED to the measured serving reference, not a compiled default | `QWEN_SD_STREAM_STRIP=1` computes newly produced warm columns; full strip pipeline remains an AR-1 candidate |
| SQ-2 direct ConvT | KEEP, default-off, not promoted | Correct and server-reached; serving result neutral/inconclusive |
| SQ-2 direct depthwise | KEEP, default-off, not promoted | Correct and server-reached; no defensible serving win |
| SQ-2 direct warm input | KEEP, default-off, not promoted | Removes materialisation locally; end-to-end serving neutral/negative |
| SQ-2 fused residual AMX | KEEP candidate, default-off, not promoted | Positive short serving signal; arithmetic/order changes require separate quality qualification. Existing quality evidence was not byte-identical to the INT8 reference, so this is not a promoted serving path. |
| SQ-2 BLAS-C residual | REJECTED / reverted | Worse C4 tail/cadence; `53fac21` reverted by `bd968f8` |
| SQ-2 one-row direct gather/quantise | REJECTED / reverted | WAVs byte-identical, but C4 STREAM_RTF p95 `0.776 → 0.954`; `aa90518` reverted by `4089bbd` |
| low-N M split / synthetic wider task count | REJECTED | More AMX tasks regressed streaming; not a P2 path |

### SQ-3 AMX quantities

The following scopes are deliberately separated.  `decoder-eligible` means the actual
Design-D INT8 ragged matrix panels for which the server emits `amx` and `fallback` panel
counters.  It is not a whole-request denominator.

| quantity | supported current value | scope and basis |
|---|---:|---|
| `amx_dispatch_share` | 100% | Decoder-eligible panels in the retained diagnostics: `amx=panels`, `fallback=0` for M=96/192/384/768 and representative K/Kp shapes. Whole request: UNKNOWN. |
| `amx_matrix_mac_share` | 100% | Decoder-eligible Design-D matrix MACs are sent through the AMX panel entry point in those diagnostics. Whole request including non-matrix decoder stages and Talker/CP: UNKNOWN. |
| `amx_addressable_mac_share` | 100% of the measured Design-D eligible subset | All measured eligible residual-conv panel work is addressable by the existing Design-D path. Whole decoder/request addressable fraction: UNKNOWN. |
| `amx_request_wall_share` | UNKNOWN | Existing phase/SDRAG timers are diagnostic and do not provide a clean whole-request denominator for the current control. The older `~10–20%` estimate is retained as historical context only, not as a current P2 measurement. |

The current evidence therefore proves path reachability and eligibility coverage, but not
that AMX dominates serving wall time.  The observed phase breakdown instead shows that
preparation, rendezvous, snake and surrounding decoder work remain material.

## Conclusion

**P2 CLOSED as a bounded implementation and measurement checkpoint.**  The trusted
Design-D decoder path and the warm range slice are in the active reference configuration;
the bounded SQ-2 candidates are either preserved as independently disable-able
default-off experiments or explicitly rejected and reverted.  Closing P2 does not claim
that all decoder fixed cost was removed and does not claim that the full strip executor
exists.  Those are architectural questions for the separate AR-1 review.

## Next action

Hand this frozen checkpoint to the separate AR-1 reviewer.  AR-1 must review this exact
committed HEAD before any post-P2 architecture is selected; this task makes no choice
between a full strip/small-quantum decoder, incremental/preemptible prefill, lead-aware
bounded scheduling, or a staged combination.
