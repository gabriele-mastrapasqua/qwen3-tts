#!/usr/bin/env python3
"""Generate tests/fixtures/: random blocks of every ggml type, plus what
llama.cpp's own dequantizer makes of them.

This is the only test in the suite that can catch a shared misunderstanding of
a format. Everything else compares ingot against ingot — a writer and a reader
that agree perfectly can still both be wrong. Here the reference comes from a
different implementation, in a different language, maintained by the people who
define the format.

The blocks are pseudo-random rather than quantized from real data, and that is
deliberate: quantizing a smooth signal leaves most of a codebook untouched,
while random bytes walk the whole grid, every sign pattern and every scale.
Blocks whose reference decode is not finite are re-rolled, because an inf
scale would compare equal for the wrong reason.

Usage (any interpreter with `gguf` installed):
    python3 tools/gen_reference.py
    /path/to/venv/bin/python tools/gen_reference.py
"""

from __future__ import annotations

import os
import sys

try:
    import numpy as np
    from gguf.constants import GGMLQuantizationType as T, GGML_QUANT_SIZES
    from gguf import quants
except ImportError:
    sys.exit("needs the `gguf` package: pip install gguf  (or use a venv that has it)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures")

BLOCKS = 64          # per type: enough to hit a wide spread of the codebook


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    manifest: list[str] = []
    skipped: list[str] = []

    # Every type's geometry goes in the manifest, decoder or not: the byte
    # accounting is a claim ingot makes about a format even when it cannot
    # decode it, and getting it wrong silently mis-sizes a whole tensor. This
    # caught Q8_1 sized at 36 bytes when ggml had long since widened it to 40.
    geometry = [f"{t.name} {t.value} {GGML_QUANT_SIZES[t][0]} {GGML_QUANT_SIZES[t][1]}"
                for t in T]
    with open(os.path.join(OUT, "geometry.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(geometry) + "\n")

    for qtype in T:
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        if block_size == 1:
            continue                      # F32/F16/I32/… are not block formats
        try:
            quants.dequantize(np.zeros(type_size, dtype=np.uint8), qtype)
        except Exception:
            skipped.append(qtype.name)    # no reference decoder in this package
            continue

        rng = np.random.default_rng(0xC0FFEE + qtype.value)
        blocks = np.empty((BLOCKS, type_size), dtype=np.uint8)
        for i in range(BLOCKS):
            for _ in range(64):           # re-roll until the decode is finite
                candidate = rng.integers(0, 256, size=type_size, dtype=np.uint8)
                decoded = quants.dequantize(candidate.reshape(1, -1), qtype)
                if np.all(np.isfinite(decoded)):
                    blocks[i] = candidate
                    break
            else:
                sys.exit(f"{qtype.name}: could not draw a finite block")

        reference = quants.dequantize(blocks, qtype).astype(np.float32).reshape(-1)
        assert reference.size == BLOCKS * block_size

        with open(os.path.join(OUT, f"{qtype.name}.bin"), "wb") as handle:
            handle.write(blocks.tobytes())
        with open(os.path.join(OUT, f"{qtype.name}.ref"), "wb") as handle:
            handle.write(reference.tobytes())
        manifest.append(f"{qtype.name} {qtype.value} {block_size} {type_size} {BLOCKS}")

    with open(os.path.join(OUT, "manifest.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(manifest) + "\n")

    print(f"wrote {len(manifest)} fixtures and {len(geometry)} geometries "
          f"to tests/fixtures/")
    if skipped:
        print("no reference decoder in this gguf package for: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
