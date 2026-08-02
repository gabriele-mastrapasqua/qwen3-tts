#!/usr/bin/env python3
"""Generate the two-file drop-in build: amalgam/ingot.h + amalgam/ingot.c.

Why: the smallest possible barrier to adoption. A project that wants to read a
GGUF should be able to copy two files in and be done — no submodule, no
Makefile change, no include path. That is the bar stb set and it is the reason
those libraries are everywhere.

How: concatenate the public headers into one, and the sources into one, with
the internal header inlined once and every intra-library #include dropped. The
feature-test macros (_POSIX_C_SOURCE and friends) are hoisted to the very top,
because they only work before the first system header.

The output is generated, never edited: src/ stays the source of truth, and
`make amalgam-test` runs the whole suite against the generated pair so the two
builds cannot drift.

Usage: python3 tools/amalgamate.py
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dependency order matters: a header may use a type declared by an earlier one.
HEADERS = [
    "include/ingot/dtype.h",
    "include/ingot/gguf.h",
    "include/ingot/safetensors.h",
    "include/ingot/quant.h",
    "include/ingot/wfile.h",
    "include/ingot/write.h",
]

# The container half: always compiled.
SOURCES = [
    "src/internal.h",
    "src/dtype.c",
    "src/gguf.c",
    "src/safetensors.c",
    "src/wfile.c",
    "src/write.c",
]

# The quantization half: wrapped in #ifndef INGOT_NO_KERNELS in the generated
# file, so the single-file build keeps the same opt-out the multi-file one has
# (there, you simply do not link these objects).
QUANT_SOURCES = [
    "src/cpu.c",
    "src/dequant.c",
    "src/dequant_iq.c",
    "src/quantize.c",
    "src/kernels.c",
    "src/generic.c",
]

INTERNAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"(ingot/[^"]+|internal\.h)".*$')
INLINE_INCLUDE = re.compile(r'^\s*#\s*include\s+"([\w.]+\.inc)".*$')
GUARD_OPEN = re.compile(r"^\s*#\s*ifndef\s+INGOT_\w+_H\s*$")
GUARD_DEFINE = re.compile(r"^\s*#\s*define\s+INGOT_\w+_H\s*$")
FEATURE_MACRO = re.compile(r"^\s*#\s*define\s+(_POSIX_C_SOURCE|_DARWIN_C_SOURCE|_GNU_SOURCE)\b")
EXTERN_C_OPEN = re.compile(r'^\s*extern\s+"C"\s*\{\s*$')


def read(path: str) -> list[str]:
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return handle.read().splitlines()


def strip_header(path: str) -> list[str]:
    """Drop the include guard, the intra-library includes and the extern "C"
    wrapper: the amalgamated header supplies one of each for the whole file."""
    out: list[str] = []
    seen_guard = False
    depth_after_guard = 0
    for line in read(path):
        if not seen_guard and GUARD_OPEN.match(line):
            seen_guard = True
            continue
        if seen_guard and depth_after_guard == 0 and GUARD_DEFINE.match(line):
            depth_after_guard = 1
            continue
        if INTERNAL_INCLUDE.match(line):
            continue
        if EXTERN_C_OPEN.match(line) or line.strip() in ('#ifdef __cplusplus', '#endif'):
            # These bracket the extern "C" block and the guard; the wrapper
            # below reinstates them once. Keeping per-file copies would nest
            # extern "C" harmlessly but leave stray #endif once the guard is
            # gone, so they go.
            continue
        if line.strip() == "}" and out and out[-1].strip() == "":
            continue
        out.append(line)
    # Trailing "#endif" of the guard was already dropped by the __cplusplus
    # filter above; anything left that is a bare "}" from extern "C" goes too.
    while out and out[-1].strip() in ("", "}"):
        out.pop()
    return out


def strip_source(path: str) -> tuple[list[str], list[str]]:
    """Returns (feature macros, body)."""
    macros: list[str] = []
    body: list[str] = []
    seen_guard = False
    for line in read(path):
        if FEATURE_MACRO.match(line):
            macros.append(line.strip())
            continue
        if INTERNAL_INCLUDE.match(line):
            continue
        inc = INLINE_INCLUDE.match(line)
        if inc:
            # Generated tables live in a .inc next to the source; the two-file
            # build has nowhere to put a third file, so it goes inline.
            body.append(f"/* ─── inlined {inc.group(1)} ─── */")
            body.extend(read(os.path.join("src", inc.group(1))))
            continue
        if path.endswith(".h"):
            if not seen_guard and GUARD_OPEN.match(line):
                seen_guard = True
                continue
            if seen_guard and GUARD_DEFINE.match(line):
                continue
            if line.strip() == "#endif":
                continue
        body.append(line)
    return macros, body


def main() -> int:
    out_dir = os.path.join(ROOT, "amalgam")
    os.makedirs(out_dir, exist_ok=True)

    banner = (
        "/* ingot — GGUF + safetensors reader/writer in C11.\n"
        " * https://github.com/mynah-org/ingot   SPDX-License-Identifier: MIT\n"
        " *\n"
        " * GENERATED FILE — do not edit. Regenerate with tools/amalgamate.py.\n"
        " * The source of truth is src/ and include/ingot/.\n"
        " *\n"
        " * Drop ingot.h and ingot.c into your project and build the .c like any\n"
        " * other source file. Needs -lpthread -lm. Define INGOT_NO_KERNELS to\n"
        " * leave out dequantization and the SIMD kernels.\n"
        " */\n"
    )

    header = [banner, "#ifndef INGOT_AMALGAM_H", "#define INGOT_AMALGAM_H", "",
              "#ifdef __cplusplus", 'extern "C" {', "#endif", ""]
    for path in HEADERS:
        header.append(f"/* ═══ {path} ═══ */")
        header.extend(strip_header(path))
        header.append("")
    header += ["#ifdef __cplusplus", "}", "#endif", "#endif /* INGOT_AMALGAM_H */", ""]

    macros: list[str] = []
    body: list[str] = []
    for path in SOURCES + QUANT_SOURCES:
        file_macros, file_body = strip_source(path)
        for macro in file_macros:
            if macro not in macros:
                macros.append(macro)
        if path == QUANT_SOURCES[0]:
            body += ["", "#ifndef INGOT_NO_KERNELS",
                     "/* ─── the optional quantization half ─────────────────────────────── */",
                     ""]
        body.append(f"/* ═══ {path} ═══ */")
        body.extend(file_body)
        body.append("")
    body += ["#endif /* INGOT_NO_KERNELS */", ""]

    source = [banner]
    if macros:
        source.append("/* Feature-test macros, hoisted: they only take effect before the")
        source.append(" * first system header, which the concatenation below would break. */")
        # _DARWIN_C_SOURCE is Apple-only; keep its guard.
        for macro in macros:
            if "_DARWIN_C_SOURCE" in macro:
                source += ["#if defined(__APPLE__)", macro, "#endif"]
            else:
                source.append(macro)
        source.append("")
    source.append('#include "ingot.h"')
    source.append("")
    source.extend(body)

    with open(os.path.join(out_dir, "ingot.h"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(header))
    with open(os.path.join(out_dir, "ingot.c"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(source))

    h_lines = len(header)
    c_lines = len(source)
    print(f"amalgam/ingot.h  {h_lines:5d} lines")
    print(f"amalgam/ingot.c  {c_lines:5d} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
