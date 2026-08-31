#!/usr/bin/env python3
"""Swap the x-vector inside a graft .qvoice, keeping TPAD and WOVR untouched.

Usage:
  python3 tests/graft_set_xvector.py <in.qvoice> <xvector.bin> <out.qvoice> [--name NAME]
"""
import sys, struct

def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 3:
        print(__doc__); return 1
    src, xvec_path, dst = a
    name = None
    if "--name" in sys.argv:
        name = sys.argv[sys.argv.index("--name") + 1]

    blob = bytearray(open(src, "rb").read())
    magic, version, enc_dim = struct.unpack("<4sII", blob[:12])
    if magic != b"QVCE":
        print(f"error: {src} is not a .qvoice (magic={magic!r})"); return 1

    xvec = open(xvec_path, "rb").read()
    want = enc_dim * 4
    if len(xvec) != want:
        print(f"error: {xvec_path} is {len(xvec)} B, this .qvoice wants {want} B "
              f"({enc_dim} floats — 0.6B=1024, 1.7B=2048)"); return 1

    blob[12:12 + want] = xvec
    open(dst, "wb").write(blob)
    print(f"wrote {dst}: x-vector from {xvec_path} ({enc_dim} floats), "
          f"TPAD+WOVR carried over from {src} ({len(blob)} B total)")
    if name:
        print("  note: --name is informational only; the META name field is left as-is")
    return 0

if __name__ == "__main__":
    sys.exit(main())
