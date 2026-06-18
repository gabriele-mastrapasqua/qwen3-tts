#!/usr/bin/env python3
"""Extract the pure x-vector (speaker embedding) from a .qvoice into a tiny legacy .bin.

WHY: a `.qvoice` ICL clone carries `ref_codes` = the codec of the REFERENCE recording, which
re-injects that recording's room acoustics ("muffled metallic / faint reverb") into every
generation — and the CSP-FT `.expr` makes it more audible (it re-attends the ref_codes harder).
The speaker x-vector alone carries abstract identity WITHOUT the room. Loading an embedding-only
`.bin` with `--xvector-only` gives a clean clone (identity preserved) and MORE force headroom for
the emotion `.expr` (sweet spot ~w1.6-2.0 @ T1.3). See plan_emo_v2.md / docs/csp-ft-emotion.md.

The .qvoice (QVCE) layout: magic[4]="QVCE", version[u32], enc_dim[u32] (v>=2), then enc_dim
float32 = the speaker embedding. This tool slices exactly those floats out → an 8KB .bin that
`--load-voice X.bin --xvector-only` reads as a complete, clean clone voice.

Usage:
  python3 tests/qvoice_to_xvec.py voices/galatea_icl.qvoice            # -> voices/galatea_icl.bin
  python3 tests/qvoice_to_xvec.py in.qvoice -o out.bin
  python3 tests/qvoice_to_xvec.py --self-test

Note: to make a clean x-vector clone FROM SCRATCH (no .qvoice yet), the engine already supports it:
  ./qwen_tts -d <base-model> --ref-audio ref24k.wav --xvector-only --save-voice voice.bin
"""
import argparse
import os
import struct
import sys


def extract(path):
    """Return (embedding_bytes, enc_dim, version) for a QVCE .qvoice file."""
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"QVCE":
            raise ValueError(f"{path} is not a .qvoice (QVCE) file")
        version = struct.unpack_from("<I", head, 4)[0]
        if version < 2:
            raise ValueError(f"{path} is QVCE v{version}; v>=2 (enc_dim header) required")
        enc_dim = struct.unpack_from("<I", head, 8)[0]
        emb = f.read(enc_dim * 4)
        if len(emb) != enc_dim * 4:
            raise ValueError(f"{path}: truncated embedding ({len(emb)} of {enc_dim*4} bytes)")
    return emb, enc_dim, version


def main():
    ap = argparse.ArgumentParser(description="Extract x-vector .bin from a .qvoice")
    ap.add_argument("qvoice", nargs="?", help="input .qvoice file")
    ap.add_argument("-o", "--out", help="output .bin (default: same name, .bin)")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not args.qvoice:
        ap.error("qvoice input required (or --self-test)")
    out = args.out or os.path.splitext(args.qvoice)[0] + ".bin"
    emb, enc_dim, version = extract(args.qvoice)
    with open(out, "wb") as f:
        f.write(emb)
    norm = sum(struct.unpack(f"<{enc_dim}f", emb)[i] ** 2 for i in range(enc_dim)) ** 0.5
    print(f"{args.qvoice} (QVCE v{version}, enc_dim={enc_dim})")
    print(f"  -> {out}  ({len(emb)} bytes, embedding L2 norm={norm:.4f})")
    print(f"  use: ./qwen_tts -d <model> --load-voice {out} --xvector-only -l <Lang> ...")


def _self_test():
    """Round-trip a synthetic QVCE header through extract() with no model/files needed."""
    import tempfile

    enc_dim = 8
    vals = [0.1 * i for i in range(enc_dim)]
    blob = b"QVCE" + struct.pack("<I", 3) + struct.pack("<I", enc_dim) + struct.pack(f"<{enc_dim}f", *vals)
    blob += b"\x00\x00\x00\x00" + b"junk after embedding"  # ref_text_len=0 + trailing sections
    with tempfile.NamedTemporaryFile(suffix=".qvoice", delete=False) as tf:
        tf.write(blob)
        p = tf.name
    emb, ed, ver = extract(p)
    os.unlink(p)
    got = list(struct.unpack(f"<{ed}f", emb))
    assert ed == enc_dim and ver == 3, (ed, ver)
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, vals)), got
    assert len(emb) == enc_dim * 4
    print("SELF-TEST PASS — extracts exactly enc_dim floats, ignores trailing ref_text/ref_codes/WDELTA.")


if __name__ == "__main__":
    sys.exit(main())
