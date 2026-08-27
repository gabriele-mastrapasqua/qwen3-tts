#!/usr/bin/env python3
"""batch_determinism.py — the cheap permanent guard behind a claim we nearly overstated.

On 2026-08-24 a quality gate found that C=4 and C=6 produced BIT-IDENTICAL audio for the
same request, which reframed RTF>1 at C=6 from "the audio degrades under load" to "correct
audio, delivered too slowly". That is a strong and useful conclusion — and it rested on
eight md5 pairs from one configuration. Eight md5s are not a theorem.

This is the test that makes it an assertion instead of an impression, and it costs seconds
rather than the half hour a capacity campaign costs:

    same text + same seed + same speaker  ->  fired at C = 2, 4, 6, 8
    every response at every C >= 2 must hash identically

WHY C>=2 AND NOT C>=1. At C=1 the engine runs B=1 and takes the GEMV path; from C=2 it takes
GEMM. The two sum in a different order, so C=1 differing is EXPECTED and is reported, not
failed. Within GEMM the result must not depend on how many columns share the batch - column
j does not depend on column k - and that is exactly what is asserted here.

    python3 tests/batch_determinism.py --url http://127.0.0.1:8000 [--conc 2,4,6,8]
"""
import argparse, concurrent.futures as cf, hashlib, json, sys, urllib.request

def one(url, path, body, timeout):
    req = urllib.request.Request(url + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = []
        while True:
            ch = r.read1(65536)
            if not ch: break
            buf.append(ch)
    return hashlib.sha256(b"".join(buf)).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--path", default="/v1/tts/stream")
    ap.add_argument("--conc", default="2,4,6,8")
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="english")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text", default="Abeg tell me wetin bi the problem, I go check am now now.")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()

    body = json.dumps({"text": a.text, "speaker": a.speaker, "language": a.language,
                       "temperature": 0.0, "seed": a.seed}).encode()
    concs = [int(x) for x in a.conc.split(",") if x.strip()]

    print(f"text={a.text!r}\nspeaker={a.speaker} language={a.language} seed={a.seed} "
          f"temperature=0.0")
    print(f"{'C':>3}  {'n':>3}  {'distinct hashes':>15}  sha256[:16]")
    seen, failures = {}, []
    for C in concs:
        with cf.ThreadPoolExecutor(max_workers=C) as ex:
            hs = list(ex.map(lambda _: one(a.url, a.path, body, a.timeout), range(C)))
        uniq = sorted(set(hs))
        print(f"{C:>3}  {len(hs):>3}  {len(uniq):>15}  {', '.join(h[:16] for h in uniq)}")
        if len(uniq) != 1:
            failures.append(f"C={C}: {len(uniq)} distinct hashes WITHIN the same wave")
        seen[C] = uniq[0] if uniq else None

    ge2 = {C: h for C, h in seen.items() if C >= 2 and h}
    if len(set(ge2.values())) > 1:
        failures.append("hashes differ ACROSS C>=2: " +
                        ", ".join(f"C={C}->{h[:16]}" for C, h in ge2.items()))
    if 1 in seen and ge2 and seen[1] not in ge2.values():
        print("\nnote: C=1 differs from C>=2, which is EXPECTED (GEMV vs GEMM sum order).")

    print()
    if failures:
        print("*** FAIL ***"); [print("  " + f) for f in failures]; sys.exit(1)
    print(f"PASS — identical output at every C in {concs} (C>=2 share one hash)")

if __name__ == "__main__":
    main()
