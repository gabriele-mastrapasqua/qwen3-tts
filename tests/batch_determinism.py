#!/usr/bin/env python3
"""batch_determinism.py — the cheap permanent guard behind a claim we nearly overstated."""
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
    ap.add_argument("--text", default="Please tell me what the problem is, I will check it now.")
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
