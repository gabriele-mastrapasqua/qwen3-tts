#!/usr/bin/env python3
"""soak_client.py — ONE closed-loop conversation of the soak, in one process."""
import argparse, json, os, sys, time, urllib.request

def load_texts(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = [x.strip() for x in ln.split("\t")]
        if len(parts) >= 8:   rows.append((parts[1], parts[-1]))
        elif len(parts) > 1:  rows.append((parts[0], parts[-1]))
        else:                 rows.append(("medium", ln.strip()))
    return rows

def one(port, text, speaker, language, seed, temperature, out_path):
    body = json.dumps({"text": text, "speaker": speaker, "language": language,
                       "seed": seed, "temperature": temperature}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/tts/stream", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    n = first = 0
    ttfa = None
    fh = open(out_path, "wb") if out_path else None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            while True:
                ch = r.read1(1 << 16)
                if not ch:
                    break
                if ttfa is None:
                    ttfa = time.time() - t0
                    first = len(ch)
                n += len(ch)
                if fh:
                    fh.write(ch)
    except Exception as e:
        return None, str(e)
    finally:
        if fh:
            fh.close()
    total = time.time() - t0
    audio = n / 2.0 / 24000.0
    rest = (n - first) / 2.0 / 24000.0
    stream = (total - ttfa) / rest if (ttfa is not None and rest > 0) else float("nan")
    return dict(ttfa_ms=(ttfa or 0) * 1000.0, total_ms=total * 1000.0, bytes=n,
                first=first, audio_s=audio, stream_rtf=stream), None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--language", default="English")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--probe-every-min", type=int, default=5)
    a = ap.parse_args()

    texts = load_texts(a.bank)
    if not texts:
        sys.exit("empty bank")
    w, i, ticked = a.worker, 0, set()
    with open(a.csv, "a", buffering=1) as csv:
        while time.time() < a.deadline:
            el = time.time() - a.t0
            minute = int(el // 60)
            probe = (a.audio_dir and minute % a.probe_every_min == 0
                     and minute not in ticked)
            if probe:
                ticked.add(minute)
                cls, text = texts[w % len(texts)]
                seed = 900 + w
                out = os.path.join(a.audio_dir, f"min{minute:03d}_w{w}.pcm")
            else:
                cls, text = texts[(w * 7 + i) % len(texts)]
                seed = 1000 + w * 100 + i
                out = ""
            rec, err = one(a.port, text, a.speaker, a.language, seed, a.temperature, out)
            t_end = time.time() - a.t0
            if err:
                csv.write(f"{t_end:.3f},{w},{i},,,,,,,{1 if probe else 0},{cls},ERROR:{err}\n")
            else:
                csv.write(f"{t_end:.3f},{w},{i},{rec['ttfa_ms']:.1f},{rec['total_ms']:.1f},"
                          f"{rec['bytes']},{rec['first']},{rec['audio_s']:.3f},"
                          f"{rec['stream_rtf']:.4f},{1 if probe else 0},{cls},\n")
            i += 1

if __name__ == "__main__":
    main()
