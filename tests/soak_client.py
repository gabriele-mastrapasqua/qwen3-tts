#!/usr/bin/env python3
"""soak_client.py — ONE closed-loop conversation of the soak, in one process.

WHY IT REPLACES bash+curl
  curl gives time_starttransfer and time_total. That is enough for TTFA, and NOT enough
  for STREAM_RTF, which needs the arrival time of the FIRST audio chunk and the amount of
  audio delivered after it. It also forked two pythons and a curl per request.

  This process holds the connection loop itself, reads chunk by chunk with read1() - the
  same call the wave harness uses, after read(65536) was found to be measuring "time to
  buffer 1.4 s of audio" rather than time to first audio - and writes ONE CSV row per
  request with everything already per-request.

⚠️ STREAM_RTF IS PER REQUEST, AND IS NEVER RECOMPUTED FROM SUMS.
  sum(wall)/sum(audio) over a window is a ratio of sums, not the statistic we froze, and
  this project has already published a table where a ratio of percentiles made a part look
  bigger than the whole. Each row here carries its own stream_rtf; the report takes
  percentiles OF THOSE VALUES.

Row: t_end_s,worker,i,ttfa_ms,total_ms,bytes,first_chunk_bytes,audio_s,stream_rtf,is_probe
  t_end_s   seconds since the run's T0, at COMPLETION - the window a request belongs to is
            the window in which it COMPLETED (documented, so nobody re-derives it).
"""
import argparse, json, os, sys, time, urllib.request


def load_texts(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        # Manifest v2 is id/class/audio_s/words/text_sha/audio_sha/retries/text, so the
        # class is column 1; the legacy banks put it in column 0. Reading column 0 blindly
        # labelled every soak request with a candidate ID - the same defect that made B1
        # run 1 benchmark a fallback sentence, in the other client.
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
                # ⚠️ THE PROBE IS A FIXED TEXT AND A FIXED SEED. Comparing minute 0 with
                # minute 25 proves nothing if they are different sentences - the ear needs
                # the same utterance. The LOAD keeps rotating; only the probe repeats.
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
