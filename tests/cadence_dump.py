#!/usr/bin/env python3
"""cadence_dump.py — reconstruct the playback timeline of individual turns, by request id."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playback_sim import analyse, audio_s

def pct(v, q):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(round(q / 100.0 * (len(v) - 1))))]

def timeline(t):
    """Independent reconstruction. Returns rows and the prebuffer derived from them."""
    marks = t["marks"]
    t_first = marks[0][0]
    rows, avail, played, prev, need = [], 0.0, 0.0, t_first, 0.0
    stall_from = None
    for i, (ta, nb) in enumerate(marks):
        dur = audio_s(nb)
        inter = (ta - marks[i - 1][0]) * 1000 if i else 0.0
        if i:
            gap = ta - prev
            want = played + gap
            deficit = want - avail
            need = max(need, (ta - t_first) - avail)
            if deficit > 0:
                if stall_from is None:
                    stall_from = prev
                played = avail
            else:
                if stall_from is not None:
                    rows[-1]["stall_end_ms"] = prev * 1000
                    stall_from = None
                played = want
            prev = ta
        else:
            deficit = 0.0
        avail += dur
        rows.append({"i": i, "arrival_ms": ta * 1000, "chunk_audio_ms": dur * 1000,
                     "interarrival_ms": inter, "cum_audio_ms": avail * 1000,
                     "playhead_ms": played * 1000,
                     "buffer_lead_ms": (avail - played) * 1000,
                     "deficit_ms": deficit * 1000,
                     "stall_start_ms": stall_from * 1000 if stall_from is not None else None,
                     "stall_end_ms": None})
    return rows, max(0.0, need) * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--per-cell", type=int, default=2)
    a = ap.parse_args()
    for d in a.dirs:
        turns = [json.loads(l) for l in open(os.path.join(d, "turns.jsonl"))]
        comp = [t for t in turns if not t.get("aborted") and not t.get("error")
                and t.get("marks") and len(t["marks"]) > 2 and t.get("min_prebuffer_ms") is not None]
        if not comp:
            continue
        comp.sort(key=lambda t: t["min_prebuffer_ms"])
        C = os.path.basename(d).replace("cs_c", "")
        print(f"\n{'='*92}\n=== {d}   C={C}   {len(comp)} completed turns with a timeline")

        pre = [t["min_prebuffer_ms"] for t in comp]
        print(f"\n  min_prebuffer over completed turns: p50 {pct(pre,50):.0f}  p95 {pct(pre,95):.0f}  max {max(pre):.0f} ms")

        durs = [audio_s(nb) * 1000 for t in comp for _ta, nb in t["marks"]]
        gaps = [(t["marks"][i][0] - t["marks"][i-1][0]) * 1000
                for t in comp for i in range(1, len(t["marks"]))]
        print(f"  chunk audio duration ms: p50 {pct(durs,50):.0f}  p95 {pct(durs,95):.0f}  "
              f"min {min(durs):.0f}  max {max(durs):.0f}  (n={len(durs)})")
        print(f"  inter-chunk gap ms:      p50 {pct(gaps,50):.0f}  p95 {pct(gaps,95):.0f}  "
              f"max {max(gaps):.0f}")
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        print(f"  gap dispersion:          mean {mean:.0f} ms  CV {var**0.5/mean:.2f}")

        early = mid = late = 0
        for t in comp:
            rows, _ = timeline(t)
            worst = max(rows, key=lambda r: r["deficit_ms"])
            f = worst["i"] / max(1, len(rows) - 1)
            if worst["deficit_ms"] <= 0:
                continue
            if f <= 0.33: early += 1
            elif f <= 0.66: mid += 1
            else: late += 1
        tot = early + mid + late
        if tot:
            print(f"  worst deficit position:  early {100*early/tot:.0f}%  "
                  f"middle {100*mid/tot:.0f}%  late {100*late/tot:.0f}%  (n={tot})")

        idx = [len(comp) // 2, min(len(comp) - 1, int(0.95 * (len(comp) - 1)))][:a.per_cell]
        for j in idx:
            t = comp[j]
            rows, recomputed = timeline(t)
            agree = abs(recomputed - t["min_prebuffer_ms"]) < 1.0
            print(f"\n  --- turn {t['text_id']} seed={t['seed']} conv={t['conversation_id']}"
                  f" turn={t['turn_index']} class={t['workload_class']}"
                  f" audio={t['delivered_audio_s']:.2f}s")
            print(f"      reported min_prebuffer {t['min_prebuffer_ms']:.1f} ms · "
                  f"recomputed from this timeline {recomputed:.1f} ms · "
                  f"{'✅ AGREE' if agree else '❌ DISAGREE'}")
            print(f"      {'i':>3}{'arrival':>10}{'chunk_ms':>10}{'inter':>8}"
                  f"{'cum_audio':>11}{'playhead':>10}{'buf_lead':>10}{'deficit':>9}")
            for r in rows[:14]:
                print(f"      {r['i']:>3}{r['arrival_ms']:>10.0f}{r['chunk_audio_ms']:>10.0f}"
                      f"{r['interarrival_ms']:>8.0f}{r['cum_audio_ms']:>11.0f}"
                      f"{r['playhead_ms']:>10.0f}{r['buffer_lead_ms']:>10.0f}"
                      f"{r['deficit_ms']:>9.0f}")
            if len(rows) > 14:
                print(f"      ... {len(rows)-14} more chunks")
    return 0

if __name__ == "__main__":
    sys.exit(main())
