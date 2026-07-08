# tools/para — paralinguistic discovery automation (plan_v4 E1)

Offline, **CPU-first** tooling to screen `[tag]` paralinguistic candidates without ear-sweeping.
Runs entirely on the M1 — **no DGX / CUDA needed** (E1 is inference + screening, not training).

## Heavy artifacts — where they live & cleanup (DO NOT COMMIT)

| artifact | location | size | gitignored |
|---|---|---|---|
| Python venv (torch etc.) | `tools/para/.venv/` | ~3 GB | yes (`.venv/` rule + explicit) |
| CNN14 AudioSet checkpoint | `~/panns_data/` | ~330 MB | outside repo |
| Whisper tiny/base weights | `~/.cache/whisper/` | ~75–140 MB | outside repo |

**Cleanup once the E1 loop is done** (tracked as a TODO in `plan_v4.md` E1):
```bash
rm -rf tools/para/.venv ~/panns_data ~/.cache/whisper
```

## Setup (one-time, ~3 GB download)
```bash
cd tools/para
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt
```

## Use — `para_judge.py` (E1.1)
```bash
# judge a folder, check each clip for one onomatopoeia
./.venv/bin/python para_judge.py --wavs /path/to/sweep --onom 哈哈哈 --tag laugh --out report.md

# manifest with per-clip tag/onom/expected (produced by para_sweep.sh, E1.3)
./.venv/bin/python para_judge.py --manifest sweep.json --out report.md

# calibration gate (E1.2): known WIN/KO clips -> precision/recall vs the ear
./.venv/bin/python para_judge.py --manifest calib.json --calibrate
```

Verdicts: `WIN_CAND` (event fired, not read literally) · `KO_LITERAL` (onomatopoeia spoken)
· `DRIFT` (a different event dominates — serendipity) · `MISS` (neutral speech).

Default device = `cpu` (keeps the M1 cool). `--device mps` to use the Apple GPU.

**Screener, not judge**: it must pass the E1.2 calibration gate against the ear-validated
WIN/KO clips (`samples/tests/*para*`) before we trust its shortlists. The ear stays the
final judge for promotions into `para_pick`.
