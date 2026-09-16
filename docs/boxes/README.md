# docs/boxes — archived dispatcher-threshold grids

`make tune-archive BOX=<name>` runs `tests/kernel_tune.sh` (`./qwen_tts --matmat-tune`) and
copies the resulting `tune.json` here as `<name>_tune.json`. `python3 tests/mm_tune_compare.py`
reads every `*_tune.json` in this directory and prints, per (format, shape, threads), which
kernel wins on which box and the measured crossover `B` — the cross-ISA comparison the
`--matmat-tune` thresholds are guessed from otherwise.

Name a box after what identifies its silicon, not the provider's SKU alone:
`aws-c8a-16c-zen5`, `gcp-c4a-16c-axion`, `m1-8c`. No hostnames, no IPs.
