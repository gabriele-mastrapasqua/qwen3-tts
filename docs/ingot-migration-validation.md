# Validating the move to the vendored loader

Record of the checks run when safetensors loading moved to the vendored `ingot`
library. Kept because it states what "equivalent" was taken to mean, not because
the numbers are current.

## What passed

- Full test suite on real weights: small and large configs, English and Italian,
  instruct, regression, error paths, emotion and emotion fine-tune, compose,
  caps, self-test.
- **Golden mel correlation: 1.00000 on the 0.6B (English, Italian, int8) and
  0.99995 on the 1.7B.**
- Server reproducibility within ±2 LSB.
- Quantized paths: int8 and int4, English and Italian.
- Old loader against new, on the same weights: byte-identical tensor payloads
  (head and tail), and bit-identical f32 conversion.
- Leak check, old binary against new: identical counts and bytes. The leaks are
  pre-existing — load-time buffers that live until exit — and the move adds none.
- Voice cloning on CustomVoice weights through the vendored path: an existing
  `.qvoice` graft loads and generates clean audio on both model sizes. A base
  model is needed only to extract a *new* embedding from raw audio
  (`--ref-audio` with `--save-voice`), not to use an embedding already extracted.

## What this does not cover

Extracting an x-vector, and voice design, both need the base checkpoints. Where
those are absent the end-to-end run fails with "Failed to load model", which is a
missing directory rather than a defect: the previous loader fails the same way.
