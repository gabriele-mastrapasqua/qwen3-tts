## Validazione migrazione ingot — 2026-08-02, Mac M1 (branch feat/migration-ingotlib)

Tutto su pesi veri. Verde:
- test-all (exit 0, 57 PASS): small 5/5, large config+EN+IT+instruct,
  regression, errors, emotion + emotion-ft, compose, caps, selftest,
  **test-golden mel-corr: 1.00000 (0.6b en/it/int8), 0.99995 (1.7b)**,
  serve-repro (riproducibilità ±2 LSB).
- test-large-quant: INT8 EN/IT + INT4 EN/IT PASS; test-modes 5/5; test-qvoice.
- Gate A/B ri-eseguito post-migrazione: 0.6b 402/402, 1.7b 404/404, payload
  head+tail 4 KiB identici, f32-conversion bit-identica.
- **leaks A/B**: 137 leak / 420096 byte IDENTICI tra binario vecchio (main,
  worktree) e nuovo — preesistenti (buffer di qwen_talker_load vivi fino a
  exit), la migrazione non ne aggiunge nessuno.

Non eseguibile qui (symlink al NAS non montato — /Volumes/shared):
- test-clone e test-voice-design (servono qwen3-tts-0.6b-base / 1.7b-base):
  il FAIL di e2e su clone_output.wav e' "Failed to load model" per dir
  assente, NON un bug: il vecchio reader fallirebbe uguale.
→ da rifare a casa col NAS montato, poi cancellare qwen_tts_safetensors.c/h
  e merge.
