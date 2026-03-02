#!/usr/bin/env python3
"""
dump_greedy_codes.py

Loads the Qwen3-TTS 0.6B custom-voice model, generates speech for
"Hello world" (speaker=Serena, language=English) with fully greedy
decoding (temperature=0 for both Talker and Code Predictor), and
dumps the raw integer codec codes (16 codebooks per frame) along
with Talker hidden-state norms for the first few frames.

The monkey-patch hooks into the Talker's forward() to capture:
  - code 0  : sampled by the Talker via codec_head
  - codes 1-15 : predicted autoregressively by the Code Predictor

No audio file is written; the purpose is diagnostic code inspection.
"""

import sys
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"
TEXT = "Hello world"
SPEAKER = "Serena"
LANGUAGE = "English"

# Fully greedy: temperature=0, do_sample still True (temp=0 triggers argmax
# inside HF generate regardless of do_sample, but we also set top_k=1 as a
# safety belt).
GENERATION_OVERRIDES = dict(
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    do_sample=False,
    subtalker_temperature=0.0,
    subtalker_top_k=1,
    subtalker_top_p=1.0,
    subtalker_dosample=False,
    repetition_penalty=1.0,   # disable rep-penalty for pure greedy
    max_new_tokens=2048,
)

# ---------------------------------------------------------------------------
# Storage for intercepted data
# ---------------------------------------------------------------------------
frame_codec_codes = []        # list of 1-D int tensors, each length num_code_groups
frame_talker_hidden_norms = []  # list of float (L2 norm of last hidden state)


# ---------------------------------------------------------------------------
# Monkey-patch: wrap Talker.forward to intercept per-frame codec codes
# and hidden states.
# ---------------------------------------------------------------------------
def install_talker_hook(talker_module):
    """
    Uses register_forward_hook on the Talker to intercept per-frame codec codes
    and hidden-state norms after each forward call.
    """
    def hook_fn(module, input, output):
        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            transformer_hiddens, codec_ids = output.hidden_states
            if codec_ids is not None:
                frame_codec_codes.append(codec_ids[0].detach().cpu())

            if output.past_hidden is not None:
                h = output.past_hidden[0, 0]
                frame_talker_hidden_norms.append(h.detach().cpu().norm().item())

    talker_module.register_forward_hook(hook_fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("dump_greedy_codes.py  --  Qwen3-TTS greedy codec dumper")
    print("=" * 70)

    # 1. Load model
    print(f"\nLoading model from {MODEL_DIR} ...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    print("Model loaded successfully.")

    inner_model = model.model  # Qwen3TTSForConditionalGeneration
    talker = inner_model.talker  # Qwen3TTSTalkerForConditionalGeneration

    # Print some useful config values
    talker_cfg = talker.config
    print(f"\n--- Talker config ---")
    print(f"  vocab_size          : {talker_cfg.vocab_size}")
    print(f"  hidden_size         : {talker_cfg.hidden_size}")
    print(f"  num_code_groups     : {talker_cfg.num_code_groups}")
    print(f"  codec_eos_token_id  : {talker_cfg.codec_eos_token_id}")
    print(f"  codec_pad_id        : {talker_cfg.codec_pad_id}")
    print(f"  codec_bos_id        : {talker_cfg.codec_bos_id}")

    cp_cfg = talker_cfg.code_predictor_config
    print(f"\n--- Code Predictor config ---")
    print(f"  vocab_size          : {cp_cfg.vocab_size}")
    print(f"  hidden_size         : {cp_cfg.hidden_size}")
    print(f"  num_code_groups     : {cp_cfg.num_code_groups}")
    print(f"  num_hidden_layers   : {cp_cfg.num_hidden_layers}")

    # 2. Install monkey-patch hook on the Talker
    print("\nInstalling Talker forward hook ...")
    install_talker_hook(talker)
    print("Hook installed.")

    # 3. Generate with greedy params
    print(f"\nGenerating speech for: \"{TEXT}\"")
    print(f"  speaker  = {SPEAKER}")
    print(f"  language = {LANGUAGE}")
    print(f"  greedy overrides = {GENERATION_OVERRIDES}")
    print()

    wavs, sample_rate = model.generate_custom_voice(
        text=TEXT,
        speaker=SPEAKER,
        language=LANGUAGE,
        **GENERATION_OVERRIDES,
    )

    print(f"\nGeneration complete.  sample_rate={sample_rate}")
    print(f"Number of frames captured: {len(frame_codec_codes)}")
    if wavs:
        print(f"Output waveform length  : {len(wavs[0])} samples "
              f"({len(wavs[0]) / sample_rate:.3f}s)")

    # 4. Dump per-frame codec codes
    num_groups = talker_cfg.num_code_groups
    eos_id = talker_cfg.codec_eos_token_id

    print("\n" + "=" * 70)
    print(f"PER-FRAME CODEC CODES  ({num_groups} codebooks per frame)")
    print(f"  Normal code range : 0 .. {cp_cfg.vocab_size - 1}")
    print(f"  EOS token id      : {eos_id}")
    print("=" * 70)

    for i, codes in enumerate(frame_codec_codes):
        codes_list = codes.tolist()
        codes_str = " ".join(f"{c:4d}" for c in codes_list)
        tag = ""
        if codes_list[0] == eos_id:
            tag = "  <-- EOS (code 0)"
        print(f"frame {i:4d} codes: {codes_str}{tag}")

    # 5. Dump Talker hidden-state norms for the first N frames
    N_NORMS = min(20, len(frame_talker_hidden_norms))
    print(f"\n{'=' * 70}")
    print(f"TALKER HIDDEN-STATE NORMS (first {N_NORMS} frames)")
    print(f"{'=' * 70}")
    for i in range(N_NORMS):
        print(f"  frame {i:4d}  hidden_norm = {frame_talker_hidden_norms[i]:.6f}")

    # 6. Summary statistics
    if frame_codec_codes:
        all_codes = torch.stack(frame_codec_codes)  # [num_frames, num_groups]
        print(f"\n{'=' * 70}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 70}")
        print(f"  Total frames         : {all_codes.shape[0]}")
        print(f"  Codebooks per frame  : {all_codes.shape[1]}")
        print(f"  Code-0 min/max       : {all_codes[:, 0].min().item()} / {all_codes[:, 0].max().item()}")
        for g in range(num_groups):
            col = all_codes[:, g]
            print(f"  Codebook {g:2d}  min={col.min().item():5d}  max={col.max().item():5d}  "
                  f"mean={col.float().mean().item():8.1f}  unique={col.unique().numel()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
