#!/usr/bin/env python3
"""Dump CP intermediate hidden states and codes for each codebook."""
import torch
import numpy as np

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def main():
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    inner = model.model
    talker = inner.talker
    cp = talker.code_predictor

    # Hook into CP's forward to capture generation_steps and hidden states
    orig_cp_forward = cp.forward

    def patched_cp_forward(*args, **kwargs):
        result = orig_cp_forward(*args, **kwargs)

        gen_steps = kwargs.get('generation_steps', None)
        if gen_steps is None and hasattr(result, 'generation_steps'):
            gen_steps = result.generation_steps

        # Get hidden states from the model output
        hidden = result.logits  # logits shape: [B, seq, vocab]
        if hidden is not None:
            # Get the raw hidden state BEFORE lm_head
            # We need to re-extract from the model's last hidden state
            pass

        logits = result.logits
        if logits is not None:
            last_logits = logits[0, -1]  # [vocab]
            argmax = last_logits.argmax().item()
            maxval = last_logits.max().item()
            gen_step_val = result.generation_steps if hasattr(result, 'generation_steps') else '?'
            print(f"  CP forward: gen_steps_out={gen_step_val} logits argmax={argmax} max={maxval:.4f} logits[0:4]={last_logits[0].item():.4f} {last_logits[1].item():.4f} {last_logits[2].item():.4f} {last_logits[3].item():.4f}")

        return result

    cp.forward = patched_cp_forward

    # Also hook the CP model's layers to capture hidden states
    cp_model = cp.model
    layer_outputs = {}

    def make_cp_layer_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            last_h = h[0, -1]
            layer_outputs[layer_idx] = last_h.detach().clone()
        return hook

    for i, layer in enumerate(cp_model.layers):
        layer.register_forward_hook(make_cp_layer_hook(i))

    # Hook final norm
    def norm_hook(module, input, output):
        last = output[0, -1]
        print(f"    CP post-norm: norm={last.norm().item():.6f} [0:4]={last[0].item():.6f} {last[1].item():.6f} {last[2].item():.6f} {last[3].item():.6f}")

    cp_model.norm.register_forward_hook(norm_hook)

    # Hook CP model input
    def cp_input_hook(module, args, kwargs):
        ie = kwargs.get('inputs_embeds', None)
        if ie is None and len(args) > 0:
            ie = args[0]
        if ie is not None:
            last = ie[0, -1]
            print(f"    CP input embed: shape={ie.shape} last norm={last.norm().item():.6f} [0:4]={last[0].item():.6f} {last[1].item():.6f} {last[2].item():.6f} {last[3].item():.6f}")

    cp_model.register_forward_pre_hook(cp_input_hook, with_kwargs=True)

    print("=== GENERATING (deterministic, 3 frames) ===")
    wavs, sr = model.generate_custom_voice(
        text="Hello world",
        speaker="Serena",
        language="English",
        temperature=0.0,
        top_k=1,
        do_sample=False,
        subtalker_temperature=0.0,
        subtalker_top_k=1,
        subtalker_dosample=False,
        repetition_penalty=1.0,
        max_new_tokens=3,
    )
    print("Done.")

if __name__ == "__main__":
    main()
