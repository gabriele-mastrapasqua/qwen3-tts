#!/usr/bin/env python3
"""Dump CP diagnostic values for frame 0 comparison with C code."""
import torch

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def main():
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    inner = model.model
    talker = inner.talker
    cp = talker.code_predictor

    frame_count = [0]

    # Hook into CP model's inner transformer to capture hidden states
    def cp_model_hook(module, input, output):
        """Hook on cp.model (the transformer) to capture hidden before final norm."""
        pass  # We'll get hidden from the CP forward hook

    # Hook into CP forward to capture inputs and outputs
    def cp_hook(module, input, output):
        # input[0] is a tuple; kwargs might have inputs_embeds
        # output is Qwen3TTSTalkerCodePredictorOutputWithPast
        if hasattr(output, 'logits') and output.logits is not None:
            logits = output.logits
            if logits.shape[1] == 2:  # Prefill with 2 positions
                last_logits = logits[0, -1]  # [vocab_size]
                argmax_idx = last_logits.argmax().item()
                print(f"\n--- CP Prefill (frame {frame_count[0]}) ---")
                print(f"  lm_head logits: argmax={argmax_idx} val={last_logits[argmax_idx].item():.4f}")
                print(f"  logits[0..3]: {last_logits[0].item():.4f} {last_logits[1].item():.4f} {last_logits[2].item():.4f} {last_logits[3].item():.4f}")

    cp.register_forward_hook(cp_hook)

    # Hook into Talker forward to capture past_hidden and code0
    def talker_hook(module, input, output):
        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            transformer_hiddens, codec_ids = output.hidden_states
            if codec_ids is not None:
                code0 = codec_ids[0, 0].item()
                if output.past_hidden is not None:
                    ph = output.past_hidden[0, 0]
                    print(f"\n=== Talker step {frame_count[0]} ===")
                    print(f"  code0 = {code0}")
                    print(f"  past_hidden norm: {ph.norm().item():.6f}")
                    print(f"  past_hidden[0..3]: {ph[0].item():.6f} {ph[1].item():.6f} {ph[2].item():.6f} {ph[3].item():.6f}")

                    # Get codec_embedding for code0
                    code0_embed = talker.get_input_embeddings()(torch.tensor([[code0]]))[0, 0]
                    print(f"  code0_embed (code {code0}) norm: {code0_embed.norm().item():.6f}")
                    print(f"  code0_embed[0..3]: {code0_embed[0].item():.6f} {code0_embed[1].item():.6f} {code0_embed[2].item():.6f} {code0_embed[3].item():.6f}")

                    print(f"  all 16 codes: {codec_ids[0].tolist()}")
                frame_count[0] += 1

    talker.register_forward_hook(talker_hook)

    # Also hook into the CP transformer to get inputs_embeds
    original_cp_model_forward = cp.model.forward

    def patched_cp_model_forward(*args, **kwargs):
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is not None and inputs_embeds.shape[1] == 2:
            pos0 = inputs_embeds[0, 0]
            pos1 = inputs_embeds[0, 1]
            print(f"  [CP transformer input] pos0 norm: {pos0.norm().item():.6f}")
            print(f"  [CP transformer input] pos0[0..3]: {pos0[0].item():.6f} {pos0[1].item():.6f} {pos0[2].item():.6f} {pos0[3].item():.6f}")
            print(f"  [CP transformer input] pos1 norm: {pos1.norm().item():.6f}")
            print(f"  [CP transformer input] pos1[0..3]: {pos1[0].item():.6f} {pos1[1].item():.6f} {pos1[2].item():.6f} {pos1[3].item():.6f}")
        output = original_cp_model_forward(*args, **kwargs)
        if inputs_embeds is not None and inputs_embeds.shape[1] == 2:
            last_hidden = output.last_hidden_state[0, -1]
            print(f"  [CP transformer output] hidden norm: {last_hidden.norm().item():.6f}")
            print(f"  [CP transformer output] hidden[0..3]: {last_hidden[0].item():.6f} {last_hidden[1].item():.6f} {last_hidden[2].item():.6f} {last_hidden[3].item():.6f}")
        return output

    cp.model.forward = patched_cp_model_forward

    # Generate
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
        max_new_tokens=10,
    )

    print(f"\nDone. Generated {frame_count[0]} frames")

if __name__ == "__main__":
    main()
