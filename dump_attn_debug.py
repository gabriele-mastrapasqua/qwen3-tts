#!/usr/bin/env python3
"""Hook directly into Layer 0 attention to capture Q/K after RoPE and attention output."""
import torch
import numpy as np

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def main():
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    inner = model.model
    talker = inner.talker

    captured = {}

    # Monkey-patch Layer 0's attention forward to capture Q, K, V after RoPE
    layer0 = talker.model.layers[0]
    sa = layer0.self_attn
    original_forward = sa.forward

    def patched_forward(*args, **kwargs):
        # Call original forward
        result = original_forward(*args, **kwargs)

        # Now manually recompute to capture intermediates
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        if hidden_states is not None and hidden_states.shape[1] > 1:
            bs, seq, _ = hidden_states.shape
            num_heads = sa.config.num_attention_heads
            num_kv_heads = sa.config.num_key_value_heads
            head_dim = sa.head_dim

            # Get the attention output (before residual, this is the self_attn output)
            if isinstance(result, tuple):
                attn_output = result[0]
            else:
                attn_output = result
            captured['attn_output'] = attn_output.detach().clone()

            # Manually get Q, K after RoPE from what the forward did internally
            # We'll do it by re-running the projections + norms + RoPE
            q = sa.q_proj(hidden_states)
            k = sa.k_proj(hidden_states)
            v = sa.v_proj(hidden_states)

            q = q.view(bs, seq, num_heads, head_dim).transpose(1, 2)  # [bs, num_heads, seq, dim]
            k = k.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)

            q = sa.q_norm(q)
            k = sa.k_norm(k)

            # RoPE - position_ids shape: [3, bs, seq] for MROPE
            # Force 3D MROPE position_ids regardless of what the model passes
            position_ids_3d = torch.arange(seq).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
            cos, sin = talker.model.rotary_emb(v, position_ids_3d)

            from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
            mrope_section = sa.rope_scaling.get("mrope_section", [24, 20, 20]) if sa.rope_scaling else [24, 20, 20]
            q_rope, k_rope = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)

            captured['q_rope'] = q_rope.detach().clone()  # [bs, num_heads, seq, head_dim]
            captured['k_rope'] = k_rope.detach().clone()
            captured['v'] = v.detach().clone()

            last = seq - 1
            # Q after RoPE, head 0, last pos
            q_last = q_rope[0, 0, last]  # [head_dim]
            print(f"  Py L0 Q_rope[{last},h0] norm={q_last.norm().item():.6f} [0:4]={q_last[0].item():.6f} {q_last[1].item():.6f} {q_last[2].item():.6f} {q_last[3].item():.6f}")
            # K after RoPE
            k_last = k_rope[0, 0, last]
            print(f"  Py L0 K_rope[{last},h0] norm={k_last.norm().item():.6f} [0:4]={k_last[0].item():.6f} {k_last[1].item():.6f} {k_last[2].item():.6f} {k_last[3].item():.6f}")
            k_0 = k_rope[0, 0, 0]
            print(f"  Py L0 K_rope[0,h0] norm={k_0.norm().item():.6f} [0:4]={k_0[0].item():.6f} {k_0[1].item():.6f} {k_0[2].item():.6f} {k_0[3].item():.6f}")

            # Compute attention scores for head 0, query pos=last
            scale = 1.0 / (head_dim ** 0.5)
            # GQA: head 0 uses kv_head 0 (heads_per_kv = num_heads // num_kv_heads = 2)
            q_h0 = q_rope[0, 0]  # [seq, head_dim]
            k_h0 = k_rope[0, 0]  # [seq, head_dim]
            v_h0 = v[0, 0]       # [seq, head_dim]

            # Compute scores for query position = last
            scores = (q_h0[last:last+1] @ k_h0.T) * scale  # [1, seq]
            # Apply causal mask
            causal_mask = torch.zeros(1, seq)
            # Already causal since Q is at last position, all K positions are accessible
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out_h0 = attn_weights @ v_h0  # [1, head_dim]

            print(f"\n  Py L0 head0 attn scores[last, :5]: {scores[0, :5].tolist()}")
            print(f"  Py L0 head0 attn scores[last, -3:]: {scores[0, -3:].tolist()}")
            print(f"  Py L0 head0 attn_weights max: pos={attn_weights.argmax().item()} val={attn_weights.max().item():.6f}")
            print(f"  Py L0 head0 attn_out[last] norm={attn_out_h0.norm().item():.6f} [0:4]={attn_out_h0[0,0].item():.6f} {attn_out_h0[0,1].item():.6f} {attn_out_h0[0,2].item():.6f} {attn_out_h0[0,3].item():.6f}")

            # Also do pos 0 for reference
            scores_0 = (q_h0[0:1] @ k_h0[:1].T) * scale  # [1, 1]
            print(f"  Py L0 head0 score[0,0]={scores_0[0,0].item():.6f}")

            # Compute full attention output across all heads for last pos
            attn_full = torch.zeros(1, num_heads * head_dim)
            for h in range(num_heads):
                kv_h = h // (num_heads // num_kv_heads)
                q_h = q_rope[0, h, last:last+1]  # [1, head_dim]
                k_h = k_rope[0, kv_h]  # [seq, head_dim]
                v_h = captured['v'][0, kv_h]  # [seq, head_dim]
                s = (q_h @ k_h.T) * scale
                w = torch.softmax(s, dim=-1)
                o = w @ v_h
                attn_full[0, h*head_dim:(h+1)*head_dim] = o
            attn_full_norm = attn_full.norm().item()
            print(f"\n  Py L0 raw_attn[last] full norm={attn_full_norm:.6f} [0:4]={attn_full[0,0].item():.6f} {attn_full[0,1].item():.6f} {attn_full[0,2].item():.6f} {attn_full[0,3].item():.6f}")

            # O projection
            o_proj_out = sa.o_proj(attn_full)
            print(f"  Py L0 o_proj[last] norm={o_proj_out.norm().item():.6f} [0:4]={o_proj_out[0,0].item():.6f} {o_proj_out[0,1].item():.6f} {o_proj_out[0,2].item():.6f} {o_proj_out[0,3].item():.6f}")

        return result

    sa.forward = patched_forward

    # Generate
    print("=== GENERATING ===")
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
