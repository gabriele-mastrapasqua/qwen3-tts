#!/usr/bin/env python3
"""Dump prefill embeddings and layer-by-layer hidden states for C comparison."""
import torch
import numpy as np

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def main():
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    inner = model.model
    talker = inner.talker

    # --- Dump text_projection for token 27 (first role token) ---
    text_emb_layer = talker.model.text_embedding
    text_proj = talker.text_projection

    tok27_emb = text_emb_layer(torch.tensor([27]))  # [1, 2048]
    print(f"text_embedding(27) norm: {tok27_emb.norm().item():.6f}")
    print(f"text_embedding(27)[0:4]: {tok27_emb[0,0].item():.6f} {tok27_emb[0,1].item():.6f} {tok27_emb[0,2].item():.6f} {tok27_emb[0,3].item():.6f}")

    tok27_proj = text_proj(tok27_emb)  # [1, 1024]
    print(f"text_projection(27) norm: {tok27_proj.norm().item():.6f}")
    print(f"text_projection(27)[0:4]: {tok27_proj[0,0].item():.6f} {tok27_proj[0,1].item():.6f} {tok27_proj[0,2].item():.6f} {tok27_proj[0,3].item():.6f}")

    # --- Dump codec embedding for CODEC_PAD=2148 ---
    codec_emb_layer = talker.model.codec_embedding
    pad_emb = codec_emb_layer(torch.tensor([2148]))  # [1, 1024]
    print(f"\ncodec_embedding(2148/PAD) norm: {pad_emb.norm().item():.6f}")
    print(f"codec_embedding(2148/PAD)[0:4]: {pad_emb[0,0].item():.6f} {pad_emb[0,1].item():.6f} {pad_emb[0,2].item():.6f} {pad_emb[0,3].item():.6f}")

    # --- Dump TTS special embeddings ---
    for name, tid in [("TTS_PAD", 151671), ("TTS_BOS", 151672), ("TTS_EOS", 151673)]:
        emb = text_emb_layer(torch.tensor([tid]))
        proj = text_proj(emb)
        print(f"\n{name} (tok {tid}):")
        print(f"  text_embedding norm: {emb.norm().item():.6f}")
        print(f"  text_projection norm: {proj.norm().item():.6f}")
        print(f"  text_projection[0:4]: {proj[0,0].item():.6f} {proj[0,1].item():.6f} {proj[0,2].item():.6f} {proj[0,3].item():.6f}")

    # --- Now hook into the model to capture prefill embeddings ---
    # We need to intercept the inputs_embeds going into the transformer
    prefill_data = {}

    # Hook the transformer model to capture its input
    def transformer_input_hook(module, args, kwargs):
        inputs_embeds = kwargs.get('inputs_embeds', None)
        if inputs_embeds is None and len(args) > 0:
            inputs_embeds = args[0]
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            prefill_data['inputs_embeds'] = inputs_embeds.detach().clone()
            seq_len = inputs_embeds.shape[1]
            print(f"\n=== PREFILL INPUT EMBEDDINGS ===")
            print(f"Shape: {inputs_embeds.shape}")
            for pos in [0, 1, 2, 3, 8, seq_len-2, seq_len-1]:
                if pos < seq_len:
                    v = inputs_embeds[0, pos]
                    print(f"  pos {pos:3d} norm={v.norm().item():.6f} [0:4]={v[0].item():.6f} {v[1].item():.6f} {v[2].item():.6f} {v[3].item():.6f}")

    talker.model.register_forward_pre_hook(transformer_input_hook, with_kwargs=True)

    # Hook each transformer layer to capture hidden states
    layer_outputs = {}

    def make_layer_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            if hidden.shape[1] > 1:  # Only capture prefill
                layer_outputs[layer_idx] = hidden.detach().clone()
        return hook

    for i, layer in enumerate(talker.model.layers):
        layer.register_forward_hook(make_layer_hook(i))

    # Hook Layer 0 attention to capture Q, K, V, attn_output
    layer0_data = {}

    def layer0_attn_pre_hook(module, args, kwargs):
        """Hook on layer 0's self_attn to capture Q/K/V after projection and RoPE."""
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        if hidden_states is not None and hidden_states.shape[1] > 1:
            layer0_data['attn_input'] = hidden_states.detach().clone()

    def layer0_attn_hook(module, input, output):
        """Hook on layer 0's self_attn output."""
        if isinstance(output, tuple):
            attn_out = output[0]
        else:
            attn_out = output
        if attn_out.shape[1] > 1:
            layer0_data['attn_output'] = attn_out.detach().clone()

    talker.model.layers[0].self_attn.register_forward_pre_hook(layer0_attn_pre_hook, with_kwargs=True)
    talker.model.layers[0].self_attn.register_forward_hook(layer0_attn_hook)

    # Hook final norm
    def norm_hook(module, input, output):
        if output.shape[1] > 1:
            prefill_data['post_norm'] = output.detach().clone()

    talker.model.norm.register_forward_hook(norm_hook)

    # Generate
    print("\n=== GENERATING ===")
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

    # Dump layer-by-layer hidden state norms for last position
    print(f"\n=== LAYER-BY-LAYER HIDDEN STATES (last position) ===")
    if 'inputs_embeds' in prefill_data:
        ie = prefill_data['inputs_embeds']
        seq_len = ie.shape[1]
        last_input = ie[0, -1]
        print(f"Input (last pos): norm={last_input.norm().item():.6f}")

    for i in sorted(layer_outputs.keys()):
        h = layer_outputs[i]
        last_h = h[0, -1]
        print(f"Layer {i:2d} output (last pos): norm={last_h.norm().item():.6f} [0:4]={last_h[0].item():.6f} {last_h[1].item():.6f} {last_h[2].item():.6f} {last_h[3].item():.6f}")

    if 'post_norm' in prefill_data:
        pn = prefill_data['post_norm']
        last_pn = pn[0, -1]
        print(f"Post-norm (last pos): norm={last_pn.norm().item():.6f} [0:4]={last_pn[0].item():.6f} {last_pn[1].item():.6f} {last_pn[2].item():.6f} {last_pn[3].item():.6f}")

    # Also dump first position norms through layers
    print(f"\n=== LAYER-BY-LAYER HIDDEN STATES (first position) ===")
    for i in sorted(layer_outputs.keys()):
        h = layer_outputs[i]
        first_h = h[0, 0]
        print(f"Layer {i:2d} output (pos 0): norm={first_h.norm().item():.6f}")

    # Dump Layer 0 attention details
    print(f"\n=== LAYER 0 ATTENTION DETAILS ===")
    if 'attn_output' in layer0_data:
        ao = layer0_data['attn_output']
        last_ao = ao[0, -1]
        print(f"  L0 attn_output[last] norm={last_ao.norm().item():.6f} [0:4]={last_ao[0].item():.6f} {last_ao[1].item():.6f} {last_ao[2].item():.6f} {last_ao[3].item():.6f}")
        first_ao = ao[0, 0]
        print(f"  L0 attn_output[0] norm={first_ao.norm().item():.6f} [0:4]={first_ao[0].item():.6f} {first_ao[1].item():.6f} {first_ao[2].item():.6f} {first_ao[3].item():.6f}")

    # Also manually compute Q/K for layer 0 to compare with C
    if 'inputs_embeds' in prefill_data:
        ie = prefill_data['inputs_embeds']
        layer0 = talker.model.layers[0]
        sa = layer0.self_attn
        num_heads = sa.config.num_attention_heads
        num_kv_heads = sa.config.num_key_value_heads
        head_dim = sa.head_dim

        # Apply input_layernorm
        x_norm = layer0.input_layernorm(ie)

        # Q/K/V projections
        q = sa.q_proj(x_norm)
        k = sa.k_proj(x_norm)
        v = sa.v_proj(x_norm)

        # Reshape for heads: [bs, seq, num_heads, head_dim]
        bs, seq, _ = q.shape
        q = q.view(bs, seq, num_heads, head_dim)
        k = k.view(bs, seq, num_kv_heads, head_dim)

        # Per-head Q/K norm
        q = sa.q_norm(q)
        k = sa.k_norm(k)

        # RoPE (need position_ids) - [1, 3, seq] for MROPE
        # rotary_emb is at model level, not attention level
        position_ids = torch.arange(seq).unsqueeze(0).unsqueeze(0).expand(1, 3, -1)
        cos, sin = talker.model.rotary_emb(v, position_ids)

        # Apply RoPE
        try:
            from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
            mrope_section = [24, 20, 20]
            q_rope, k_rope = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        except Exception as e:
            print(f"  Warning: apply_multimodal_rotary_pos_emb failed: {e}")
            q_rope = q
            k_rope = k

        last = seq - 1
        # Q last pos, head 0
        q_last_h0 = q_rope[0, last, 0]
        print(f"\n  L0 Q[{last},h0] norm={q_last_h0.norm().item():.6f} [0:4]={q_last_h0[0].item():.6f} {q_last_h0[1].item():.6f} {q_last_h0[2].item():.6f} {q_last_h0[3].item():.6f}")
        # K last pos, kv_head 0
        k_last_h0 = k_rope[0, last, 0]
        print(f"  L0 K[{last},h0] norm={k_last_h0.norm().item():.6f} [0:4]={k_last_h0[0].item():.6f} {k_last_h0[1].item():.6f} {k_last_h0[2].item():.6f} {k_last_h0[3].item():.6f}")
        # K pos 0, kv_head 0
        k_0_h0 = k_rope[0, 0, 0]
        print(f"  L0 K[0,h0] norm={k_0_h0.norm().item():.6f} [0:4]={k_0_h0[0].item():.6f} {k_0_h0[1].item():.6f} {k_0_h0[2].item():.6f} {k_0_h0[3].item():.6f}")
        # V pos 0, kv_head 0
        v_0_h0 = v[0, 0, :head_dim]
        print(f"  L0 V[0,h0] norm={v_0_h0.norm().item():.6f} [0:4]={v_0_h0[0].item():.6f} {v_0_h0[1].item():.6f} {v_0_h0[2].item():.6f} {v_0_h0[3].item():.6f}")

    # Save prefill embeddings to binary for exact comparison
    if 'inputs_embeds' in prefill_data:
        ie_np = prefill_data['inputs_embeds'][0].numpy()
        np.save("/tmp/python_prefill_embeds.npy", ie_np)
        print(f"\nSaved prefill embeddings to /tmp/python_prefill_embeds.npy ({ie_np.shape})")

    print("\nDone.")

if __name__ == "__main__":
    main()
