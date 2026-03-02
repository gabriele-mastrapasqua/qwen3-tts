#!/usr/bin/env python3
"""
Full speech decoder in Python - reconstruct audio from codec codes.
Matches the C implementation structure for comparison.
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
import soundfile as sf
from safetensors.torch import load_file

MODEL_DIR = "qwen3-tts-0.6b"
CODES_FILE = "/tmp/test_codes.bin"
OUTPUT_PYTHON = "/tmp/python_full_decode.wav"

print("=" * 60)
print("Full Speech Decoder in Python")
print("=" * 60)

# Load codes
with open(CODES_FILE, "rb") as f:
    n_frames, n_codebooks = np.frombuffer(f.read(8), dtype=np.int32)
    codes = np.frombuffer(f.read(), dtype=np.int32).reshape(n_frames, n_codebooks)

print(f"Loaded codes: {n_frames} frames x {n_codebooks} codebooks")

# Load weights
st_path = f"{MODEL_DIR}/speech_tokenizer/model.safetensors"
W = load_file(st_path)

with open(f"{MODEL_DIR}/speech_tokenizer/config.json") as f:
    cfg = json.load(f)
dec_cfg = cfg["decoder_config"]

# ============ STEP 1: VQ Dequantization ============
print("\n[Step 1] VQ Dequantization...")

# Load and dequantize codebooks
emb0 = W["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"]
usage0 = W["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
cb0 = emb0 / usage0.clamp(min=1e-5).unsqueeze(1)

cb_rest = []
for k in range(15):
    emb = W[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.embedding_sum"]
    usage = W[f"decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.cluster_usage"]
    cb_rest.append(emb / usage.clamp(min=1e-5).unsqueeze(1))

codes_t = torch.tensor(codes, dtype=torch.long)

# Codebook 0
q_first = F.embedding(codes_t[:, 0], cb0)  # [T, 256]
q_first = q_first.T.unsqueeze(0)  # [1, 256, T]

# Codebooks 1-15
q_rest_sum = torch.zeros(1, 256, n_frames)
for k in range(15):
    q_rest_sum += F.embedding(codes_t[:, k + 1], cb_rest[k]).T.unsqueeze(0)

# Apply output projections
out_proj_first = W["decoder.quantizer.rvq_first.output_proj.weight"]
out_proj_rest = W["decoder.quantizer.rvq_rest.output_proj.weight"]

q_first_proj = F.conv1d(q_first, out_proj_first)  # [1, 512, T]
q_rest_proj = F.conv1d(q_rest_sum, out_proj_rest)  # [1, 512, T]

vq_out = q_first_proj + q_rest_proj  # [1, 512, T]
print(f"  VQ output: {vq_out.shape}, min={vq_out.min():.4f}, max={vq_out.max():.4f}")

# ============ STEP 2: Pre-conv ============
print("\n[Step 2] Pre-conv (causal k=3)...")

pre_conv_w = W["decoder.pre_conv.conv.weight"]
pre_conv_b = W["decoder.pre_conv.conv.bias"]

# Causal padding: pad_left = (3-1)*1 = 2
vq_padded = F.pad(vq_out, (2, 0))
pre_conv_out = F.conv1d(vq_padded, pre_conv_w, pre_conv_b)  # [1, 1024, T]
print(f"  Pre-conv out: {pre_conv_out.shape}, min={pre_conv_out.min():.4f}, max={pre_conv_out.max():.4f}")

# ============ STEP 3: Input Projection ============
print("\n[Step 3] Input projection...")

inp_w = W["decoder.pre_transformer.input_proj.weight"]
inp_b = W["decoder.pre_transformer.input_proj.bias"]

latent = pre_conv_out.squeeze(0).T  # [T, 1024]
hidden = F.linear(latent, inp_w, inp_b)  # [T, 512]
print(f"  Hidden: {hidden.shape}, min={hidden.min():.4f}, max={hidden.max():.4f}")

# ============ STEP 4: Pre-transformer (8 layers) ============
print("\n[Step 4] Pre-transformer (8 layers)...")

def rms_norm(x, weight, eps=1e-5):
    norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / norm * weight

def apply_rope(x, cos, sin, head_dim):
    """Interleaved RoPE"""
    num_pairs = head_dim // 2
    x = x.view(*x.shape[:-1], num_pairs, 2)
    x_even = x[..., 0]
    x_odd = x[..., 1]
    cos_exp = cos.unsqueeze(0).unsqueeze(0)
    sin_exp = sin.unsqueeze(0).unsqueeze(0)
    y_even = x_even * cos_exp - x_odd * sin_exp
    y_odd = x_odd * cos_exp + x_even * sin_exp
    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return y

# Load pre-transformer weights
pre_layers = []
for i in range(8):
    layer = {
        'attn_norm': W[f"decoder.pre_transformer.layers.{i}.attn_norm.weight"],
        'attn_q': W[f"decoder.pre_transformer.layers.{i}.attn_q.weight"],
        'attn_k': W[f"decoder.pre_transformer.layers.{i}.attn_k.weight"],
        'attn_v': W[f"decoder.pre_transformer.layers.{i}.attn_v.weight"],
        'attn_o': W[f"decoder.pre_transformer.layers.{i}.attn_o.weight"],
        'ffn_norm': W[f"decoder.pre_transformer.layers.{i}.ffn_norm.weight"],
        'ffn_up': W[f"decoder.pre_transformer.layers.{i}.ffn_up.weight"],
        'ffn_down': W[f"decoder.pre_transformer.layers.{i}.ffn_down.weight"],
    }
    pre_layers.append(layer)

# RoPE cache
head_dim = 64
rope_theta = 10000.0
num_pairs = head_dim // 2
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, num_pairs).float() / num_pairs))
positions = torch.arange(n_frames).float().unsqueeze(1)
angle = positions * inv_freq.unsqueeze(0)
rope_cos = torch.cos(angle)
rope_sin = torch.sin(angle)

# Forward through pre-transformer
x = hidden  # [T, 512]
n_heads = 16
qkv_dim = n_heads * head_dim  # 1024

for i, layer in enumerate(pre_layers):
    # RMSNorm
    x_norm = rms_norm(x, layer['attn_norm'])
    
    # QKV projection
    q = F.linear(x_norm, layer['attn_q'])  # [T, 1024]
    k = F.linear(x_norm, layer['attn_k'])  # [T, 1024]
    v = F.linear(x_norm, layer['attn_v'])  # [T, 1024]
    
    # Reshape for RoPE
    q = q.view(n_frames, n_heads, head_dim)
    k = k.view(n_frames, n_heads, head_dim)
    v = v.view(n_frames, n_heads, head_dim)
    
    # Apply RoPE
    q = apply_rope(q, rope_cos, rope_sin, head_dim)
    k = apply_rope(k, rope_cos, rope_sin, head_dim)
    
    # Reshape back
    q = q.view(n_frames, qkv_dim)
    k = k.view(n_frames, qkv_dim)
    v = v.view(n_frames, qkv_dim)
    
    # Sliding window attention (window=72)
    window = 72
    scale = 1.0 / (head_dim ** 0.5)
    attn_out = torch.zeros_like(q)
    
    for t in range(n_frames):
        k_start = max(0, t - window + 1)
        k_end = t + 1
        q_t = q[t:t+1]  # [1, 1024]
        k_slice = k[k_start:k_end]  # [win, 1024]
        v_slice = v[k_start:k_end]  # [win, 1024]
        
        # Attention scores
        scores = (q_t @ k_slice.T) * scale  # [1, win]
        weights = F.softmax(scores, dim=-1)  # [1, win]
        
        # Output
        attn_out[t:t+1] = weights @ v_slice  # [1, 1024]
    
    # Output projection + residual
    x = x + F.linear(attn_out, layer['attn_o'])
    
    # FFN
    x_norm = rms_norm(x, layer['ffn_norm'])
    ffn_up = F.linear(x_norm, layer['ffn_up'])  # [T, 1024]
    ffn_down = F.linear(F.silu(ffn_up), layer['ffn_down'])  # [T, 512]
    x = x + ffn_down

print(f"  Pre-transformer output: {x.shape}, min={x.min():.4f}, max={x.max():.4f}")

# ============ STEP 5: Output Projection ============
print("\n[Step 5] Output projection...")

out_w = W["decoder.pre_transformer.output_proj.weight"]
out_b = W["decoder.pre_transformer.output_proj.bias"]
latent_out = F.linear(x, out_w, out_b)  # [T, 1024]
print(f"  Latent out: {latent_out.shape}, min={latent_out.min():.4f}, max={latent_out.max():.4f}")

# ============ STEP 6: ConvNeXt Upsample (2 blocks) ============
print("\n[Step 6] ConvNeXt upsample (2 blocks)...")

def conv_transpose1d_causal(x, weight, bias, stride):
    """Causal ConvTranspose1d"""
    # weight: [in_ch, out_ch, kernel]
    in_ch, out_ch, kernel = weight.shape
    # Output length: (L-1)*stride + kernel
    L = x.shape[-1]
    out_len = (L - 1) * stride + kernel
    # Trim right by (kernel - stride)
    trim = kernel - stride
    out_len = out_len - trim
    
    # Full conv transpose
    full_out = F.conv_transpose1d(x, weight, bias=None, stride=stride)  # [B, out_ch, full_len]
    # Trim right
    out = full_out[:, :, :out_len]
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out

signal = latent_out.T.unsqueeze(0)  # [1, 1024, T]

for b in range(2):
    cn_conv_w = W[f"decoder.upsample.{b}.conv.weight"]
    cn_conv_b = W[f"decoder.upsample.{b}.conv.bias"]
    cn_dwconv_w = W[f"decoder.upsample.{b}.dwconv.weight"]
    cn_dwconv_b = W[f"decoder.upsample.{b}.dwconv.bias"]
    cn_pwconv1_w = W[f"decoder.upsample.{b}.pwconv1.weight"]
    cn_pwconv1_b = W[f"decoder.upsample.{b}.pwconv1.bias"]
    cn_pwconv2_w = W[f"decoder.upsample.{b}.pwconv2.weight"]
    cn_pwconv2_b = W[f"decoder.upsample.{b}.pwconv2.bias"]
    cn_gamma = W[f"decoder.upsample.{b}.gamma"]
    cn_norm_w = W[f"decoder.upsample.{b}.norm.weight"]
    cn_norm_b = W[f"decoder.upsample.{b}.norm.bias"]
    
    # Upsample ConvTranspose1d
    up_out = conv_transpose1d_causal(signal, cn_conv_w, cn_conv_b, stride=2)
    signal = up_out
    
    # Residual
    residual = signal.clone()
    
    # Depthwise conv (causal k=7)
    dw_out = F.pad(signal, (6, 0))  # pad_left=6
    dw_out = F.conv1d(dw_out, cn_dwconv_w, cn_dwconv_b, groups=1024)
    signal = dw_out
    
    # LayerNorm
    signal = signal.permute(0, 2, 1)  # [B, L, C]
    signal = F.layer_norm(signal, (1024,), cn_norm_w, cn_norm_b)
    signal = signal.permute(0, 2, 1)  # [B, C, L]
    
    # Pointwise 1
    signal = signal.permute(0, 2, 1)  # [B, L, C]
    pw1_out = F.linear(signal, cn_pwconv1_w, cn_pwconv1_b)  # [B, L, 4096]
    pw1_out = F.gelu(pw1_out)
    
    # Pointwise 2
    pw2_out = F.linear(pw1_out, cn_pwconv2_w, cn_pwconv2_b)  # [B, L, 1024]
    pw2_out = pw2_out.permute(0, 2, 1)  # [B, C, L]
    
    # Gamma
    pw2_out = pw2_out * cn_gamma.view(1, -1, 1)
    
    # Add residual
    signal = residual + pw2_out

print(f"  After ConvNeXt: {signal.shape}, min={signal.min():.4f}, max={signal.max():.4f}")

# ============ STEP 7: Initial Conv ============
print("\n[Step 7] Initial conv (k=7)...")

init_conv_w = W["decoder.decoder.initial.conv.weight"]
init_conv_b = W["decoder.decoder.initial.conv.bias"]

signal_padded = F.pad(signal, (6, 0))  # pad_left=6
signal = F.conv1d(signal_padded, init_conv_w, init_conv_b)
print(f"  After initial conv: {signal.shape}, min={signal.min():.4f}, max={signal.max():.4f}")

# ============ STEP 8: 4 Upsample Blocks ============
print("\n[Step 8] 4 Upsample blocks...")

cur_ch = 1536
for b in range(4):
    ub_rates = dec_cfg["upsample_rates"]
    rate = ub_rates[b]
    
    up_w = W[f"decoder.decoder.{b}.upsample.conv.weight"]
    up_b = W[f"decoder.decoder.{b}.upsample.conv.bias"]
    
    # Snake activation (simplified - just use the signal)
    # Full snake: x + alpha*sin^2(beta*x)
    
    # Upsample
    up_out = conv_transpose1d_causal(signal, up_w, up_b, stride=rate)
    signal = up_out
    cur_ch = 96
    
    # 3 residual blocks
    for r in range(3):
        dilation = 3 ** r
        conv1_pad = (7 - 1) * dilation
        
        res1_conv1_w = W[f"decoder.decoder.{b}.residual.{r}.conv1.weight"]
        res1_conv1_b = W[f"decoder.decoder.{b}.residual.{r}.conv1.bias"]
        res1_conv2_w = W[f"decoder.decoder.{b}.residual.{r}.conv2.weight"]
        res1_conv2_b = W[f"decoder.decoder.{b}.residual.{r}.conv2.bias"]
        
        residual = signal.clone()
        
        # Conv1 (causal with dilation)
        sig_padded = F.pad(signal, (conv1_pad, 0))
        conv1_out = F.conv1d(sig_padded, res1_conv1_w, res1_conv1_b, dilation=dilation)
        
        # Conv2 (k=1)
        conv2_out = F.conv1d(conv1_out, res1_conv2_w, res1_conv2_b)
        
        signal = residual + conv2_out

print(f"  After upsample blocks: {signal.shape}, min={signal.min():.4f}, max={signal.max():.4f}")

# ============ STEP 9: Final Conv ============
print("\n[Step 9] Final conv...")

final_conv_w = W["decoder.final_conv.weight"]
final_conv_b = W["decoder.final_conv.bias"]

signal_padded = F.pad(signal, (6, 0))
audio = F.conv1d(signal_padded, final_conv_w, final_conv_b)  # [1, 1, L]
audio = audio.squeeze()

# Clamp to [-1, 1]
audio = torch.clamp(audio, -1, 1)

print(f"  Final audio: {audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}")
print(f"  Duration: {len(audio)/24000:.2f}s @ 24kHz")

# Save
sf.write(OUTPUT_PYTHON, audio.numpy(), 24000)
print(f"\nSaved: {OUTPUT_PYTHON}")
print("=" * 60)
