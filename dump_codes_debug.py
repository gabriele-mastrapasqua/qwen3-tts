#!/usr/bin/env python3
"""Dump codec codes from Python in deterministic mode for comparison with C."""
import torch
import numpy as np

MODEL_DIR = "/Users/gabrielemastrapasqua/source/personal/qwen-tts/qwen3-tts-0.6b"

def main():
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, device_map="cpu")
    inner = model.model

    # Hook to capture codec codes before speech decoder
    captured = {}

    orig_decode = inner.speech_tokenizer.decode
    def patched_decode(codes):
        captured['codes'] = codes
        print(f"\n=== CODEC CODES ===")
        print(f"Type: {type(codes)}")
        if isinstance(codes, list):
            print(f"List length: {len(codes)}")
            for i, item in enumerate(codes):
                if torch.is_tensor(item):
                    print(f"  [{i}] tensor shape={item.shape}")
                elif isinstance(item, np.ndarray):
                    print(f"  [{i}] ndarray shape={item.shape}")
                else:
                    print(f"  [{i}] {type(item)}")
            # Inspect first element
            item = codes[0]
            if isinstance(item, dict):
                print(f"  Dict keys: {list(item.keys())}")
                for k, v in item.items():
                    if torch.is_tensor(v):
                        print(f"    '{k}': tensor shape={v.shape} dtype={v.dtype}")
                        if v.dim() <= 2:
                            print(f"    values: {v}")
                    elif isinstance(v, np.ndarray):
                        print(f"    '{k}': ndarray shape={v.shape} dtype={v.dtype}")
                        if v.ndim <= 2 and v.size < 200:
                            print(f"    values: {v}")
                    else:
                        print(f"    '{k}': {type(v)} = {v}")
        elif torch.is_tensor(codes):
            c = codes.squeeze()
            print(f"Tensor shape: {c.shape}")
            if c.dim() == 2:
                for f in range(min(c.shape[-1], 10)):
                    row = [str(c[cb, f].item()) for cb in range(c.shape[0])]
                    print(f"  frame {f} codes: {' '.join(row)}")
        return orig_decode(codes)

    inner.speech_tokenizer.decode = patched_decode

    # Also hook the talker to see what code0 values are generated
    talker_codes = []
    orig_talker_generate = None

    # Hook the main generate to capture intermediate codec codes
    # The generate_custom_voice calls self.model.generate() which produces talker tokens
    # Let's hook at a lower level to capture the raw codes

    print("=== GENERATING (deterministic) ===")
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

    if 'codes' in captured:
        codes = captured['codes']
        try:
            if isinstance(codes, list):
                if torch.is_tensor(codes[0]):
                    c = codes[0].squeeze()
                else:
                    c = torch.tensor(np.array(codes)).squeeze()
            else:
                c = codes.squeeze() if torch.is_tensor(codes) else torch.tensor(codes).squeeze()
            if c.dim() == 2:
                print(f"\nFull code matrix ({c.shape[0]} codebooks x {c.shape[1]} frames):")
                for f in range(c.shape[1]):
                    row = [str(c[cb, f].item()) for cb in range(c.shape[0])]
                    print(f"  frame {f}: {' '.join(row)}")
        except Exception as e:
            print(f"Error printing codes: {e}")

    print(f"\nAudio: {wavs[0].shape[0]} samples, {wavs[0].shape[0]/sr:.2f}s")
    print("Done.")

if __name__ == "__main__":
    main()
