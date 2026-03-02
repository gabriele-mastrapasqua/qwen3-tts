import torch
from transformers import AutoModel, AutoProcessor
import soundfile as sf

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TEXT = "Hello world"

print(f"Loading {MODEL_ID} from HF...")
try:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    
    print(f"Generating '{TEXT}'...")
    # Try generate method
    if hasattr(model, 'generate'):
        audio = model.generate(TEXT)
        sf.write("/tmp/hf_audio.wav", audio.squeeze().numpy(), 24000)
        print(f"Saved /tmp/hf_audio.wav")
    else:
        print("No generate method found")
        print(f"Model attrs: {[a for a in dir(model) if not a.startswith('_')][:20]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
