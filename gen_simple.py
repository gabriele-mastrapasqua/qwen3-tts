import torch
import soundfile as sf
import sys
sys.path.insert(0, 'qwen_tts')

# Try direct model loading
from transformers import AutoModel

MODEL = "qwen3-tts-0.6b"
print(f"Loading {MODEL}...")

try:
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True)
    print("Generating 'Hello world'...")
    audio = model.generate("Hello world", language="English")
    sf.write("/tmp/py_hello.wav", audio.squeeze().numpy(), 24000)
    print(f"Saved /tmp/py_hello.wav ({len(audio.squeeze())/24000:.2f}s)")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
