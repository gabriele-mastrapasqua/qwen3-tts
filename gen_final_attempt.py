import sys
sys.path.insert(0, 'qwen_tts')

from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

MODEL = "qwen3-tts-0.6b"
print(f"Loading {MODEL}...")

# Load with explicit config
tts = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cpu",
    torch_dtype=torch.float32,
)

print("Generating 'Hello world'...")

wavs, sr = tts.generate_custom_voice(
    text="Hello world",
    language="English",
    speaker="Serena",
    max_new_tokens=512,  # Limit frames
)

sf.write("/tmp/py_final_hello.wav", wavs[0], sr)
print(f"Saved /tmp/py_final_hello.wav ({len(wavs[0])/sr:.2f}s @ {sr}Hz)")
