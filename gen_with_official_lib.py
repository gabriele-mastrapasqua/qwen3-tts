import sys
sys.path.insert(0, 'qwen_tts')

from qwen_tts import Qwen3TTSModel
import soundfile as sf

MODEL = "qwen3-tts-0.6b"
print(f"Loading {MODEL}...")

tts = Qwen3TTSModel.from_pretrained(MODEL)
print("Generating 'Hello world'...")

wavs, sr = tts.generate_custom_voice(
    text="Hello world",
    language="English",
    speaker="Serena",
)

sf.write("/tmp/py_official_hello.wav", wavs[0], sr)
print(f"Saved /tmp/py_official_hello.wav ({len(wavs[0])/sr:.2f}s @ {sr}Hz)")
