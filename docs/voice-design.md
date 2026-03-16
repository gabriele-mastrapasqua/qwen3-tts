# VoiceDesign

Create entirely new voices from natural language descriptions. Instead of cloning
an existing voice, you describe the voice you want and the model generates it.

## Requirements

- **1.7B VoiceDesign model** (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`)
- Does **not** work with the 0.6B model — the engine will refuse to run
- Model type is auto-detected from the config
- `--instruct` is required to describe the desired voice
- No `--speaker` is needed

## Quick Start

```bash
# Download the VoiceDesign model
./download_model.sh --model voice-design

# Deep British male
./qwen_tts -d qwen3-tts-voice-design -l English \
    --instruct "A deep male voice with a British accent, speaking slowly and calmly" \
    --text "Hello, this is a test of the voice design system." -o british.wav

# Young energetic female
./qwen_tts -d qwen3-tts-voice-design -l English \
    --instruct "Young energetic female, cheerful and fast-paced" \
    --text "Oh my gosh, this is so exciting!" -o cheerful.wav

# Chinese loli voice
./qwen_tts -d qwen3-tts-voice-design -l Chinese \
    --instruct "萝莉女声，撒娇稚嫩" \
    --text "你好，这是一个语音设计的测试。" -o loli.wav
```

## Tips

- Be specific about gender, age, accent, speaking speed, and emotional tone
- You can use the target language for the description (e.g., Chinese descriptions for Chinese voices)
- Try different `--seed` values — the model interprets descriptions stochastically
- Combine with `--temperature` to control how closely the model follows the description
  (lower = more predictable, higher = more varied)

## How It Differs from Voice Cloning

| | VoiceDesign | Voice Cloning |
|---|---|---|
| **Input** | Text description | Reference audio WAV |
| **Model** | VoiceDesign (1.7B only) | Base (0.6B or 1.7B) |
| **Voice consistency** | Varies with seed | Consistent (from embedding) |
| **Use case** | Creative / fictional voices | Reproducing a real voice |
| **Saveable** | No | Yes (`.qvoice`) |
