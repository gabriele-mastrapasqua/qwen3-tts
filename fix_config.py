import json

# Load and fix config
with open('qwen3-tts-0.6b/config.json') as f:
    cfg = json.load(f)

# Add pad_token_id to talker_config if missing
if 'talker_config' in cfg:
    tc = cfg['talker_config']
    if 'pad_token_id' not in tc:
        tc['pad_token_id'] = tc.get('codec_pad_id', 2148)
        print(f"Added pad_token_id={tc['pad_token_id']} to talker_config")

# Save fixed config
with open('qwen3-tts-0.6b/config_fixed.json', 'w') as f:
    json.dump(cfg, f, indent=2)

print("Saved config_fixed.json")
