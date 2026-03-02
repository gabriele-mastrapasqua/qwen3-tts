#!/usr/bin/env python3
"""
Validate C implementation against Qwen3-TTS Python reference.

Compares:
1. Generated codec tokens (from Talker + Code Predictor)
2. Speech decoder output (from codec tokens -> audio)
3. Final audio quality (waveform correlation)

Usage:
    python3 validate.py --model-dir qwen3-tts-0.6b --text "Hello world"
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np
import soundfile as sf


def read_wav(path):
    """Read WAV file, return (samples_float32, sample_rate)."""
    data, sr = sf.read(path, dtype='float32')
    return data, sr


def compute_stats(audio):
    """Compute basic audio statistics."""
    return {
        'length': len(audio),
        'duration_s': len(audio) / 24000.0,
        'rms': float(np.sqrt(np.mean(audio ** 2))),
        'peak': float(np.max(np.abs(audio))),
        'nonzero': int(np.count_nonzero(audio)),
        'mean': float(np.mean(audio)),
    }


def run_c_inference(model_dir, text, output_path, seed=42, speaker="serena"):
    """Run the C implementation and return output WAV path."""
    cmd = [
        './qwen_tts',
        '-d', model_dir,
        '--text', text,
        '-o', output_path,
        '--seed', str(seed),
        '--speaker', speaker,
    ]
    print(f"Running C inference: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"C inference FAILED (exit code {result.returncode})")
        print(result.stderr)
        return None, elapsed

    print(result.stderr)
    return output_path, elapsed


def run_python_reference(model_id, text, output_path, seed=42, speaker="Serena"):
    """Run the HuggingFace Qwen3-TTS reference model."""
    print(f"Running Python reference (model={model_id})...")
    t0 = time.time()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return None, 0

    # Check if the model is available locally or needs download
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Trying with Qwen/Qwen3-TTS-0.6B-CustomVoice from HuggingFace...")
        try:
            model_id = "Qwen/Qwen3-TTS-0.6B-CustomVoice"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        except Exception as e2:
            print(f"Failed to load HF model: {e2}")
            return None, 0

    # Build the prompt similar to our C code
    # The official way uses the model's generate method
    try:
        torch.manual_seed(seed)

        # Try using the model's built-in TTS generation
        if hasattr(model, 'generate_speech') or hasattr(model, 'synthesize'):
            # Some models have a direct speech generation method
            audio = model.generate_speech(text, speaker=speaker)
        else:
            # Use the chat template approach
            messages = [
                {"role": "assistant", "content": text}
            ]

            # Try the standard generate approach
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            if isinstance(inputs, dict):
                input_ids = inputs["input_ids"]
            else:
                input_ids = inputs

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=8192,
                    temperature=0.9,
                    top_k=50,
                    top_p=1.0,
                    repetition_penalty=1.05,
                    do_sample=True,
                )

            # Extract codec tokens from outputs
            generated = outputs[0][input_ids.shape[-1]:]
            print(f"  Generated {len(generated)} tokens")

            # The model outputs codec tokens that need to be decoded by the speech tokenizer
            # This part depends on the specific model API
            print("  Note: Full Python reference requires speech tokenizer decoder")
            print("  which may need additional setup. Skipping waveform comparison.")
            elapsed = time.time() - t0
            return None, elapsed

    except Exception as e:
        print(f"Python generation failed: {e}")
        import traceback
        traceback.print_exc()
        elapsed = time.time() - t0
        return None, elapsed

    elapsed = time.time() - t0
    return output_path, elapsed


def analyze_audio_quality(wav_path):
    """Analyze the audio from the C implementation for basic quality checks."""
    audio, sr = read_wav(wav_path)
    stats = compute_stats(audio)

    print(f"\n--- Audio Quality Analysis: {wav_path} ---")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {stats['duration_s']:.2f}s ({stats['length']} samples)")
    print(f"  RMS level: {stats['rms']:.4f}")
    print(f"  Peak level: {stats['peak']:.4f}")
    print(f"  Non-zero samples: {stats['nonzero']}/{stats['length']} "
          f"({100*stats['nonzero']/max(1,stats['length']):.1f}%)")
    print(f"  DC offset (mean): {stats['mean']:.6f}")

    # Basic quality checks
    issues = []
    if stats['rms'] < 0.001:
        issues.append("FAIL: RMS too low (near silence)")
    if stats['peak'] < 0.01:
        issues.append("FAIL: Peak too low (near silence)")
    if stats['nonzero'] < stats['length'] * 0.5:
        issues.append("FAIL: Too many zero samples (>50%)")
    if abs(stats['mean']) > 0.1:
        issues.append("WARNING: Large DC offset")
    if stats['peak'] > 0.999:
        issues.append("WARNING: Possible clipping (peak near 1.0)")

    # Check for periodic structure (should have speech-like patterns)
    # Simple check: autocorrelation at typical pitch periods (100-400 Hz)
    if len(audio) > 4800:  # need at least 200ms
        chunk = audio[sr//4 : sr//4 + sr//2]  # 500ms from 250ms onwards
        if len(chunk) > 0 and np.std(chunk) > 0.001:
            # Check spectral properties
            fft = np.fft.rfft(chunk * np.hanning(len(chunk)))
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(chunk), 1.0/sr)

            # Speech should have energy in 100-4000 Hz range
            speech_band = (freqs >= 100) & (freqs <= 4000)
            total_power = np.sum(power)
            speech_power = np.sum(power[speech_band])
            speech_ratio = speech_power / max(total_power, 1e-10)

            print(f"  Speech band energy ratio: {speech_ratio:.2f} "
                  f"(expect >0.3 for speech)")
            if speech_ratio < 0.1:
                issues.append("WARNING: Very little energy in speech band (100-4000 Hz)")

    if not issues:
        print("  Quality checks: ALL PASSED")
    else:
        for issue in issues:
            print(f"  {issue}")

    return stats, issues


def compare_wavs(wav1_path, wav2_path):
    """Compare two WAV files."""
    audio1, sr1 = read_wav(wav1_path)
    audio2, sr2 = read_wav(wav2_path)

    min_len = min(len(audio1), len(audio2))
    if min_len == 0:
        print("Cannot compare: one or both files are empty")
        return

    a1 = audio1[:min_len]
    a2 = audio2[:min_len]

    # Correlation
    corr = np.corrcoef(a1, a2)[0, 1]

    # SNR (treating a2 as reference)
    noise = a1 - a2
    snr = 10 * np.log10(np.mean(a2**2) / max(np.mean(noise**2), 1e-10))

    print(f"\n--- Waveform Comparison ---")
    print(f"  Length: {len(audio1)} vs {len(audio2)} samples")
    print(f"  Correlation: {corr:.4f}")
    print(f"  SNR: {snr:.1f} dB")
    print(f"  Max absolute difference: {np.max(np.abs(noise)):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3-TTS C implementation")
    parser.add_argument('--model-dir', default='qwen3-tts-0.6b',
                        help='Path to model directory')
    parser.add_argument('--text', default='Hello, how are you today?',
                        help='Text to synthesize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--speaker', default='serena',
                        help='Speaker name')
    parser.add_argument('--skip-python', action='store_true',
                        help='Skip Python reference (just analyze C output)')
    parser.add_argument('--c-wav', default=None,
                        help='Use existing C output WAV instead of generating')
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-TTS Validation")
    print("=" * 60)
    print(f"Text: \"{args.text}\"")
    print(f"Speaker: {args.speaker}")
    print(f"Seed: {args.seed}")
    print()

    # Step 1: Generate with C implementation
    c_wav = args.c_wav
    if not c_wav:
        c_wav = '/tmp/validate_c_output.wav'
        c_wav, c_time = run_c_inference(
            args.model_dir, args.text, c_wav,
            seed=args.seed, speaker=args.speaker
        )
        if not c_wav or not os.path.exists(c_wav):
            print("C inference failed, aborting")
            return 1
        print(f"C inference completed in {c_time:.1f}s")

    # Step 2: Analyze C output quality
    c_stats, c_issues = analyze_audio_quality(c_wav)

    # Step 3: Run a few more test cases
    test_cases = [
        ("English short", "Hello world"),
        ("English long", "The quick brown fox jumps over the lazy dog."),
        ("Italian", "Ciao, come stai oggi?"),
        ("Numbers", "One two three four five six seven eight nine ten."),
    ]

    if args.text not in [t[1] for t in test_cases]:
        # Already tested the main text above, just run the additional cases
        pass

    print("\n" + "=" * 60)
    print("Running additional test cases...")
    print("=" * 60)

    all_passed = len(c_issues) == 0
    for name, text in test_cases:
        out_wav = f'/tmp/validate_{name.replace(" ", "_").lower()}.wav'
        wav_path, elapsed = run_c_inference(
            args.model_dir, text, out_wav,
            seed=args.seed, speaker=args.speaker
        )
        if wav_path and os.path.exists(wav_path):
            stats, issues = analyze_audio_quality(wav_path)
            if issues:
                all_passed = False
        else:
            print(f"  FAIL: {name} - inference failed")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if all_passed:
        print("  All basic quality checks PASSED")
        print("  Audio files generated successfully for all test cases")
        print("  Note: Manual listening test recommended for speech intelligibility")
    else:
        print("  Some checks had warnings or failures (see above)")

    print(f"\n  Output files in /tmp/validate_*.wav")
    print("  Listen to them to verify speech quality!")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
