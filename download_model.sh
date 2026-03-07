#!/bin/bash
# Download Qwen3-TTS model files from HuggingFace.
#
# Usage:
#   ./download_model.sh
#   ./download_model.sh --model small
#   ./download_model.sh --model large --dir my-model-dir
#
# Options:
#   --model small|large   Choose 0.6B (small) or 1.7B (large)
#   --dir DIR             Override output directory

set -e

MODEL_CHOICE=""
MODEL_DIR=""

usage() {
    echo "Usage: $0 [--model small|large] [--dir DIR]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_CHOICE="$2"
            shift 2
            ;;
        --dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

choose_model_interactive() {
    echo "Select model size to download:"
    echo "  1) small (Qwen3-TTS-12Hz-0.6B-CustomVoice)"
    echo "  2) large (Qwen3-TTS-12Hz-1.7B-CustomVoice)"
    echo ""
    while true; do
        read -r -p "Enter choice [1/2]: " ans
        case "$ans" in
            1|small|Small|SMALL)
                MODEL_CHOICE="small"
                return
                ;;
            2|large|Large|LARGE)
                MODEL_CHOICE="large"
                return
                ;;
            *)
                echo "Please choose 1 (small) or 2 (large)."
                ;;
        esac
    done
}

if [[ -z "$MODEL_CHOICE" ]]; then
    choose_model_interactive
fi

case "$MODEL_CHOICE" in
    small|0.6b|0.6B)
        MODEL_ID="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-tts-0.6b"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "tokenizer_config.json"
            "preprocessor_config.json"
            "model.safetensors"
            "vocab.json"
            "merges.txt"
        )
        SPEECH_TOKENIZER_FILES=(
            "config.json"
            "configuration.json"
            "model.safetensors"
            "preprocessor_config.json"
        )
        ;;
    large|1.7b|1.7B)
        MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-tts-1.7b"; fi
        FILES=(
            "config.json"
            "generation_config.json"
            "tokenizer_config.json"
            "preprocessor_config.json"
            "model.safetensors"
            "vocab.json"
            "merges.txt"
        )
        SPEECH_TOKENIZER_FILES=(
            "config.json"
            "configuration.json"
            "model.safetensors"
            "preprocessor_config.json"
        )
        ;;
    *)
        echo "Invalid --model value: $MODEL_CHOICE"
        echo "Use: --model small or --model large"
        exit 1
        ;;
esac

echo "Downloading ${MODEL_ID} to ${MODEL_DIR}/"
echo ""

mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}/speech_tokenizer"

BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

echo "=== Main model files ==="
for file in "${FILES[@]}"; do
    dest="${MODEL_DIR}/${file}"
    if [[ -f "${dest}" ]]; then
        echo "  [skip] ${file} (already exists)"
    else
        echo "  [download] ${file}..."
        curl -fL -o "${dest}" "${BASE_URL}/${file}" --progress-bar
        echo "  [done] ${file}"
    fi
done

echo ""
echo "=== Speech tokenizer files ==="
for file in "${SPEECH_TOKENIZER_FILES[@]}"; do
    dest="${MODEL_DIR}/speech_tokenizer/${file}"
    if [[ -f "${dest}" ]]; then
        echo "  [skip] speech_tokenizer/${file} (already exists)"
    else
        echo "  [download] speech_tokenizer/${file}..."
        curl -fL -o "${dest}" "${BASE_URL}/speech_tokenizer/${file}" --progress-bar
        echo "  [done] speech_tokenizer/${file}"
    fi
done

echo ""
echo "Download complete. Files in ${MODEL_DIR}/"
ls -lh "${MODEL_DIR}/"
echo ""
echo "Speech tokenizer files:"
ls -lh "${MODEL_DIR}/speech_tokenizer/"
