#!/bin/bash
# Download Qwen3-TTS model files from HuggingFace.
#
# Usage:
#   ./download_model.sh
#   ./download_model.sh --model small
#   ./download_model.sh --model large --dir my-model-dir
#   ./download_model.sh --model voice-design
#   ./download_model.sh --model base-small
#   ./download_model.sh --model base-large
#
# Options:
#   --model small|large|voice-design|base-small|base-large
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
    echo "  3) voice-design (Qwen3-TTS-12Hz-1.7B-VoiceDesign)"
    echo "  4) base-small (Qwen3-TTS-12Hz-0.6B-Base, for voice cloning)"
    echo "  5) base-large (Qwen3-TTS-12Hz-1.7B-Base, for voice cloning)"
    echo ""
    while true; do
        read -r -p "Enter choice [1/2/3/4/5]: " ans
        case "$ans" in
            1|small|Small|SMALL)
                MODEL_CHOICE="small"
                return
                ;;
            2|large|Large|LARGE)
                MODEL_CHOICE="large"
                return
                ;;
            3|voice-design|VoiceDesign)
                MODEL_CHOICE="voice-design"
                return
                ;;
            4|base-small)
                MODEL_CHOICE="base-small"
                return
                ;;
            5|base-large)
                MODEL_CHOICE="base-large"
                return
                ;;
            *)
                echo "Please choose 1-5."
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
    voice-design|voicedesign|VoiceDesign)
        MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-tts-voice-design"; fi
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
    base-small|base-0.6b|base-0.6B)
        MODEL_ID="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-tts-0.6b-base"; fi
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
    base-large|base-1.7b|base-1.7B)
        MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        if [[ -z "$MODEL_DIR" ]]; then MODEL_DIR="qwen3-tts-1.7b-base"; fi
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
        echo "Use: --model small|large|voice-design|base-small|base-large"
        exit 1
        ;;
esac

echo "Downloading ${MODEL_ID} to ${MODEL_DIR}/"
echo ""

mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}/speech_tokenizer"

BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

# Scarica in modo ATOMICO: curl scrive su <dest>.part e SOLO un curl riuscito
# rinomina in <dest>.
#
# PERCHE' NON BASTA `curl -o "$dest"`, ed e' il motivo per cui questa funzione esiste:
# lo "[skip] (already exists)" qui sotto e' sicuro solo se la presenza del file implica
# la sua COMPLETEZZA. Scrivendo diretto sulla destinazione, un download interrotto —
# Ctrl-C, rete che cade, box spento, timeout ssh — lascia un model.safetensors TRONCATO
# al posto giusto. Ogni run successivo lo salta dicendo "already exists", il modello si
# carica senza un errore, e i pesi sono spazzatura a meta' file: il sintomo arriva molto
# piu' tardi, come audio sbagliato, e non come fallimento del download. Con .part+mv un
# file interrotto non e' MAI scambiabile per un file valido.
#
# `-C -` riprende un .part parziale invece di ributtare via i GB gia' scesi, e viene
# passato solo se il .part c'e' davvero (curl con -C - su un file assente e' legale ma
# inutile, e cosi' l'intento resta leggibile).
fetch() {
    local dest="$1" url="$2" label="$3"
    if [[ -f "${dest}" ]]; then
        echo "  [skip] ${label} (already exists)"
        return 0
    fi
    local resume=()
    [[ -f "${dest}.part" ]] && { resume=(-C -); echo "  [resume] ${label} (ripresa di un .part parziale)"; }
    echo "  [download] ${label}..."
    if curl -fL "${resume[@]}" -o "${dest}.part" "${url}" --progress-bar; then
        mv -f "${dest}.part" "${dest}"
        echo "  [done] ${label}"
    else
        echo "  🚨 download FALLITO: ${label} — il parziale resta in ${dest}.part (rilancia per riprenderlo)"
        return 1
    fi
}

echo "=== Main model files ==="
for file in "${FILES[@]}"; do
    fetch "${MODEL_DIR}/${file}" "${BASE_URL}/${file}" "${file}"
done

echo ""
echo "=== Speech tokenizer files ==="
for file in "${SPEECH_TOKENIZER_FILES[@]}"; do
    fetch "${MODEL_DIR}/speech_tokenizer/${file}" \
          "${BASE_URL}/speech_tokenizer/${file}" "speech_tokenizer/${file}"
done

echo ""
echo "Download complete. Files in ${MODEL_DIR}/"
ls -lh "${MODEL_DIR}/"
echo ""
echo "Speech tokenizer files:"
ls -lh "${MODEL_DIR}/speech_tokenizer/"
