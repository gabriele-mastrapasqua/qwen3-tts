#!/usr/bin/env bash
set -uo pipefail
cd ~/qwen-ft/docker
echo "### building qwen-ft image..."
docker build -t qwen-ft:latest . 2>&1 | tail -15
echo "### smoke: torch+torchaudio+qwen-tts"
docker run --rm --gpus all qwen-ft:latest python3 -c "
import torch, torchaudio
print(\"torch\", torch.__version__, \"ta\", torchaudio.__version__)
print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\")
from qwen_tts import Qwen3TTSTokenizer
print(\"qwen-tts import OK\")
x=torch.randn(2048,2048,device=\"cuda\"); torch.cuda.synchronize(); print(\"gpu matmul OK\")
" 2>&1 | tail -8
echo "### BUILD_DONE"
