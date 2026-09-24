#!/usr/bin/env bash
# Runs on the GPU box: installs uv, pulls the SGLang image and starts all model downloads in the background.
set -euo pipefail
cd ~/aeing
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=$HOME/.local/bin:$PATH
uv sync -q
sudo mkdir -p /opt/hf && sudo chown "$USER" /opt/hf
nohup bash -c 'for m in "$@"; do HF_HOME=/opt/hf uvx -q --from "huggingface_hub[hf_xet]" hf download "$m" >/dev/null && echo "done $m"; done' _ \
  google/gemma-4-31B-it Qwen/Qwen3.6-35B-A3B-FP8 openai/gpt-oss-120b Qwen/Qwen3.6-27B-FP8 > ~/downloads.log 2>&1 &
docker pull -q lmsysorg/sglang:latest
