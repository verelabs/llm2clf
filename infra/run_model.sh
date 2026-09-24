#!/usr/bin/env bash
# Runs on the GPU box: serves one model with SGLang, puts jevlocal in front, runs the accuracy and throughput benchmarks.
set -euo pipefail
MODEL=$1
NAME=$2
shift 2
cd ~/aeing
export PATH=$HOME/.local/bin:$PATH HF_HOME=/opt/hf

until grep -q "done $MODEL" ~/downloads.log; do sleep 10; done

docker rm -f sglang >/dev/null 2>&1 || true
docker run -d --name sglang --gpus all --ipc=host --shm-size 32g -p 127.0.0.1:30000:30000 -v /opt/hf:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path "$MODEL" --host 0.0.0.0 --port 30000 \
  --context-length 8192 --mem-fraction-static 0.85 "$@" >/dev/null
until curl -sf 127.0.0.1:30000/health >/dev/null; do
  docker ps -q -f name=sglang | grep -q . || { docker logs sglang 2>&1 | tail -30; exit 1; }
  sleep 5
done

pkill -f "jevlocal-serve" || true
HF_HUB_OFFLINE=1 nohup uv run jevlocal-serve --model-id "$MODEL" --name "$NAME" --port 8000 > ~/jevlocal-$NAME.log 2>&1 &
until curl -sf 127.0.0.1:8000/v1/models >/dev/null; do sleep 2; done

uv run python -m bench.run --name "$NAME" --url http://127.0.0.1:8000 --workers 32
uv run python -m bench.throughput --url http://127.0.0.1:8000 --out "bench/results/$NAME.throughput.json"
