"""Runs every benchmark item against Jev or a jevlocal server and appends per-item probabilities to results/<name>.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bench.data import TASKS, load

RESULTS = Path(__file__).parent / "results"


def option_probs(item: dict, answer: dict) -> list[float]:
    q = item["request"]["questions"]["q"]
    if answer["type"] == "noul":
        return [answer["noul"], 1 - answer["noul"]]
    if answer["type"] == "choice":
        return [answer["probabilities"][o] for o in q["criteria"]]
    return [answer["probabilities"][str(i)] for i in range(len(q["criteria"]))]


def jev_caller(model: str):
    sys.path.insert(0, str(Path.home() / ".claude/skills/jev"))
    os.environ.setdefault("JEV_CACHE", str(RESULTS / "jev-cache"))
    from client import ask

    def call(item):
        response = ask(item["request"]["state"], item["request"]["questions"], model=model)
        answer = response["answers"]["q"]
        return {"probs": option_probs(item, answer), "input_tokens": response["usage"]["input_tokens"]}

    return call


def local_caller(url: str, permutations: int):
    client = httpx.Client(timeout=600)

    def call(item):
        body = item["request"] | {"model": "local", "jevlocal": {"raw": True, "calibrate": False, "permutations": permutations}}
        response = client.post(f"{url.rstrip('/')}/v1/systemone", json=body)
        response.raise_for_status()
        data = response.json()
        answer = data["answers"]["q"]
        return {"probs": answer["raw_probabilities"], "first_order_probs": answer["order_probabilities"][0],
                "input_tokens": data["usage"]["input_tokens"]}

    return call


def bedrock_caller(model_id: str, region: str, permutations: int):
    from jevlocal.backends import BedrockBackend
    from jevlocal.engine import Engine

    engine = Engine(BedrockBackend(model_id, region=region), model_id)

    def call(item):
        data = engine.evaluate(item["request"], permutations=permutations, calibrate=False, raw=True)
        answer = data["answers"]["q"]
        return {"probs": answer["raw_probabilities"], "first_order_probs": answer["order_probabilities"][0],
                "input_tokens": data["usage"]["input_tokens"]}

    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--jev", help="Jev model version, e.g. jev-1.13.0")
    parser.add_argument("--url", help="jevlocal server URL")
    parser.add_argument("--bedrock", help="Bedrock model id, run in-process")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.jev:
        call = jev_caller(args.jev)
    elif args.bedrock:
        call = bedrock_caller(args.bedrock, args.region, args.permutations)
    else:
        call = local_caller(args.url, args.permutations)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    items = [dict(item, task=task) for task in args.tasks.split(",") for item in load(task) if item["id"] not in done]
    lock = threading.Lock()
    started = time.time()

    def run(item):
        result = call(item)
        record = {k: item[k] for k in ("id", "task", "split", "kind", "gold")} | result
        with lock, out.open("a") as f:
            f.write(json.dumps(record) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, items))
    print(f"{args.name}: {len(items)} items in {time.time() - started:.1f}s ({len(done)} cached)")


if __name__ == "__main__":
    main()
