"""Measures billable input tokens per second under load and converts it to dollars per million at a GPU hourly price."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from bench.data import AG_NEWS, MNLI, SST5, load


def jev_style_request(passage: str) -> dict:
    """One state with five questions of all three kinds, the shape Jev is built for."""
    return {"model": "local", "state": {"document": passage}, "questions": {
        "about_people": {"type": "noul", "instructions": "Is `document` mainly about a person or group of people?"},
        "has_numbers": {"type": "noul", "instructions": "Does `document` state any specific figure, date or quantity?"},
        "section": {"type": "choice", "instructions": "Which section of a news site would `document` fit best?", "criteria": AG_NEWS},
        "stance": {"type": "choice", "instructions": "How does `document` bear on the claim that the topic is widely known?", "criteria": MNLI},
        "tone": {"type": "score", "instructions": "What tone does `document` take?", "criteria": SST5},
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--gpu-dollars-per-hour", type=float, default=6.88)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--permutations", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--out")
    args = parser.parse_args()

    passages = [item["request"]["state"]["passage"] for item in load("boolq")]
    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=args.concurrency))
    results = []
    for perms in args.permutations:
        requests = [jev_style_request(p) | {"llm2clf": {"permutations": perms}} for p in passages]
        client.post(f"{args.url}/v1/systemone", json=requests[0]).raise_for_status()
        started = time.time()
        with ThreadPoolExecutor(args.concurrency) as pool:
            usages = list(pool.map(lambda r: client.post(f"{args.url}/v1/systemone", json=r).json()["usage"]["input_tokens"], requests))
        seconds = time.time() - started
        tps = sum(usages) / seconds
        row = {"permutations": perms, "requests": len(requests), "seconds": round(seconds, 2),
               "requests_per_second": round(len(requests) / seconds, 2), "input_tokens_per_second": round(tps),
               "mean_input_tokens": round(sum(usages) / len(usages)),
               "dollars_per_million_input": round(args.gpu_dollars_per_hour / 3600 / tps * 1e6, 4)}
        results.append(row)
        print(json.dumps(row))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
