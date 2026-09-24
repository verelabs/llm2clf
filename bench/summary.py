"""One comparison table: accuracy and calibration error per task at the best label-free and labelled levels, plus cost per 1,000 questions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.report import RESULTS, levels, metrics

PRICE_PER_MILLION_INPUT = {
    "jev-1.13.0": 0.042, "gpt-oss-120b": 0.15, "qwen3-235b": 0.22, "mistral-large-3": 0.50,
    "kimi-k2.5": 0.60, "deepseek-v3.2": 0.62, "glm-5": 1.00,
}
TASKS = ["boolq", "mnli", "ag_news", "sst5"]


def summarize(name: str) -> dict:
    rows = [json.loads(line) for line in (RESULTS / f"{name}.jsonl").read_text().splitlines()]
    by_level = levels(rows)
    by_level.pop("_temps")
    free = "L0" if "L0" in by_level else "served"
    test = defaultdict(list)
    for r in rows:
        if r["split"] == "test":
            test[r["task"]].append(r)
    out = {"model": name, "n": len(rows)}
    for level in (free, "L1"):
        for task in TASKS:
            items = test.get(task, [])
            if items:
                m = metrics([by_level[level][r["id"]] for r in items], [r["gold"] for r in items], items[0]["kind"])
                out[f"{level if level == 'L1' else 'free'}:{task}"] = m
    tokens = np.mean([r["input_tokens"] for r in rows])
    price = PRICE_PER_MILLION_INPUT.get(name)
    out["tokens_per_question"] = float(tokens)
    out["dollars_per_1k_questions"] = float(tokens * price / 1e3) if price else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    names = args.names or sorted(p.stem for p in RESULTS.glob("*.jsonl"))
    summaries = [summarize(n) for n in names]

    print("Accuracy on the test split (label-free level; Jev as served)")
    print("| model | BoolQ | MNLI | AG News | SST-5 | mean |")
    print("|---|---|---|---|---|---|")
    for s in summaries:
        accs = [s.get(f"free:{t}", {}).get("acc") for t in TASKS]
        cells = [f"{a:.1%}" if a is not None else "-" for a in accs]
        mean = np.mean([a for a in accs if a is not None])
        print(f"| {s['model']} | " + " | ".join(cells) + f" | {mean:.1%} |")

    print("\nCalibration error (ECE, lower is better): label-free / after temperature fitted on 150 labels per task")
    print("| model | BoolQ | MNLI | AG News | SST-5 |")
    print("|---|---|---|---|---|")
    for s in summaries:
        cells = []
        for t in TASKS:
            free, l1 = s.get(f"free:{t}"), s.get(f"L1:{t}")
            cells.append(f"{free['ece']:.3f} / {l1['ece']:.3f}" if free else "-")
        print(f"| {s['model']} | " + " | ".join(cells) + " |")

    print("\nCost")
    print("| model | $ per 1M input | input tokens per question | $ per 1,000 questions |")
    print("|---|---|---|---|")
    for s in summaries:
        price = PRICE_PER_MILLION_INPUT.get(s["model"])
        cost = f"${s['dollars_per_1k_questions']:.3f}" if s["dollars_per_1k_questions"] is not None else "-"
        print(f"| {s['model']} | {f'${price}' if price else '-'} | {s['tokens_per_question']:.0f} | {cost} |")


if __name__ == "__main__":
    main()
