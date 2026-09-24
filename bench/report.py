"""Scores results/*.jsonl on the test split: accuracy, ECE, Brier, NLL, score MAE, at raw, L0 and L1 levels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jevlocal.calibration import apply_temperature, fit_temperature

RESULTS = Path(__file__).parent / "results"


def ece(probs: list[np.ndarray], gold: list[int], bins: int = 10) -> float:
    conf = np.array([p.max() for p in probs])
    right = np.array([p.argmax() == g for p, g in zip(probs, gold)], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - right[mask].mean())
    return float(total)


def metrics(probs: list[np.ndarray], gold: list[int], kind: str) -> dict:
    out = {
        "acc": float(np.mean([p.argmax() == g for p, g in zip(probs, gold)])),
        "ece": ece(probs, gold),
        "brier": float(np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(probs, gold)])),
        "nll": float(-np.mean([np.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)])),
    }
    if kind == "score":
        out["mae"] = float(np.mean([abs(np.dot(np.arange(len(p)), p) - g) for p, g in zip(probs, gold)]))
    return out


def levels(rows: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Per level, the probabilities each item gets. L1 temperatures are fitted per kind on the calib split."""
    served = "L0" if "first_order_probs" in rows[0] else "served"
    by_level = {served: {r["id"]: np.array(r["probs"]) for r in rows}}
    if served == "L0":
        by_level["raw"] = {r["id"]: np.array(r["first_order_probs"]) for r in rows}
    temps = {}
    for kind in {r["kind"] for r in rows}:
        calib = [r for r in rows if r["kind"] == kind and r["split"] == "calib"]
        temps[kind] = fit_temperature([by_level[served][r["id"]] for r in calib], [r["gold"] for r in calib])
    by_level["L1"] = {r["id"]: apply_temperature(by_level[served][r["id"]], temps[r["kind"]]) for r in rows}
    return by_level | {"_temps": temps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--json", help="write the full table here")
    args = parser.parse_args()
    names = args.names or sorted(p.stem for p in RESULTS.glob("*.jsonl"))

    table = []
    for name in names:
        rows = [json.loads(line) for line in (RESULTS / f"{name}.jsonl").read_text().splitlines()]
        by_level = levels(rows)
        temps = by_level.pop("_temps")
        tasks = defaultdict(list)
        for r in rows:
            if r["split"] == "test":
                tasks[r["task"]].append(r)
        for level, probs in by_level.items():
            for task, items in sorted(tasks.items()):
                m = metrics([probs[r["id"]] for r in items], [r["gold"] for r in items], items[0]["kind"])
                table.append({"model": name, "level": level, "task": task, "n": len(items),
                              "temperature": temps[items[0]["kind"]] if level == "L1" else None} | m)

    print(f"{'model':28} {'level':5} {'task':8} {'n':>4} {'acc':>6} {'ece':>6} {'brier':>6} {'nll':>6} {'mae':>6}")
    for t in table:
        mae = f"{t['mae']:6.3f}" if "mae" in t else "      "
        print(f"{t['model']:28} {t['level']:5} {t['task']:8} {t['n']:4d} {t['acc']:6.3f} {t['ece']:6.3f} {t['brier']:6.3f} {t['nll']:6.3f} {mae}")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
