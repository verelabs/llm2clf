"""Gold-labelled tasks shaped as System One requests, sampled deterministically from public datasets."""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

PARQUET_API = "https://datasets-server.huggingface.co/parquet"
DATA_DIR = Path(__file__).parent / "data"


def fetch_rows(dataset: str, config: str, split: str, n: int, seed: int = 7) -> list[dict]:
    import pyarrow.parquet as pq

    query = urllib.parse.urlencode(dict(dataset=dataset))
    with urllib.request.urlopen(f"{PARQUET_API}?{query}", timeout=60) as response:
        files = [f["url"] for f in json.load(response)["parquet_files"] if f["config"] == config and f["split"] == split]
    tables = []
    for url in files:
        local = DATA_DIR / "raw" / urllib.parse.quote(url, safe="")
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, local)
        tables.extend(pq.read_table(local).to_pylist())
    offsets = sorted(random.Random(seed).sample(range(len(tables)), n))
    return [tables[i] | {"_offset": i} for i in offsets]


def boolq(row):
    request = {"state": {"passage": row["passage"]},
               "questions": {"q": {"type": "noul", "instructions": f"Based on `passage`: {row['question']}?"}}}
    return request, 0 if row["answer"] else 1


MNLI = {"entailment": "`premise` shows that `hypothesis` is true",
        "neutral": "`premise` does not settle whether `hypothesis` is true or false",
        "contradiction": "`premise` shows that `hypothesis` is false"}


def mnli(row):
    request = {"state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
               "questions": {"q": {"type": "choice", "instructions": "How does `premise` bear on `hypothesis`?",
                                   "criteria": MNLI}}}
    return request, row["label"]


AG_NEWS = {"world": "International news, politics, conflicts, diplomacy",
           "sports": "Sports events, teams, athletes",
           "business": "Companies, markets, economy, finance",
           "science and technology": "Science, technology, computing, the internet, space, health research"}


def ag_news(row):
    request = {"state": {"article": row["text"]},
               "questions": {"q": {"type": "choice", "instructions": "Which section of a news site does `article` belong in?",
                                   "criteria": AG_NEWS}}}
    return request, row["label"]


SST5 = ["Very negative", "Negative", "Neutral or mixed", "Positive", "Very positive"]


def sst5(row):
    request = {"state": {"review_excerpt": row["text"]},
               "questions": {"q": {"type": "score", "instructions": "What sentiment does `review_excerpt` express about the movie?",
                                   "criteria": SST5}}}
    return request, row["label"]


TASKS = {
    "boolq": ("google/boolq", "default", "validation", boolq, "noul"),
    "mnli": ("nyu-mll/glue", "mnli", "validation_matched", mnli, "choice"),
    "ag_news": ("fancyzhx/ag_news", "default", "test", ag_news, "choice"),
    "sst5": ("SetFit/sst5", "default", "validation", sst5, "score"),
}


def load(task: str, n: int = 300) -> list[dict]:
    """Items {id, split, kind, request, gold}; the first half is the calibration split, the rest is test."""
    path = DATA_DIR / f"{task}-{n}.jsonl"
    if not path.exists():
        dataset, config, split, build, kind = TASKS[task]
        lines = []
        for i, row in enumerate(fetch_rows(dataset, config, split, n)):
            request, gold = build(row)
            lines.append(json.dumps({"id": f"{task}-{row['_offset']}", "split": "calib" if i % 2 == 0 else "test",
                                     "kind": kind, "request": request, "gold": gold}))
        path.write_text("\n".join(lines) + "\n")
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    for name in TASKS:
        items = load(name)
        print(name, len(items), items[0]["request"]["questions"]["q"]["type"])
