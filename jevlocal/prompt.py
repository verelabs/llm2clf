"""Renders a (state, question, option order) into a chat prompt whose next token is the answer label."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from typing import Any, Sequence

SYSTEM = (
    "You are a decision function. You are given a state and one question about it. "
    "Answer the question as written, using the state and common knowledge. "
    "Reply with the answer label only: no words, no punctuation, no explanation."
)

CHOICE_LABELS = list(string.ascii_uppercase + string.ascii_lowercase)
SCORE_LABELS = [str(i) for i in range(10)]
NOUL_LABELS = ["Yes", "No"]
MAX_CHOICE_OPTIONS = len(CHOICE_LABELS)
MAX_SCORE_LEVELS = len(SCORE_LABELS)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class Question:
    kind: str
    instructions: Any
    options: tuple[str, ...] = ()
    descriptions: tuple[Any, ...] = ()

    @classmethod
    def from_api(cls, q: dict) -> "Question":
        kind = q.get("type")
        if kind not in ("noul", "choice", "score"):
            raise ValueError(f"unknown question type {kind!r}")
        if "instructions" not in q:
            raise ValueError("instructions is required")
        criteria = q.get("criteria")
        if kind == "noul":
            criteria = criteria or {}
            return cls(kind, q["instructions"], ("true", "false"), (criteria.get("true"), criteria.get("false")))
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError("choice criteria must map at least two options to descriptions")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice supports at most {MAX_CHOICE_OPTIONS} options")
            return cls(kind, q["instructions"], tuple(criteria), tuple(criteria.values()))
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score criteria must be a list of 2 to {MAX_SCORE_LEVELS} levels")
        return cls(kind, q["instructions"], tuple(str(i) for i in range(len(criteria))), tuple(criteria))

    @property
    def k(self) -> int:
        return len(self.options)

    def labels(self) -> list[str]:
        if self.kind == "noul":
            return NOUL_LABELS
        if self.kind == "score":
            return SCORE_LABELS[: self.k]
        return CHOICE_LABELS[: self.k]


def cyclic_orders(k: int, n: int) -> list[list[int]]:
    """Up to n cyclic shifts spread evenly (0, k/2, k/4, 3k/4, ...); order[j] is the option shown at position j."""
    shifts, seen, d = [0], {0}, 2
    while len(shifts) < min(k, n) and d <= 4 * k:
        for num in range(1, d, 2):
            s = int(k * num / d) % k
            if s not in seen and len(shifts) < min(k, n):
                seen.add(s)
                shifts.append(s)
        d *= 2
    return [[(j + s) % k for j in range(k)] for s in shifts]


def orders_for(q: Question, permutations: int) -> list[list[int]]:
    if q.kind == "score" or permutations <= 1:
        return [list(range(q.k))]
    if q.kind == "noul":
        return [[0, 1], [1, 0]]
    return cyclic_orders(q.k, permutations)


def render_state(state: Any) -> str:
    return "State:\n" + (as_text(state) or "(empty)") + "\n\n"


def render_question(q: Question, order: Sequence[int]) -> str:
    """Question text for one option order. Choice and score labels name positions; noul labels name the answer."""
    lines = ["Question:", as_text(q.instructions)]
    labels = q.labels()
    if q.kind == "noul":
        true_desc, false_desc = (as_text(d) for d in q.descriptions)
        if true_desc or false_desc:
            lines.append("")
            lines.append(f"Yes means: {true_desc or 'the answer is yes'}")
            lines.append(f"No means: {false_desc or 'the answer is no'}")
        first, second = (NOUL_LABELS[i] for i in order)
        lines += ["", f"Answer {first} or {second}."]
    elif q.kind == "score":
        lines += ["", "Levels, from lowest to highest:"]
        lines += [f"{labels[j]}. {as_text(q.descriptions[i])}" for j, i in enumerate(order)]
        lines += ["", "Answer with the level number only."]
    else:
        lines += ["", "Options:"]
        for j, i in enumerate(order):
            desc = as_text(q.descriptions[i])
            lines.append(f"{labels[j]}. {q.options[i]}" + (f": {desc}" if desc else ""))
        lines += ["", "Answer with the option letter only."]
    return "\n".join(lines)


def option_labels(q: Question, order: Sequence[int]) -> list[str]:
    """Label the model emits for each option (by option index) when options are shown in this order."""
    if q.kind == "noul":
        return list(NOUL_LABELS)
    labels = q.labels()
    out = [""] * q.k
    for position, option in enumerate(order):
        out[option] = labels[position]
    return out
