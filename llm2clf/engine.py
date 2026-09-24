"""Turns a System One request into prompts, reads label probabilities from a backend, and builds Jev-shaped answers."""

from __future__ import annotations

from typing import Any

import numpy as np

from llm2clf.backends import Backend
from llm2clf.calibration import Calibration, log_softmax
from llm2clf.prompt import NOUL_LABELS, SYSTEM, Question, option_labels, orders_for, render_question, render_state


def confidence(probs: np.ndarray) -> float:
    k = len(probs)
    return float(np.clip((k * probs.max() - 1) / (k - 1), 0.0, 1.0))


def spellings(label: str) -> list[str]:
    """Ways a model may emit a label: bare or space-prefixed, and any casing for Yes/No."""
    forms = [label, label.lower(), label.upper(), label.capitalize()] if label in NOUL_LABELS else [label]
    return list(dict.fromkeys(f for form in forms for f in (form, " " + form)))


class Engine:
    def __init__(self, backend: Backend, model_name: str, calibration: Calibration | None = None, permutations: int = 4):
        self.backend = backend
        self.model_name = model_name
        self.calibration = calibration or Calibration()
        self.permutations = permutations

    def evaluate(self, request: dict, permutations: int | None = None, calibrate: bool = True, raw: bool = False) -> dict:
        questions = {qid: Question.from_api(q) for qid, q in request["questions"].items()}
        permutations = self.permutations if permutations is None else permutations
        state_text = render_state(request.get("state"))

        prompts, option_spellings, owners = [], [], []
        for qid, q in questions.items():
            for order in orders_for(q, permutations):
                prompts.append((SYSTEM, state_text + render_question(q, order)))
                option_spellings.append([spellings(label) for label in option_labels(q, order)])
                owners.append(qid)

        results = self.backend.option_logprobs(prompts, option_spellings)

        per_question: dict[str, list[np.ndarray]] = {qid: [] for qid in questions}
        for qid, (option_lp, _) in zip(owners, results):
            per_question[qid].append(log_softmax(option_lp))

        answers = {}
        for qid, q in questions.items():
            probs = np.exp(log_softmax(np.mean(per_question[qid], axis=0)))
            answers[qid] = self._answer(q, self.calibration.apply(q.kind, probs) if calibrate else probs)
            if raw:
                answers[qid]["raw_probabilities"] = probs.round(6).tolist()
                answers[qid]["order_probabilities"] = [np.exp(v).round(6).tolist() for v in per_question[qid]]

        usage = {"input_tokens": int(sum(tokens for _, tokens in results)), "output_tokens": 0}
        return {"model": self.model_name, "answers": answers, "usage": usage}

    def _answer(self, q: Question, probs: np.ndarray) -> dict[str, Any]:
        if q.kind == "noul":
            return {"type": "noul", "noul": round(float(probs[0]), 6)}
        rounded = {option: round(float(p), 6) for option, p in zip(q.options, probs)}
        if q.kind == "choice":
            return {"type": "choice", "choice": q.options[int(probs.argmax())], "probabilities": rounded,
                    "confidence": round(confidence(probs), 6)}
        return {"type": "score", "score": round(float(np.dot(np.arange(q.k), probs)), 6),
                "legend": {str(i): str(d) for i, d in enumerate(q.descriptions)},
                "probabilities": rounded, "confidence": round(confidence(probs), 6)}
