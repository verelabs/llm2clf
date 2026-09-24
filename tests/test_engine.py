import numpy as np
from jevlocal.calibration import apply_temperature, fit_temperature
from jevlocal.engine import Engine
from jevlocal.prompt import Question, cyclic_orders, option_labels

class FakeBackend:
    """Puts all probability on whichever option is labelled on the prompt line that mentions `target_text`."""

    def __init__(self, target_text):
        self.target_text = target_text

    def option_logprobs(self, prompts, spellings):
        out = []
        for (_, user), options in zip(prompts, spellings):
            label = next(l for l in user.splitlines() if self.target_text in l).split(".")[0]
            out.append((np.array([0.0 if label in option else -20.0 for option in options]), 10))
        return out


def test_choice_follows_option_across_permutations():
    engine = Engine(FakeBackend("technical"), "t", permutations=3)
    request = {"state": "x", "questions": {"d": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}}}}
    result = engine.evaluate(request)
    assert result["answers"]["d"]["choice"] == "technical"
    assert result["answers"]["d"]["probabilities"]["technical"] > 0.99
    assert result["usage"]["input_tokens"] == 30


def test_option_labels_map_positions_back_to_options():
    q = Question.from_api({"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None, "c": None}})
    assert option_labels(q, [2, 0, 1]) == ["B", "C", "A"]
    assert sorted(map(tuple, cyclic_orders(4, 4))) == sorted(tuple((j + s) % 4 for j in range(4)) for s in range(4))


def test_temperature_softens_overconfident_probs():
    probs = [np.array([0.99, 0.01])] * 50 + [np.array([0.01, 0.99])] * 50
    labels = [0] * 35 + [1] * 15 + [1] * 35 + [0] * 15
    t = fit_temperature(probs, labels)
    assert t > 1
    assert abs(apply_temperature(np.array([0.99, 0.01]), t)[0] - 0.7) < 0.05


def test_validation_rejects_bad_questions():
    import pytest

    with pytest.raises(ValueError):
        Question.from_api({"type": "choice", "instructions": "q", "criteria": {"only": None}})
    with pytest.raises(ValueError):
        Question.from_api({"type": "score", "instructions": "q", "criteria": ["x"] * 11})
