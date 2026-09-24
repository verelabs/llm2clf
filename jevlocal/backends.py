"""Backends score answer labels at the first answer position: one log-probability per option, summed over its spellings."""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Sequence

import httpx
import numpy as np

Prompt = tuple[str, str]
Spellings = list[list[str]]


class Backend(Protocol):
    def option_logprobs(self, prompts: Sequence[Prompt], spellings: Sequence[Spellings]) -> list[tuple[np.ndarray, int]]:
        """For each prompt, (per-option log-probability, prompt tokens processed)."""
        ...


class LocalBackend:
    """Shared by backends that see raw token ids: renders the chat template and maps spellings to single tokens."""

    def __init__(self, tokenizer, assistant_prefix: str | None = None):
        self.tokenizer = tokenizer
        self.assistant_prefix = self._detect_prefix() if assistant_prefix is None else assistant_prefix
        self._ids: dict[str, int | None] = {}

    def _detect_prefix(self) -> str:
        """Reasoning models in the harmony format must be pointed at their final channel to answer directly."""
        probe = self._template([{"role": "user", "content": "x"}])
        return "<|channel|>final<|message|>" if probe.rstrip().endswith("<|start|>assistant") else ""

    def _template(self, messages: list[dict]) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def render(self, prompt: Prompt) -> str:
        system, user = prompt
        return self._template([{"role": "system", "content": system}, {"role": "user", "content": user}]) + self.assistant_prefix

    def token_id(self, spelling: str) -> int | None:
        if spelling not in self._ids:
            toks = self.tokenizer.encode(spelling, add_special_tokens=False)
            self._ids[spelling] = toks[0] if len(toks) == 1 else None
        return self._ids[spelling]

    def option_ids(self, spellings: Spellings) -> list[list[int]]:
        groups = [sorted({i for s in option if (i := self.token_id(s)) is not None}) for option in spellings]
        flat = [i for g in groups for i in g]
        if any(not g for g in groups) or len(set(flat)) != len(flat):
            raise ValueError(f"answer labels are not distinct single tokens: {[o[0] for o in spellings]}")
        return groups

    def _score(self, texts: list[str], groups: list[list[list[int]]]) -> list[tuple[np.ndarray, int]]:
        raise NotImplementedError

    def option_logprobs(self, prompts, spellings):
        return self._score([self.render(p) for p in prompts], [self.option_ids(s) for s in spellings])


def combine(logprob_of: dict[int, float], groups: list[list[int]]) -> np.ndarray:
    return np.array([np.logaddexp.reduce([logprob_of.get(i, -np.inf) for i in g]) for g in groups])


class SGLangBackend(LocalBackend):
    """Talks to a running SGLang server; its radix cache reuses the shared state prefix across questions."""

    def __init__(self, tokenizer, url: str = "http://127.0.0.1:30000", timeout: float = 600.0):
        super().__init__(tokenizer)
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def _score(self, texts, groups):
        body = {
            "text": texts,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "token_ids_logprob": [[i for g in option_groups for i in g] for option_groups in groups],
            "logprob_start_len": -1,
        }
        response = self.client.post(f"{self.url}/generate", json=body)
        response.raise_for_status()
        items = response.json()
        items = [items] if isinstance(items, dict) else items
        out = []
        for item, option_groups in zip(items, groups):
            first = item["meta_info"]["output_token_ids_logprobs"][0]
            out.append((combine({int(tok): lp for lp, tok, *_ in first}, option_groups), item["meta_info"]["prompt_tokens"]))
        return out


class HFBackend(LocalBackend):
    """Plain transformers forward pass, for local development and tests on small models."""

    def __init__(self, model_id: str, device: str | None = None, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        super().__init__(tokenizer)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def _score(self, texts, groups):
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            with self.lock, self.torch.no_grad():
                enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                logits = self.model(**enc).logits[:, -1, :].float()
                logprobs = self.torch.log_softmax(logits, dim=-1).cpu().numpy()
                lengths = enc["attention_mask"].sum(dim=1).tolist()
            for row, option_groups, length in zip(logprobs, groups[start : start + self.batch_size], lengths):
                out.append((combine(dict(enumerate(row.tolist())), option_groups), int(length)))
        return out


class BedrockBackend:
    """Amazon Bedrock chat models via InvokeModel with top-20 logprobs; labels outside the top 20 get a floor below the lowest seen."""

    TOP = 20

    def __init__(self, model_id: str, region: str = "us-west-2", workers: int = 8):
        import boto3
        from botocore.config import Config

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region,
                                   config=Config(retries={"max_attempts": 12, "mode": "adaptive"}, max_pool_connections=64))
        self.harmony = "gpt-oss" in model_id
        self.pool = ThreadPoolExecutor(workers)

    def _call(self, prompt: Prompt, spellings: Spellings) -> tuple[np.ndarray, int]:
        system, user = prompt
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": 400 if self.harmony else 1, "temperature": 0, "logprobs": True, "top_logprobs": self.TOP}
        if self.harmony:
            body["reasoning_effort"] = "low"
        response = json.loads(self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))["body"].read())
        tokens = response["choices"][0]["logprobs"]["content"]
        position = self._answer_position(tokens)
        if position is None:
            return np.zeros(len(spellings)) - math.log(len(spellings)), response["usage"]["prompt_tokens"]
        top: dict[str, float] = {}
        for t in tokens[position]["top_logprobs"]:
            top[t["token"]] = np.logaddexp(top.get(t["token"], -np.inf), t["logprob"])
        floor = min(top.values()) - 1.0
        scores = np.array([np.logaddexp.reduce([top.get(s, -np.inf) for s in option]) for option in spellings])
        return np.where(np.isfinite(scores), scores, floor), response["usage"]["prompt_tokens"]

    def _answer_position(self, tokens: list[dict]) -> int | None:
        if not self.harmony:
            return 0
        for i in range(len(tokens) - 2):
            if tokens[i]["token"] == "final" and tokens[i + 1]["token"] == "<|message|>":
                return i + 2
        return None

    def option_logprobs(self, prompts, spellings):
        return list(self.pool.map(self._call, prompts, spellings))
