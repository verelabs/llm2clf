# llm2clf

Turn any open-weights LLM into a calibrated classifier. Ask yes/no, multiple-choice or rating questions and get back a label with probabilities, read straight from the model's next-token distribution. Nothing is generated, so output tokens cost nothing.

Inspired by TypeSafe's [Jev](https://docs.typesafe.ai). The API matches its `/v1/systemone` (Noul, Choice, Score), so existing clients work by changing the URL.

Write-up: [Turning open LLMs into calibrated classifiers](https://enclave.md/d/qcC763bnAVdDWEea)

## How it works

- Options are shown as `A.`, `B.`, ... (levels as `0.`, `1.`, ...; yes/no as `Yes`/`No`), and the probability of each label is read at the first answer token.
- **Debiasing, no labels needed:** options are shown in up to 4 orders and averaged per option, which cancels position bias.
- **Calibration, with labels:** one temperature per question type, fitted on a labelled set and loaded with `--calibration`.

Debiasing and calibration code is ported from [AnyJev](https://github.com/nokia-applied-research/AnyJev) (Apache-2.0).

## Run

Any open model on Amazon Bedrock:

```bash
uv sync
uv run llm2clf-serve --backend bedrock --model-id qwen.qwen3-235b-a22b-2507-v1:0
```

A local model for development:

```bash
uv sync --extra hf
uv run llm2clf-serve --backend hf --model-id Qwen/Qwen3-0.6B
```

There is also an SGLang backend for self-hosting on a GPU (`--sglang-url`); it has not been benchmarked yet.

## Results

Six open models on Bedrock against Jev as a purpose-built classifier baseline. Four public tasks: BoolQ (yes/no), MNLI and AG News (choice), SST-5 (5-level rating). 150 test items each; gaps under about 4 points are within noise.

Accuracy, no labels used. Sizes are from each model card; TypeSafe has not published Jev's size.

| Model | Size (total / active) | BoolQ | MNLI | AG News | SST-5 | Mean |
|---|---|---|---|---|---|---|
| Kimi K2.5 | 1T / 32B | 94.0% | 89.3% | 88.7% | 57.3% | 82.3% |
| Qwen3-235B | 235B / 22B | 94.0% | 86.7% | 89.3% | 57.3% | 81.8% |
| GLM-5 | 744B / 40B | 91.3% | 82.0% | 90.0% | 60.7% | 81.0% |
| Jev (baseline) | - | 94.0% | 79.3% | 90.0% | 59.3% | 80.7% |
| Mistral Large 3 | 675B / 41B | 88.7% | 85.3% | 89.3% | 56.7% | 80.0% |
| DeepSeek V3.2 | 671B / 37B | 90.0% | 79.3% | 90.7% | 53.3% | 78.3% |
| gpt-oss-120b | 117B / 5.1B | 92.7% | 76.7% | 87.3% | 56.7% | 78.3% |

Calibration error (ECE, lower is better), before / after fitting a temperature on 150 labels:

| Model | BoolQ | MNLI | AG News | SST-5 |
|---|---|---|---|---|
| Jev (baseline) | 0.031 / 0.017 | 0.090 / 0.108 | 0.080 / 0.089 | 0.167 / 0.134 |
| Qwen3-235B | 0.063 / 0.009 | 0.103 / 0.029 | 0.088 / 0.067 | 0.348 / 0.060 |
| Kimi K2.5 | 0.049 / 0.027 | 0.078 / 0.045 | 0.096 / 0.060 | 0.275 / 0.072 |
| GLM-5 | 0.071 / 0.037 | 0.135 / 0.068 | 0.089 / 0.037 | 0.299 / 0.054 |
| Mistral Large 3 | 0.113 / 0.052 | 0.140 / 0.047 | 0.107 / 0.025 | 0.394 / 0.053 |
| DeepSeek V3.2 | 0.084 / 0.042 | 0.112 / 0.062 | 0.089 / 0.071 | 0.359 / 0.103 |
| gpt-oss-120b | 0.073 / 0.009 | 0.230 / 0.122 | 0.121 / 0.031 | 0.433 / 0.036 |

Cost:

| Model | Size (total / active) | $ per 1M input tokens | $ per 1,000 questions |
|---|---|---|---|
| Jev (baseline) | - | $0.042 | $0.017 |
| gpt-oss-120b | 117B / 5.1B | $0.15 | $0.094 |
| Qwen3-235B | 235B / 22B | $0.22 | $0.108 |
| Mistral Large 3 | 675B / 41B | $0.50 | $0.238 |
| DeepSeek V3.2 | 671B / 37B | $0.62 | $0.289 |
| Kimi K2.5 | 1T / 32B | $0.60 | $0.291 |
| GLM-5 | 744B / 40B | $1.00 | $0.471 |

On Bedrock every question resends the full state, so a request with 5 questions about one passage uses 4 to 5 times the input tokens Jev bills. Self-hosting with the SGLang backend should remove that overhead, because its prefix cache computes the state once for all questions. We project this would be much cheaper than Bedrock, but it is not measured yet, and it only holds if the GPU stays busy. It also depends on model size: the most accurate models above need a multi-GPU machine.

## Reproduce

```bash
uv run python -m bench.run --name qwen3-235b --bedrock qwen.qwen3-235b-a22b-2507-v1:0
uv run python -m bench.summary
```

Per-item results are in `bench/results/`.
