# Can an open model be Jev? We built a clone and measured it

TypeSafe's Jev is a strange and useful kind of model. You don't chat with it. You hand it a state and a few typed questions (a yes/no Noul, a Choice among options, a Score on a scale), and it hands back probabilities your code can threshold. It is fast, calibrated, and very cheap: $0.042 per million input tokens, with output free.

We wanted to know whether any open-weights model could be turned into the same thing. So we built jevlocal, a harness that serves Jev's exact API on top of open models, and benchmarked six of the strongest against Jev on the same gold-labelled data.

The short answer: the best open models match Jev on accuracy and beat it on calibration, but through a hosted API they cost 6 to 28 times more per question.

## The trick: read probabilities, never generate

Every language model already computes a probability for every possible next token. If you ask a question and list the options as `A.`, `B.`, `C.`, the probability the model puts on the token `A` at the first answer position is its belief that A is right. Normalize over the option labels and you have a Choice answer. Yes and No give a Noul. Level numbers `0` to `9` give a Score, and the probability-weighted level gives the score value.

Nothing is generated, so there are no output tokens to pay for. One prefill per question is the whole cost.

Two refinements matter:

- **Debiasing without labels.** Models prefer some positions: listing the same options in a different order can flip the answer. We show each choice in up to four rotated orders and average the log-probabilities per option, which cancels the position bias (Zheng et al., 2024). On a tiny 0.6B model this was worth 11 points of accuracy. On the large models it was worth 0 to 3.
- **Calibration with a few labels.** Raw probabilities from chat models are overconfident. A single temperature per question type, fitted on 150 labelled examples, fixes most of it (Guo et al., 2017).

The debiasing and calibration code is ported from [AnyJev](https://github.com/nokia-applied-research/AnyJev) (Apache-2.0).

## Somebody is already doing this

Two projects appeared in the last two weeks. [openjev-sglang](https://github.com/ekzhang/openjev-sglang) serves the same API on SGLang, but its probabilities are uncalibrated, it is tied to one model, and it has no license. AnyJev has good calibration but no server and only a Hugging Face backend. Nobody had published an independent check of these clones against Jev itself, so that became the point of this work.

## The benchmark

Four public tasks, 300 items each, split evenly into calibration and test halves:

- **BoolQ** (Noul): does this passage answer yes to the question?
- **MNLI** (Choice): does the premise support, contradict, or say nothing about the hypothesis? This is the closest to how Jev is used for grounding claims against evidence.
- **AG News** (Choice): which of four news sections an article belongs to.
- **SST-5** (Score): movie review sentiment on five levels.

The open models ran on Amazon Bedrock, which returns the top 20 token probabilities. That is enough, since the answer labels are almost always among them. Jev ran through its own API. The whole run cost about $1.80 on Bedrock and a few cents on Jev.

## Results

Accuracy on the test half, with no labels used:

| Model | BoolQ | MNLI | AG News | SST-5 | Mean |
|---|---|---|---|---|---|
| Kimi K2.5 | 94.0% | 89.3% | 88.7% | 57.3% | 82.3% |
| Qwen3-235B | 94.0% | 86.7% | 89.3% | 57.3% | 81.8% |
| GLM-5 | 91.3% | 82.0% | 90.0% | 60.7% | 81.0% |
| **Jev 1.13** | 94.0% | 79.3% | 90.0% | 59.3% | 80.7% |
| Mistral Large 3 | 88.7% | 85.3% | 89.3% | 56.7% | 80.0% |
| DeepSeek V3.2 | 90.0% | 79.3% | 90.7% | 53.3% | 78.3% |
| gpt-oss-120b | 92.7% | 76.7% | 87.3% | 56.7% | 78.3% |

With 150 items per task, gaps of 3 or 4 points are within noise, so the top five are effectively tied. The clear exception is MNLI, where Kimi and Qwen beat Jev by 7 to 10 points.

Calibration error (ECE, lower is better), before and after fitting a temperature on 150 labels:

| Model | BoolQ | MNLI | AG News | SST-5 |
|---|---|---|---|---|
| **Jev 1.13** | 0.031 / 0.017 | 0.090 / 0.108 | 0.080 / 0.089 | 0.167 / 0.134 |
| Qwen3-235B | 0.063 / 0.009 | 0.103 / 0.029 | 0.088 / 0.067 | 0.348 / 0.060 |
| Kimi K2.5 | 0.049 / 0.027 | 0.078 / 0.045 | 0.096 / 0.060 | 0.275 / 0.072 |
| GLM-5 | 0.071 / 0.037 | 0.135 / 0.068 | 0.089 / 0.037 | 0.299 / 0.054 |
| Mistral Large 3 | 0.113 / 0.052 | 0.140 / 0.047 | 0.107 / 0.025 | 0.394 / 0.053 |
| DeepSeek V3.2 | 0.084 / 0.042 | 0.112 / 0.062 | 0.089 / 0.071 | 0.359 / 0.103 |
| gpt-oss-120b | 0.073 / 0.009 | 0.230 / 0.122 | 0.121 / 0.031 | 0.433 / 0.036 |

Out of the box, Jev is better calibrated than every open model, most visibly on the five-level sentiment score. After 150 labels, Qwen3-235B, Kimi K2.5 and GLM-5 are better calibrated than Jev on all four tasks. That is the number that decides how much traffic you can auto-accept at a threshold like 0.8.

gpt-oss-120b is the odd one out. It reasons before answering, so by the time it emits the answer token the verdict is settled, and its probabilities are almost always 0 or 1. It had the worst raw calibration and the weakest MNLI score.

## Cost

| Model | $ per 1M input tokens | $ per 1,000 questions |
|---|---|---|
| **Jev 1.13** | $0.042 | $0.017 |
| gpt-oss-120b | $0.15 | $0.094, plus reasoning tokens |
| Qwen3-235B | $0.22 | $0.108 |
| Mistral Large 3 | $0.50 | $0.238 |
| DeepSeek V3.2 | $0.62 | $0.289 |
| Kimi K2.5 | $0.60 | $0.291 |
| GLM-5 | $1.00 | $0.471 |

The gap is list price, not harness overhead: jevlocal sends about the same number of tokens per question as Jev bills (roughly 470 against 404).

The only path to Jev's price is running the model yourself on a GPU that stays busy, where prefill-only work is cheap and a shared state is computed once for all its questions. jevlocal has an SGLang backend and a self-terminating H100 script for exactly that measurement, but on the day we ran this AWS had no H100 capacity in any us-west-2 zone. That number is still to come.

## What to take from this

- If you want Jev-level answers with probabilities you can set a threshold on, a large open model plus 150 labels per question type gets you there today.
- Qwen3-235B is the best value of the six: top-tier accuracy, the best calibration after fitting, and the second-lowest price.
- If price is what matters most, Jev is still far ahead of hosted open models. Self-hosting is the open question.
- Skip reasoning models for this job. Their confidence has already collapsed by the time they answer.
