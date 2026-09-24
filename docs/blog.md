# Turning open LLMs into calibrated classifiers

A classifier should give you a label and a probability you can trust. Chat models give you text. We wanted to know how close an open model gets to a purpose-built classifier if you skip the text entirely.

## How

Ask the question, list the options as `A`, `B`, `C`, and read the probability the model puts on each label at the first answer token. No generation, so no output cost.

Two fixes make the probabilities usable:

- Show the options in a few different orders and average, so the model's position bias cancels out.
- Fit one temperature on about 150 labelled examples, so a 0.8 means right 80% of the time.

## Results

Six open models on Amazon Bedrock, against Jev (a purpose-built classifier) as the baseline. Four public tasks, 150 test items each.

| Model | BoolQ | MNLI | AG News | SST-5 | Mean | $ per 1k questions |
|---|---|---|---|---|---|---|
| Kimi K2.5 | 94.0% | 89.3% | 88.7% | 57.3% | 82.3% | $0.291 |
| Qwen3-235B | 94.0% | 86.7% | 89.3% | 57.3% | 81.8% | $0.108 |
| GLM-5 | 91.3% | 82.0% | 90.0% | 60.7% | 81.0% | $0.471 |
| Jev (baseline) | 94.0% | 79.3% | 90.0% | 59.3% | 80.7% | $0.017 |
| Mistral Large 3 | 88.7% | 85.3% | 89.3% | 56.7% | 80.0% | $0.238 |
| DeepSeek V3.2 | 90.0% | 79.3% | 90.7% | 53.3% | 78.3% | $0.289 |
| gpt-oss-120b | 92.7% | 76.7% | 87.3% | 56.7% | 78.3% | $0.094 |

## What we learned

- The best open models classify as well as the baseline, and better on entailment (MNLI).
- After calibration, Qwen3-235B, Kimi K2.5 and GLM-5 give more trustworthy probabilities than the baseline on every task.
- Hosted, they cost 6 to 28 times more per question. Self-hosting is the next thing to measure.
- Reasoning models make poor classifiers: their probabilities collapse to 0 or 1.
