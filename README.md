# jevlocal

Jev-compatible typed judgments (Noul, Choice, Score) from any open-weights LLM. It never generates text: each question is one prefill, and the answer is read from the next-token probabilities of the answer labels, so output tokens are free.

## How it works

- **Same API as Jev.** `POST /v1/systemone` takes and returns the same shapes as `api.typesafe.ai`, so existing Jev clients work by changing the URL.
- **Readout.** Options are shown as `A.`, `B.`, ... (levels as `0.`, `1.`, ...; yes/no as `Yes`/`No`), and the probability of each label token is read at the first answer position. Spellings (`Yes`, ` yes`, `YES`) are summed.
- **Shared state.** The state comes first in every prompt, so SGLang's radix cache computes it once for all questions about it.
- **L0 debiasing, no labels.** Choice options are shown in up to 4 cyclic orders and yes/no in both phrasings; log-probabilities are averaged per option, which removes position bias (Zheng et al. 2024).
- **L1 calibration, with labels.** A temperature per question kind, fitted on a labelled set and loaded with `--calibration`.
- Reasoning models are handled: thinking is disabled for Qwen and Gemma, and gpt-oss is pointed at its final channel.

Calibration and permutation code is ported from [AnyJev](https://github.com/nokia-applied-research/AnyJev) (Apache-2.0). [openjev-sglang](https://github.com/ekzhang/openjev-sglang) has the same API idea but no license, so none of its code is used.

## Run

```bash
uv sync --extra hf
uv run jevlocal-serve --backend hf --model-id Qwen/Qwen3-0.6B --port 8000
```

Any open model on Amazon Bedrock (DeepSeek, Kimi, GLM, Qwen, Mistral, gpt-oss) works with no GPU, using its top-20 logprobs:

```bash
uv run jevlocal-serve --backend bedrock --model-id qwen.qwen3-235b-a22b-2507-v1:0
```

On a GPU box, start SGLang and point jevlocal at it (see `infra/run_model.sh`):

```bash
uv run jevlocal-serve --model-id google/gemma-4-31B-it --sglang-url http://127.0.0.1:30000
```

Extra request field, ignored by Jev: `"jevlocal": {"permutations": 1, "calibrate": false, "raw": true}`.

## Benchmark

Four gold-labelled tasks, 300 items each (half calibration, half test): BoolQ (noul), MNLI and AG News (choice), SST-5 (score).

```bash
uv run python -m bench.run --name jev-1.13.0 --jev jev-1.13.0
uv run python -m bench.run --name gemma-4-31b --url http://127.0.0.1:8000
uv run python -m bench.throughput --url http://127.0.0.1:8000
uv run python -m bench.report
```

`bench.summary` prints the comparison table; the latest run is in [bench/RESULTS.md](bench/RESULTS.md).

Levels in the report: `raw` is one prompt in the listed order, `L0` adds permutation debiasing, `L1` adds a temperature fitted on the calibration split. Jev's own answers are `served`.

## AWS

`infra/launch.sh` starts one `p5.4xlarge` (1x H100 80GB) that terminates itself after 4 hours, with SSH open only to the caller's IP. `infra/setup.sh` and `infra/run_model.sh` run on the box.

## Results so far (2026-09-24)

Six open models on Bedrock against `jev-1.13.0`, 150 test items per task:

- Accuracy: Kimi K2.5 (82.3% mean), Qwen3-235B (81.8%) and GLM-5 (81.0%) match Jev (80.7%); on MNLI, Kimi and Qwen beat Jev by 7 to 10 points.
- Calibration: after a temperature fitted on 150 labels per task, Qwen3-235B, Kimi K2.5 and GLM-5 have lower calibration error than Jev on all four tasks.
- Cost: $0.09 to $0.47 per 1,000 questions on Bedrock, against $0.017 for Jev. Self-hosted cost on an H100 is not measured yet.
