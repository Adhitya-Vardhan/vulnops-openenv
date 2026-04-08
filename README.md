---
title: VulnOps Reasoning Benchmark
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# VulnOps OpenEnv

`vulnops` is an OpenEnv benchmark for open-source vulnerability operations. The agent plays the role of a maintainer or security analyst working through incoming vulnerability cases, revealing supporting evidence, filling a structured draft, and submitting the correct next maintainer action.

This benchmark is intentionally not a bug-fixing environment and not a generic classifier. It models a real workflow: validating advisories, identifying affected packages and versions, weighing severity versus exploitability, and deciding whether to patch or publish an advisory.

## Data sources

The benchmark now pulls case data from live public vulnerability feeds at runtime:

- OSV for package identity, advisory details, affected ranges, and references
- NVD for normalized CVE descriptions and CVSS severity metadata
- EPSS for exploitability scoring signals

The environment normalizes those live responses into hidden ground truth on `reset()`. To keep tests, local development, and offline execution stable, each task also includes a bundled fallback snapshot that is used when the APIs are unavailable.

In addition to the task-specific fallbacks, the container now ships with a broader cache of 200 provider-backed fallback snapshots under `data/snapshots/`. That keeps the image self-sufficient and gives us room to expand the benchmark without depending entirely on live API availability.

## Why this is useful

- Real-world utility: OSS maintainers triage reports like these every week.
- Deterministic grading: each case has hidden ground truth and a reproducible scorer.
- Multi-step rewards: the agent earns signal for revealing good evidence and filling the draft correctly before final submission.
- Lightweight deployment: no VM, browser, or external datasets are required at runtime.

## Environment interface

The environment implements the standard OpenEnv APIs:

- `reset(task_id=...) -> VulnTriageObservation`
- `step(VulnTriageAction) -> VulnTriageObservation`
- `state -> VulnTriageState`

### Action space

`VulnTriageAction` has these fields:

- `action_type`: one of `read_report`, `inspect_evidence`, `search_nvd_database`, `fetch_commit_diff`, `message_maintainer`, `set_validity`, `set_affected_package`, `set_affected_versions`, `set_severity`, `set_exploitability`, `set_next_action`, `set_missing_information`, `request_more_info`, `submit_triage`
- `evidence_id`: used with `inspect_evidence`
- `value`: generic value for label-setting and missing-information actions
- `rationale`: optional free-form note

### Observation space

`VulnTriageObservation` returns:

- task metadata: `task_id`, `difficulty`, `objective`
- `report_summary`
- `visible_evidence`
- `available_evidence`
- `draft`
- `action_history`
- `steps_remaining`
- `score_breakdown`
- `final_score`
- standard OpenEnv fields: `reward`, `done`, `metadata`

## Task ladder

### 1. GuardDog Path Traversal
- Difficulty: easy
- Goal: Validate the report, identify the package and fixed range, and choose `patch`.

### 2. Invenio Multi-Branch XSS
- Difficulty: medium
- Goal: Resolve affected versions across multiple release lines and extract truth despite decoy severity signals.

### 3. Requests Auth Header Leak
- Difficulty: medium
- Goal: Ignore severe threat-intel decoys and use `fetch_commit_diff` to read the Python fix manually.

### 4. Gradio Upload XSS
- Difficulty: hard
- Goal: Actively `message_maintainer` to discover the lack of a patch and avoid catastrophic penalties by choosing `request_info`.

## Baseline Scores

The benchmark includes a baseline evaluation script (`inference.py`). Tested against **Qwen3:30b** using the interactive action space:

- **Average Score (0-1.0):** `0.3104`
- **Reasoning Gap:** `68.96%`

*Frontier models struggle with proactive tool-use (`search_nvd_database`, `fetch_commit_diff`, `message_maintainer`) instead of passive reading, creating a massive optimization valley for RL evaluation.*

## Reward design

Per-step reward is shaped to encourage realistic behavior:

- positive reward for reading the report, revealing new relevant evidence, and setting a draft field correctly
- negative reward for repeated evidence inspection, empty or incorrect updates, and premature or low-evidence submission
- final submission reward equals the normalized grader score in `[0.0, 1.0]`, with a small penalty for submitting with too little evidence

### Grader weights

- validity: `0.20`
- affected package: `0.10`
- affected versions: `0.10`
- severity: `0.20`
- exploitability: `0.15`
- next action: `0.15`
- missing-information handling: `0.10`

## Project structure

```text
.
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
└── server
    ├── app.py
    ├── cases.py
    ├── Dockerfile
    ├── graders.py
    └── vuln_triage_env_environment.py
```

## Setup

### Local Python setup

```bash
python -m pip install -e ".[dev]"
```

### Run the environment locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Validate the environment

```bash
openenv validate .
```

## Inference baseline

The required root-level `inference.py` supports two modes:

- `--policy openai`: uses the OpenAI Python client, reading credentials from `OPENAI_API_KEY` or `HF_TOKEN`, model name from `MODEL_NAME`, and optional base URL from `API_BASE_URL`
- `--policy heuristic`: deterministic offline smoke test for local development

### Local direct benchmark run

```bash
python inference.py --policy heuristic
```

### Against a running local or remote server

```bash
export ENV_BASE_URL=http://localhost:8000
python inference.py --policy openai --model "$MODEL_NAME"
```

## Docker

Build and run:

```bash
docker build -t vulnops .
docker run -p 8000:8000 vulnops
```

## Hugging Face Space deployment

This project is packaged for a container-based FastAPI Space. The Space should be tagged with `openenv` and pointed at the provided `Dockerfile`.

## Expected baseline behavior

The heuristic policy should score `1.0` on all three bundled fallback snapshots. The OpenAI baseline is intended as the hackathon submission baseline and should be reproducible with `temperature=0`.

## Local LoRA learnability check

This repo now includes a local LoRA pipeline for a quick "is the environment learnable?" check with `Qwen/Qwen3.5-4B`.

On Apple Silicon, the recommended path is now `MLX`, not the older PyTorch `MPS` path.

### What it does

- generates deterministic heuristic transitions from the environment
- expands them into prompt-variant SFT examples
- runs LoRA SFT with checkpointing
- evaluates the base model and adapted model back on `vulnops`
- writes append-only logs so interrupted runs still leave useful evidence

### Install the training extra

```bash
python -m pip install -e ".[train]"
```

### Recommended MLX path

```bash
python -m pip install mlx mlx-lm
./scripts/start_mlx_training.sh
```

Artifacts are written under `artifacts/mlx_qwen3_4b/`:

- `run_manifest.json`: current status and latest known checkpoint
- `data/train.jsonl`: MLX-ready SFT records
- `logs/mlx_train.log`: main training log
- `logs/nohup.out`: launcher stdout/stderr
- `metrics/speed_mlx.json`: parsed speed summary
- `adapters/`: MLX adapter artifacts
- `training_summary.json`: final run status

If you stop the run midway, rerun `python scripts/run_mlx_training.py --model Qwen/Qwen3.5-4B --output-root artifacts/mlx_qwen3_4b`.
It will reuse the prepared dataset and resume from the saved adapter file when present.

### Current speed comparison

On this Mac, the saved local benchmark showed:

- PyTorch `MPS`: about `72.5s/step`
- MLX: about `16.4s/step`

See [artifacts/speed_comparison.json](/Users/adithyavardhan/Tweeks/hack/artifacts/speed_comparison.json).
