"""Dump a full raw generation from the MLX model for one vulnops observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from server.vuln_triage_env_environment import VulnTriageEnvironment
from training_utils import render_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", default="artifacts/mlx_qwen3_4b/adapters")
    parser.add_argument("--task-id", default="task_easy_guarddog")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--output-file",
        default="artifacts/mlx_qwen3_4b/inspection/task_easy_guarddog_latest_raw_output.json",
    )
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    env = VulnTriageEnvironment()
    observation = env.reset(task_id=args.task_id).model_dump()
    prompt = render_prompt(observation, "Return only the best next action in JSON.")
    raw_output = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=False,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0),
    )

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": args.task_id,
        "model": args.model,
        "adapter_path": args.adapter_path,
        "max_tokens": args.max_tokens,
        "prompt": prompt,
        "raw_output": raw_output,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
