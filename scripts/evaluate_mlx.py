"""Evaluate base or MLX-adapted Qwen models on the local vulnops environment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from models import VulnTriageAction
from server.cases import TASK_ORDER
from server.vuln_triage_env_environment import VulnTriageEnvironment
from training_utils import render_prompt


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_last_json_object(text: str) -> str | None:
    cleaned = THINK_BLOCK_RE.sub("", text).strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    last_candidate = None
    candidate_start = None
    for index, ch in enumerate(cleaned):
        if ch == "\\" and in_string and not escape:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        escape = False
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                candidate_start = index
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and candidate_start is not None:
                last_candidate = cleaned[candidate_start : index + 1]
    return last_candidate


def parse_action_output(text: str) -> Dict[str, object] | None:
    candidate = extract_last_json_object(text)
    if candidate is None:
        return None
    try:
        payload = json.loads(candidate)
        action = VulnTriageAction.model_validate(payload)
    except Exception:
        return None
    return action.model_dump(exclude_none=True)


def next_action(model, tokenizer, observation: Dict[str, object]) -> Dict[str, object]:
    prompt = render_prompt(
        observation=observation,
        prompt_variant="Return only the best next action in JSON.",
    )
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=False,
        max_tokens=192,
        sampler=make_sampler(temp=0.0),
    )
    payload = parse_action_output(output)
    if payload is None:
        return {
            "action_type": "submit_triage",
            "rationale": f"Fallback because model output could not be parsed: {output[:120]}",
        }
    return payload


def run_episode(model, tokenizer, task_id: str) -> Dict[str, object]:
    env = VulnTriageEnvironment()
    observation = env.reset(task_id=task_id).model_dump()
    actions: List[Dict[str, object]] = []
    while not observation["done"]:
        action_payload = next_action(model, tokenizer, observation)
        action = VulnTriageAction.model_validate(action_payload)
        actions.append(action.model_dump(exclude_none=True))
        observation = env.step(action).model_dump()
    return {
        "task_id": task_id,
        "difficulty": observation["difficulty"],
        "final_score": float(observation.get("final_score") or 0.0),
        "score_breakdown": observation["score_breakdown"],
        "steps_used": len(actions),
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    episodes = [run_episode(model, tokenizer, task_id) for task_id in TASK_ORDER]
    average_score = round(sum(item["final_score"] for item in episodes) / len(episodes), 4)
    payload = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "average_score": average_score,
        "episodes": episodes,
    }
    if args.output_json:
        out = Path(args.output_json)
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
