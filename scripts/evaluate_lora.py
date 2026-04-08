"""Evaluate a base or LoRA-adapted model on the local vulnops environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.cases import TASK_ORDER
from training_utils import (
    detect_device,
    maybe_parse_action,
    preferred_torch_dtype,
    render_prompt,
    set_default_env,
)
from models import VulnTriageAction
from server.vuln_triage_env_environment import VulnTriageEnvironment


def load_model(model_name: str, adapter_path: str | None, output_root: Path):
    set_default_env(output_root)
    device = detect_device()
    torch_dtype = preferred_torch_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to evaluate a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    if device in {"cuda", "mps"}:
        model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def next_action(model, tokenizer, device: str, observation: Dict[str, object]) -> Dict[str, object]:
    prompt = render_prompt(
        observation=observation,
        prompt_variant="Return only the best next action in JSON.",
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=192,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_length = encoded["input_ids"].shape[1]
    output_text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
    payload = maybe_parse_action(output_text)
    if payload is None:
        return {
            "action_type": "submit_triage",
            "rationale": f"Fallback because model output could not be parsed: {output_text[:120]}",
        }
    return payload


def run_episode(model, tokenizer, device: str, task_id: str) -> Dict[str, object]:
    env = VulnTriageEnvironment()
    observation = env.reset(task_id=task_id).model_dump()
    actions: List[Dict[str, object]] = []
    while not observation["done"]:
        action_payload = next_action(model, tokenizer, device, observation)
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
    parser.add_argument("--output-root", default="artifacts/lora_qwen3_4b")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve()
    model, tokenizer, device = load_model(args.model, args.adapter_path, output_root)
    episodes = [run_episode(model, tokenizer, device, task_id) for task_id in TASK_ORDER]
    average_score = round(sum(item["final_score"] for item in episodes) / len(episodes), 4)
    payload = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "device": device,
        "average_score": average_score,
        "episodes": episodes,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = (ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
