"""Generate resumable SFT data from deterministic heuristic rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_utils import (
    PROMPT_VARIANTS,
    append_jsonl,
    build_text_example,
    generate_heuristic_transitions,
    split_for_key,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/lora_qwen3_4b")
    parser.add_argument("--augmentations", type=int, default=12)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve()
    data_dir = output_root / "data"
    transitions_path = data_dir / "transitions.jsonl"
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / "eval.jsonl"
    manifest_path = output_root / "run_manifest.json"

    if args.force:
        for path in (transitions_path, train_path, eval_path):
            if path.exists():
                path.unlink()

    if transitions_path.exists() and train_path.exists() and eval_path.exists():
        print(json.dumps({"status": "already_exists", "output_root": str(output_root)}, indent=2))
        return

    transition_count = 0
    train_examples = 0
    eval_examples = 0

    for transition in generate_heuristic_transitions():
        record = {
            "task_id": transition.task_id,
            "difficulty": transition.difficulty,
            "step_index": transition.step_index,
            "observation": transition.observation,
            "action": transition.action,
            "reward_after_action": transition.reward_after_action,
            "score_after_action": transition.score_after_action,
            "done": transition.done,
        }
        append_jsonl(transitions_path, record)
        transition_count += 1

        for augmentation_index in range(args.augmentations):
            prompt_variant = PROMPT_VARIANTS[augmentation_index % len(PROMPT_VARIANTS)]
            example = build_text_example(
                observation=transition.observation,
                action=transition.action,
                prompt_variant=prompt_variant,
            )
            example_record = {
                "id": f"{transition.task_id}-step{transition.step_index}-aug{augmentation_index}",
                "task_id": transition.task_id,
                "difficulty": transition.difficulty,
                "step_index": transition.step_index,
                "prompt_variant": prompt_variant,
                **example,
            }
            split = split_for_key(example_record["id"], args.eval_ratio)
            append_jsonl(train_path if split == "train" else eval_path, example_record)
            if split == "train":
                train_examples += 1
            else:
                eval_examples += 1

        write_json(
            manifest_path,
            {
                "status": "data_ready",
                "output_root": str(output_root),
                "transition_count": transition_count,
                "train_examples": train_examples,
                "eval_examples": eval_examples,
                "augmentations": args.augmentations,
                "eval_ratio": args.eval_ratio,
            },
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(output_root),
                "transition_count": transition_count,
                "train_examples": train_examples,
                "eval_examples": eval_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
