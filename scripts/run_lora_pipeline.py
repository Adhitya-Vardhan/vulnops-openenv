"""Run the full resumable local LoRA pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_utils import latest_checkpoint, write_json


def run_step(name: str, command: list[str], log_path: Path, output_root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n===== {name} =====\n")
        log_handle.flush()
        write_json(
            output_root / "run_manifest.json",
            {
                "status": "running_step",
                "current_step": name,
                "command": command,
                "latest_checkpoint": str(latest_checkpoint(output_root / "checkpoints")) if (output_root / "checkpoints").exists() else None,
            },
        )
        process = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise SystemExit(process.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-root", default="artifacts/lora_qwen3_4b")
    parser.add_argument("--augmentations", type=int, default=12)
    parser.add_argument("--skip-base-eval", action="store_true")
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve()
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_base_eval and not (output_root / "metrics" / "eval_before.json").exists():
        run_step(
            "eval_base",
            [
                sys.executable,
                "scripts/evaluate_lora.py",
                "--model",
                args.model,
                "--output-root",
                str(output_root),
                "--output-json",
                str(output_root / "metrics" / "eval_before.json"),
            ],
            logs_dir / "eval_base.log",
            output_root,
        )

    if not (output_root / "data" / "train.jsonl").exists():
        run_step(
            "generate_data",
            [
                sys.executable,
                "scripts/generate_sft_data.py",
                "--output-root",
                str(output_root),
                "--augmentations",
                str(args.augmentations),
            ],
            logs_dir / "generate_data.log",
            output_root,
        )

    run_step(
        "train_lora",
        [
            sys.executable,
            "scripts/train_lora_sft.py",
            "--model",
            args.model,
            "--output-root",
            str(output_root),
        ],
        logs_dir / "train_lora.log",
        output_root,
    )

    run_step(
        "eval_adapter",
        [
            sys.executable,
            "scripts/evaluate_lora.py",
            "--model",
            args.model,
            "--adapter-path",
            str(output_root / "adapter"),
            "--output-root",
            str(output_root),
            "--output-json",
            str(output_root / "metrics" / "eval_after.json"),
        ],
        logs_dir / "eval_adapter.log",
        output_root,
    )

    write_json(
        output_root / "run_manifest.json",
        {
            "status": "finished",
            "output_root": str(output_root),
            "eval_before": str(output_root / "metrics" / "eval_before.json"),
            "training_summary": str(output_root / "training_summary.json"),
            "eval_after": str(output_root / "metrics" / "eval_after.json"),
        },
    )
    print(
        json.dumps(
            {
                "status": "finished",
                "output_root": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
