"""Run MLX LoRA training as the default local Mac training path."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--source-root", default="artifacts/lora_qwen3_4b/data")
    parser.add_argument("--output-root", default="artifacts/mlx_qwen3_4b")
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--steps-per-report", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--include-valid", action="store_true")
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve()
    data_root = output_root / "data"
    log_path = output_root / "logs" / "mlx_train.log"
    manifest_path = output_root / "run_manifest.json"
    adapter_root = output_root / "adapters"
    adapter_file = adapter_root / "adapters.safetensors"
    speed_path = output_root / "metrics" / "speed_mlx.json"

    output_root.mkdir(parents=True, exist_ok=True)
    if args.fresh_start:
        for rel in [log_path, speed_path, output_root / "training_summary.json", adapter_file]:
            if rel.exists():
                rel.unlink()

    prepare_cmd = [
        sys.executable,
        "scripts/prepare_mlx_data.py",
        "--source-root",
        args.source_root,
        "--output-root",
        str(data_root.relative_to(ROOT)),
        "--model",
        args.model,
        "--max-seq-length",
        str(args.max_seq_length),
        "--force",
    ]
    if args.include_valid:
        prepare_cmd.append("--include-valid")
    subprocess.run(prepare_cmd, cwd=ROOT, check=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(data_root),
        "--mask-prompt",
        "--num-layers",
        str(args.num_layers),
        "--batch-size",
        str(args.batch_size),
        "--iters",
        str(args.iters),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        "1000000",
        "--save-every",
        str(args.save_every),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--grad-checkpoint",
        "--adapter-path",
        str(adapter_root),
        "--max-seq-length",
        str(args.max_seq_length),
        "--seed",
        str(args.seed),
    ]
    if not args.fresh_start and adapter_file.exists():
        cmd.extend(["--resume-adapter-file", str(adapter_file)])

    write_json(
        manifest_path,
        {
            "status": "starting_training",
            "trainer": "mlx_lm_lora",
            "model": args.model,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "command": cmd,
            "fresh_start": args.fresh_start,
        },
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n===== mlx_lm_lora =====\n")
        handle.write("COMMAND: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)

    subprocess.run([sys.executable, "scripts/save_mlx_speed.py", "--log-path", str(log_path), "--output-path", str(speed_path)], cwd=ROOT, check=False)

    summary = {
        "status": "finished" if process.returncode == 0 else "failed",
        "trainer": "mlx_lm_lora",
        "return_code": process.returncode,
        "log_path": str(log_path),
        "speed_path": str(speed_path),
        "adapter_root": str(adapter_root),
    }
    write_json(output_root / "training_summary.json", summary)
    write_json(manifest_path, summary)
    if process.returncode != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
