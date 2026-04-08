"""Save a small speed summary from an MLX LoRA training log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_RE = re.compile(r"Iter\s+(\d+):\s+Train loss.*?It/sec\s+([0-9.]+)", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="artifacts/mlx_qwen3_4b/logs/mlx_lora_benchmark.log")
    parser.add_argument("--output-path", default="artifacts/mlx_qwen3_4b/metrics/speed_mlx.json")
    args = parser.parse_args()

    log_path = (ROOT / args.log_path).resolve()
    output_path = (ROOT / args.output_path).resolve()
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    records = []
    for step, it_per_sec in REPORT_RE.findall(text):
        itps = float(it_per_sec)
        records.append(
            {
                "step": int(step),
                "iterations_per_second": itps,
                "seconds_per_step_estimate": 1.0 / itps if itps > 0 else None,
            }
        )

    payload = {
        "method": "mlx_lm_lora",
        "source_log": str(log_path),
        "records": records,
        "latest_seconds_per_step": records[-1]["seconds_per_step_estimate"] if records else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
