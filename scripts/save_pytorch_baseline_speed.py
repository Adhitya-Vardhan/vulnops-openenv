"""Save a small speed summary from the current PyTorch training log."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "artifacts" / "lora_qwen3_4b" / "logs" / "train_lora_manual.log"
OUT_PATH = ROOT / "artifacts" / "lora_qwen3_4b" / "metrics" / "speed_baseline_pytorch.json"


STEP_RE = re.compile(r"(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[(\d+):(\d+)<")


def main() -> None:
    text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    matches = STEP_RE.findall(text)
    records = []
    for _pct, step, total, mins, secs in matches:
        step_num = int(step)
        elapsed_s = int(mins) * 60 + int(secs)
        if step_num > 0:
            records.append(
                {
                    "step": step_num,
                    "total_steps": int(total),
                    "elapsed_seconds": elapsed_s,
                    "seconds_per_step_estimate": elapsed_s / step_num,
                }
            )

    payload = {
        "method": "pytorch_mps_lora",
        "source_log": str(LOG_PATH),
        "records": records,
        "latest_seconds_per_step": records[-1]["seconds_per_step_estimate"] if records else None,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
