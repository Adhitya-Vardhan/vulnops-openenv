"""Compare saved PyTorch and MLX speed summaries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PT_PATH = ROOT / "artifacts" / "lora_qwen3_4b" / "metrics" / "speed_baseline_pytorch.json"
MLX_PATH = ROOT / "artifacts" / "mlx_qwen3_4b" / "metrics" / "speed_mlx.json"
OUT_PATH = ROOT / "artifacts" / "speed_comparison.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pt = load(PT_PATH)
    mlx = load(MLX_PATH)
    pt_s = pt.get("latest_seconds_per_step")
    mlx_s = mlx.get("latest_seconds_per_step")
    payload = {
        "pytorch_mps_seconds_per_step": pt_s,
        "mlx_seconds_per_step": mlx_s,
        "speedup_factor_mlx_vs_pytorch": (pt_s / mlx_s) if pt_s and mlx_s else None,
        "notes": [
            "PyTorch baseline uses the existing PEFT/Transformers trainer on MPS.",
            "MLX benchmark uses a lower-memory LoRA config: 8 layers and max_seq_length 1024.",
        ],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
