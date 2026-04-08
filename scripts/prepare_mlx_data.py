"""Prepare MLX-LM-compatible train/valid files from existing SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TRUNCATION_MARKER = "\n...[truncated observation]...\n"


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def trim_prompt_to_budget(prompt: str, tokenizer, budget: int) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) <= budget:
        return prompt

    marker_ids = tokenizer.encode(TRUNCATION_MARKER, add_special_tokens=False)
    marker_len = len(marker_ids)
    if budget <= marker_len + 8:
        return tokenizer.decode(prompt_ids[-budget:])

    remaining = budget - marker_len
    head_len = max(1, int(remaining * 0.55))
    tail_len = max(1, remaining - head_len)
    trimmed_ids = prompt_ids[:head_len] + marker_ids + prompt_ids[-tail_len:]
    if len(trimmed_ids) > budget:
        trimmed_ids = trimmed_ids[:budget]
    return tokenizer.decode(trimmed_ids, skip_special_tokens=False)


def rendered_length(prompt: str, completion: str, tokenizer) -> int:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    return len(tokenizer.apply_chat_template(messages, return_dict=False))


def normalize_record(record: Dict[str, object], tokenizer, max_seq_length: int) -> tuple[Dict[str, object] | None, Dict[str, int]]:
    prompt = str(record["prompt"])
    completion = str(record["completion"])

    stats = {"trimmed": 0, "dropped": 0}
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    prompt_budget = max_seq_length - len(completion_ids) - 32
    if prompt_budget <= 0:
        stats["dropped"] = 1
        return None, stats

    normalized_prompt = trim_prompt_to_budget(prompt, tokenizer, prompt_budget)
    while rendered_length(normalized_prompt, completion, tokenizer) > max_seq_length and prompt_budget > 64:
        prompt_budget = max(64, int(prompt_budget * 0.9))
        normalized_prompt = trim_prompt_to_budget(prompt, tokenizer, prompt_budget)
    if rendered_length(normalized_prompt, completion, tokenizer) > max_seq_length:
        stats["dropped"] = 1
        return None, stats

    if normalized_prompt != prompt:
        stats["trimmed"] = 1

    text = f"{normalized_prompt}\n{completion}"
    normalized = dict(record)
    normalized["prompt"] = normalized_prompt
    normalized["text"] = text
    return normalized, stats


def transform_split(src: Path, dst: Path, tokenizer, max_seq_length: int) -> Dict[str, int]:
    rows = load_jsonl(src)
    normalized_rows: List[Dict[str, object]] = []
    stats = {"input_examples": len(rows), "written_examples": 0, "trimmed_examples": 0, "dropped_examples": 0}

    for row in rows:
        normalized, row_stats = normalize_record(row, tokenizer, max_seq_length)
        stats["trimmed_examples"] += row_stats["trimmed"]
        stats["dropped_examples"] += row_stats["dropped"]
        if normalized is not None:
            normalized_rows.append(normalized)

    stats["written_examples"] = len(normalized_rows)
    dump_jsonl(dst, normalized_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="artifacts/lora_qwen3_4b/data")
    parser.add_argument("--output-root", default="artifacts/mlx_qwen3_4b/data")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--include-valid", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_root = (ROOT / args.source_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    mapping = {source_root / "train.jsonl": output_root / "train.jsonl"}
    if args.include_valid:
        mapping[source_root / "eval.jsonl"] = output_root / "valid.jsonl"

    summary: Dict[str, object] = {
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "splits": {},
    }
    for src, dst in mapping.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        if dst.exists() and not args.force:
            continue
        summary["splits"][dst.stem] = transform_split(src, dst, tokenizer, args.max_seq_length)

    valid_path = output_root / "valid.jsonl"
    if not args.include_valid and valid_path.exists():
        valid_path.unlink()

    summary_path = output_root.parent / "prepare_stats.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
