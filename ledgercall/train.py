"""Agentic GRPO training entrypoint for LedgerCall on CohereLabs/tiny-aya-global.

Designed to fit comfortably on a single 16 GB Kaggle/Colab T4 (using fp16, since
Turing GPUs lack native bf16) or 24 GB L4 (using bf16) with LoRA r=16 on all
linear layers, gradient checkpointing, and `use_vllm=False`.

Example usage:
    python -m ledgercall.train --train_file data/train.jsonl --output_dir out/grpo --max_steps 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .rewards import correctness, efficiency, tool_choice
from .tools import TOOLS


def _ensure_vendor_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for rel in ("vendor/AgentTune/src", "vendor/AuditKIT/src"):
        candidate = repo_root / rel
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))


def detect_precision() -> tuple[bool, bool, str]:
    """Return `(bf16, fp16, torch_dtype_name)` tuned for T4 (fp16), L4/A100 (bf16), or CPU (fp32)."""
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, False, "bfloat16"
        # NVIDIA T4 (SM 7.5) has no native bfloat16; use float16 instead.
        return False, True, "float16"
    return False, False, "float32"


def load_train_dataset(train_file: str | Path) -> Any:
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Training file '{train_file}' is empty.")
    return Dataset.from_list(rows)


def build_trainer_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    bf16, fp16, dtype_name = detect_precision()
    train_dataset = load_train_dataset(args.train_file)

    peft_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
    }

    kwargs: dict[str, Any] = {
        "algorithm": "grpo",
        "backend": "trl",
        "model": args.model,
        "train_dataset": train_dataset,
        "tools": TOOLS,
        "reward_funcs": [correctness, tool_choice, efficiency],
        "reward_weights": [1.0, 0.5, 0.5],
        "peft_config": peft_config,
        "use_vllm": False,
        "bf16": bf16,
        "fp16": fp16,
        "model_init_kwargs": {"torch_dtype": dtype_name},
        "gradient_checkpointing": True,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps_per_turn": args.max_steps_per_turn,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "warmup_steps": min(15, max(1, args.max_steps // 10)),
        "logging_steps": 5,
        "save_steps": max(25, args.max_steps // 3),
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": "none",
        "log_completions": True,
    }
    if args.push_to_hub:
        kwargs["push_to_hub"] = True
        if args.hub_model_id:
            kwargs["hub_model_id"] = args.hub_model_id
    return kwargs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LedgerCall Tiny Aya finance tool agent with GRPO.")
    parser.add_argument("--model", type=str, default="CohereLabs/tiny-aya-global", help="Base HuggingFace model ID")
    parser.add_argument("--train_file", "--train-file", dest="train_file", type=str, default="data/train.jsonl", help="Training JSONL path")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="out/grpo", help="Output directory for LoRA adapter + trajectories")
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=150, help="GRPO optimizer steps (default: 150)")
    parser.add_argument("--num_generations", "--num-generations", dest="num_generations", type=int, default=4, help="Rollouts per prompt (default: 4)")
    parser.add_argument(
        "--per_device_train_batch_size",
        "--per-device-train-batch-size",
        dest="per_device_train_batch_size",
        type=int,
        default=4,
        help="Per-device batch size; must be a multiple of num_generations (default: 4)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument("--max_steps_per_turn", "--max-steps-per-turn", dest="max_steps_per_turn", type=int, default=4, help="Max tool-call turns per rollout (default: 4)")
    parser.add_argument("--max_completion_length", "--max-completion-length", dest="max_completion_length", type=int, default=384, help="Max tokens per generation turn (default: 384)")
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=5e-5, help="LoRA learning rate (default: 5e-5)")
    parser.add_argument("--lora_r", "--lora-r", dest="lora_r", type=int, default=16, help="LoRA rank r (default: 16)")
    parser.add_argument("--lora_alpha", "--lora-alpha", dest="lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--push_to_hub", "--push-to-hub", dest="push_to_hub", action="store_true", help="Push trained LoRA adapter to Hugging Face Hub")
    parser.add_argument("--hub_model_id", "--hub-model-id", dest="hub_model_id", type=str, default=None, help="Target Hub repo ID (e.g. <user>/ledgercall-tiny-aya-grpo)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _ensure_vendor_paths()
    from agenttune.core.backend_factory import create_agentic_trainer

    args = parse_args(argv)
    trainer_kwargs = build_trainer_kwargs(args)
    print(
        f"[ledgercall.train] Initializing GRPO trainer for {args.model} "
        f"(bf16={trainer_kwargs['bf16']}, fp16={trainer_kwargs['fp16']}, max_steps={args.max_steps})"
    )
    trainer = create_agentic_trainer(**trainer_kwargs)
    stats = trainer.train()
    print(
        f"[ledgercall.train] Done -> {args.output_dir} "
        f"(steps={stats.get('total_steps')}, final_loss={stats.get('final_loss')})"
    )
    return stats


if __name__ == "__main__":
    main()
