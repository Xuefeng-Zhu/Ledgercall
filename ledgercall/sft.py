"""Supervised agentic fine-tuning (trajectory SFT) for LedgerCall.

Why SFT and not GRPO on a T4: GRPO with Tiny Aya (3.35B) at 4 generations ran out of
memory on a 16 GB T4 before its first step. Instead we build *gold* tool-calling
trajectories (our tools are deterministic, so every correct call and result is known),
replay them through AgentTune's own rollout loop so the training prompts are rendered
byte-for-byte the way evaluation renders them (same chat template, same tool fallback
prompt, same folded "Tool result for ..." turns), and fine-tune a LoRA adapter on the
assistant turns only.

Steps:
    1. capture: rollout with a scripted engine that answers each turn with the gold
       completion -> list of (prompt_ids, completion_ids) per assistant turn.
    2. train: LoRA SFT, loss on completion tokens only.

Usage:
    python -m ledgercall.sft --model <base> --train_file data/train.jsonl --output_dir out/sft
    python -m ledgercall.sft --model <base> --capture_only --output_dir out/sft   # CPU ok
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from .evaluate import _ensure_vendor_paths, load_jsonl
from .tools import TOOLS, TOOL_MAP

FINAL_TEMPLATE = "The answer is *{answer}*."


def gold_turns(row: dict[str, Any]) -> list[str]:
    """Gold assistant texts: one JSON tool call per turn, then the starred final answer."""
    turns = []
    for call in row["gold_calls"]:
        turns.append(json.dumps([{"tool_name": call["name"], "parameters": call["arguments"]}]))
    final = None
    for call in row["gold_calls"]:
        final = TOOL_MAP[call["name"]](**call["arguments"])
    turns.append(FINAL_TEMPLATE.format(answer=f"{final:.2f}"))
    return turns


def _stop_ids(tokenizer) -> list[int]:
    ids = []
    for tok in ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def capture_examples(tokenizer, rows: list[dict[str, Any]], max_steps: int = 4) -> tuple[list[dict], dict]:
    """Replay gold trajectories through AgentTune's rollout loop and record every turn."""
    _ensure_vendor_paths()
    import torch
    from agenttune.agentic.rollout_engines.rollout_factory import (
        DEFAULT_TOOL_RESULT_FORMAT,
        DEFAULT_TOOLS_FALLBACK_PROMPT,
        create_rollout_fn,
    )
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    stop_ids = _stop_ids(tokenizer)
    end_turn = [tokenizer.convert_tokens_to_ids(t) for t in ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>")]

    class _Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(1, 1)
            self.generation_config = None

    class ScriptedEngine(TransformersRolloutEngine):
        def __init__(self, tok):
            super().__init__(_Dummy(), tok)
            self.script: list[str] = []
            self.records: list[dict] = []

        def generate(self, prompts, tools, gen_cfg):
            conv = prompts if (isinstance(prompts[0], dict) and "role" in prompts[0]) else prompts[0]
            n_asst = sum(1 for m in conv if m.get("role") == "assistant")
            text = self.script[min(n_asst, len(self.script) - 1)]
            enc = self._apply_chat_template([conv], tools if tools else None, False)
            prompt_ids = enc["input_ids"][0].tolist()
            if "attention_mask" in enc:
                mask = enc["attention_mask"][0].tolist()
                prompt_ids = [t for t, m in zip(prompt_ids, mask) if m]
            comp_ids = tokenizer.encode(text, add_special_tokens=False) + end_turn
            self.records.append({"prompt_ids": prompt_ids, "completion_ids": comp_ids, "text": text})
            structured = self._parse_completion(comp_ids, text)
            return {
                "completions": [text],
                "completions_raw": [text],
                "completions_structured": [structured],
                "logprobs": [[0.0] * len(comp_ids)],
                "metadata": {"backend": "scripted", "tools_enabled": bool(tools), "completion_ids": [comp_ids],
                             "can_parse": self._can_parse, "device": "cpu", "multi_device": False,
                             "prompt_len": len(prompt_ids)},
            }

    engine = ScriptedEngine(tokenizer)
    trajs: list[dict] = []
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=TOOLS,
        max_steps=max_steps,
        on_trajectory_end=lambda t: trajs.append(t),
        tools_fallback_prompt=DEFAULT_TOOLS_FALLBACK_PROMPT,
        tool_result_format=DEFAULT_TOOL_RESULT_FORMAT,
    )
    examples: list[dict] = []
    ok = 0
    for row in rows:
        engine.script = gold_turns(row)
        engine.records = []
        rollout_fn([{"prompt": row["prompt"], "answer": row["answer"]}], max_new_tokens=128, do_sample=False)
        recs = engine.records
        # Keep the row only if the replay consumed exactly the scripted turns.
        if len(recs) == len(engine.script) and [r["text"] for r in recs] == engine.script:
            ok += 1
            examples.extend({"prompt_ids": r["prompt_ids"], "completion_ids": r["completion_ids"]} for r in recs)
    stats = {"rows": len(rows), "rows_replayed_ok": ok, "turn_examples": len(examples),
             "stop_ids": stop_ids, "end_turn_ids": end_turn}
    return examples, stats


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_jsonl(args.train_file)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples, stats = capture_examples(tokenizer, rows)
    print("[ledgercall.sft] capture", stats, flush=True)
    with open(out / "sft_examples_preview.json", "w") as f:
        json.dump([{"prompt_tail": tokenizer.decode(e["prompt_ids"][-120:]),
                    "completion": tokenizer.decode(e["completion_ids"])} for e in examples[:6]], f, indent=2)
    if args.capture_only:
        json.dump(stats, open(out / "capture_stats.json", "w"), indent=2)
        return stats

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map={"": 0} if torch.cuda.is_available() else None)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                             task_type="CAUSAL_LM"))
    # LoRA params in fp32 for stable fp16 mixed precision
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    rng = random.Random(0)
    order = list(range(len(examples)))
    total_steps = math.ceil(len(examples) * args.epochs / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    device = next(model.parameters()).device
    step, micro, t0, losses = 0, 0, time.time(), []
    log = []
    model.train()
    for epoch in range(math.ceil(args.epochs)):
        rng.shuffle(order)
        for i in order:
            if step >= total_steps or (time.time() - t0) > args.max_minutes * 60:
                break
            ex = examples[i]
            ids = ex["prompt_ids"] + ex["completion_ids"]
            input_ids = torch.tensor([ids], device=device)
            comp = torch.tensor(ex["completion_ids"], device=device)
            # Only materialise logits for the completion span: Tiny Aya's 262k vocab
            # makes full-sequence logits the dominant memory cost on a 16 GB T4.
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(input_ids=input_ids, logits_to_keep=len(ex["completion_ids"]) + 1).logits
            logits = logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, comp) / args.grad_accum
            del logits
            scaler.scale(loss).backward()
            losses.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum == 0:
                lr = args.lr * min(1.0, (step + 1) / max(1, int(0.05 * total_steps))) * (0.5 * (1 + math.cos(math.pi * step / total_steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    avg = sum(losses[-args.grad_accum * 10:]) / len(losses[-args.grad_accum * 10:])
                    log.append({"step": step, "loss": round(avg, 4), "lr": lr, "elapsed_s": round(time.time() - t0, 1)})
                    print(f"[ledgercall.sft] step {step}/{total_steps} loss {avg:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    summary = {**stats, "steps": step, "epochs": args.epochs, "grad_accum": args.grad_accum, "lr": args.lr,
               "lora_r": args.lora_r, "train_time_s": round(time.time() - t0, 1), "loss_log": log}
    json.dump(summary, open(out / "train_summary.json", "w"), indent=2)
    return summary


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="LedgerCall trajectory SFT (LoRA)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--output_dir", default="out/sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_minutes", type=float, default=60.0)
    ap.add_argument("--max_steps", type=int, default=0, help="cap optimizer steps so the cosine schedule completes")
    ap.add_argument("--capture_only", action="store_true")
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
