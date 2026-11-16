"""Evaluate a base or LoRA-tuned model on LedgerCall test tasks with AgentTune + AuditKIT.

Runs multi-turn tool-calling rollouts via AgentTune's `transformers` rollout engine
(`use_vllm=False`, keeping AgentTune's automatic Tiny Aya tool-call prompt fallback),
writes `trajectories.jsonl` in portable AgentTune format, and scores recorded episodes
with AuditKIT (`auditkit.agent_eval`) alongside LedgerCall's finance metrics.

Example usage:
    python -m ledgercall.evaluate --model CohereLabs/tiny-aya-global --data data/test.jsonl --tag base --limit 100
    python -m ledgercall.evaluate --model out/grpo --data data/test.jsonl --tag tuned --limit 100
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .rewards import is_numeric_match, ordered_tool_recall, parse_starred_number
from .tools import TOOL_NAMES, TOOLS


def _ensure_vendor_paths() -> None:
    """Allow running directly from a repo checkout when vendor packages aren't pip-installed."""
    repo_root = Path(__file__).resolve().parent.parent
    for rel in ("vendor/AuditKIT/src", "vendor/AgentTune/src"):
        candidate = repo_root / rel
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _select_torch_dtype() -> Any:
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_or_adapter: str,
    base_model_override: str | None = None,
    adapter_override: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load either a base HuggingFace model or a base model + PEFT LoRA adapter."""
    _ensure_vendor_paths()
    from transformers import AutoTokenizer

    from agenttune.agentic.rollout_engines.rollout_factory import load_causal_or_multimodal_model

    dtype = _select_torch_dtype()
    adapter_path: str | None = adapter_override
    model_path_obj = Path(model_or_adapter)

    if adapter_path is None and (model_path_obj / "adapter_config.json").is_file():
        adapter_path = model_or_adapter

    # Also detect remote Hub PEFT adapter repos when adapter_override isn't passed
    if adapter_path is None and not model_path_obj.exists():
        try:
            from peft import PeftConfig

            peft_probe = PeftConfig.from_pretrained(model_or_adapter)
            if getattr(peft_probe, "base_model_name_or_path", None):
                adapter_path = model_or_adapter
        except Exception:
            adapter_path = None

    if adapter_path is not None:
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(adapter_path)
        base_id = base_model_override or peft_cfg.base_model_name_or_path
        if not base_id:
            raise ValueError(f"Could not determine base_model_name_or_path for adapter '{adapter_path}'.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_id)
        base_model = load_causal_or_multimodal_model(
            base_id,
            device_map="auto",
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        info = {
            "base_model": base_id,
            "adapter": adapter_path,
            "dtype": str(dtype),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_or_adapter)
        model = load_causal_or_multimodal_model(
            model_or_adapter,
            device_map="auto",
            torch_dtype=dtype,
        )
        model.eval()
        info = {
            "base_model": model_or_adapter,
            "adapter": None,
            "dtype": str(dtype),
        }

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Stop on Tiny Aya's real end-of-turn markers. Its generation_config only lists
    # <EOS_TOKEN>, so without this every turn runs to max_new_tokens.
    stop_ids = []
    for tok in ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    if stop_ids and getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = stop_ids
        info["stop_token_ids"] = stop_ids
    return model, tokenizer, info


_TOOL_ATTEMPT_HINT_RE = re.compile(
    r'("tool_name"\s*:|<\|START_ACTION\|>|<tool_call>|"function"\s*:)',
    re.IGNORECASE,
)


def _trajectory_has_malformed_call(raw_traj: dict[str, Any]) -> bool:
    """Detect malformed or failed tool-call attempts in one serialized AgentTune trajectory."""
    steps = raw_traj.get("steps") or []
    allowed = set(TOOL_NAMES)
    for step in steps:
        if not isinstance(step, dict):
            continue
        smeta = step.get("metadata") or {}
        if int(smeta.get("tool_failure_count") or 0) > 0:
            return True
        obs = str(step.get("observation") or "")
        action = step.get("action") or {}
        tcs = action.get("tool_calls") if isinstance(action, dict) else None
        if isinstance(tcs, list) and tcs:
            for tc in tcs:
                if not isinstance(tc, dict):
                    return True
                name = tc.get("name") or tc.get("tool_name")
                args = tc.get("arguments")
                if not name or name not in allowed:
                    return True
                # _portable_call leaves unparseable JSON arguments as a raw str
                if not isinstance(args, dict):
                    return True
            if "'error':" in obs or '"error":' in obs:
                return True
        else:
            thought = str(step.get("thought") or "")
            if _TOOL_ATTEMPT_HINT_RE.search(thought):
                return True
    return False


def _extract_traj_called_tools(raw_traj: dict[str, Any]) -> list[str]:
    called: list[str] = []
    for step in raw_traj.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or {}
        tcs = action.get("tool_calls") if isinstance(action, dict) else None
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict) and tc.get("name"):
                    called.append(str(tc["name"]))
    return called


def score_trajectories(
    test_rows: list[dict[str, Any]],
    trajectories_path: str | Path,
    *,
    latencies_s: list[float] | None = None,
    completion_tokens: list[int] | None = None,
    model_info: dict[str, Any] | None = None,
    tag: str = "eval",
) -> dict[str, Any]:
    """Score an AgentTune `trajectories.jsonl` file using AuditKIT + LedgerCall metrics."""
    _ensure_vendor_paths()
    from auditkit.agent_eval import (
        AgentEvalRunner,
        AgentEvalSpec,
        CustomPredicate,
        case_from_episode,
        episodes_from_agenttune,
    )

    traj_path = Path(trajectories_path)
    raw_trajs = load_jsonl(traj_path)
    if len(raw_trajs) != len(test_rows):
        raise ValueError(
            f"Trajectory count ({len(raw_trajs)}) does not match test row count ({len(test_rows)})."
        )

    episodes = episodes_from_agenttune(str(traj_path))
    cases = []
    aligned_episodes = []
    for i, (row, ep) in enumerate(zip(test_rows, episodes, strict=True)):
        cid = str(ep.source_id or f"case_{i}")
        ep_aligned = dataclasses.replace(ep, case_id=cid)
        ep_aligned.metadata["target"] = row["answer"]
        base_case = case_from_episode(ep_aligned, case_id=cid)
        ref_turns = [[{"name": t}] for t in row.get("expected_tools", [])]
        oracle = CustomPredicate(
            lambda ep_obj, _case_obj, gold=row["answer"]: is_numeric_match(
                parse_starred_number(ep_obj.final_answer), gold, rel_tol=0.01
            ),
            version="1",
            name="numeric_star_match",
        )
        enriched_case = dataclasses.replace(
            base_case,
            allowed_tools=list(TOOL_NAMES),
            reference_turns=ref_turns,
            outcome=oracle,
        )
        cases.append(enriched_case)
        aligned_episodes.append(ep_aligned)

    scorers = [
        "tool_call_f1_name",
        "tool_call_validity",
        "redundant_tool_calls",
        "agent_loop_detection",
    ]
    ak_result = AgentEvalRunner().run(
        AgentEvalSpec(
            cases=cases,
            episodes=aligned_episodes,
            mode="recorded",
            scorers=scorers,
            name=f"ledgercall-{tag}",
        )
    )

    # Aggregate AuditKIT diagnostic scorers over eligible cases
    ak_diag_means: dict[str, float | None] = {}
    for scorer_name in scorers:
        vals = [
            float(r.diagnostics[scorer_name]["value"])
            for r in ak_result.rows
            if scorer_name in (r.diagnostics or {})
            and isinstance(r.diagnostics[scorer_name], dict)
            and r.diagnostics[scorer_name].get("value") is not None
        ]
        ak_diag_means[scorer_name] = round(statistics.mean(vals), 4) if vals else None

    ak_verdicts = [r.outcome.get("verdict") for r in ak_result.rows]
    ak_verified_success = (
        round(ak_verdicts.count("success") / len(ak_verdicts), 4) if ak_verdicts else 0.0
    )

    # LedgerCall task & trajectory metrics
    successes: list[float] = []
    tool_recalls: list[float] = []
    exact_seq: list[float] = []
    malformed_flags: list[float] = []
    call_counts: list[int] = []

    for row, raw_traj in zip(test_rows, raw_trajs, strict=True):
        final_resp = str(raw_traj.get("final_response") or "")
        pred_num = parse_starred_number(final_resp)
        successes.append(1.0 if is_numeric_match(pred_num, row["answer"], rel_tol=0.01) else 0.0)

        expected = [str(t) for t in row.get("expected_tools", [])]
        called = _extract_traj_called_tools(raw_traj)
        tool_recalls.append(ordered_tool_recall(called, expected))
        exact_seq.append(1.0 if called == expected else 0.0)
        malformed_flags.append(1.0 if _trajectory_has_malformed_call(raw_traj) else 0.0)

        tmeta = raw_traj.get("metadata") or {}
        n_calls = int(tmeta.get("tool_call_count", len(called)))
        call_counts.append(max(n_calls, len(called)))

    n = len(test_rows)
    total_time_s = float(sum(latencies_s)) if latencies_s else 0.0
    avg_latency = round(total_time_s / n, 4) if (latencies_s and n > 0) else 0.0
    total_tokens = int(sum(completion_tokens)) if completion_tokens else 0
    tokens_per_sec = round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0.0
    avg_tokens = round(total_tokens / n, 2) if (completion_tokens and n > 0) else 0.0

    return {
        "tag": tag,
        "n_evaluated": n,
        "model_info": model_info or {},
        "trajectories_path": str(traj_path),
        "task_success": round(statistics.mean(successes), 4) if successes else 0.0,
        "tool_call_accuracy": round(statistics.mean(tool_recalls), 4) if tool_recalls else 0.0,
        "exact_tool_sequence_rate": round(statistics.mean(exact_seq), 4) if exact_seq else 0.0,
        "malformed_call_rate": round(statistics.mean(malformed_flags), 4) if malformed_flags else 0.0,
        "avg_tool_calls": round(statistics.mean(call_counts), 4) if call_counts else 0.0,
        "avg_latency_s": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "avg_completion_tokens": avg_tokens,
        "total_completion_tokens": total_tokens,
        "total_time_s": round(total_time_s, 3),
        "auditkit": {
            "verified_success_rate": ak_verified_success,
            "diagnostics": ak_diag_means,
            "summary": ak_result.summary(),
        },
    }


def run_evaluation(
    model_or_adapter: str,
    data_path: str | Path,
    *,
    tag: str = "base",
    out_dir: str | Path = "results",
    trajectories_path: str | Path | None = None,
    limit: int | None = None,
    max_steps_per_turn: int = 4,
    max_new_tokens: int = 384,
    base_model_override: str | None = None,
    adapter_override: str | None = None,
) -> dict[str, Any]:
    """Run AgentTune rollouts over `data_path`, write `trajectories.jsonl`, and score with AuditKIT."""
    _ensure_vendor_paths()
    from agenttune.agentic.rollout_engines.rollout_factory import (
        DEFAULT_TOOL_RESULT_FORMAT,
        DEFAULT_TOOLS_FALLBACK_PROMPT,
        create_rollout_engine,
        create_rollout_fn,
    )
    from agenttune.agentic.trajectory.dataset import trajectory_writer

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    traj_out = Path(trajectories_path) if trajectories_path else (out_dir_path / f"{tag}_trajectories.jsonl")
    if traj_out.exists():
        traj_out.unlink()

    test_rows = load_jsonl(data_path, limit=limit)
    model, tokenizer, model_info = load_model_and_tokenizer(
        model_or_adapter,
        base_model_override=base_model_override,
        adapter_override=adapter_override,
    )

    engine = create_rollout_engine(
        backend="transformers",
        model=model,
        tokenizer=tokenizer,
    )
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=TOOLS,
        max_steps=max_steps_per_turn,
        on_trajectory_end=trajectory_writer(traj_out),
        tools_fallback_prompt=DEFAULT_TOOLS_FALLBACK_PROMPT,
        tool_result_format=DEFAULT_TOOL_RESULT_FORMAT,
    )

    latencies_s: list[float] = []
    completion_tokens: list[int] = []

    for idx, row in enumerate(test_rows):
        t0 = time.perf_counter()
        batch = rollout_fn(
            [{"prompt": row["prompt"], "answer": row["answer"]}],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - t0
        latencies_s.append(elapsed)
        comp_ids = (batch.get("completion_ids") or [[]])[0]
        completion_tokens.append(len(comp_ids))
        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_rows):
            print(f"[ledgercall.evaluate:{tag}] {idx + 1}/{len(test_rows)} completed (last={elapsed:.2f}s)")

    metrics = score_trajectories(
        test_rows,
        traj_out,
        latencies_s=latencies_s,
        completion_tokens=completion_tokens,
        model_info=model_info,
        tag=tag,
    )
    metrics_path = out_dir_path / f"{tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ledgercall.evaluate:{tag}] Saved metrics -> {metrics_path}")
    print(metrics["auditkit"]["summary"])
    return metrics


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate base or GRPO-tuned model on LedgerCall test set.")
    parser.add_argument("--model", type=str, default="CohereLabs/tiny-aya-global", help="HF model ID or local PEFT adapter directory")
    parser.add_argument("--base-model", type=str, default=None, help="Optional base HF model ID override when loading an adapter")
    parser.add_argument("--adapter", type=str, default=None, help="Optional explicit PEFT adapter directory or Hub ID")
    parser.add_argument("--data", type=str, default="data/test.jsonl", help="Evaluation dataset JSONL path")
    parser.add_argument("--tag", type=str, default="base", help="Run tag -> writes results/<tag>_metrics.json")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory for metrics and trajectories")
    parser.add_argument("--trajectories", type=str, default=None, help="Custom trajectories.jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N test examples")
    parser.add_argument("--max-steps-per-turn", type=int, default=4, help="Max tool-calling steps per episode")
    parser.add_argument("--max-new-tokens", type=int, default=384, help="Max new tokens per generation turn")
    parser.add_argument("--score-only", action="store_true", help="Score an existing trajectories.jsonl offline without loading a model")
    args = parser.parse_args(argv)

    if args.score_only:
        if not args.trajectories:
            raise ValueError("--score-only requires --trajectories <path>")
        rows = load_jsonl(args.data, limit=args.limit)
        metrics = score_trajectories(rows, args.trajectories, tag=args.tag)
        out_dir_path = Path(args.out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir_path / f"{args.tag}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[ledgercall.evaluate:{args.tag}] Scored existing trajectories -> {metrics_path}")
        return metrics

    return run_evaluation(
        model_or_adapter=args.model,
        data_path=args.data,
        tag=args.tag,
        out_dir=args.out_dir,
        trajectories_path=args.trajectories,
        limit=args.limit,
        max_steps_per_turn=args.max_steps_per_turn,
        max_new_tokens=args.max_new_tokens,
        base_model_override=args.base_model,
        adapter_override=args.adapter,
    )


if __name__ == "__main__":
    main()
