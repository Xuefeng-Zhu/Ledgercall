"""GRPO reward functions for LedgerCall (TRL + AgentTune signature compatible).

All reward functions follow the TRL signature:
    (prompts, completions, answer=None, expected_tools=None, **kwargs) -> list[float]
and are defensive: they never raise on malformed model outputs or missing fields.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

# Matches single-asterisk spans *123.45* while ignoring Markdown bold **...**
_SINGLE_STAR_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_NUMERIC_CLEAN_RE = re.compile(r"(?:USD|EUR|GBP|JPY|INR|[$€£¥₹,%])", re.IGNORECASE)


def parse_starred_number(text: Any) -> float | None:
    """Extract the last finite float wrapped in single asterisks `*...*` from `text`."""
    if not isinstance(text, str) or "*" not in text:
        return None
    candidates = _SINGLE_STAR_RE.findall(text)
    for raw in reversed(candidates):
        cleaned = _NUMERIC_CLEAN_RE.sub("", raw).replace(",", "").strip()
        if not cleaned:
            continue
        try:
            val = float(cleaned)
        except ValueError:
            continue
        if math.isfinite(val):
            return val
    return None


def is_numeric_match(pred: float | None, gold: Any, rel_tol: float = 0.01) -> bool:
    """Check whether `pred` is within `rel_tol` (1%) relative tolerance of `gold`."""
    if pred is None or not math.isfinite(pred):
        return False
    try:
        target = float(str(gold).replace(",", "").strip())
    except (TypeError, ValueError):
        return False
    if not math.isfinite(target):
        return False
    if target == 0.0:
        return abs(pred) <= rel_tol + 1e-9
    return abs(pred - target) / abs(target) <= rel_tol + 1e-9


def _extract_name_from_call_obj(obj: Any) -> list[str]:
    """Extract tool name(s) from a parsed JSON dict or list of tool-call dicts."""
    names: list[str] = []
    if isinstance(obj, list):
        for item in obj:
            names.extend(_extract_name_from_call_obj(item))
        return names
    if not isinstance(obj, dict):
        return names
    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        names.append(str(fn["name"]))
        return names
    for key in ("tool_name", "name"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            # Ensure this looks like a tool-call payload rather than arbitrary JSON
            if (
                key == "tool_name"
                or "arguments" in obj
                or "parameters" in obj
                or "args" in obj
                or obj.get("type") == "function"
            ):
                names.append(val.strip())
                return names
    return names


def parse_tool_names_from_text(text: Any) -> list[str]:
    """Scan `text` for embedded JSON tool-call lists/objects (Cohere, Tiny Aya fallback, Hermes, OpenAI)."""
    if not isinstance(text, str) or not text:
        return []
    decoder = json.JSONDecoder()
    names: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ("[", "{"):
            try:
                parsed, end_idx = decoder.raw_decode(text, i)
                extracted = _extract_name_from_call_obj(parsed)
                if extracted:
                    names.extend(extracted)
                    i = end_idx
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
        i += 1
    return names


def _extract_final_text(completion_item: Any, response_text: Any = None) -> str:
    """Resolve the assistant text to inspect for the final `*...*` answer."""
    if isinstance(response_text, str) and "*" in response_text:
        return response_text
    if isinstance(completion_item, str):
        return completion_item
    if isinstance(completion_item, dict):
        return str(completion_item.get("content") or "")
    if isinstance(completion_item, list):
        assistant_texts = [
            str(m.get("content") or "")
            for m in completion_item
            if isinstance(m, dict) and m.get("role", "assistant") == "assistant" and m.get("content")
        ]
        if assistant_texts:
            return "\n".join(assistant_texts)
    if isinstance(response_text, str):
        return response_text
    return str(completion_item or "")


def extract_called_tools(
    completion_item: Any,
    trajectory: Any = None,
    conversation: Any = None,
) -> list[str]:
    """Extract ordered list of tool names called in one rollout from AgentTune or TRL structures."""
    # 1. Authoritative AgentTune Trajectory object / dict if provided in kwargs
    if trajectory is not None:
        steps = getattr(trajectory, "steps", None)
        if steps is None and isinstance(trajectory, dict):
            steps = trajectory.get("steps")
        if isinstance(steps, list) and steps:
            traj_names: list[str] = []
            for step in steps:
                action = getattr(step, "action", None)
                if action is None and isinstance(step, dict):
                    action = step.get("action")
                if isinstance(action, dict):
                    tcs = action.get("tool_calls")
                    if isinstance(tcs, list):
                        for tc in tcs:
                            if isinstance(tc, dict):
                                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                                name = fn.get("name") or tc.get("name") or tc.get("tool_name")
                                if name:
                                    traj_names.append(str(name))
                    else:
                        name = action.get("name") or action.get("tool_name")
                        if name:
                            traj_names.append(str(name))
            return traj_names

    # 2. AgentTune reward_conversation / conversation list[dict] if provided in kwargs
    if isinstance(conversation, list) and conversation:
        conv_names: list[str] = []
        for msg in conversation:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                for tc in tcs:
                    if isinstance(tc, dict):
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        name = fn.get("name") or tc.get("name") or tc.get("tool_name")
                        if name:
                            conv_names.append(str(name))
            elif msg.get("content"):
                conv_names.extend(parse_tool_names_from_text(msg.get("content")))
        return conv_names

    # 3. Direct TRL completion (str, dict, or list[dict] of messages)
    if isinstance(completion_item, str):
        return parse_tool_names_from_text(completion_item)
    if isinstance(completion_item, dict):
        tcs = completion_item.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            out: list[str] = []
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name") or tc.get("tool_name")
                    if name:
                        out.append(str(name))
            return out
        return parse_tool_names_from_text(completion_item.get("content"))
    if isinstance(completion_item, list):
        out = []
        for msg in completion_item:
            if isinstance(msg, str):
                out.extend(parse_tool_names_from_text(msg))
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("role", "assistant") != "assistant":
                continue
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                for tc in tcs:
                    if isinstance(tc, dict):
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        name = fn.get("name") or tc.get("name") or tc.get("tool_name")
                        if name:
                            out.append(str(name))
            elif msg.get("content"):
                out.extend(parse_tool_names_from_text(msg.get("content")))
        return out
    return []


def _normalize_inputs(
    prompts: Any,
    completions: Any,
) -> tuple[Any, list[Any]]:
    """Support both `(prompts, completions, ...)` and `(completions=..., ...)` call patterns."""
    if completions is None and prompts is not None:
        completions = prompts
        prompts = None
    if completions is None:
        return prompts, []
    try:
        comp_list = list(completions)
    except TypeError:
        return prompts, []
    return prompts, comp_list


def _item_at(seq: Any, idx: int, default: Any = None) -> Any:
    if seq is None:
        return default
    if isinstance(seq, (list, tuple)):
        return seq[idx] if idx < len(seq) else default
    return seq


def ordered_tool_recall(called_tools: list[str], expected: list[str]) -> float:
    """Fraction of `expected` tools called in sequential order (`[0.0, 1.0]`)."""
    if not expected:
        return 1.0 if not called_tools else 0.0
    exp_idx = 0
    for name in called_tools:
        if name == expected[exp_idx]:
            exp_idx += 1
            if exp_idx == len(expected):
                break
    return exp_idx / len(expected)


def correctness(
    prompts: Any = None,
    completions: Any = None,
    answer: Any = None,
    expected_tools: Any = None,
    **kwargs: Any,
) -> list[float]:
    """Reward 1.0 when `*...*` numeric answer is within 1% relative tolerance of `answer`, else 0.0."""
    try:
        _, comp_list = _normalize_inputs(prompts, completions)
        responses = kwargs.get("responses")
        scores: list[float] = []
        for i, comp in enumerate(comp_list):
            try:
                gold = _item_at(answer, i)
                resp_text = _item_at(responses, i)
                text = _extract_final_text(comp, resp_text)
                pred = parse_starred_number(text)
                scores.append(1.0 if is_numeric_match(pred, gold, rel_tol=0.01) else 0.0)
            except Exception:
                scores.append(0.0)
        return scores
    except Exception:
        return []


def tool_choice(
    prompts: Any = None,
    completions: Any = None,
    answer: Any = None,
    expected_tools: Any = None,
    **kwargs: Any,
) -> list[float]:
    """Reward in `[0.0, 1.0]` equal to the fraction of `expected_tools` called in order."""
    try:
        _, comp_list = _normalize_inputs(prompts, completions)
        trajectories = kwargs.get("trajectories")
        conversations = kwargs.get("conversations")
        scores: list[float] = []
        for i, comp in enumerate(comp_list):
            try:
                exp = _item_at(expected_tools, i, default=[])
                if isinstance(exp, str):
                    exp_list = [exp]
                elif isinstance(exp, (list, tuple)):
                    exp_list = [str(x) for x in exp]
                else:
                    exp_list = []
                called = extract_called_tools(
                    comp,
                    trajectory=_item_at(trajectories, i),
                    conversation=_item_at(conversations, i),
                )
                scores.append(float(ordered_tool_recall(called, exp_list)))
            except Exception:
                scores.append(0.0)
        return scores
    except Exception:
        return []


def efficiency(
    prompts: Any = None,
    completions: Any = None,
    answer: Any = None,
    expected_tools: Any = None,
    **kwargs: Any,
) -> list[float]:
    """Non-positive penalty (`[-1.0, 0.0]`) for tool calls exceeding `len(expected_tools)`.

    Returns `0.0` when `actual_calls <= expected_calls` (avoiding a positive reward
    for skipping tools) and `-0.25` per extra call beyond expected (capped at `-1.0`).
    """
    try:
        _, comp_list = _normalize_inputs(prompts, completions)
        trajectories = kwargs.get("trajectories")
        conversations = kwargs.get("conversations")
        tool_call_counts = kwargs.get("tool_call_counts")
        scores: list[float] = []
        for i, comp in enumerate(comp_list):
            try:
                exp = _item_at(expected_tools, i, default=[])
                if isinstance(exp, str):
                    exp_count = 1
                elif isinstance(exp, (list, tuple)):
                    exp_count = len(exp)
                else:
                    exp_count = 0

                called = extract_called_tools(
                    comp,
                    trajectory=_item_at(trajectories, i),
                    conversation=_item_at(conversations, i),
                )
                actual_count = len(called)
                recorded_count = _item_at(tool_call_counts, i)
                if isinstance(recorded_count, (int, float)) and not isinstance(recorded_count, bool):
                    actual_count = max(actual_count, int(recorded_count))

                excess = max(0, actual_count - exp_count)
                penalty = 0.0 if excess == 0 else -min(1.0, 0.25 * excess)
                scores.append(float(penalty))
            except Exception:
                scores.append(0.0)
        return scores
    except Exception:
        return []


REWARD_FUNCS = [correctness, tool_choice, efficiency]
