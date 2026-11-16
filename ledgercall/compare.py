"""Compare base vs SFT-tuned evaluation metrics and compute Cost/ROI analysis.

Reads `results/base_metrics.json` and `results/tuned_metrics.json`, prints a
Markdown before/after delta table plus Cost/ROI summary, and writes
`results/comparison.md`.

Example usage:
    python -m ledgercall.compare
    python -m ledgercall.compare --base results/base_metrics.json --tuned results/tuned_metrics.json --out results/comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# (key_path, label, direction, fmt)
_METRIC_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("task_success", "Task Success (numeric ≤1% tol)", "↑ higher", "pct"),
    ("tool_call_accuracy", "Tool Call Accuracy (ordered recall)", "↑ higher", "pct"),
    ("exact_tool_sequence_rate", "Exact Tool Sequence Match", "↑ higher", "pct"),
    ("malformed_call_rate", "Malformed / Failed Call Rate", "↓ lower", "pct"),
    ("avg_tool_calls", "Avg Tool Calls / Task", "↓ efficient", "num"),
    ("avg_latency_s", "Avg Latency / Task (s)", "↓ lower", "sec"),
    ("avg_completion_tokens", "Avg Completion Tokens / Task", "↓ lower", "num"),
    ("tokens_per_sec", "Throughput (completion tok/s)", "↑ higher", "num"),
    ("auditkit.verified_success_rate", "AuditKIT Verified Success", "↑ higher", "pct"),
    ("auditkit.diagnostics.tool_call_f1_name", "AuditKIT Tool Call F1 (name)", "↑ higher", "pct"),
    ("auditkit.diagnostics.tool_call_validity", "AuditKIT Tool Call Validity", "↑ higher", "pct"),
    ("auditkit.diagnostics.redundant_tool_calls", "AuditKIT Redundant Tool Calls", "↓ lower", "num"),
    ("auditkit.diagnostics.agent_loop_detection", "AuditKIT No-Loop Score", "↑ higher", "pct"),
)


def _lookup(d: dict[str, Any], dotted_key: str) -> float | None:
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _fmt_val(val: float | None, fmt: str) -> str:
    if val is None:
        return "n/a"
    if fmt == "pct":
        return f"{val * 100:.1f}%"
    if fmt == "sec":
        return f"{val:.2f}s"
    return f"{val:.2f}"


def _fmt_delta(base_val: float | None, tuned_val: float | None, fmt: str) -> str:
    if base_val is None or tuned_val is None:
        return "n/a"
    diff = tuned_val - base_val
    sign = "+" if diff >= 0 else ""
    if fmt == "pct":
        return f"{sign}{diff * 100:.1f} pp"
    if fmt == "sec":
        return f"{sign}{diff:.2f}s"
    return f"{sign}{diff:.2f}"


def build_cost_roi_section(
    base_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
    *,
    t4_hourly_usd: float = 0.35,
    gpt4o_mini_input_per_1m_usd: float = 0.15,
    gpt4o_mini_output_per_1m_usd: float = 0.60,
    assumed_prompt_tokens_per_task: int = 650,
) -> str:
    """Build a Markdown Cost & ROI breakdown from measured evaluation throughput/latency.

    Note: Proprietary API pricing and prompt token overhead are clearly marked
    as placeholder assumptions to verify at submission time.
    """
    tuned_lat = float(tuned_metrics.get("avg_latency_s") or 0.0)
    base_lat = float(base_metrics.get("avg_latency_s") or 0.0)
    tuned_tps = float(tuned_metrics.get("tokens_per_sec") or 0.0)
    base_tps = float(base_metrics.get("tokens_per_sec") or 0.0)
    tuned_out_toks = float(tuned_metrics.get("avg_completion_tokens") or 120.0)
    if tuned_out_toks <= 0:
        tuned_out_toks = 120.0

    # Self-hosted T4 cost per 1,000 tasks = (hours for 1,000 tasks) * $0.35/hr
    tuned_hours_per_1k = (tuned_lat * 1000.0) / 3600.0 if tuned_lat > 0 else 0.0
    base_hours_per_1k = (base_lat * 1000.0) / 3600.0 if base_lat > 0 else 0.0
    tuned_t4_cost_1k = tuned_hours_per_1k * t4_hourly_usd
    base_t4_cost_1k = base_hours_per_1k * t4_hourly_usd

    # Placeholder hosted API comparison (GPT-4o-mini list-price assumption)
    api_in_cost_1k = (assumed_prompt_tokens_per_task * 1000.0 / 1_000_000.0) * gpt4o_mini_input_per_1m_usd
    api_out_cost_1k = (tuned_out_toks * 1000.0 / 1_000_000.0) * gpt4o_mini_output_per_1m_usd
    api_total_cost_1k = api_in_cost_1k + api_out_cost_1k

    t4_cost_str = f"${tuned_t4_cost_1k:.4f}" if tuned_lat > 0 else "n/a (offline score-only run)"
    base_cost_str = f"${base_t4_cost_1k:.4f}" if base_lat > 0 else "n/a (offline score-only run)"

    lines = [
        "## Cost & ROI Analysis",
        "",
        "| Deployment Option | Throughput (tok/s) | Avg Latency / Task | Est. Cost / 1k Tasks | Data Governance |",
        "| :--- | ---: | ---: | ---: | :--- |",
        f"| **Tiny Aya Base (T4 @ ${t4_hourly_usd:.2f}/hr)** | {base_tps:.1f} | {base_lat:.2f}s | {base_cost_str} | Self-hosted / VPC |",
        f"| **LedgerCall Tuned LoRA (T4 @ ${t4_hourly_usd:.2f}/hr)** | {tuned_tps:.1f} | {tuned_lat:.2f}s | {t4_cost_str} | Self-hosted / VPC |",
        "",
        "> **Assumptions:**",
        f"> - T4-equivalent cloud spot/on-demand rate assumed at **${t4_hourly_usd:.2f}/hr** (sequential batch=1 rollout latency; continuous batching lowers unit cost further).",
    ]
    return "\n".join(lines)


def build_comparison_markdown(
    base_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
) -> str:
    """Render the full before/after Markdown comparison report."""
    n_base = base_metrics.get("n_evaluated", "?")
    n_tuned = tuned_metrics.get("n_evaluated", "?")

    lines = [
        "# LedgerCall Evaluation Comparison: Base vs SFT-Tuned",
        "",
        f"- **Base run (`{base_metrics.get('tag', 'base')}`)**: `n={n_base}`, model `{base_metrics.get('model_info', {}).get('base_model', 'CohereLabs/tiny-aya-global')}`",
        f"- **Tuned run (`{tuned_metrics.get('tag', 'tuned')}`)**: `n={n_tuned}`, adapter `{tuned_metrics.get('model_info', {}).get('adapter', 'out/grpo')}`",
        "",
        "## Before / After Metrics",
        "",
        "| Metric | Direction | Base | SFT-Tuned | Delta (Tuned - Base) |",
        "| :--- | :---: | ---: | ---: | ---: |",
    ]

    for key_path, label, direction, fmt in _METRIC_SPECS:
        b_val = _lookup(base_metrics, key_path)
        t_val = _lookup(tuned_metrics, key_path)
        lines.append(
            f"| {label} | {direction} | {_fmt_val(b_val, fmt)} | {_fmt_val(t_val, fmt)} | **{_fmt_delta(b_val, t_val, fmt)}** |"
        )

    lines.append("")
    lines.append(build_cost_roi_section(base_metrics, tuned_metrics))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="Compare base and tuned LedgerCall metrics.")
    parser.add_argument("--base", type=str, default="results/base_metrics.json", help="Base metrics JSON path")
    parser.add_argument("--tuned", type=str, default="results/tuned_metrics.json", help="Tuned metrics JSON path")
    parser.add_argument("--out", type=str, default="results/comparison.md", help="Output Markdown report path")
    args = parser.parse_args(argv)

    with open(args.base, encoding="utf-8") as f:
        base_metrics = json.load(f)
    with open(args.tuned, encoding="utf-8") as f:
        tuned_metrics = json.load(f)

    report_md = build_comparison_markdown(base_metrics, tuned_metrics)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    print(report_md)
    print(f"[ledgercall.compare] Wrote comparison report -> {out_path}")
    return report_md


if __name__ == "__main__":
    main()
