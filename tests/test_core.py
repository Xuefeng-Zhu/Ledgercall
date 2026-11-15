"""CPU-only unit tests for LedgerCall tools, data generator, rewards, offline AuditKIT eval, and comparison."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from ledgercall.compare import build_comparison_markdown
from ledgercall.data import check_leakage, generate_splits
from ledgercall.data import main as data_main
from ledgercall.evaluate import score_trajectories
from ledgercall.rewards import correctness, efficiency, parse_starred_number, tool_choice
from ledgercall.tools import (
    TOOL_NAMES,
    TOOLS,
    compound_interest,
    convert_currency,
    loan_payment,
    percent_change,
)


# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------


def test_tools_registry_and_google_docstrings() -> None:
    assert len(TOOLS) == 4
    assert TOOL_NAMES == [
        "convert_currency",
        "compound_interest",
        "percent_change",
        "loan_payment",
    ]
    for fn in TOOLS:
        doc = inspect.getdoc(fn) or ""
        assert "Args:" in doc, f"{fn.__name__} missing Google-style Args section"
        assert "Returns:" in doc, f"{fn.__name__} missing Google-style Returns section"
        for param in inspect.signature(fn).parameters:
            assert f"{param}:" in doc, f"{fn.__name__} docstring missing parameter '{param}'"


def test_convert_currency() -> None:
    assert convert_currency(100.0, "USD", "EUR") == 92.00
    assert convert_currency(92.0, "EUR", "USD") == 100.00
    assert convert_currency(100.0, "usd", "jpy") == 15200.00
    assert convert_currency(123.456, "GBP", "GBP") == 123.46
    with pytest.raises(ValueError):
        convert_currency(50.0, "CHF", "USD")


def test_compound_interest() -> None:
    # 1000 * (1 + 0.08/4)**8 = 1171.65938... -> 1171.66
    assert compound_interest(1000.0, 8.0, 2, 4) == 1171.66
    assert compound_interest(2500.0, 5.0, 0, 12) == 2500.00
    with pytest.raises(ValueError):
        compound_interest(1000.0, 5.0, -1, 4)
    with pytest.raises(ValueError):
        compound_interest(1000.0, 5.0, 2, 0)


def test_percent_change() -> None:
    assert percent_change(100.0, 125.0) == 25.00
    assert percent_change(200.0, 150.0) == -25.00
    assert percent_change(3.0, 4.0) == 33.33
    with pytest.raises(ValueError):
        percent_change(0.0, 10.0)


def test_loan_payment() -> None:
    # 12000 at 6% APR over 12 months -> monthly rate 0.005 -> 1032.80
    assert loan_payment(12000.0, 6.0, 12) == 1032.80
    assert loan_payment(1200.0, 0.0, 12) == 100.00
    with pytest.raises(ValueError):
        loan_payment(5000.0, 5.0, 0)


# ---------------------------------------------------------------------------
# 2. Data determinism & leak check
# ---------------------------------------------------------------------------


def test_data_determinism_and_zero_leakage() -> None:
    train_a, test_a, stats_a = generate_splits(n_train=600, n_test=150, seed=42)
    train_b, test_b, _ = generate_splits(n_train=600, n_test=150, seed=42)
    train_c, _, _ = generate_splits(n_train=40, n_test=10, seed=99)

    assert train_a == train_b
    assert test_a == test_b
    assert train_a[:40] != train_c

    assert len(train_a) == 600
    assert len(test_a) == 150
    assert stats_a["leak_overlap_count"] == 0
    assert check_leakage(train_a, test_a) == 0
    # Verify leak detector catches an injected duplicate
    assert check_leakage(train_a, [train_a[0], *test_a[:5]]) == 1

    single_steps = [r for r in train_a if len(r["expected_tools"]) == 1]
    two_steps = [r for r in train_a if len(r["expected_tools"]) == 2]
    assert len(single_steps) > 200
    assert len(two_steps) > 200

    for row in test_a[:20]:
        assert isinstance(row["prompt"], list)
        assert any("*" in m["content"] for m in row["prompt"])
        float(row["answer"])
        assert set(row["expected_tools"]).issubset(set(TOOL_NAMES))
        assert row["language"] == "en"


def test_data_cli_writes_files(tmp_path: Path) -> None:
    stats = data_main(["--out", str(tmp_path), "--train-size", "32", "--test-size", "8", "--seed", "7"])
    assert stats["leak_overlap_count"] == 0
    assert (tmp_path / "train.jsonl").is_file()
    assert (tmp_path / "test.jsonl").is_file()
    assert (tmp_path / "stats.json").is_file()


# ---------------------------------------------------------------------------
# 3. GRPO Reward functions
# ---------------------------------------------------------------------------


def test_correctness_reward_tolerance_and_parsing() -> None:
    assert parse_starred_number("**Final Answer:** *1,234.56*") == 1234.56
    assert parse_starred_number("Unwrapped 123.45") is None

    prompts = [[{"role": "user", "content": "q"}]] * 5
    completions = [
        [{"role": "assistant", "content": "Result: *100.00*"}],      # exact
        [{"role": "assistant", "content": "**Ans:** *100.95*"}],     # +0.95% <= 1%
        [{"role": "assistant", "content": "Result: *101.50*"}],      # +1.50% > 1%
        [{"role": "assistant", "content": "Result is 100.00"}],      # missing stars
        [{"role": "assistant", "content": "Broken *not_a_number*"}], # non-numeric
    ]
    answers = ["100.00"] * 5
    scores = correctness(prompts, completions, answer=answers)
    assert scores == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_tool_choice_across_agenttune_and_text_formats() -> None:
    expected = [
        ["compound_interest", "convert_currency"],
        ["compound_interest", "convert_currency"],
        ["compound_interest", "convert_currency"],
        ["percent_change"],
    ]
    # 0: Tiny Aya fallback JSON list across turns -> 2/2 = 1.0
    comp_0 = (
        '[{"tool_name": "compound_interest", "parameters": {"principal": 1000, "annual_rate_pct": 5, "years": 2, "compounds_per_year": 1}}]\n'
        '[{"tool_name": "convert_currency", "parameters": {"amount": 1102.5, "from_ccy": "USD", "to_ccy": "EUR"}}]\n*1014.30*'
    )
    # 1: Reversed order -> 1/2 = 0.5
    comp_1 = [
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": {"name": "convert_currency", "arguments": "{}"}},
                {"type": "function", "function": {"name": "compound_interest", "arguments": "{}"}},
            ],
        }
    ]
    # 2: Only first tool called -> 1/2 = 0.5
    comp_2 = '<|START_ACTION|>[{"tool_name": "compound_interest", "parameters": {}}]<|END_ACTION|>'
    # 3: Wrong tool called -> 0/1 = 0.0
    comp_3 = '[{"tool_name": "loan_payment", "parameters": {"principal": 100, "annual_rate_pct": 5, "months": 12}}]'

    scores = tool_choice([None] * 4, [comp_0, comp_1, comp_2, comp_3], expected_tools=expected)
    assert scores == [1.0, 0.5, 0.5, 0.0]

    # Verify AgentTune `conversations` kwarg path
    conv = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "percent_change", "arguments": "{}"}}]},
        {"role": "tool", "name": "percent_change", "content": "25.0"},
        {"role": "assistant", "content": "*25.00*"},
    ]
    assert tool_choice([None], ["*25.00*"], expected_tools=[["percent_change"]], conversations=[conv]) == [1.0]


def test_efficiency_penalty_and_robustness() -> None:
    expected = [["convert_currency"], ["convert_currency"], ["convert_currency"], ["convert_currency"]]
    completions = [
        "",  # 0 calls -> 0.0 (no reward-hacking bonus for skipping tools)
        '[{"tool_name": "convert_currency", "parameters": {}}]',  # 1 call == expected -> 0.0
        '[{"tool_name": "convert_currency", "parameters": {}}, {"tool_name": "convert_currency", "parameters": {}}]',  # +1 excess -> -0.25
        '[{"tool_name": "convert_currency", "parameters": {}}]' * 6,  # +5 excess -> capped -1.0
    ]
    assert efficiency([None] * 4, completions, expected_tools=expected) == [0.0, 0.0, -0.25, -1.0]

    # Never raise on malformed/None inputs
    for fn in (correctness, tool_choice, efficiency):
        assert fn(None, None) == []
        assert fn(None, [None, 12345, {"bad": object()}], answer=[None, "bad", "10.0"], expected_tools=None) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 4. Offline AuditKIT evaluation & comparison report
# ---------------------------------------------------------------------------


def test_offline_auditkit_scoring_and_comparison(tmp_path: Path) -> None:
    test_rows = [
        {
            "prompt": [{"role": "user", "content": "Convert 100 USD to EUR. *123.45*"}],
            "answer": "92.00",
            "expected_tools": ["convert_currency"],
            "task_type": "single_convert_currency",
            "language": "en",
        }
    ]
    base_traj = {
        "task": test_rows[0]["prompt"],
        "trajectory_id": "base-1",
        "reward": None,
        "final_response": "I estimate *85.00*.",
        "steps": [{"step_number": 0, "state": "s", "action": {}, "observation": "I estimate *85.00*.", "thought": "I estimate *85.00*.", "reward": None}],
        "metadata": {"conversation": [], "tool_call_count": 0, "tool_failure_count": 0, "answer": "92.00"},
    }
    tuned_traj = {
        "task": test_rows[0]["prompt"],
        "trajectory_id": "tuned-1",
        "reward": None,
        "final_response": "Converted amount: *92.00*.",
        "steps": [
            {
                "step_number": 0,
                "state": "s0",
                "action": {"tool_calls": [{"id": "c1", "name": "convert_currency", "arguments": {"amount": 100.0, "from_ccy": "USD", "to_ccy": "EUR"}}]},
                "observation": "{'convert_currency': '92.0'}",
                "thought": "call",
                "reward": None,
            },
            {"step_number": 1, "state": "s1", "action": {}, "observation": "Converted amount: *92.00*.", "thought": "Converted amount: *92.00*.", "reward": None},
        ],
        "metadata": {"conversation": [], "tool_call_count": 1, "tool_failure_count": 0, "answer": "92.00"},
    }

    base_p = tmp_path / "base_traj.jsonl"
    tuned_p = tmp_path / "tuned_traj.jsonl"
    base_p.write_text(json.dumps(base_traj) + "\n", encoding="utf-8")
    tuned_p.write_text(json.dumps(tuned_traj) + "\n", encoding="utf-8")

    m_base = score_trajectories(test_rows, base_p, latencies_s=[0.8], completion_tokens=[20], tag="base")
    m_tuned = score_trajectories(test_rows, tuned_p, latencies_s=[0.9], completion_tokens=[36], tag="tuned")

    assert m_base["task_success"] == 0.0
    assert m_tuned["task_success"] == 1.0
    assert m_tuned["auditkit"]["verified_success_rate"] == 1.0
    assert m_tuned["auditkit"]["diagnostics"]["tool_call_f1_name"] == 1.0
    assert m_tuned["auditkit"]["diagnostics"]["tool_call_validity"] == 1.0

    md = build_comparison_markdown(m_base, m_tuned)
    assert "+100.0 pp" in md
    assert "Cost & ROI Analysis" in md
