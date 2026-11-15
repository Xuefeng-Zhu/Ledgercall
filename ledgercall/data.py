"""Deterministic synthetic dataset generator for LedgerCall finance tool-calling tasks.

Generates single-tool and 2-step sequential tool-composition tasks with verified
ground-truth outputs and an explicit train/test prompt leakage check.

CLI usage:
    python -m ledgercall.data --out data/
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .tools import (
    SUPPORTED_CURRENCIES,
    compound_interest,
    convert_currency,
    loan_payment,
    percent_change,
)

SYSTEM_PROMPT = (
    "You are a precise financial operations assistant. Always call the provided "
    "finance tools instead of doing mental math. After receiving tool results, "
    "provide your final numeric answer wrapped in single asterisks, like *123.45*."
)

_COMPOUND_LABELS: dict[int, str] = {
    1: "annually",
    2: "semi-annually",
    4: "quarterly",
    12: "monthly",
}

_LOAN_MONTHS: tuple[int, ...] = (12, 18, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360)


def _prompt_key(row: dict[str, Any]) -> str:
    """Canonical key for exact-prompt deduplication and train/test leakage checks."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        user_turns = [
            str(m.get("content", "")).strip()
            for m in prompt
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_turns:
            return "\n".join(user_turns)
    return json.dumps(prompt, sort_keys=True)


def check_leakage(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> int:
    """Return the count of test rows whose prompt appears identically in train_rows."""
    train_keys = {_prompt_key(r) for r in train_rows}
    return sum(1 for r in test_rows if _prompt_key(r) in train_keys)


def _make_row(user_text: str, answer_val: float, expected_tools: list[str], task_type: str) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "answer": f"{answer_val:.2f}",
        "expected_tools": expected_tools,
        "task_type": task_type,
        "language": "en",
    }


def _sample_ccy_pair(rng: random.Random) -> tuple[str, str]:
    src, dst = rng.sample(list(SUPPORTED_CURRENCIES), 2)
    return src, dst


def _gen_single_convert_currency(rng: random.Random) -> dict[str, Any]:
    amount = round(rng.uniform(25.0, 25000.0), 2)
    src, dst = _sample_ccy_pair(rng)
    templates = (
        f"Convert {amount:.2f} {src} to {dst} using the currency conversion tool. "
        "Wrap your final numeric answer like *123.45*.",
        f"A vendor invoice totals {amount:.2f} {src}. How much is that in {dst}? "
        "Wrap the final converted number in asterisks like *123.45*.",
        f"Treasury needs {amount:.2f} {src} expressed in {dst}. Use the FX tool and "
        "return your final answer wrapped like *123.45*.",
    )
    ans = convert_currency(amount, src, dst)
    return _make_row(rng.choice(templates), ans, ["convert_currency"], "single_convert_currency")


def _gen_single_compound_interest(rng: random.Random) -> dict[str, Any]:
    principal = round(rng.uniform(500.0, 50000.0), 2)
    rate = round(rng.uniform(1.25, 14.50), 2)
    years = rng.randint(1, 20)
    n = rng.choice((1, 2, 4, 12))
    freq = _COMPOUND_LABELS[n]
    templates = (
        f"Calculate the final total maturity balance on a principal of {principal:.2f} USD "
        f"at {rate:.2f}% annual interest for {years} years, compounded {freq} ({n} times per year). "
        "Wrap your final numeric answer like *123.45*.",
        f"A reserve account deposits {principal:.2f} USD earning {rate:.2f}% APR over {years} years, "
        f"compounded {freq} ({n} compounds per year). What is the ending balance? "
        "Return the final number wrapped like *123.45*.",
    )
    ans = compound_interest(principal, rate, years, n)
    return _make_row(rng.choice(templates), ans, ["compound_interest"], "single_compound_interest")


def _gen_single_percent_change(rng: random.Random) -> dict[str, Any]:
    old = round(rng.uniform(50.0, 20000.0), 2)
    multiplier = rng.uniform(0.45, 1.85)
    new = round(old * multiplier, 2)
    if abs(new - old) < 1.0:
        new = round(old + 12.50, 2)
    templates = (
        f"Operating revenue moved from {old:.2f} to {new:.2f}. Calculate the percentage change "
        f"from {old:.2f} to {new:.2f} and wrap your final numeric answer like *123.45*.",
        f"Unit cost was {old:.2f} last period and is {new:.2f} this period. What is the "
        "percentage change from old to new? Wrap the final answer like *123.45*.",
    )
    ans = percent_change(old, new)
    return _make_row(rng.choice(templates), ans, ["percent_change"], "single_percent_change")


def _gen_single_loan_payment(rng: random.Random) -> dict[str, Any]:
    principal = round(rng.uniform(2500.0, 250000.0), 2)
    rate = round(rng.uniform(2.0, 15.0), 2)
    months = rng.choice(_LOAN_MONTHS)
    templates = (
        f"Compute the fixed monthly payment on an amortizing loan of {principal:.2f} USD "
        f"at {rate:.2f}% annual interest repaid over {months} months. "
        "Wrap your final numeric answer like *123.45*.",
        f"Equipment financing of {principal:.2f} USD carries a {rate:.2f}% annual interest rate "
        f"for a term of {months} months. What is the monthly loan payment? "
        "Return the final number wrapped like *123.45*.",
    )
    ans = loan_payment(principal, rate, months)
    return _make_row(rng.choice(templates), ans, ["loan_payment"], "single_loan_payment")


def _gen_two_step_ci_fx(rng: random.Random) -> dict[str, Any]:
    principal = round(rng.uniform(1000.0, 40000.0), 2)
    rate = round(rng.uniform(1.5, 12.5), 2)
    years = rng.randint(1, 15)
    n = rng.choice((1, 2, 4, 12))
    freq = _COMPOUND_LABELS[n]
    src, dst = _sample_ccy_pair(rng)
    maturity = compound_interest(principal, rate, years, n)
    ans = convert_currency(maturity, src, dst)
    user_text = (
        f"First compute the final maturity balance of {principal:.2f} {src} invested at "
        f"{rate:.2f}% annual interest for {years} years, compounded {freq} ({n} times per year). "
        f"Then convert that maturity balance from {src} to {dst}. "
        "Wrap your final numeric answer like *123.45*."
    )
    return _make_row(
        user_text,
        ans,
        ["compound_interest", "convert_currency"],
        "two_step_ci_fx",
    )


def _gen_two_step_loan_fx(rng: random.Random) -> dict[str, Any]:
    principal = round(rng.uniform(5000.0, 180000.0), 2)
    rate = round(rng.uniform(2.25, 13.75), 2)
    months = rng.choice(_LOAN_MONTHS)
    src, dst = _sample_ccy_pair(rng)
    monthly = loan_payment(principal, rate, months)
    ans = convert_currency(monthly, src, dst)
    user_text = (
        f"First calculate the monthly payment on a {principal:.2f} {src} loan at "
        f"{rate:.2f}% annual interest over {months} months. "
        f"Then convert that monthly payment from {src} to {dst}. "
        "Wrap your final numeric answer like *123.45*."
    )
    return _make_row(
        user_text,
        ans,
        ["loan_payment", "convert_currency"],
        "two_step_loan_fx",
    )


def _gen_two_step_fx_pct(rng: random.Random) -> dict[str, Any]:
    amount = round(rng.uniform(200.0, 30000.0), 2)
    src, dst = _sample_ccy_pair(rng)
    converted = convert_currency(amount, src, dst)
    baseline = round(converted * rng.uniform(0.55, 1.55), 2)
    if baseline == 0.0 or abs(converted - baseline) < 0.5:
        baseline = round(converted + 25.0, 2)
    ans = percent_change(baseline, converted)
    user_text = (
        f"Last quarter's regional spend baseline was {baseline:.2f} {dst}. "
        f"This quarter's spend was {amount:.2f} {src}. "
        f"First convert {amount:.2f} {src} to {dst}, then calculate the percentage change "
        f"from {baseline:.2f} {dst} to the converted amount. "
        "Wrap your final numeric answer like *123.45*."
    )
    return _make_row(
        user_text,
        ans,
        ["convert_currency", "percent_change"],
        "two_step_fx_pct",
    )


def _gen_two_step_ci_pct(rng: random.Random) -> dict[str, Any]:
    principal = round(rng.uniform(1000.0, 35000.0), 2)
    rate = round(rng.uniform(2.0, 12.0), 2)
    years = rng.randint(1, 15)
    n = rng.choice((1, 2, 4, 12))
    freq = _COMPOUND_LABELS[n]
    maturity = compound_interest(principal, rate, years, n)
    benchmark = round(maturity * rng.uniform(0.70, 1.30), 2)
    if benchmark == 0.0 or abs(maturity - benchmark) < 0.5:
        benchmark = round(maturity + 50.0, 2)
    ans = percent_change(benchmark, maturity)
    user_text = (
        f"A treasury deposit of {principal:.2f} USD earns {rate:.2f}% annual interest for "
        f"{years} years, compounded {freq} ({n} times per year). "
        f"First compute its final maturity balance, then calculate the percentage change "
        f"from a target benchmark of {benchmark:.2f} USD to that maturity balance. "
        "Wrap your final numeric answer like *123.45*."
    )
    return _make_row(
        user_text,
        ans,
        ["compound_interest", "percent_change"],
        "two_step_ci_pct",
    )


_GENERATORS = (
    _gen_single_convert_currency,
    _gen_single_compound_interest,
    _gen_single_percent_change,
    _gen_single_loan_payment,
    _gen_two_step_ci_fx,
    _gen_two_step_loan_fx,
    _gen_two_step_fx_pct,
    _gen_two_step_ci_pct,
)


def generate_splits(
    n_train: int = 600,
    n_test: int = 150,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministically generate non-overlapping train and test splits plus stats."""
    rng = random.Random(seed)
    total_needed = n_train + n_test
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()

    idx = 0
    max_attempts = total_needed * 20 + 100
    attempts = 0
    while len(rows) < total_needed:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not generate {total_needed} unique tasks within {max_attempts} attempts.")
        gen_fn = _GENERATORS[idx % len(_GENERATORS)]
        idx += 1
        candidate = gen_fn(rng)
        key = _prompt_key(candidate)
        if key in seen_prompts:
            continue
        seen_prompts.add(key)
        rows.append(candidate)

    rng.shuffle(rows)
    train_rows = rows[:n_train]
    test_rows = rows[n_train : n_train + n_test]

    leak_count = check_leakage(train_rows, test_rows)
    if leak_count != 0:
        raise AssertionError(f"Prompt leakage detected between train and test: {leak_count} overlapping rows")

    train_types = dict(sorted(Counter(r["task_type"] for r in train_rows).items()))
    test_types = dict(sorted(Counter(r["task_type"] for r in test_rows).items()))
    stats: dict[str, Any] = {
        "seed": seed,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "leak_overlap_count": leak_count,
        "train_single_step": sum(1 for r in train_rows if len(r["expected_tools"]) == 1),
        "train_two_step": sum(1 for r in train_rows if len(r["expected_tools"]) == 2),
        "test_single_step": sum(1 for r in test_rows if len(r["expected_tools"]) == 1),
        "test_two_step": sum(1 for r in test_rows if len(r["expected_tools"]) == 2),
        "train_by_task_type": train_types,
        "test_by_task_type": test_types,
        "languages": ["en"],
    }
    return train_rows, test_rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate LedgerCall train/test splits.")
    parser.add_argument("--out", type=str, default="data/", help="Output directory for train.jsonl, test.jsonl, stats.json")
    parser.add_argument("--train-size", type=int, default=600, help="Number of training rows (default: 600)")
    parser.add_argument("--test-size", type=int, default=150, help="Number of test rows (default: 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows, stats = generate_splits(
        n_train=args.train_size,
        n_test=args.test_size,
        seed=args.seed,
    )
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(
        f"[ledgercall.data] Wrote {len(train_rows)} train and {len(test_rows)} test tasks to {out_dir} "
        f"(leak_overlap_count={stats['leak_overlap_count']})"
    )
    return stats


if __name__ == "__main__":
    main()
