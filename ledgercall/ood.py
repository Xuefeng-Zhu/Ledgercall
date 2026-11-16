"""Out-of-distribution (OOD) evaluation set for LedgerCall.

Nothing here is used for training. Four shifts away from the training templates:

* ``para_*``   - new English phrasings: no "First ... then ..." scaffolding, no tool hints,
                 shorthand numbers ("$12.5k", "6.25 percent", "a 5-year loan").
* ``zh_*``     - Simplified Chinese prompts (Tiny Aya is multilingual; training was English only).
* ``es_*``     - Spanish prompts.
* ``three_*``  - 3-step chains never seen in training (compound -> FX -> % change,
                 loan -> FX -> % change).

CLI:
    python -m ledgercall.ood --out data/ood.jsonl --per-family 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .data import SYSTEM_PROMPT, check_leakage
from .tools import SUPPORTED_CURRENCIES, compound_interest, convert_currency, loan_payment, percent_change

FREQ_EN = {1: "once a year", 2: "twice a year", 4: "every quarter", 12: "every month"}
FREQ_ZH = {1: "每年复利一次", 2: "每半年复利一次", 4: "每季度复利一次", 12: "每月复利一次"}
FREQ_ES = {1: "una vez al año", 2: "dos veces al año", 4: "trimestralmente", 12: "mensualmente"}
ASK_EN = "Give the final figure wrapped in asterisks, e.g. *123.45*."
ASK_ZH = "请把最终数字用星号包起来，例如 *123.45*。"
ASK_ES = "Escribe el resultado final entre asteriscos, por ejemplo *123.45*."


def _row(text: str, ans: float, calls: list[dict[str, Any]], task_type: str, lang: str) -> dict[str, Any]:
    return {
        "gold_calls": calls,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "answer": f"{ans:.2f}",
        "expected_tools": [c["name"] for c in calls],
        "task_type": task_type,
        "language": lang,
        "split": "ood",
    }


def _pair(rng: random.Random) -> tuple[str, str]:
    a, b = rng.sample(list(SUPPORTED_CURRENCIES), 2)
    return a, b


def _ci_args(rng):
    return round(rng.uniform(1000, 40000), 2), round(rng.uniform(1.5, 12), 2), rng.randint(2, 15), rng.choice((1, 2, 4, 12))


def _loan_args(rng):
    return round(rng.uniform(5000, 150000), 2), round(rng.uniform(2.5, 12), 2), rng.choice((2, 3, 4, 5, 10, 15, 30))


# ---------------------------------------------------------------- paraphrased English
def para_fx(rng):
    amt = round(rng.uniform(100, 20000), 2); s, d = _pair(rng)
    text = f"Our Berlin office got billed {amt:,.2f} {s}. What does that come to in {d}? {ASK_EN}"
    return _row(text, convert_currency(amt, s, d), [{"name": "convert_currency", "arguments": {"amount": amt, "from_ccy": s, "to_ccy": d}}], "para_fx", "en")


def para_loan(rng):
    p, r, y = _loan_args(rng); p = round(p / 100) * 100.0
    text = f"I'm taking out a {y}-year loan for ${p/1000:g}k at {r} percent. How much is each monthly installment? {ASK_EN}"
    return _row(text, loan_payment(p, r, y * 12), [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": y * 12}}], "para_loan", "en")


def para_ci_fx(rng):
    p, r, y, n = _ci_args(rng); s, d = _pair(rng)
    m = compound_interest(p, r, y, n)
    text = (f"If I park {p:,.2f} {s} in a deposit paying {r}% a year, compounding {FREQ_EN[n]}, and leave it for {y} years, "
            f"how much will I have once it's converted into {d}? {ASK_EN}")
    calls = [{"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}},
             {"name": "convert_currency", "arguments": {"amount": m, "from_ccy": s, "to_ccy": d}}]
    return _row(text, convert_currency(m, s, d), calls, "para_ci_fx", "en")


def para_fx_pct(rng):
    amt = round(rng.uniform(500, 25000), 2); s, d = _pair(rng)
    conv = convert_currency(amt, s, d); base = round(conv * rng.uniform(0.6, 1.5), 2)
    text = (f"Marketing spent {amt:,.2f} {s} this month versus {base:,.2f} {d} last month. "
            f"By what percent did spend change, measured in {d}? {ASK_EN}")
    calls = [{"name": "convert_currency", "arguments": {"amount": amt, "from_ccy": s, "to_ccy": d}},
             {"name": "percent_change", "arguments": {"old": base, "new": conv}}]
    return _row(text, percent_change(base, conv), calls, "para_fx_pct", "en")


# ---------------------------------------------------------------- Chinese
def zh_fx(rng):
    amt = round(rng.uniform(100, 20000), 2); s, d = _pair(rng)
    text = f"一张供应商发票金额为 {amt:.2f} {s}，换算成 {d} 是多少？{ASK_ZH}"
    return _row(text, convert_currency(amt, s, d), [{"name": "convert_currency", "arguments": {"amount": amt, "from_ccy": s, "to_ccy": d}}], "zh_fx", "zh")


def zh_ci(rng):
    p, r, y, n = _ci_args(rng)
    text = f"本金 {p:.2f} 美元，年利率 {r}%，{FREQ_ZH[n]}，存 {y} 年后到期余额是多少？{ASK_ZH}"
    return _row(text, compound_interest(p, r, y, n), [{"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}}], "zh_ci", "zh")


def zh_loan_fx(rng):
    p, r, y = _loan_args(rng); s, d = _pair(rng); mo = y * 12
    pay = loan_payment(p, r, mo)
    text = f"一笔 {p:.2f} {s} 的贷款，年利率 {r}%，期限 {mo} 个月。每月还款额换算成 {d} 是多少？{ASK_ZH}"
    calls = [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": mo}},
             {"name": "convert_currency", "arguments": {"amount": pay, "from_ccy": s, "to_ccy": d}}]
    return _row(text, convert_currency(pay, s, d), calls, "zh_loan_fx", "zh")


def zh_pct(rng):
    old = round(rng.uniform(100, 20000), 2); new = round(old * rng.uniform(0.5, 1.8), 2)
    text = f"上季度营收 {old:.2f}，本季度营收 {new:.2f}，环比变化百分比是多少？{ASK_ZH}"
    return _row(text, percent_change(old, new), [{"name": "percent_change", "arguments": {"old": old, "new": new}}], "zh_pct", "zh")


# ---------------------------------------------------------------- Spanish
def es_loan(rng):
    p, r, y = _loan_args(rng); mo = y * 12
    text = f"Un préstamo de {p:.2f} USD al {r}% de interés anual a {mo} meses. ¿Cuál es la cuota mensual? {ASK_ES}"
    return _row(text, loan_payment(p, r, mo), [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": mo}}], "es_loan", "es")


def es_ci_pct(rng):
    p, r, y, n = _ci_args(rng); m = compound_interest(p, r, y, n); bench = round(m * rng.uniform(0.7, 1.3), 2)
    text = (f"Un depósito de {p:.2f} USD rinde {r}% anual con capitalización {FREQ_ES[n]} durante {y} años. "
            f"¿Cuál es la variación porcentual entre la meta de {bench:.2f} USD y el saldo final? {ASK_ES}")
    calls = [{"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}},
             {"name": "percent_change", "arguments": {"old": bench, "new": m}}]
    return _row(text, percent_change(bench, m), calls, "es_ci_pct", "es")


# ---------------------------------------------------------------- 3-step chains
def three_ci_fx_pct(rng):
    p, r, y, n = _ci_args(rng); s, d = _pair(rng)
    m = compound_interest(p, r, y, n); conv = convert_currency(m, s, d); bench = round(conv * rng.uniform(0.7, 1.3), 2)
    text = (f"A {p:.2f} {s} reserve grows at {r}% per year, compounded {FREQ_EN[n]}, for {y} years. "
            f"Convert the ending balance to {d}, then give the percentage change from a {bench:.2f} {d} target to that converted balance. {ASK_EN}")
    calls = [{"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}},
             {"name": "convert_currency", "arguments": {"amount": m, "from_ccy": s, "to_ccy": d}},
             {"name": "percent_change", "arguments": {"old": bench, "new": conv}}]
    return _row(text, percent_change(bench, conv), calls, "three_ci_fx_pct", "en")


def three_loan_fx_pct(rng):
    p, r, y = _loan_args(rng); s, d = _pair(rng); mo = y * 12
    pay = loan_payment(p, r, mo); conv = convert_currency(pay, s, d); budget = round(conv * rng.uniform(0.7, 1.3), 2)
    text = (f"Work out the monthly payment on a {p:.2f} {s} loan at {r}% annual interest over {mo} months, convert it to {d}, "
            f"and then report the percentage change from a monthly budget of {budget:.2f} {d} to that payment. {ASK_EN}")
    calls = [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": mo}},
             {"name": "convert_currency", "arguments": {"amount": pay, "from_ccy": s, "to_ccy": d}},
             {"name": "percent_change", "arguments": {"old": budget, "new": conv}}]
    return _row(text, percent_change(budget, conv), calls, "three_loan_fx_pct", "en")


FAMILIES = (para_fx, para_loan, para_ci_fx, para_fx_pct, zh_fx, zh_ci, zh_loan_fx, zh_pct,
            es_loan, es_ci_pct, three_ci_fx_pct, three_loan_fx_pct)


def generate_ood(per_family: int = 7, seed: int = 2026) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [fam(rng) for _ in range(per_family) for fam in FAMILIES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/ood.jsonl")
    ap.add_argument("--per-family", type=int, default=7)
    ap.add_argument("--train", default="data/train.jsonl")
    a = ap.parse_args()
    rows = generate_ood(a.per_family)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    leak = 0
    if Path(a.train).exists():
        train = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
        leak = check_leakage(train, rows)
    print(json.dumps({"rows": len(rows), "families": Counter(r["task_type"] for r in rows), "leak_overlap": leak}, ensure_ascii=False))


if __name__ == "__main__":
    main()
