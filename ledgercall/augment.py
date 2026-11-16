"""Training-only augmentation targeting the failure modes seen on the OOD eval.

Fixes three things the v3 adapter got wrong, using wording that is NOT reused from
ledgercall/ood.py (OOD prompts stay unseen; exact-prompt leakage is checked):
  1. formatted numbers in the prompt (1,234.56 / $85k / 1.2M) -> plain JSON numbers
  2. loan terms given in years -> months argument
  3. three-step chains (more than one follow-up call)

    python -m ledgercall.augment --out data/aug.jsonl --per-family 45
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .data import SYSTEM_PROMPT, check_leakage, write_jsonl
from .tools import SUPPORTED_CURRENCIES, compound_interest, convert_currency, loan_payment, percent_change

ASKS = (
    "Put the final number between asterisks like *123.45*.",
    "Reply with the result wrapped in asterisks, for example *123.45*.",
    "Wrap the final value as *123.45*.",
)
FREQ = {1: "yearly", 2: "every six months", 4: "four times a year", 12: "each month"}


def _row(text, ans, calls, task_type):
    return {
        "gold_calls": calls,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "answer": f"{ans:.2f}",
        "expected_tools": [c["name"] for c in calls],
        "task_type": task_type,
        "language": "en",
        "split": "aug",
    }


def _pair(rng):
    return tuple(rng.sample(list(SUPPORTED_CURRENCIES), 2))


def _money(rng, p, usd=True):
    """Render a principal in one of several human formats; returns (text, exact value)."""
    p = round(p, 2)
    style = rng.choice(("comma", "k", "plain", "M") if p >= 100000 else ("comma", "k", "plain"))
    if style == "k":
        p = round(p / 100) * 100.0
        txt = f"{p/1000:g}k"
    elif style == "M":
        p = round(p / 10000) * 10000.0
        txt = f"{p/1e6:g}M"
    elif style == "comma":
        txt = f"{p:,.2f}"
    else:
        txt = f"{p:.2f}"
    return (("$" + txt) if usd else txt), p


def _term(rng):
    """Returns (adjective form, noun phrase, months)."""
    y = rng.choice((1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30))
    if rng.random() < 0.6:
        return f"{y}-year", f"{y} year{'s' if y > 1 else ''}", y * 12
    return f"{y*12}-month", f"{y*12} months", y * 12


def fmt_fx(rng):
    amt = round(rng.uniform(1000, 90000), 2); s, d = _pair(rng)
    t = rng.choice((
        f"A client in Toronto wired us {amt:,.2f} {s}. Express that in {d}.",
        f"Please convert {amt:,.2f} {s} into {d}.",
        f"How many {d} is {amt:,.2f} {s}?",
    ))
    return _row(f"{t} {rng.choice(ASKS)}", convert_currency(amt, s, d),
                [{"name": "convert_currency", "arguments": {"amount": amt, "from_ccy": s, "to_ccy": d}}], "aug_fmt_fx")


def fmt_loan(rng):
    p0 = rng.uniform(8000, 900000); P, p = _money(rng, p0); adj, phrase, mo = _term(rng); r = round(rng.uniform(2.0, 13.0), 2)
    t = rng.choice((
        f"What is the monthly payment on a {adj} mortgage of {P} at {r}% APR?",
        f"A {P} car loan at {r}% interest, paid back over {phrase}. What's the monthly payment?",
        f"Calculate monthly repayments for borrowing {P} at {r}% per year over {phrase}.",
    ))
    return _row(f"{t} {rng.choice(ASKS)}", loan_payment(p, r, mo),
                [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": mo}}], "aug_fmt_loan")


def fmt_ci(rng):
    p0 = rng.uniform(2000, 400000); P, p = _money(rng, p0); r = round(rng.uniform(1.0, 11.0), 2)
    y = rng.randint(2, 25); n = rng.choice((1, 2, 4, 12))
    t = rng.choice((
        f"I invest {P} at {r}% annual interest, compounded {FREQ[n]}. What is it worth after {y} years?",
        f"Savings of {P} earn {r}% a year with interest added {FREQ[n]}. Balance in {y} years?",
    ))
    return _row(f"{t} {rng.choice(ASKS)}", compound_interest(p, r, y, n),
                [{"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}}], "aug_fmt_ci")


def fmt_pct(rng):
    old = round(rng.uniform(1000, 500000), 2); new = round(old * rng.uniform(0.4, 1.9), 2)
    t = rng.choice((
        f"Revenue went from {old:,.2f} to {new:,.2f}. What is the percent change?",
        f"Headcount cost was {old:,.2f} last year and {new:,.2f} this year. Percentage change?",
    ))
    return _row(f"{t} {rng.choice(ASKS)}", percent_change(old, new),
                [{"name": "percent_change", "arguments": {"old": old, "new": new}}], "aug_fmt_pct")


def loan_fx(rng):
    p = round(rng.uniform(5000, 200000), 2); adj, phrase, mo = _term(rng); r = round(rng.uniform(2.0, 12.0), 2); s, d = _pair(rng)
    pay = loan_payment(p, r, mo)
    t = f"We borrowed {p:,.2f} {s} at {r}% for {phrase}. What is the monthly repayment in {d}?"
    calls = [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": mo}},
             {"name": "convert_currency", "arguments": {"amount": pay, "from_ccy": s, "to_ccy": d}}]
    return _row(f"{t} {rng.choice(ASKS)}", convert_currency(pay, s, d), calls, "aug_loan_fx")


def fx_pct(rng):
    amt = round(rng.uniform(1000, 60000), 2); s, d = _pair(rng)
    conv = convert_currency(amt, s, d); ref = round(conv * rng.uniform(0.6, 1.5), 2)
    t = f"This quarter's bill is {amt:,.2f} {s}; last quarter's was {ref:,.2f} {d}. Compare them in {d}: what is the percent change?"
    calls = [{"name": "convert_currency", "arguments": {"amount": amt, "from_ccy": s, "to_ccy": d}},
             {"name": "percent_change", "arguments": {"old": ref, "new": conv}}]
    return _row(f"{t} {rng.choice(ASKS)}", percent_change(ref, conv), calls, "aug_fx_pct")


def chain_ci_fx_pct(rng):
    p = round(rng.uniform(1000, 60000), 2); r = round(rng.uniform(1.0, 11.0), 2); y = rng.randint(2, 20); n = rng.choice((1, 2, 4, 12))
    s, d = _pair(rng); m = compound_interest(p, r, y, n); conv = convert_currency(m, s, d); goal = round(conv * rng.uniform(0.7, 1.3), 2)
    t = (f"Step 1: grow {p:,.2f} {s} at {r}% for {y} years, compounded {FREQ[n]}. Step 2: convert the result to {d}. "
         f"Step 3: how far is that from a goal of {goal:,.2f} {d}, as a percent change from the goal?")
    calls = [{"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}},
             {"name": "convert_currency", "arguments": {"amount": m, "from_ccy": s, "to_ccy": d}},
             {"name": "percent_change", "arguments": {"old": goal, "new": conv}}]
    return _row(f"{t} {rng.choice(ASKS)}", percent_change(goal, conv), calls, "aug_chain_ci_fx_pct")


def chain_loan_fx_pct(rng):
    p = round(rng.uniform(5000, 200000), 2); adj, phrase, mo = _term(rng); r = round(rng.uniform(2.0, 12.0), 2); s, d = _pair(rng)
    pay = loan_payment(p, r, mo); conv = convert_currency(pay, s, d); cap = round(conv * rng.uniform(0.7, 1.3), 2)
    t = (f"A {p:,.2f} {s} loan at {r}% runs for {phrase}. Find the monthly payment, express it in {d}, "
         f"and tell me the percent change versus our {cap:,.2f} {d} monthly cap.")
    calls = [{"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": mo}},
             {"name": "convert_currency", "arguments": {"amount": pay, "from_ccy": s, "to_ccy": d}},
             {"name": "percent_change", "arguments": {"old": cap, "new": conv}}]
    return _row(f"{t} {rng.choice(ASKS)}", percent_change(cap, conv), calls, "aug_chain_loan_fx_pct")


FAMILIES = (fmt_fx, fmt_loan, fmt_ci, fmt_pct, loan_fx, fx_pct, chain_ci_fx_pct, chain_loan_fx_pct)


def generate(per_family: int = 45, seed: int = 7) -> list[dict]:
    rng = random.Random(seed); rows, seen = [], set()
    for fam in FAMILIES:
        k = 0
        while k < per_family:
            r = fam(rng); key = r["prompt"][1]["content"]
            if key in seen:
                continue
            seen.add(key); rows.append(r); k += 1
    rng.shuffle(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/aug.jsonl")
    ap.add_argument("--per-family", type=int, default=45)
    ap.add_argument("--check", nargs="*", default=["data/test.jsonl", "data/ood.jsonl"])
    a = ap.parse_args(argv)
    rows = generate(a.per_family)
    for p in a.check:
        if Path(p).exists():
            other = [json.loads(l) for l in open(p)]
            leak = check_leakage(rows, other)
            assert leak == 0, f"leak vs {p}: {leak}"
            print(f"[ledgercall.augment] leak vs {p}: 0")
    write_jsonl(Path(a.out), rows)
    print(f"[ledgercall.augment] wrote {len(rows)} rows", dict(Counter(r["task_type"] for r in rows)))


if __name__ == "__main__":
    main()
