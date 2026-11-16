"""Blind OOD set #2 (written Nov 15 11:25 PM PST, after v4 trained, before any model saw it).

Never used for training or for designing training data. New wording, new languages (fr, de, ja),
currency symbols, number words, and new chain shapes.

    python -m ledgercall.ood2 --out data/ood2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .data import SYSTEM_PROMPT, check_leakage, write_jsonl
from .tools import SUPPORTED_CURRENCIES, compound_interest, convert_currency, loan_payment, percent_change

SYM = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 10: "ten", 15: "fifteen", 20: "twenty"}
ASK = {
    "en": "Finish with the number in asterisks, like *123.45*.",
    "fr": "Donne le résultat final entre astérisques, par exemple *123.45*.",
    "de": "Gib das Endergebnis in Sternchen an, z. B. *123.45*.",
    "ja": "最終的な数値をアスタリスクで囲んでください（例：*123.45*）。",
    "zh": "最后的数字请写成 *123.45* 这样的格式。",
    "es": "Da la cifra final entre asteriscos, como *123.45*.",
}


def _row(text, ans, calls, task_type, lang):
    return {"gold_calls": calls,
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{text} {ASK[lang]}"}],
            "answer": f"{ans:.2f}", "expected_tools": [c["name"] for c in calls],
            "task_type": task_type, "language": lang, "split": "ood2"}


def _pair(rng):
    return tuple(rng.sample(list(SUPPORTED_CURRENCIES), 2))


def _fx(a, s, d):
    return {"name": "convert_currency", "arguments": {"amount": a, "from_ccy": s, "to_ccy": d}}


def _ci(p, r, y, n):
    return {"name": "compound_interest", "arguments": {"principal": p, "annual_rate_pct": r, "years": y, "compounds_per_year": n}}


def _loan(p, r, m):
    return {"name": "loan_payment", "arguments": {"principal": p, "annual_rate_pct": r, "months": m}}


def _pct(o, n):
    return {"name": "percent_change", "arguments": {"old": o, "new": n}}


def en_fx_symbol(rng):
    a = round(rng.uniform(50, 30000), 2); s, d = _pair(rng)
    return _row(f"My hotel bill came to {SYM[s]}{a:,.2f}. Roughly what is that in {d}?", convert_currency(a, s, d), [_fx(a, s, d)], "en_fx_symbol", "en")


def en_loan_words(rng):
    y = rng.choice(list(WORDS)); p = round(rng.uniform(5000, 90000), 2); r = round(rng.uniform(3, 12), 2)
    return _row(f"For a {WORDS[y]}-year personal loan of {p:.2f} USD at {r}% a year, what would the monthly bill be?",
                loan_payment(p, r, y * 12), [_loan(p, r, y * 12)], "en_loan_words", "en")


def en_pct_decline(rng):
    o = round(rng.uniform(20, 900), 2); n = round(o * rng.uniform(0.3, 0.95), 2)
    return _row(f"Our share price slid from {o:.2f} to {n:.2f}. What was the percentage move?", percent_change(o, n), [_pct(o, n)], "en_pct_decline", "en")


def en_ci_pct(rng):
    p = round(rng.uniform(1000, 30000), 2); r = round(rng.uniform(2, 10), 2); y = rng.randint(3, 20); n = rng.choice((1, 4, 12))
    m = compound_interest(p, r, y, n)
    return _row(f"Ignoring inflation, by what percent will {p:.2f} USD have grown after {y} years at {r}% with {n} compounding periods per year?",
                percent_change(p, m), [_ci(p, r, y, n), _pct(p, m)], "en_ci_pct", "en")


def en_fx_m(rng):
    a = round(round(rng.uniform(1.1, 9.9), 1) * 1e6, 2); d = rng.choice(["USD", "EUR", "GBP", "INR"])
    return _row(f"A Tokyo supplier quotes ¥{a/1e6:g}M. What's that in {d}?", convert_currency(a, "JPY", d), [_fx(a, "JPY", d)], "en_fx_m", "en")


def en_three_fx_ci_fx(rng):
    a = round(rng.uniform(1000, 20000), 2); s, d = _pair(rng); r = round(rng.uniform(2, 9), 2); y = rng.randint(2, 10)
    c1 = convert_currency(a, s, d); m = compound_interest(c1, r, y, 1); back = convert_currency(m, d, s)
    return _row(f"I move {a:.2f} {s} into a {d} account, let it earn {r}% compounded annually for {y} years, then bring it all back to {s}. How much {s} do I end up with?",
                back, [_fx(a, s, d), _ci(c1, r, y, 1), _fx(m, d, s)], "en_three_fx_ci_fx", "en")


def fr_fx(rng):
    a = round(rng.uniform(100, 20000), 2); s, d = _pair(rng)
    return _row(f"Combien font {a:.2f} {s} en {d} ?", convert_currency(a, s, d), [_fx(a, s, d)], "fr_fx", "fr")


def fr_loan_fx(rng):
    p = round(rng.uniform(5000, 80000), 2); r = round(rng.uniform(2, 10), 2); m = rng.choice((24, 36, 48, 60, 120)); s, d = _pair(rng)
    pay = loan_payment(p, r, m)
    return _row(f"Prêt de {p:.2f} {s} sur {m} mois à {r}% par an : quelle est la mensualité en {d} ?",
                convert_currency(pay, s, d), [_loan(p, r, m), _fx(pay, s, d)], "fr_loan_fx", "fr")


def de_ci(rng):
    p = round(rng.uniform(1000, 50000), 2); r = round(rng.uniform(1, 9), 2); y = rng.randint(2, 20); n = rng.choice((1, 4, 12))
    lab = {1: "jährlicher", 4: "vierteljährlicher", 12: "monatlicher"}[n]
    return _row(f"Wie hoch ist das Guthaben nach {y} Jahren, wenn {p:.2f} EUR zu {r}% p.a. mit {lab} Verzinsung angelegt werden?",
                compound_interest(p, r, y, n), [_ci(p, r, y, n)], "de_ci", "de")


def ja_loan(rng):
    p = round(rng.uniform(5000, 90000), 2); r = round(rng.uniform(2, 10), 2); m = rng.choice((12, 24, 36, 60, 84))
    return _row(f"{p:.2f} USD を年利 {r}%、{m} か月で借りた場合、毎月の返済額はいくらですか？", loan_payment(p, r, m), [_loan(p, r, m)], "ja_loan", "ja")


def zh_ci_fx(rng):
    p = round(rng.uniform(1000, 40000), 2); r = round(rng.uniform(1, 9), 2); y = rng.randint(2, 15); s, d = _pair(rng)
    m = compound_interest(p, r, y, 12)
    return _row(f"把 {p:.2f} {s} 按年利率 {r}%、按月复利存 {y} 年，到期后换成 {d} 有多少？",
                convert_currency(m, s, d), [_ci(p, r, y, 12), _fx(m, s, d)], "zh_ci_fx", "zh")


def es_fx_pct(rng):
    a = round(rng.uniform(500, 20000), 2); s, d = _pair(rng); c = convert_currency(a, s, d); ref = round(c * rng.uniform(0.6, 1.4), 2)
    return _row(f"El año pasado pagamos {ref:.2f} {d} de alquiler; este año son {a:.2f} {s}. ¿Cuál es la variación porcentual en {d}?",
                percent_change(ref, c), [_fx(a, s, d), _pct(ref, c)], "es_fx_pct", "es")


FAMILIES = (en_fx_symbol, en_loan_words, en_pct_decline, en_ci_pct, en_fx_m, en_three_fx_ci_fx,
            fr_fx, fr_loan_fx, de_ci, ja_loan, zh_ci_fx, es_fx_pct)


def generate(per_family=6, seed=1115):
    rng = random.Random(seed)
    return [fam(rng) for fam in FAMILIES for _ in range(per_family)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/ood2.jsonl")
    ap.add_argument("--check", nargs="*", default=["data/train.jsonl", "data/aug.jsonl", "data/test.jsonl", "data/ood.jsonl"])
    a = ap.parse_args(); rows = generate()
    for p in a.check:
        if Path(p).exists():
            assert check_leakage([json.loads(l) for l in open(p)], rows) == 0, p
    write_jsonl(Path(a.out), rows)
    print(f"[ledgercall.ood2] {len(rows)} rows", dict(Counter(r["task_type"] for r in rows)))


if __name__ == "__main__":
    main()
