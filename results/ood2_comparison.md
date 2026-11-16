## Blind OOD set #2 (72 tasks, 12 families x 6)

Written after v4 finished training and evaluated once, with no changes afterwards (ledgercall/ood2.py).
New shifts: currency symbols ($, €, ¥), "¥9.2M", number words ("four-year"), French, German, Japanese,
new chain shapes (FX -> compound -> FX back). Exact-prompt leakage vs train/aug/test/OOD#1 = 0.

| Metric | Base | v3 SFT | v4 SFT + aug |
|---|---|---|---|
| Task success | 0.0% (0/72) | 41.7% (30/72) | 84.7% (61/72) |
| Tool call accuracy | 71.3% | 61.1% | 96.5% |
| Exact tool sequence | 56.9% | 48.6% | 93.1% |
| Malformed call rate | 58.3% | 9.7% | 0.0% |
| Avg tool calls | 1.40 | 0.90 | 1.50 (gold 1.50) |
| Avg completion tokens | 213 | 70 | 98 |
| Avg latency (s) | 18.6 | 13.6 | 18.1 |

v4 by family (out of 6): 6/6 on currency symbols, % decline, compound->%, French FX, French loan->FX,
German compound, Japanese loan; 5/6 ¥…M and Chinese compound->FX; 3/6 number-word terms, 3-step FX->compound->FX,
Spanish FX->%.

Remaining v4 failures (11):
- Number words for terms ("four-year") -> wrong months.
- Unfamiliar chain shape: starts with loan_payment because every training 3-step chain began with loan/compound.
- Spanish FX->%: converts in the wrong direction or uses the pre-conversion amount.
