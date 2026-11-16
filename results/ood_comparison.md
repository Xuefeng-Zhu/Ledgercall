## Out-of-distribution eval (84 tasks, 12 families x 7; never seen in training)

| Metric | Base | SFT-Tuned |
|---|---|---|
| Task success | 1.2% (1/84) | 39.3% (33/84) |
| Tool call accuracy | 54.6% | 61.5% |
| Exact tool sequence | 36.9% | 50.0% |
| Malformed/failed call rate | 56.0% | 27.4% |
| Avg tool calls | 1.13 | 1.00 |
| Avg completion tokens | 185 | 81 |
| Avg latency (s) | 15.0 | 14.5 |
| AuditKIT verified success | 1/84 | 33/84 |

Success by family (out of 7 each), base -> tuned:
- Chinese: single-tool currency 1->7, compound interest 0->7, percent change 0->7; two-step loan->FX 0->0
- Spanish: loan 0->7, compound interest->percent change 0->5
- English paraphrases: currency 0->0, currency->percent 0->0, compound->FX 0->0, loan 0->0
- English 3-step chains: 0->0 (both families)

Failure modes of the tuned model (read from trajectories):
1. Thousands separators copied into JSON ("amount": 2,470.49) -> unparseable call.
2. Unit normalisation in paraphrases ("$133.2k", "10-year loan") -> principal 133.2, months 10.
3. Never chains beyond what it saw: avg 1.00 call on OOD; stops after the first tool on two-step zh, skips steps in 3-step chains.
Takeaway: SFT taught the call format and transfers across languages, but not argument normalisation or longer planning. Next: augment with formatted numbers/units and 3-step chains, then GRPO with the existing rewards.py.
