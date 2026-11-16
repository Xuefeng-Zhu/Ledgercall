## v4: targeted data augmentation (same OOD set, 84 tasks)

v4 adds 360 training rows (ledgercall/augment.py) aimed at the three v3 failure modes:
formatted numbers (1,234.56 / $85k / 1.2M), loan terms in years, and 3-step chains.
Wording is new (no template reused from ood.py; exact-prompt leakage vs train/test/OOD = 0).
Training: 960 rows -> 2,487 turn examples, LoRA r=16, 100 steps, 74 min on one T4.

| Metric | Base | v3 SFT | v4 SFT + aug |
|---|---|---|---|
| OOD task success | 1.2% (1/84) | 39.3% (33/84) | 89.3% (75/84) |
| OOD tool call accuracy | 54.6% | 61.5% | 100% |
| OOD exact tool sequence | 36.9% | 50.0% | 100% |
| OOD malformed call rate | 56.0% | 27.4% | 0% |
| OOD avg tool calls | 1.13 | 1.00 | 1.67 (gold 1.67) |
| In-distribution task success (100) | 0% | 100% | 100% |

v4 success by family (out of 7): every family 7/7 except English paraphrased loan 0/7 and Chinese loan->FX 5/7.

Remaining failures (9), all argument copying in loan_payment:
- "$133.2k" style principals with one decimal shift by 10x (794000 for $79.4k).
- Month counts miscopied (120 -> 124, 24 -> 28) when the term is written as "10-year" or in Chinese.

Caveat: the augmentation was designed after reading v3's OOD failures, so this OOD set is no longer a
blind test of v4. It measures whether targeted data fixes known failure types; the wording is unseen.
