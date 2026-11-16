# LedgerCall

**Teaching a 3.35B open-weights model to call finance tools instead of guessing numbers: 0% → 100% on held-out tasks, and 0% → 84.7% on a blind multilingual OOD set written after training.**
Cohere `tiny-aya-global` + LoRA, trained with AgentTune-rendered trajectory SFT and audited with AuditKIT on multi-step finance tasks.

- **Track 3: Agentic Fine-tuning**, problem statements **3.1 Reliable tool calling** and **3.2 Efficient trajectories**
- Model (v4 adapter): [frankzhu94/ledgercall-tiny-aya-sft](https://huggingface.co/frankzhu94/ledgercall-tiny-aya-sft) · Dataset: [frankzhu94/ledgercall-finance-toolbench](https://huggingface.co/datasets/frankzhu94/ledgercall-finance-toolbench) · Demo: [Space](https://huggingface.co/spaces/frankzhu94/ledgercall-demo) · Collection: [LedgerCall](https://huggingface.co/collections/frankzhu94/ledgercall-lexsi-hackathon-track-3-6afac2cda7a31348026aee1f) · Code: [github.com/Xuefeng-Zhu/Ledgercall](https://github.com/Xuefeng-Zhu/Ledgercall)
- Team: solo, Frank Zhu ([@Xuefeng-Zhu](https://github.com/Xuefeng-Zhu), HF [frankzhu94](https://huggingface.co/frankzhu94))

> **For judges, one minute.** Base Tiny Aya Global invents argument names and then does the math in its head, so it gets 0% of finance tasks right. LedgerCall teaches it the tool-call protocol by replaying gold trajectories through AgentTune itself (train and eval prompts are byte-identical). The headline that matters is generalization: on **72 blind tasks** in French, German, Japanese, Chinese, Spanish and English, with currency symbols and new chain shapes, written after training and run once, success goes **0% → 84.7%** (95% Wilson CI 74.7–91.2%) and malformed calls go **58.3% → 0%**.

## Headline: generalization (tasks never seen in training)

| Eval set | Base | v3 SFT | v4 SFT + targeted aug (published) |
| :--- | ---: | ---: | ---: |
| **Blind OOD #2**, 72 tasks (fr/de/ja/zh/es/en, symbols, new chains) | 0.0% | 41.7% | **84.7%** (61/72, CI 74.7–91.2%) |
| OOD #1, 84 tasks (paraphrase, zh/es, 3-step) | 1.2% | 39.3% | **89.3%*** (75/84) |
| Malformed call rate, blind OOD #2 | 58.3% | 9.7% | **0.0%** |
| Exact tool sequence, blind OOD #2 | 56.9% | 48.6% | **93.1%** |

*OOD #1 guided the v4 augmentation, so it is not blind for v4; OOD #2 was written after v4 trained and run once. Details: `results/ood_comparison_v4.md`, `results/ood2_comparison.md`.

## In-distribution result (100 held-out test tasks, v4 adapter, greedy, T4)

Same 8 task families as training, disjoint prompts and numbers. This measures reliable execution of the learned protocol, not generalization (see above for that).

| Metric | Direction | Base tiny-aya-global | LedgerCall v4 (SFT LoRA) | Change |
| :--- | :---: | ---: | ---: | ---: |
| Task Success (numeric ≤1% tol) | ↑ higher | 0.0% | 100.0% | **+100.0 pp** |
| Tool Call Accuracy (ordered recall) | ↑ higher | 77.5% | 100.0% | **+22.5 pp** |
| Exact Tool Sequence Match | ↑ higher | 48.0% | 100.0% | **+52.0 pp** |
| Malformed / Failed Call Rate | ↓ lower | 71.0% | 0.0% | **-71.0 pp** |
| Avg Tool Calls / Task | ↓ efficient | 1.69 | 1.48 | **-0.21** |
| Avg Latency / Task (s) | ↓ lower | 24.41s | 17.76s | **-6.65s** |
| Avg Completion Tokens / Task | ↓ lower | 308.24 | 98.77 | **-209.47** |
| AuditKIT Verified Success | ↑ higher | 0.0% | 100.0% | **+100.0 pp** |
| AuditKIT Tool Call F1 (name) | ↑ higher | 76.0% | 100.0% | **+24.0 pp** |
| AuditKIT Tool Call Validity | ↑ higher | 100.0% | 100.0% | **+0.0 pp** |
| AuditKIT Redundant Tool Calls | ↓ lower | 0.12 | 0.00 | **-0.12** |
| AuditKIT No-Loop Score | ↑ higher | 77.0% | 100.0% | **+23.0 pp** |

AuditKIT `agent_eval` (recorded mode over AgentTune `trajectories.jsonl`) verified **0/100 → 100/100** episodes as successful. v3 and v4 give identical final answers on all 100 in-distribution tasks (greedy decoding), so the gains from v4 show up only out of distribution.

Note on throughput: raw completion tok/s fell (12.6 → 5.6) because the LoRA was evaluated unmerged and answers are much shorter, so fixed per-turn overhead dominates. Per-task latency, the number a user feels, improved 27%. Merging the adapter removes the LoRA overhead.

## Abstract

Small open models are attractive for finance back-office work because ledgers can stay inside a private network, but they fail in a dangerous way: they emit fluent, plausible numbers. Our baseline makes this concrete. Given four deterministic tools (`convert_currency`, `compound_interest`, `percent_change`, `loan_payment`), base Tiny Aya Global solved **0%** of 100 held-out tasks: 71% of episodes contained a malformed or failed call (typically an invented argument name such as `annual_rate_ptc`), after which the model did the arithmetic in its head.

LedgerCall fixes this with **trajectory SFT rendered by AgentTune itself**. Because the tools are deterministic, every task has a known gold trajectory. We replay those trajectories through AgentTune's own `create_rollout_fn` with a scripted rollout engine, so each training prompt is byte-identical to what the model sees at evaluation time: the Tiny Aya chat template, AgentTune's tool fallback prompt, real tool executions and folded `Tool result for …` turns. A first adapter (v3, 600 tasks) reached 100% in distribution but only 39–42% on unseen phrasings. Reading its OOD failures, we added 360 targeted training rows (formatted numbers like `$85k`, loan terms in years, 3-step chains) with new wording. The published v4 adapter trains on 960 tasks (2,487 assistant-turn examples), rank-16 LoRA on assistant tokens only, 100 optimizer steps in 74 minutes on one free Kaggle T4. Evaluated with AgentTune rollouts and AuditKIT `agent_eval`, task success rises to **100%** in distribution and **84.7%** on a blind 72-task multilingual OOD set, malformed calls fall to **0%**, and each task costs **99** generated tokens instead of 308.

## The failure, concretely (held-out task 0)

> A treasury deposit of 25310.50 USD earns 2.76% annual interest for 12 years, compounded semi-annually … compute its maturity balance, then the percentage change from a benchmark.

- **Base:** calls `compound_interest(principal=2531.5, annual_rate_ptc=2.76, …)` — wrong principal, invented argument → tool error → writes out the formula and guesses.
- **LedgerCall:** calls `compound_interest(principal=25310.5, annual_rate_pct=2.76, years=12, compounds_per_year=2)` → 35168.64, then `percent_change(old=26768.29, new=35168.64)` → 31.38, and answers `The answer is *31.38*.` (correct).

## Method

```mermaid
flowchart LR
  D["ledgercall/data.py<br/>600 train · 150 test · seed 42"] --> M["+ ledgercall/augment.py<br/>360 targeted rows → 960 train"]
  M --> R["AgentTune create_rollout_fn<br/>scripted engine replays gold calls<br/>(real tools, Tiny Aya fallback prompt)"]
  R --> S["ledgercall/sft.py<br/>LoRA r=16, loss on assistant turns<br/>logits_to_keep (262k vocab)"]
  S --> E["ledgercall/evaluate.py<br/>AgentTune rollouts → trajectories.jsonl<br/>test · OOD #1 · blind OOD #2"]
  B["base tiny-aya-global"] --> E
  E --> A["AuditKIT agent_eval<br/>verified success, tool F1,<br/>validity, redundancy, loops"]
  A --> C["ledgercall/compare.py<br/>before/after table"]
```

- **AgentTune** (`Lexsi-Labs/AgentTune`): rollout loop, tool execution, Tiny Aya tool-call fallback prompt and parser, trajectory serialization — used both to *render training data* and to *evaluate*. This guarantees the train/eval prompt format cannot drift.
- **AuditKIT** (`Lexsi-Labs/AuditKIT`): `agent_eval` over recorded AgentTune episodes with a custom numeric-tolerance predicate, plus `tool_call_f1_name`, `tool_call_validity`, `redundant_tool_calls`, `agent_loop_detection`.
- **Training (v4, published):** 960 rows → 2,487 assistant-turn examples. LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down; fp16 autocast; AdamW lr 2e-4, warmup 5%, cosine schedule capped at 100 optimizer steps (`--max_steps 100`, ≈0.32 epoch at grad accumulation 8) so the schedule completes; 74 minutes; final loss 0.0028. (`epochs: 1.0` in `train_summary_v4.json` is the configured value; the step cap ends training first.)
- **Training (v3, ablation):** 600 rows → 1,497 examples, 72 of 188 steps (≈0.38 epoch, stopped by the 55-minute budget), final loss 0.0003.
- **Eval:** greedy, max 384 new tokens/turn, max 4 tool steps, stop on `<|END_RESPONSE|>`/`<|END_OF_TURN_TOKEN|>` for **both** models (Tiny Aya's `generation_config` lists only `<EOS_TOKEN>`, which otherwise makes every turn run to the token cap). Success = starred final number within 1% of the gold answer.

## Metrics and why

- **Task success** — the outcome that matters: correct final number. Higher is better.
- **Ordered tool-call accuracy / exact sequence** — right tools in the right order for multi-step compositions. Higher is better.
- **Malformed / failed call rate** — reliability of the call payload itself (bad names, bad keys, tool errors). Lower is better.
- **Avg tool calls, completion tokens, latency** — trajectory efficiency: correct answers should not cost more steps or tokens. Lower is better (tool calls should match gold).
- **AuditKIT diagnostics** — an independent, kit-standard view of the same episodes.
- **Automatic judge:** none needed. Success is a deterministic numeric check against the gold answer computed by the same tools, so there is no LLM-judge agreement to validate.

## Data

| Split | Rows | Source | Languages |
| :--- | ---: | :--- | :--- |
| train | 600 | `ledgercall/data.py`, 8 families (4 single-tool, 4 two-step) | en |
| aug | 360 | `ledgercall/augment.py`, formatted numbers, loan terms in years, 3-step chains | en |
| test (in-distribution) | 150 (100 evaluated) | `ledgercall/data.py` | en |
| OOD #1 | 84 | `ledgercall/ood.py`, 12 families × 7 | en, zh, es |
| Blind OOD #2 | 72 | `ledgercall/ood2.py`, 12 families × 6 | en, fr, de, ja, zh, es |

All rows store `prompt`, `answer`, `expected_tools`, `gold_calls`, `task_type`. Leak check: normalized exact-prompt match across every pair of splits = 0. All files are in the [HF dataset](https://huggingface.co/datasets/frankzhu94/ledgercall-finance-toolbench) along with every trajectory.

## What did not work (honest notes)

- **GRPO** via AgentTune `create_agentic_trainer` (`ledgercall/train.py`, rewards in `ledgercall/rewards.py`) ran out of memory on the 16 GB T4 before its first step (3.35B model, 4 generations, multi-turn rollouts). We kept the reward code; GRPO warm-started from the SFT adapter is the natural next step on a 24–40 GB GPU.
- **v3 alone did not generalize:** 39–42% on OOD sets, failing on thousands separators copied into JSON, `$133.2k`-style units, and chains longer than it had seen. Targeted augmentation (v4) fixed most of this.
- The gated Hugging Face repo was not yet accessible from our account during the sprint, so weights were loaded from Cohere's official Kaggle Models mirror (`cohere-labs/tiny-aya/transformers/global/1`), the same checkpoint.

## Limitations

- Only 4 deterministic tools and a static FX table; no live data, no abstention cases where no tool applies.
- Remaining blind-OOD failures: number words for terms ("four-year" → wrong months), unseen chain shapes (FX → compound → FX starts with the wrong tool), and Spanish FX → % converting in the wrong direction.
- OOD #1 is not blind for v4, since its failures shaped the augmentation.
- Robustness to adversarial input (e.g. instructions injected inside tool output) is not yet measured.
- 100 of 150 in-distribution test tasks evaluated to fit the free GPU budget; OOD sets are small (72–84), hence the confidence intervals.

## Threat model (brief)

The agent's job is to move numbers from a request into tool arguments and report the tool's result. The failures that matter are silent ones: a wrong argument that still returns a plausible number, or an answer computed without calling a tool. LedgerCall's metrics target exactly these (malformed calls, ordered tool recall, verified numeric success). Out of scope today: a malicious user or poisoned tool result trying to redirect calls; that is the first red-team test planned below.

## Cost / ROI

| Deployment Option | Avg Latency / Task | Est. Cost / 1k Tasks | Data Governance |
| :--- | ---: | ---: | :--- |
| Tiny Aya Base (T4 @ $0.35/hr) | 24.41s | $2.37 (0% correct) | Self-hosted / VPC |
| **LedgerCall v4 LoRA (T4 @ $0.35/hr)** | 17.76s | **$1.73** | Self-hosted / VPC |

Assumes sequential batch=1 rollouts on one T4 at $0.35/hr; continuous batching (vLLM) and a merged adapter lower unit cost further. Compute used for the whole project: free Kaggle T4 quota (≈3 hours of training plus evaluation).

## With more credits

| Planned step | Expected before → after |
| :--- | :--- |
| GRPO from the v4 adapter with `rewards.py` (correctness + tool choice + efficiency penalty) on L4/A100 | Blind OOD #2 84.7% → ~92%; unseen chain shapes 3/6 → 5/6 |
| Number-word and multilingual term templates in augmentation | Number-word terms 3/6 → 6/6 |
| Abstention cases + more tools (amortization, NPV, FX with dates) | New abstention metric; keep ≥95% in-distribution |
| AuditKIT injection probes through tool results | First measured attack success rate, then a hardening pass |
| Merge LoRA + vLLM serving | Throughput back above base, cost per 1k tasks well under $1 |

## Repo layout

| Path | What it does |
| :--- | :--- |
| `ledgercall/tools.py` | The 4 deterministic finance tools and their schemas |
| `ledgercall/data.py` | Generates train/test splits with gold calls and leak check |
| `ledgercall/augment.py` | v4 targeted augmentation (360 rows) |
| `ledgercall/ood.py`, `ledgercall/ood2.py` | OOD #1 and blind OOD #2 generators |
| `ledgercall/sft.py` | AgentTune-rendered trajectory SFT (LoRA) |
| `ledgercall/evaluate.py` | AgentTune rollouts + AuditKIT scoring → `results/<tag>_metrics.json` |
| `ledgercall/compare.py` | Before/after markdown tables and cost |
| `ledgercall/train.py`, `ledgercall/rewards.py` | GRPO path (OOM on T4, kept for future work) |
| `scripts/kaggle_run.py` | Single-file Kaggle job used for the official run |
| `scripts/push_to_hub.py` | Uploads adapter, data and results to HF |
| `notebooks/` | Kaggle notebook |
| `space/`, `space_static/` | Demo Space (static version is the live one) |
| `results/`, `data/` | Metrics, comparisons, eval sets (full trajectories in the HF dataset) |
| `tests/test_core.py` | Unit tests for tools, data and scoring |

## Reproduce

```bash
pip install "git+https://github.com/Lexsi-Labs/AgentTune" "auditkit[transformers]" peft datasets sentencepiece
python -m pytest -q tests/

# data
python -m ledgercall.data --out data/
python -m ledgercall.augment --out data/aug.jsonl --per-family 45
python -m ledgercall.ood --out data/ood.jsonl
python -m ledgercall.ood2 --out data/ood2.jsonl
cat data/train.jsonl data/aug.jsonl > data/train_v4.jsonl

# baseline
python -m ledgercall.evaluate --model <tiny-aya-global path> --data data/test.jsonl --tag base --limit 100

# v4 adapter
python -m ledgercall.sft --model <tiny-aya-global path> --train_file data/train_v4.jsonl --output_dir out/sft_v4 --max_steps 100
python -m ledgercall.evaluate --model out/sft_v4 --base-model <tiny-aya-global path> --data data/test.jsonl --tag tuned_v4 --limit 100
python -m ledgercall.evaluate --model out/sft_v4 --base-model <tiny-aya-global path> --data data/ood.jsonl  --tag ood_tuned_v4
python -m ledgercall.evaluate --model out/sft_v4 --base-model <tiny-aya-global path> --data data/ood2.jsonl --tag ood2_v4
python -m ledgercall.compare --base results/base_metrics.json --tuned results/tuned_v4_metrics.json --out results/comparison.md
```

Or skip training and evaluate the published adapter: `--base-model <tiny-aya-global path> --adapter frankzhu94/ledgercall-tiny-aya-sft`. Hardware: one 16 GB T4 is enough for SFT and eval.

## Licenses

Code and synthetic data: MIT. The LoRA adapter is a derivative of `CohereLabs/tiny-aya-global` and inherits **CC BY-NC 4.0** (non-commercial). AgentTune and AuditKIT are under Lexsi Labs source-available licenses.

## Declaration

Built solo by Frank Zhu during the Lexsi AI Safety & Alignment Hackathon on Nov 15, 2026. No pre-existing project code was reused; AgentTune and AuditKIT are used as libraries.

# Appendix: OOD details

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

### v4: targeted data augmentation (same OOD set, 84 tasks)

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

### Blind OOD set #2 (72 tasks, 12 families x 6)

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
