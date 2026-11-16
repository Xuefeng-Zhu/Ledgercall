# LedgerCall Evaluation Comparison: Base vs SFT-Tuned

- **Base run (`base`)**: `n=100`, model `/kaggle/input/models/cohere-labs/tiny-aya/transformers/global/1`
- **Tuned run (`tuned`)**: `n=100`, adapter `out/sft`

## Before / After Metrics

| Metric | Direction | Base | SFT-Tuned | Delta (Tuned - Base) |
| :--- | :---: | ---: | ---: | ---: |
| Task Success (numeric ≤1% tol) | ↑ higher | 0.0% | 100.0% | **+100.0 pp** |
| Tool Call Accuracy (ordered recall) | ↑ higher | 77.5% | 100.0% | **+22.5 pp** |
| Exact Tool Sequence Match | ↑ higher | 48.0% | 100.0% | **+52.0 pp** |
| Malformed / Failed Call Rate | ↓ lower | 71.0% | 0.0% | **-71.0 pp** |
| Avg Tool Calls / Task | ↓ efficient | 1.69 | 1.48 | **-0.21** |
| Avg Latency / Task (s) | ↓ lower | 24.41s | 18.06s | **-6.34s** |
| Avg Completion Tokens / Task | ↓ lower | 308.24 | 98.77 | **-209.47** |
| Throughput (completion tok/s) | ↑ higher | 12.63 | 5.47 | **-7.16** |
| AuditKIT Verified Success | ↑ higher | 0.0% | 100.0% | **+100.0 pp** |
| AuditKIT Tool Call F1 (name) | ↑ higher | 76.0% | 100.0% | **+24.0 pp** |
| AuditKIT Tool Call Validity | ↑ higher | 100.0% | 100.0% | **+0.0 pp** |
| AuditKIT Redundant Tool Calls | ↓ lower | 0.12 | 0.00 | **-0.12** |
| AuditKIT No-Loop Score | ↑ higher | 77.0% | 100.0% | **+23.0 pp** |

## Cost & ROI Analysis

| Deployment Option | Throughput (tok/s) | Avg Latency / Task | Est. Cost / 1k Tasks | Data Governance |
| :--- | ---: | ---: | ---: | :--- |
| **Tiny Aya Base (T4 @ $0.35/hr)** | 12.6 | 24.41s | $2.3727 | Self-hosted / VPC |
| **LedgerCall Tuned LoRA (T4 @ $0.35/hr)** | 5.5 | 18.06s | $1.7559 | Self-hosted / VPC |

> **Assumptions:**
> - T4-equivalent cloud spot/on-demand rate assumed at **$0.35/hr** (sequential batch=1 rollout latency; continuous batching lowers unit cost further).
