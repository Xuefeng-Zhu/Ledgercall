"""Push LedgerCall LoRA adapter, datasets/trajectories/metrics, and HF Collections to Hub.

Reads `HF_USER` (and optional `HF_TOKEN`) from environment variables.

Usage:
    HF_USER=<your-hf-username> python scripts/push_to_hub.py \
        --adapter-dir out/grpo \
        --data-dir data \
        --results-dir results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def _build_model_card(hf_user: str, model_repo: str, dataset_repo: str) -> str:
    return f"""---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: other
tags:
  - agenttune
  - auditkit
  - grpo
  - tool-calling
  - finance
  - tiny-aya
datasets:
  - {dataset_repo}
---

# {model_repo}

LoRA (`r=16`, `all-linear`) GRPO adapter for `CohereLabs/tiny-aya-global` trained with **AgentTune** and evaluated with **AuditKIT** for reliable, efficient deterministic finance tool calling (**LedgerCall**, Lexsi AI Safety & Alignment Hackathon Track 3).

- **Base model:** `CohereLabs/tiny-aya-global` (check upstream model card for CC-BY-NC / gating terms)
- **Tools:** `convert_currency`, `compound_interest`, `percent_change`, `loan_payment`
- **Dataset & trajectories:** [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})
- **Author:** `{hf_user}`
"""


def _build_dataset_card(hf_user: str, model_repo: str, dataset_repo: str) -> str:
    return f"""---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - agenttune
  - auditkit
  - tool-calling
  - finance
  - trajectories
---

# {dataset_repo}

Synthetic deterministic finance tool-calling dataset (`train.jsonl`, `test.jsonl`, `stats.json`), AgentTune rollout trajectories (`results/*_trajectories.jsonl`), and AuditKIT evaluation reports (`results/*_metrics.json`, `results/comparison.md`) for **LedgerCall**.

- **Associated GRPO LoRA adapter:** [`{model_repo}`](https://huggingface.co/{model_repo})
- **Prompt leakage check:** 0 overlapping prompts between `train.jsonl` (600) and `test.jsonl` (150)
- **Author:** `{hf_user}`
"""


def push_all(
    *,
    hf_user: str,
    adapter_dir: Path,
    data_dir: Path,
    results_dir: Path,
    model_repo_name: str = "ledgercall-tiny-aya-grpo",
    dataset_repo_name: str = "ledgercall-finance-toolbench",
    dry_run: bool = False,
) -> dict[str, Any]:
    model_repo = f"{hf_user}/{model_repo_name}"
    dataset_repo = f"{hf_user}/{dataset_repo_name}"

    summary: dict[str, Any] = {
        "hf_user": hf_user,
        "model_repo": model_repo,
        "dataset_repo": dataset_repo,
        "dry_run": dry_run,
    }
    if dry_run:
        print(f"[push_to_hub] DRY RUN — would push adapter={adapter_dir} -> {model_repo} and data/results -> {dataset_repo}")
        return summary

    from huggingface_hub import HfApi, add_collection_item, create_collection

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    # 1. Push LoRA adapter repository
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    api.create_repo(repo_id=model_repo, repo_type="model", private=False, exist_ok=True)
    readme_model = adapter_dir / "README.md"
    if not readme_model.exists():
        readme_model.write_text(_build_model_card(hf_user, model_repo, dataset_repo), encoding="utf-8")
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(adapter_dir),
        ignore_patterns=["checkpoint-*", "checkpoint-*/*"],
        commit_message="Upload LedgerCall GRPO LoRA adapter and provenance",
    )
    print(f"[push_to_hub] Uploaded model repo: https://huggingface.co/{model_repo}")

    # 2. Push dataset + trajectories + AuditKIT metrics repository
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=_build_dataset_card(hf_user, model_repo, dataset_repo).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="Add LedgerCall dataset card",
    )
    if data_dir.is_dir():
        api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo="data",
            commit_message="Upload train/test splits and leak-check stats",
        )
    if results_dir.is_dir():
        api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=str(results_dir),
            path_in_repo="results",
            commit_message="Upload AgentTune trajectories and AuditKIT metrics",
        )
    for extra_name in ("trajectories.jsonl", "lexsi_provenance.json", "training_stats.json"):
        extra_file = adapter_dir / extra_name
        if extra_file.is_file():
            api.upload_file(
                path_or_fileobj=str(extra_file),
                path_in_repo=f"training/{extra_name}",
                repo_id=dataset_repo,
                repo_type="dataset",
                commit_message=f"Upload training {extra_name}",
            )
    print(f"[push_to_hub] Uploaded dataset repo: https://huggingface.co/datasets/{dataset_repo}")

    # 3. Create public Model + Dataset Collections
    model_col = create_collection(
        title="LedgerCall Models (Lexsi Hackathon Track 3)",
        namespace=hf_user,
        description="GRPO LoRA adapter teaching CohereLabs/tiny-aya-global reliable finance tool calling.",
        private=False,
        token=token,
        exists_ok=True,
    )
    add_collection_item(
        collection_slug=model_col.slug,
        item_id=model_repo,
        item_type="model",
        token=token,
        exists_ok=True,
    )

    dataset_col = create_collection(
        title="LedgerCall Datasets & Trajectories (Lexsi Hackathon Track 3)",
        namespace=hf_user,
        description="LedgerCall synthetic finance benchmark, AgentTune trajectories, and AuditKIT reports.",
        private=False,
        token=token,
        exists_ok=True,
    )
    add_collection_item(
        collection_slug=dataset_col.slug,
        item_id=dataset_repo,
        item_type="dataset",
        token=token,
        exists_ok=True,
    )

    summary["model_collection_url"] = model_col.url
    summary["dataset_collection_url"] = dataset_col.url
    print(f"[push_to_hub] Public Model Collection:   {model_col.url}")
    print(f"[push_to_hub] Public Dataset Collection: {dataset_col.url}")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Push LedgerCall artifacts and create public HF Collections.")
    parser.add_argument("--hf-user", type=str, default=os.environ.get("HF_USER", ""), help="Hugging Face username/org (defaults to $HF_USER)")
    parser.add_argument("--adapter-dir", type=str, default="out/grpo", help="Trained LoRA adapter directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory (train.jsonl, test.jsonl, stats.json)")
    parser.add_argument("--results-dir", type=str, default="results", help="Evaluation trajectories and metrics directory")
    parser.add_argument("--model-repo-name", type=str, default="ledgercall-tiny-aya-grpo", help="Model repo name under HF_USER")
    parser.add_argument("--dataset-repo-name", type=str, default="ledgercall-finance-toolbench", help="Dataset repo name under HF_USER")
    parser.add_argument("--dry-run", action="store_true", help="Print planned Hub actions without network calls")
    args = parser.parse_args(argv)

    hf_user = args.hf_user.strip()
    if not hf_user or hf_user.startswith("TODO"):
        raise ValueError("Set a valid HF_USER environment variable or pass --hf-user <username>.")

    return push_all(
        hf_user=hf_user,
        adapter_dir=Path(args.adapter_dir),
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        model_repo_name=args.model_repo_name,
        dataset_repo_name=args.dataset_repo_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
