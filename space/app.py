"""LedgerCall demo: recorded base vs SFT-tuned Tiny Aya tool-calling trajectories (CPU only)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import gradio as gr

from tools import TOOL_MAP

HERE = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


TEST = load_jsonl(HERE / "test.jsonl")
BASE = load_jsonl(HERE / "base_trajectories.jsonl")
TUNED = load_jsonl(HERE / "tuned_trajectories.jsonl")
COMPARISON = (HERE / "comparison.md").read_text(encoding="utf-8") if (HERE / "comparison.md").exists() else ""
N = min(len(TEST), max(len(BASE), len(TUNED)) or len(TEST))
STAR = re.compile(r"\*\s*(-?[\d,]*\.?\d+)\s*\*")


TAGS = re.compile(r"<\|[A-Z_]+\|>|<EOS_TOKEN>")


def clean(x) -> str:
    return TAGS.sub("", str(x)).strip()


def final_answer(traj: dict) -> str:
    steps = traj.get("steps") or []
    return str(steps[-1].get("observation") or "") if steps else ""


def is_success(traj: dict, answer: str) -> bool:
    m = STAR.search(final_answer(traj))
    if not m:
        return False
    try:
        got, want = float(m.group(1).replace(",", "")), float(answer)
    except ValueError:
        return False
    return abs(got - want) <= max(0.01, abs(want) * 0.01)


def render(traj: dict | None, answer: str) -> str:
    if not traj:
        return "_No recorded trajectory._"
    out = []
    steps = traj.get("steps") or []
    for s in steps:
        calls = (s.get("action") or {}).get("tool_calls") or []
        if calls:
            for c in calls:
                out.append(f"**Tool call** `{c.get('name')}`\n```json\n{json.dumps(c.get('arguments'), indent=1)[:800]}\n```")
            out.append(f"**Tool result:** `{str(s.get('observation'))[:400]}`")
        else:
            out.append(f"**Final answer:**\n\n{clean(s.get('observation'))[:1200]}")
    ok = is_success(traj, answer)
    out.append(f"\n---\n**Tool calls:** {sum(len((s.get('action') or {}).get('tool_calls') or []) for s in steps)}  ·  "
               f"**Correct:** {'✅' if ok else '❌'} (expected {answer})")
    return "\n\n".join(out)


def show(idx: int):
    idx = int(idx)
    row = TEST[idx]
    q = row["prompt"][-1]["content"]
    meta = f"**Task {idx}** · type `{row.get('task_type')}` · expected tools `{', '.join(row.get('expected_tools', []))}`\n\n> {q}"
    b = BASE[idx] if idx < len(BASE) else None
    t = TUNED[idx] if idx < len(TUNED) else None
    return meta, render(b, row["answer"]), render(t, row["answer"])


def run_tool(name: str, args_json: str) -> str:
    try:
        args = json.loads(args_json)
        return json.dumps({"result": TOOL_MAP[name](**args)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


EXAMPLES = {
    "convert_currency": '{"amount": 1000, "from_ccy": "USD", "to_ccy": "EUR"}',
    "compound_interest": '{"principal": 10000, "annual_rate_pct": 5, "years": 10, "compounds_per_year": 12}',
    "percent_change": '{"old": 120, "new": 150}',
    "loan_payment": '{"principal": 250000, "annual_rate_pct": 6.5, "months": 360}',
}

with gr.Blocks(title="LedgerCall") as demo:
    gr.Markdown("# LedgerCall\nTiny Aya Global (3.35B) before and after AgentTune-rendered trajectory SFT on 4 deterministic finance tools. "
                "Trajectories below were recorded on held-out test tasks during evaluation (greedy decoding, T4).")
    if COMPARISON:
        with gr.Accordion("Before / after metrics", open=True):
            gr.Markdown(COMPARISON)
    with gr.Tab("Compare trajectories"):
        idx = gr.Slider(0, max(N - 1, 0), value=0, step=1, label="Held-out test task")
        meta = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base tiny-aya-global")
                base_md = gr.Markdown()
            with gr.Column():
                gr.Markdown("### LedgerCall (SFT LoRA)")
                tuned_md = gr.Markdown()
        idx.change(show, idx, [meta, base_md, tuned_md])
        demo.load(show, idx, [meta, base_md, tuned_md])
    with gr.Tab("Run the tools"):
        name = gr.Dropdown(list(TOOL_MAP), value="loan_payment", label="Tool")
        args = gr.Code(EXAMPLES["loan_payment"], language="json", label="Arguments")
        name.change(lambda n: EXAMPLES[n], name, args)
        res = gr.Code(language="json", label="Result")
        gr.Button("Run").click(run_tool, [name, args], res)

if __name__ == "__main__":
    demo.launch()
