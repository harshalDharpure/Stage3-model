#!/usr/bin/env python3
"""
Greedy-generate on flattened dialogue-level test SFT pairs, then aggregate metrics (same stack as run_eval.py).

Use for **Stage 1 (M1)** or **Stage 2 (M2)** HF `final/` folders (or any compatible causal LM checkpoint).

Examples:

  python3 -u q1_3stage_pipeline/evaluation/checkpoint_test_eval.py \\
    --eval-name stage1_m1_test \\
    --model-path q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed43_qlora/final \\
    --test-jsonl q1_3stage_pipeline/data/test_20_dialogues.jsonl \\
    --refs-jsonl q1_3stage_pipeline/logs/evaluation/test_flat_refs.jsonl \\
    --preds-jsonl q1_3stage_pipeline/logs/evaluation/stage1_m1_test_predictions.jsonl \\
    --results-json q1_3stage_pipeline/logs/evaluation/stage1_m1_test_results.json \\
    --results-md q1_3stage_pipeline/logs/evaluation/stage1_m1_test_results.md

  python3 -u q1_3stage_pipeline/evaluation/checkpoint_test_eval.py \\
    --eval-name stage2_m2_test \\
    --model-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final \\
    ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.utils import DatasetBuilder, load_jsonl, set_global_seed


def main() -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-name", required=True, help="Label stored in metrics JSON, e.g. stage1_m1_test.")
    ap.add_argument("--model-path", required=True, help="HF folder with weights + tokenizer (e.g. .../final).")
    ap.add_argument("--test-jsonl", required=True, help="Dialogue-level test JSONL.")
    ap.add_argument(
        "--refs-jsonl",
        default="",
        help="Optional: write flattened ref pairs (same order as preds). Empty = skip.",
    )
    ap.add_argument("--preds-jsonl", required=True, help="One JSON object per line (candidate + ids).")
    ap.add_argument("--results-json", required=True, help="Aggregate metrics JSON.")
    ap.add_argument("--results-md", required=True, help="Human-readable metrics markdown.")
    ap.add_argument("--markdown-title", default="", help="Override markdown H1; default uses --eval-name.")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--max-examples", type=int, default=0, help="If >0, only first N pairs (smoke test).")
    ap.add_argument(
        "--bertscore",
        action="store_true",
        help="Include BERTScore (slow; large download). Default: off.",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)

    model_path = args.model_path if os.path.isabs(args.model_path) else str(_REPO / args.model_path)
    test_path = args.test_jsonl if os.path.isabs(args.test_jsonl) else str(_REPO / args.test_jsonl)
    preds_path = args.preds_jsonl if os.path.isabs(args.preds_jsonl) else str(_REPO / args.preds_jsonl)
    results_json = args.results_json if os.path.isabs(args.results_json) else str(_REPO / args.results_json)
    results_md = args.results_md if os.path.isabs(args.results_md) else str(_REPO / args.results_md)
    refs_path = args.refs_jsonl.strip()
    if refs_path and not os.path.isabs(refs_path):
        refs_path = str(_REPO / refs_path)

    for p in (preds_path, results_json, results_md) + ((refs_path,) if refs_path else ()):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_path)
    flat: list[dict] = []
    for ex in DatasetBuilder(rows).build_sft():
        flat.append(
            {
                "dialogue_id": ex["dialogue_id"],
                "turn_index": ex["turn_index"],
                "language": ex.get("language", ""),
                "statutes_cited": ex.get("statutes_cited", []),
                "metadata": ex.get("metadata", {}),
                "input": ex["prompt"],
                "output": ex["output"],
            }
        )
    if args.max_examples > 0:
        flat = flat[: int(args.max_examples)]

    tag = args.eval_name
    print(f"[{tag}] Flattened test examples: {len(flat)}", flush=True)
    if refs_path:
        with open(refs_path, "w", encoding="utf-8") as f:
            for row in flat:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{tag}] Wrote refs JSONL: {refs_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = next(model.parameters()).device

    refs: list[str] = []
    cands: list[str] = []
    statutes: list = []

    with open(preds_path, "w", encoding="utf-8") as fout:
        for row in tqdm(flat, desc="generate", file=sys.stdout, dynamic_ncols=True):
            prompt = str(row["input"]).strip()
            ref = str(row["output"]).strip()
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=int(args.max_input_tokens),
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = out[0, in_len:]
            cand = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            refs.append(ref)
            cands.append(cand)
            statutes.append(row.get("statutes_cited", []) or [])
            fout.write(
                json.dumps(
                    {
                        "dialogue_id": row.get("dialogue_id", ""),
                        "turn_index": row.get("turn_index", 0),
                        "candidate": cand,
                        "reference": ref,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    from q1_3stage_pipeline.evaluation.legal_metrics import statute_correctness_score
    from q1_3stage_pipeline.evaluation.metrics import calculate_batch_metrics, calculate_nli_score, calculate_response_length_stats
    from q1_3stage_pipeline.evaluation.safety_metrics import harmful_output_flag, refusal_flag

    metrics: dict = {
        "eval_name": args.eval_name,
        "model_path": model_path,
        "test_jsonl": test_path,
        "n_examples": len(flat),
        "max_new_tokens": int(args.max_new_tokens),
        "max_input_tokens": int(args.max_input_tokens),
        "seed": int(args.seed),
        "bertscore_enabled": bool(args.bertscore),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics.update(calculate_batch_metrics(refs, cands, lang="en", include_bertscore=bool(args.bertscore)))
    metrics.update(calculate_response_length_stats(refs, cands))
    try:
        metrics.update(calculate_nli_score(refs, cands))
    except Exception as e:
        metrics["nli_error"] = str(e)

    legal_scores = [statute_correctness_score(statutes_cited=s, candidate=c) for s, c in zip(statutes, cands)]
    if legal_scores:
        metrics["statute_precision"] = sum(x["statute_precision"] for x in legal_scores) / len(legal_scores)
        metrics["statute_recall"] = sum(x["statute_recall"] for x in legal_scores) / len(legal_scores)
        metrics["statute_f1"] = sum(x["statute_f1"] for x in legal_scores) / len(legal_scores)
    metrics["harmful_rate"] = sum(harmful_output_flag(c) for c in cands) / max(len(cands), 1)
    metrics["refusal_rate"] = sum(refusal_flag(c) for c in cands) / max(len(cands), 1)

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    title = (args.markdown_title or "").strip() or f"Test-set generation metrics ({args.eval_name})"

    def _fmt_md(m: dict) -> str:
        lines = [
            f"# {title}",
            "",
            f"- **eval_name:** `{m.get('eval_name', '')}`",
            f"- **Model:** `{m.get('model_path', '')}`",
            f"- **Test JSONL:** `{m.get('test_jsonl', '')}`",
            f"- **Flattened pairs:** {m.get('n_examples', 0)}",
            f"- **max_new_tokens:** {m.get('max_new_tokens')}",
            f"- **max_input_tokens:** {m.get('max_input_tokens')}",
            f"- **UTC time:** {m.get('generated_at_utc', '')}",
            "",
            "## Aggregate metrics",
            "",
        ]
        skip = {"eval_name", "model_path", "test_jsonl", "generated_at_utc"}
        for k in sorted(m.keys()):
            if k in skip:
                continue
            v = m[k]
            if isinstance(v, float):
                lines.append(f"- **{k}:** {v:.6f}")
            else:
                lines.append(f"- **{k}:** {v}")
        lines.extend(["", "## Files", "", f"- Predictions: `{preds_path}`", f"- JSON metrics: `{results_json}`", ""])
        return "\n".join(lines)

    with open(results_md, "w", encoding="utf-8") as f:
        f.write(_fmt_md(metrics))

    print(json.dumps({k: v for k, v in metrics.items() if k not in {"model_path", "test_jsonl"}}, indent=2), flush=True)
    print(f"[{tag}] Wrote {results_json} and {results_md}", flush=True)


if __name__ == "__main__":
    main()
