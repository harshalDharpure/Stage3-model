#!/usr/bin/env python3
"""
Stage 1: supervised fine-tuning (causal LM CE on assistant tokens only).

Example:
  python q1_3stage_pipeline/stage1_sft/train.py \
    --config q1_3stage_pipeline/configs/pipeline_default.yaml \
    --train-jsonl q1_3stage_pipeline/data/splits/train.jsonl \
    --val-jsonl q1_3stage_pipeline/data/splits/val.jsonl \
    --output-dir q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.stage1_sft.dataset import LegalSFTDataset, collate_sft_batch
from q1_3stage_pipeline.utils import load_jsonl, set_global_seed


def load_yaml(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_run_config(dst_dir: str, cfg_path: str, extra: dict) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        with open(cfg_path, encoding="utf-8") as src:
            f.write(src.read())
    with open(os.path.join(dst_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2, ensure_ascii=False)


class JsonlLossLogger(TrainerCallback):
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {
            "type": "train_step",
            "ts": datetime.now(timezone.utc).isoformat(),
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()},
        }
        if isinstance(row.get("loss"), (int, float)):
            row["generation_loss"] = float(row["loss"])
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        help="If set, fully fine-tune ALL parameters (no LoRA, no quantization). This is the strict default for Q1.",
    )
    ap.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA (memory fallback). Only use if you cannot full-finetune.",
    )
    ap.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="If set, load the base model in 4-bit (QLoRA-style). Recommended with --use-lora to avoid OOM.",
    )
    ap.add_argument(
        "--metrics-dir",
        default="",
        help="Separate folder to store loss/metrics logs (jsonl + summary). If empty, uses <logs_root>/stage1_metrics/<run_name>.",
    )
    ap.add_argument("--run-name", default="", help="Optional run name for metrics folder naming.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP (multi-GPU full fine-tuning). Run via `accelerate launch --num_processes <N> ...`.",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)
    cfg_path = args.config if os.path.isabs(args.config) else str(_REPO / args.config)
    cfg = load_yaml(cfg_path)
    base = cfg["project"]["base_model"]
    tc = cfg.get("training", {})
    paths = cfg.get("paths", {})

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # STRICT Stage 1 requirement: fully fine-tune all parameters, no freezing.
    # LoRA is allowed only as a memory fallback.
    if args.use_lora and args.full_finetune:
        raise SystemExit("Choose only one of --full-finetune or --use-lora.")
    if (not args.use_lora) and (not args.full_finetune):
        # Make strict behavior the default.
        args.full_finetune = True

    if args.full_finetune:
        # For FSDP / distributed training we must NOT use device_map="auto".
        device_map = None if args.fsdp else ("auto" if torch.cuda.is_available() else None)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map=device_map,
        )
    else:
        # Memory fallback: LoRA fine-tuning. Prefer 4-bit base weights when enabled.
        quant = None
        if args.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant,
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)

    train_path = args.train_jsonl if os.path.isabs(args.train_jsonl) else str(_REPO / args.train_jsonl)
    val_path = (
        args.val_jsonl
        if args.val_jsonl and os.path.isabs(args.val_jsonl)
        else (str(_REPO / args.val_jsonl) if args.val_jsonl else "")
    )
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path and os.path.isfile(val_path) else None

    max_len = int(tc.get("max_length", 512))
    train_ds = LegalSFTDataset(train_rows, tokenizer, max_len)
    val_ds = LegalSFTDataset(val_rows, tokenizer, max_len) if val_rows else None

    collate_fn = lambda b: collate_sft_batch(b, tokenizer)

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else str(_REPO / args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_name = args.run_name.strip() or os.path.basename(os.path.abspath(out_dir)) or f"stage1_seed{args.seed}"
    logs_root = paths.get("logs_root", "q1_3stage_pipeline/logs")
    default_metrics_dir = os.path.join(logs_root, "stage1_metrics", run_name)
    metrics_dir = args.metrics_dir.strip() or default_metrics_dir
    metrics_dir = metrics_dir if os.path.isabs(metrics_dir) else str(_REPO / metrics_dir)
    os.makedirs(metrics_dir, exist_ok=True)
    loss_jsonl = os.path.join(metrics_dir, "loss_log.jsonl")
    _save_run_config(
        metrics_dir,
        cfg_path,
        {
            "stage": "stage1_sft",
            "train_jsonl": train_path,
            "val_jsonl": val_path if val_path else "",
            "output_dir": out_dir,
            "metrics_dir": metrics_dir,
            "seed": args.seed,
            "full_finetune": bool(args.full_finetune),
            "use_lora": bool(args.use_lora),
        },
    )

    ta_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(tc.get("per_device_batch_size", 2)),
        gradient_accumulation_steps=int(tc.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tc.get("learning_rate", 5e-5)),
        num_train_epochs=float(tc.get("num_epochs", 3)),
        logging_steps=int(tc.get("logging_steps", 50)),
        save_steps=int(tc.get("save_steps", 500)),
        save_total_limit=int(tc.get("save_total_limit", 3)),
        fp16=bool(tc.get("fp16", True)),
        report_to="none",
        seed=args.seed,
    )
    # Full fine-tune loads weights in bfloat16; fp16=True enables the AMP GradScaler and can
    # raise "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'".
    if args.full_finetune and torch.cuda.is_available():
        ta_kwargs["fp16"] = False
        ta_kwargs["bf16"] = torch.cuda.is_bf16_supported()
    if args.fsdp:
        # Minimal FSDP setup for LLaMA-style transformer blocks.
        # Use accelerate launcher + multiple GPUs.
        ta_kwargs["fsdp"] = "full_shard auto_wrap"
        ta_kwargs["fsdp_transformer_layer_cls_to_wrap"] = "LlamaDecoderLayer"
        ta_kwargs["gradient_checkpointing"] = True
    if val_ds:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["eval_steps"] = int(tc.get("eval_steps", 500))
        ta_kwargs["load_best_model_at_end"] = True
        ta_kwargs["metric_for_best_model"] = "loss"
    else:
        ta_kwargs["eval_strategy"] = "no"

    args_tr = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[JsonlLossLogger(loss_jsonl)],
    )
    train_out = trainer.train()

    # Perplexity (from final eval loss if available).
    ppl = None
    try:
        if val_ds:
            eval_metrics = trainer.evaluate()
            if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] is not None:
                ppl = float(torch.exp(torch.tensor(eval_metrics["eval_loss"])).item())
    except Exception:
        ppl = None

    # Save a compact summary (separate from checkpoints).
    summary = {
        "base_model": base,
        "seed": args.seed,
        "run_name": run_name,
        "train_jsonl": train_path,
        "val_jsonl": val_path if val_path else "",
        "output_dir": out_dir,
        "metrics_dir": metrics_dir,
        "full_finetune": bool(args.full_finetune),
        "use_lora": bool(args.use_lora),
        "train_runtime": float(getattr(train_out, "metrics", {}).get("train_runtime", 0.0)) if train_out else 0.0,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows) if val_rows else 0,
        "final_metrics": getattr(train_out, "metrics", {}) if train_out else {},
        "val_perplexity": ppl,
    }
    with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    trainer.save_model(os.path.join(out_dir, "final"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final"))
    print("Saved", os.path.join(out_dir, "final"))
    print("Metrics", metrics_dir)


if __name__ == "__main__":
    main()

