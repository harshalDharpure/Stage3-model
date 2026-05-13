#!/usr/bin/env python3
"""
Stage 3: DPO alignment (TRL) on top of M2.

STRICT:
- Dataset derives from master dialogue JSONL (split) only.
- Chosen = ground truth.
- Rejected = dynamic hard negatives (NOT stored in dataset).
- Reference model = M2 (frozen).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.stage2_multi_objective.hard_negatives import (
    corrupt_legal_text,
    cross_sample_negative,
    model_negative_generate,
    select_hard_negative,
)
from q1_3stage_pipeline.utils import DatasetBuilder, load_jsonl, set_global_seed

def load_prefs(path: str) -> dict:
    prompts, chosen, rejected = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            chosen.append(row["chosen"])
            rejected.append(row["rejected"])
    return {"prompt": prompts, "chosen": chosen, "rejected": rejected}


def main() -> None:
    try:
        import torch
        from datasets import Dataset
        from transformers import BitsAndBytesConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        from sentence_transformers import SentenceTransformer
        from peft import AutoPeftModelForCausalLM
        from tqdm.auto import tqdm
    except ImportError as e:
        raise SystemExit(f"Install trl peft datasets: {e}") from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-path", required=True, help="Stage2 final checkpoint folder (HF) for policy init + reference.")
    ap.add_argument("--train-jsonl", default="", help="Dialogue-level split JSONL (strict). If set, preferences are generated dynamically.")
    ap.add_argument("--preferences", default="", help="Optional: prebuilt preferences JSONL (not recommended for strict mode).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--load-in-4bit", action="store_true", help="Load policy/ref in 4-bit (recommended if VRAM is limited).")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=1280)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing (uses more VRAM).")
    ap.add_argument("--preferences-cache", default="", help="Path to JSONL cache for mined preferences. Defaults to <output-dir>/preferences.jsonl.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume DPO from the latest Hugging Face checkpoint under --output-dir (e.g. checkpoint-200). "
        "Use the same hyperparameters and --load-in-4bit setting as the original run.",
    )
    ap.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Resume from this checkpoint directory (e.g. .../checkpoint-200). Overrides --resume last-checkpoint search.",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)

    m2_path = args.m2_path if os.path.isabs(args.m2_path) else str(_REPO / args.m2_path)
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else str(_REPO / args.output_dir)

    def _bnb_4bit_cfg() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    def _load_tok(path: str):
        tok = AutoTokenizer.from_pretrained(path, use_fast=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _load_policy_and_ref(path: str):
        """
        Supports both:
        - Full HF checkpoints (AutoModelForCausalLM)
        - PEFT adapter checkpoints (AutoPeftModelForCausalLM)
        """
        is_adapter = os.path.isfile(os.path.join(path, "adapter_config.json"))
        if is_adapter:
            policy = AutoPeftModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=_bnb_4bit_cfg() if args.load_in_4bit else None,
            )
            ref = AutoPeftModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=_bnb_4bit_cfg() if args.load_in_4bit else None,
            )
        else:
            policy = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=_bnb_4bit_cfg() if args.load_in_4bit else None,
            )
            ref = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=_bnb_4bit_cfg() if args.load_in_4bit else None,
            )
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        return policy, ref

    if args.preferences and args.train_jsonl:
        raise SystemExit("Provide only one of --train-jsonl (strict dynamic) or --preferences.")
    if not args.preferences and not args.train_jsonl:
        raise SystemExit("Provide --train-jsonl (recommended) or --preferences.")

    if args.preferences:
        raw = load_prefs(args.preferences if os.path.isabs(args.preferences) else str(_REPO / args.preferences))
        dataset = Dataset.from_dict(raw)
    else:
        os.makedirs(out_dir, exist_ok=True)
        cache_path = args.preferences_cache or os.path.join(out_dir, "preferences.jsonl")
        if not os.path.isabs(cache_path):
            cache_path = str(_REPO / cache_path)

        if os.path.isfile(cache_path):
            print(f"[stage3] Loading cached preferences from {cache_path}", flush=True)
            raw = load_prefs(cache_path)
            dataset = Dataset.from_dict(raw)
        else:
            split_path = args.train_jsonl if os.path.isabs(args.train_jsonl) else str(_REPO / args.train_jsonl)
            rows = load_jsonl(split_path)
            builder = DatasetBuilder(rows)
            sft = builder.build_sft()

            # Frozen SBERT for filtering/hard mining.
            sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            for p in sbert.parameters():
                p.requires_grad = False

            # We'll load policy model first (used for generation of model negative).
            tok = _load_tok(m2_path)
            # Policy used only for negative generation; prefer 4-bit if enabled.
            is_adapter = os.path.isfile(os.path.join(m2_path, "adapter_config.json"))
            if is_adapter:
                policy_for_gen = AutoPeftModelForCausalLM.from_pretrained(
                    m2_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=_bnb_4bit_cfg() if args.load_in_4bit else None,
                )
            else:
                policy_for_gen = AutoModelForCausalLM.from_pretrained(
                    m2_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=_bnb_4bit_cfg() if args.load_in_4bit else None,
                )
            gen_device = next(policy_for_gen.parameters()).device

            rng = random.Random(args.seed)
            prompts, chosen, rejected = [], [], []
            tmp_path = cache_path + ".tmp"
            print(f"[stage3] Mining preferences over {len(sft)} examples -> {cache_path}", flush=True)
            with open(tmp_path, "w", encoding="utf-8") as fout:
                for i, ex in enumerate(tqdm(sft, desc="mining preferences", file=sys.stdout, dynamic_ncols=True), start=1):
                    x = ex["prompt"]
                    y_pos = ex["output"]
                    cand1 = model_negative_generate(policy_for_gen, tok, x, gen_device, rng)
                    cand2 = corrupt_legal_text(y_pos)
                    cand3 = cross_sample_negative(sft, rng, avoid_dialogue_id=str(ex.get("dialogue_id", "")))
                    y_neg = select_hard_negative(
                        x=x,
                        y_pos=y_pos,
                        candidates=[cand1, cand2, cand3],
                        sentence_encoder=sbert,
                        sim_high_threshold=0.2,
                    )
                    prompts.append(x)
                    chosen.append(y_pos)
                    rejected.append(y_neg)
                    fout.write(json.dumps({"prompt": x, "chosen": y_pos, "rejected": y_neg}, ensure_ascii=False) + "\n")
                    if i % 50 == 0:
                        fout.flush()
            os.replace(tmp_path, cache_path)
            print(f"[stage3] Saved preferences cache: {cache_path}", flush=True)

            del policy_for_gen
            torch.cuda.empty_cache()
            dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})

    tok = _load_tok(m2_path)

    # Policy init = M2; Reference = M2 frozen (strict spec).
    model, ref_model = _load_policy_and_ref(m2_path)
    if not args.no_grad_checkpoint:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    os.makedirs(out_dir, exist_ok=True)

    resume_ckpt: str | None = None
    if args.resume_from_checkpoint:
        r = args.resume_from_checkpoint
        resume_ckpt = r if os.path.isabs(r) else str(_REPO / r)
        if not os.path.isdir(resume_ckpt):
            raise SystemExit(f"--resume-from-checkpoint is not a directory: {resume_ckpt}")
    elif args.resume:
        from transformers.trainer_utils import get_last_checkpoint

        resume_ckpt = get_last_checkpoint(out_dir)
        if not resume_ckpt:
            raise SystemExit(f"--resume set but no checkpoint found under {out_dir}")

    # TRL versions differ: some use max_prompt_length on DPOConfig, newer ones only max_length.
    _dpo_kw = dict(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=10,
        save_steps=200,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=(not args.no_grad_checkpoint),
        optim="paged_adamw_8bit",
        seed=args.seed,
        overwrite_output_dir=False,
    )
    try:
        dpo_args = DPOConfig(**_dpo_kw, max_prompt_length=args.max_prompt_length)
    except TypeError:
        dpo_args = DPOConfig(**_dpo_kw)

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tok,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_args,
            train_dataset=dataset,
            tokenizer=tok,
        )

    if resume_ckpt:
        print(f"[stage3] Resuming from checkpoint: {resume_ckpt}", flush=True)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(os.path.join(out_dir, "final"))
    tok.save_pretrained(os.path.join(out_dir, "final"))
    print("Saved", os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()

