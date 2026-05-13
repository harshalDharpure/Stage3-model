#!/usr/bin/env python3
"""
Stage 2: multi-objective training (L_gen + λ1 L_entail + λ2 L_triplet).

Example:
  python q1_3stage_pipeline/stage2_multi_objective/train.py \
    --config q1_3stage_pipeline/configs/pipeline_default.yaml \
    --init-from base \
    --ablation full \
    --train-jsonl q1_3stage_pipeline/data/splits/train.jsonl \
    --val-jsonl q1_3stage_pipeline/data/splits/val.jsonl \
    --output-dir q1_3stage_pipeline/logs/checkpoints/stage2/M2_base_full_seed42 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import math
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.stage1_sft.dataset import LegalSFTDataset, collate_sft_batch
from q1_3stage_pipeline.stage2_multi_objective.hard_negatives import (
    corrupt_legal_text,
    cross_sample_negative,
    model_negative_generate,
    select_hard_negative,
)
from q1_3stage_pipeline.stage2_multi_objective.losses import (
    FrozenSentenceEncoder,
    EntailmentStudentHead,
    FrozenNLITeacher,
    kl_teacher_student,
    pooled_assistant_hidden,
    triplet_margin_loss,
)
from q1_3stage_pipeline.utils import load_jsonl, set_global_seed


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_run_config(dst_dir: str, cfg_path: str, extra: dict) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        with open(cfg_path, encoding="utf-8") as src:
            f.write(src.read())
    with open(os.path.join(dst_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2, ensure_ascii=False)


def _checkpoint_paths(output_dir: str) -> tuple[str, str]:
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    return ckpt_dir, os.path.join(ckpt_dir, "latest.pt")


def _save_training_checkpoint(
    *,
    output_dir: str,
    epoch: int,
    step_i: int,
    opt_steps: int,
    best_val: float | None,
    model,
    triplet_proj: torch.nn.Module,
    entail_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    ckpt_dir, ckpt_path = _checkpoint_paths(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    is_peft = hasattr(model, "peft_config")
    payload = {
        "epoch": int(epoch),
        "step_i": int(step_i),
        "opt_steps": int(opt_steps),
        "best_val": None if best_val is None else float(best_val),
        # Saving an 8B full state_dict every few steps is extremely slow and can stall training.
        # If this is a PEFT model, save only adapter weights (plus heads/optimizer/scaler).
        "is_peft": bool(is_peft),
        "model_state_dict": (
            {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
            if is_peft
            else {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ),
        "triplet_proj_state_dict": {k: v.detach().cpu() for k, v in triplet_proj.state_dict().items()},
        "entail_head_state_dict": {k: v.detach().cpu() for k, v in entail_head.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(payload, ckpt_path)


def _load_training_checkpoint(
    *,
    output_dir: str,
    model,
    triplet_proj: torch.nn.Module,
    entail_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, int | float | None]:
    _, ckpt_path = _checkpoint_paths(output_dir)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    if bool(payload.get("is_peft", False)):
        set_peft_model_state_dict(model, payload["model_state_dict"])
    else:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    triplet_proj.load_state_dict(payload["triplet_proj_state_dict"], strict=True)
    entail_head.load_state_dict(payload["entail_head_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    return {
        "epoch": int(payload.get("epoch", 0)),
        "step_i": int(payload.get("step_i", 0)),
        "opt_steps": int(payload.get("opt_steps", 0)),
        "best_val": payload.get("best_val", None),
    }


def load_llama_qlora_model_only(base_name: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb,
        device_map="auto",
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora)


def load_init_from_exp3(exp3_rel: str, *, torch_dtype: torch.dtype = torch.float16):
    """Load merged Exp3 HF folder, then attach a new LoRA for Stage 2."""
    path = _REPO / exp3_rel
    if not path.is_dir():
        raise FileNotFoundError(f"exp3 checkpoint not found: {path}")
    print(f"Loading merged Exp3 weights from {path}")
    m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch_dtype, device_map="auto")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(m, lora)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", choices=["base", "m1", "exp3"], default="m1")
    ap.add_argument("--m1-path", default="", help="Path to Stage1 'final' HF folder (required if --init-from m1).")
    ap.add_argument("--ablation", choices=["gen_only", "gen_entail", "gen_triplet", "full"], default="full")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=0, help="If >0, run validation every N optimizer steps.")
    ap.add_argument("--gen-max-new-tokens", type=int, default=128, help="Max new tokens for model-generated negatives.")
    ap.add_argument("--entail-max-new-tokens", type=int, default=48, help="Max new tokens for greedy decode used in NLI teacher.")
    ap.add_argument("--entail-every", type=int, default=2, help="Compute entailment loss every N optimizer steps (saves time).")
    ap.add_argument("--entail-cache-size", type=int, default=2048, help="LRU cache size for greedy y_hat generations (0 disables).")
    ap.add_argument("--lang-tag-prefix", type=str, default="", help="Optional prefix added to every prompt input, e.g. [HI_EN_LEGAL].")
    ap.add_argument("--fixed-eval-every", type=int, default=200, help="Every N optimizer steps, run 5 fixed prompt checks and log outputs.")
    ap.add_argument("--debug-fast", action="store_true", help="Fast debug: 100 samples, no triplet, entail every 5, low token caps.")
    ap.add_argument(
        "--skip-grad-norm-threshold",
        type=float,
        default=0.0,
        help="If >0, skip optimizer step when grad_norm exceeds this value (after unscale+clip). 0 disables skipping.",
    )
    ap.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=1.0,
        help="Max grad norm for clipping (applied after AMP unscale).",
    )
    ap.add_argument("--load-in-4bit", action="store_true", help="Load policy LM in 4-bit to fit on smaller GPUs.")
    ap.add_argument("--nli-on-cpu", action="store_true", help="Run frozen NLI teacher on CPU to save GPU VRAM.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output_dir/checkpoints/latest.pt if present (weights + optimizer + scaler + counters).",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save a resumable checkpoint every N optimizer steps (writes output_dir/checkpoints/latest.pt).",
    )
    ap.add_argument(
        "--per-device-batch-size",
        type=int,
        default=0,
        help="Override config training.per_device_batch_size if >0.",
    )
    ap.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=0,
        help="Override config training.gradient_accumulation_steps if >0.",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)
    cfg_path = os.path.join(_REPO, args.config) if not os.path.isabs(args.config) else args.config
    cfg = load_yaml(cfg_path)
    base = cfg["project"]["base_model"]
    exp3_path = cfg["project"].get("exp3_checkpoint", "")
    s2 = cfg.get("stage2", {})
    lam_e = float(s2.get("lambda_entail", 0.5))
    lam_t = float(s2.get("lambda_triplet", 0.5))
    # Slightly lower default margin reduces saturation; config can override.
    margin = float(s2.get("triplet_margin", 0.2))
    emb_name = s2.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
    train_cfg = cfg.get("training", {})

    train_rows = load_jsonl(os.path.join(_REPO, args.train_jsonl) if not os.path.isabs(args.train_jsonl) else args.train_jsonl)
    val_path = args.val_jsonl
    if val_path and not os.path.isabs(val_path):
        val_path = str(_REPO / val_path)
    val_rows = load_jsonl(val_path) if val_path and os.path.isfile(val_path) else []

    # Debug-fast overrides (keeps the same codepath, just cheaper).
    if args.debug_fast:
        train_rows = train_rows[:100]
        val_rows = val_rows[:5] if val_rows else []
        args.entail_every = max(int(args.entail_every), 5)
        args.entail_max_new_tokens = min(int(args.entail_max_new_tokens), 16)
        args.gen_max_new_tokens = min(int(args.gen_max_new_tokens), 16)
        lam_t = 0.0
        if args.ablation == "full":
            args.ablation = "gen_entail"

    # Optional dataset-aware language tag prefix for code-mixed prompts.
    if args.lang_tag_prefix:
        prefix = args.lang_tag_prefix.strip()
        if not prefix.endswith("\n"):
            prefix = prefix + "\n"
        for r in train_rows:
            r["input"] = prefix + str(r.get("input", "")).strip()
        for r in val_rows:
            r["input"] = prefix + str(r.get("input", "")).strip()

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match Stage 1 on Ampere+: bf16 weights + autocast; GradScaler must stay off for bf16.
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    if args.init_from == "m1":
        if not args.m1_path:
            raise SystemExit("--m1-path is required when --init-from m1")
        m1 = args.m1_path if os.path.isabs(args.m1_path) else str(_REPO / args.m1_path)
        # M1 can be either a full HF checkpoint OR a PEFT adapter folder (QLoRA/LoRA).
        if os.path.isfile(os.path.join(m1, "adapter_config.json")):
            from peft import PeftModel

            if args.load_in_4bit:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base,
                    quantization_config=bnb,
                    device_map="auto",
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=amp_dtype, device_map="auto")
            model = PeftModel.from_pretrained(base_model, m1)
        else:
            if args.load_in_4bit:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    m1,
                    quantization_config=bnb,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(m1, torch_dtype=amp_dtype, device_map="auto")
    elif args.init_from == "exp3" and exp3_path:
        model = load_init_from_exp3(exp3_path, torch_dtype=amp_dtype)
    else:
        # Fallback: base model (QLoRA) if you want quick experiments.
        model = load_llama_qlora_model_only(base)

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    st_model = FrozenSentenceEncoder(emb_name)
    st_dim = st_model.encoder.get_sentence_embedding_dimension()
    triplet_proj = torch.nn.Linear(hidden_size, st_dim).to(device)
    entail_head = EntailmentStudentHead(hidden_size).to(device)
    nli_teacher = FrozenNLITeacher("microsoft/deberta-large-mnli")
    # NLI teacher device: CPU saves VRAM; GPU fp16 is faster.
    if args.nli_on_cpu:
        nli_device = torch.device("cpu")
        nli_teacher.move_to(nli_device)
    else:
        nli_device = device
        nli_teacher.move_to(nli_device, dtype=torch.float16 if nli_device.type == "cuda" else None)

    lr = float(train_cfg.get("learning_rate", 5e-5))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(triplet_proj.parameters()) + list(entail_head.parameters()),
        lr=lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype != torch.bfloat16))

    max_len = int(train_cfg.get("max_length", 512))
    # LegalSFTDataset supports dialogue-level rows (with `turns`) and will flatten them
    # into (input, output) examples in-memory. We then use those flattened examples
    # for entailment/triplet refs/negs.
    train_ds = LegalSFTDataset(train_rows, tokenizer, max_len, return_row_index=True)
    train_examples = train_ds.examples
    bs_cfg = int(train_cfg.get("per_device_batch_size", 1))
    ga_cfg = int(train_cfg.get("gradient_accumulation_steps", 8))
    bs = max(1, int(args.per_device_batch_size) if int(args.per_device_batch_size) > 0 else bs_cfg)
    ga = max(1, int(args.gradient_accumulation_steps) if int(args.gradient_accumulation_steps) > 0 else ga_cfg)
    loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, tokenizer),
        num_workers=0,
    )

    rng = random.Random(args.seed)
    # Entailment y_hat LRU cache (prompt -> generated) to avoid recomputation.
    yhat_cache: OrderedDict[str, str] = OrderedDict()
    win = 50
    ma_loss = deque(maxlen=win)
    ma_ent = deque(maxlen=win)
    ma_triplet_sat = deque(maxlen=win)  # 1 if "near margin", else 0
    ma_ent_high = deque(maxlen=win)  # 1 if loss_entail > 0.8

    # Fixed evaluation samples (5).
    fixed_examples: list[dict[str, Any]] = []
    if int(args.fixed_eval_every) > 0 and val_rows:
        fixed_ds = LegalSFTDataset(val_rows, tokenizer, max_len, return_row_index=False)
        fixed_examples = fixed_ds.examples[:5]

    os.makedirs(args.output_dir, exist_ok=True)
    _, latest_ckpt = _checkpoint_paths(args.output_dir)
    resume_state: dict[str, int | float | None] | None = None
    if args.resume:
        if os.path.isfile(latest_ckpt):
            resume_state = _load_training_checkpoint(
                output_dir=args.output_dir,
                model=model,
                triplet_proj=triplet_proj,
                entail_head=entail_head,
                optimizer=opt,
                scaler=scaler,
            )
            print(f"Resumed training from checkpoint: {latest_ckpt} ({resume_state})")
        else:
            print(f"--resume set but no checkpoint found at {latest_ckpt}; starting fresh.")

    log_mode = "a" if (args.resume and os.path.isfile(os.path.join(args.output_dir, "train_log.jsonl"))) else "w"
    log_f = open(os.path.join(args.output_dir, "train_log.jsonl"), log_mode, encoding="utf-8")
    best_val = None if resume_state is None or resume_state.get("best_val", None) is None else float(resume_state["best_val"])  # type: ignore[arg-type]
    best_dir = os.path.join(args.output_dir, "best")
    _save_run_config(
        args.output_dir,
        cfg_path,
        {
            "stage": "stage2_multi_objective",
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "init_from": args.init_from,
            "m1_path": args.m1_path,
            "ablation": args.ablation,
            "lambda_entail": lam_e,
            "lambda_triplet": lam_t,
            "triplet_margin": margin,
            "embedding_model": emb_name,
            "nli_teacher": "microsoft/deberta-large-mnli",
            "eval_every": args.eval_every,
            "resume": bool(args.resume),
            "checkpoint_every": int(args.checkpoint_every),
            "entail_every": int(args.entail_every),
            "entail_cache_size": int(args.entail_cache_size),
            "fixed_eval_every": int(args.fixed_eval_every),
            "lang_tag_prefix": str(args.lang_tag_prefix),
            "debug_fast": bool(args.debug_fast),
        },
    )

    def evaluate(rows: list[dict[str, Any]]) -> dict[str, float]:
        """Validation pass: no weight updates, no negative caching to disk."""
        if not rows:
            return {}
        model.eval()
        triplet_proj.eval()
        entail_head.eval()
        ds = LegalSFTDataset(rows, tokenizer, max_len, return_row_index=True)
        examples = ds.examples
        dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=lambda b: collate_sft_batch(b, tokenizer), num_workers=0)
        totals = {"loss": 0.0, "loss_gen": 0.0, "loss_entail": 0.0, "loss_triplet": 0.0}
        n_batches = 0
        with torch.no_grad():
            for batch in dl:
                row_indices = batch.pop("_row_indices")
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                    output_hidden_states=True,
                )
                loss_gen = out.loss
                hidden = out.hidden_states[-1]
                mask = (labels != -100).long()
                pooled = pooled_assistant_hidden(hidden, mask).float()

                sub_rows = [examples[ri] for ri in row_indices]
                refs: list[str] = [r.get("output", "").strip() for r in sub_rows]
                prompts: list[str] = [str(r.get("input", "")).strip() for r in sub_rows]

                negs: list[str] = []
                for j, r in enumerate(sub_rows):
                    x = prompts[j]
                    y_pos = refs[j]
                    # Use deterministic corruption + cross-sample for val; avoid slow generation here.
                    cand2 = corrupt_legal_text(y_pos)
                    cand3 = cross_sample_negative(examples, rng, avoid_dialogue_id=str(r.get("dialogue_id", "")))
                    y_neg = select_hard_negative(
                        x=x,
                        y_pos=y_pos,
                        candidates=[cand2, cand3],
                        sentence_encoder=st_model.encoder,
                        sim_high_threshold=0.2,
                        sim_pos_gap_min=0.02,
                        sim_pos_gap_max=0.35,
                    )
                    negs.append(y_neg)

                ref_emb = st_model.encode_texts(refs, torch.device("cpu")).to(device)
                neg_emb = st_model.encode_texts(negs, torch.device("cpu")).to(device)

                # Teacher probs using greedy decode (no sampling)
                y_hats: list[str] = []
                for x in prompts:
                    inputs = tokenizer(x, return_tensors="pt", add_special_tokens=False).to(device)
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=int(args.entail_max_new_tokens),
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_ids = gen[0, inputs["input_ids"].shape[1] :]
                    y_hats.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                teacher_p = nli_teacher.probs(refs, y_hats, device=nli_device).to(device)
                student_logits = entail_head(pooled)
                loss_e = kl_teacher_student(teacher_p, student_logits)

                anchor = triplet_proj(pooled)
                loss_tr = triplet_margin_loss(anchor, ref_emb, neg_emb, margin=margin)

                if args.ablation == "gen_only":
                    loss = loss_gen
                elif args.ablation == "gen_entail":
                    loss = loss_gen + lam_e * loss_e
                elif args.ablation == "gen_triplet":
                    loss = loss_gen + lam_t * loss_tr
                else:
                    loss = loss_gen + lam_e * loss_e + lam_t * loss_tr

                totals["loss"] += float(loss.item())
                totals["loss_gen"] += float(loss_gen.item())
                totals["loss_entail"] += float(loss_e.item())
                totals["loss_triplet"] += float(loss_tr.item())
                n_batches += 1
        if n_batches == 0:
            return {}
        return {k: v / n_batches for k, v in totals.items()}

    def _get_schedule_weights(progress: float) -> tuple[float, float]:
        # First 30%: (0.3, 0.3), mid: (0.7, 0.5), late: (1.0, 0.7)
        if progress < 0.30:
            return 0.3, 0.3
        if progress < 0.70:
            return 0.7, 0.5
        return 1.0, 0.7

    def _maybe_cached_greedy(prompt: str) -> str:
        """Greedy decode with small LRU cache (prompt -> generation)."""
        if int(args.entail_cache_size) <= 0:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            gen = model.generate(
                **inputs,
                max_new_tokens=int(args.entail_max_new_tokens),
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = gen[0, inputs["input_ids"].shape[1] :]
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if prompt in yhat_cache:
            y = yhat_cache.pop(prompt)
            yhat_cache[prompt] = y
            return y
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=int(args.entail_max_new_tokens),
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = gen[0, inputs["input_ids"].shape[1] :]
        y = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        yhat_cache[prompt] = y
        while len(yhat_cache) > int(args.entail_cache_size):
            yhat_cache.popitem(last=False)
        return y

    def _log_fixed_eval(opt_step: int) -> None:
        """Every N steps: generate on 5 fixed prompts + log teacher entail prob."""
        if not fixed_examples:
            return
        prompts = [str(r.get("input", "")).strip() for r in fixed_examples]
        refs = [str(r.get("output", "")).strip() for r in fixed_examples]
        with torch.no_grad():
            hyps: list[str] = [_maybe_cached_greedy(p) for p in prompts]
            teacher_p = nli_teacher.probs(refs, hyps, device=nli_device).to(device)
            entail_scores = teacher_p[:, 2].detach().float().cpu().tolist()  # [c, n, e]
        log_f.write(
            json.dumps(
                {
                    "type": "fixed_eval",
                    "opt_step": int(opt_step),
                    "items": [
                        {"prompt": prompts[i], "ref": refs[i], "gen": hyps[i], "entail_p": float(entail_scores[i])}
                        for i in range(len(prompts))
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_f.flush()

    # Monotonic micro-batch counter across epochs (one row per dataloader batch in train_log.jsonl).
    micro_batch_global = {"count": 0}

    def run_epoch(epoch: int, *, start_step_i: int = 0, start_opt_steps: int = 0) -> float:
        nonlocal best_val
        model.train()
        triplet_proj.train()
        entail_head.train()
        total = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        step_i = int(start_step_i)

        opt_steps = int(start_opt_steps)
        skipped_updates = 0
        # Rolling grad norm stats for diagnostics.
        ma_gn = deque(maxlen=50)
        ma_skipped = deque(maxlen=50)  # 1 if skipped update, else 0
        # Rough total optimizer steps for scheduling (based on dataloader length).
        total_opt_steps = max(1, int(math.ceil((len(loader) * max(1, args.num_epochs)) / max(1, ga))))
        pbar = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            micro_batch_global["count"] += 1
            progress_sched = float(opt_steps) / float(total_opt_steps)
            w_sched_e, w_sched_t = _get_schedule_weights(progress_sched)
            row_indices = batch.pop("_row_indices")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                    output_hidden_states=True,
                )
                loss_gen = out.loss.float()
                hidden = out.hidden_states[-1]
            mask = (labels != -100).long()
            pooled = pooled_assistant_hidden(hidden, mask).float()

            # row_indices refer to the flattened examples inside LegalSFTDataset.
            sub_rows = [train_examples[ri] for ri in row_indices]
            batch_refs: list[str] = [r.get("output", "").strip() for r in sub_rows]
            batch_prompts: list[str] = [str(r.get("input", "")).strip() for r in sub_rows]

            # Dynamic hard negatives (DO NOT STORE in dataset):
            # 1) model negative
            # 2) contradictory legal corruption of y+
            # 3) cross-sample negative from another dialogue
            batch_negs: list[str] = []
            for j, r in enumerate(sub_rows):
                x = batch_prompts[j]
                y_pos = batch_refs[j]
                cand1 = model_negative_generate(
                    model,
                    tokenizer,
                    x,
                    device,
                    rng,
                    max_new_tokens=int(args.gen_max_new_tokens),
                )
                cand2 = corrupt_legal_text(y_pos)
                cand3 = cross_sample_negative(train_examples, rng, avoid_dialogue_id=str(r.get("dialogue_id", "")))

                # Filter + hard mine using frozen Sentence-BERT similarities.
                y_neg = select_hard_negative(
                    x=x,
                    y_pos=y_pos,
                    candidates=[cand1, cand2, cand3],
                    sentence_encoder=st_model.encoder,
                    sim_high_threshold=0.2,
                    sim_pos_gap_min=0.02,
                    sim_pos_gap_max=0.35,
                )
                batch_negs.append(y_neg)

            with torch.no_grad():
                ref_emb = st_model.encode_texts(batch_refs, torch.device("cpu")).to(device)
                neg_emb = st_model.encode_texts(batch_negs, torch.device("cpu")).to(device)

            # L_entail (spec): DeBERTa-large MNLI teacher KL (expensive).
            # premise = ground truth (y+), hypothesis = model output (y_hat).
            # Teacher forcing only; no sampling gradients (we decode y_hat under no_grad).
            compute_entail = (int(args.entail_every) > 0) and (opt_steps % int(args.entail_every) == 0)
            if compute_entail and args.ablation in ("gen_entail", "full"):
                with torch.no_grad():
                    y_hats: list[str] = [_maybe_cached_greedy(x) for x in batch_prompts]
                    teacher_p = nli_teacher.probs(batch_refs, y_hats, device=nli_device).to(device)
                student_logits = entail_head(pooled)
                loss_e = kl_teacher_student(teacher_p, student_logits)
            else:
                loss_e = torch.zeros((), device=device, dtype=torch.float32)
            anchor = triplet_proj(pooled)
            loss_tr = triplet_margin_loss(anchor, ref_emb, neg_emb, margin=margin)

            if args.ablation == "gen_only":
                loss = loss_gen
                contrib_gen = float(loss_gen.item())
                contrib_ent = 0.0
                contrib_tri = 0.0
            elif args.ablation == "gen_entail":
                contrib_gen = float(loss_gen.item())
                contrib_ent = float((lam_e * w_sched_e) * loss_e.item())
                contrib_tri = 0.0
                loss = loss_gen + (lam_e * w_sched_e) * loss_e
            elif args.ablation == "gen_triplet":
                contrib_gen = float(loss_gen.item())
                contrib_ent = 0.0
                contrib_tri = float((lam_t * w_sched_t) * loss_tr.item())
                loss = loss_gen + (lam_t * w_sched_t) * loss_tr
            else:
                contrib_gen = float(loss_gen.item())
                contrib_ent = float((lam_e * w_sched_e) * loss_e.item())
                contrib_tri = float((lam_t * w_sched_t) * loss_tr.item())
                loss = loss_gen + (lam_e * w_sched_e) * loss_e + (lam_t * w_sched_t) * loss_tr

            entail_loss_logged = (
                float(loss_e.item())
                if compute_entail and args.ablation in ("gen_entail", "full")
                else None
            )
            schedule_lambda_entail = (lam_e * w_sched_e) if args.ablation in ("gen_entail", "full") else 0.0
            schedule_lambda_triplet = (lam_t * w_sched_t) if args.ablation in ("gen_triplet", "full") else 0.0

            # Live three-loss readout on the tqdm bar (raw L_gen / L_entail / L_tri for this micro-batch).
            _ent_disp = f"{entail_loss_logged:.4f}" if entail_loss_logged is not None else "—"
            pbar.set_postfix_str(
                f"Lgen={float(loss_gen.item()):.4f} Lent={_ent_disp} Ltri={float(loss_tr.item()):.4f}",
                refresh=True,
            )

            loss = loss / ga
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            total += loss.item() * ga
            n += 1
            step_i += 1

            if step_i % ga == 0:
                # IMPORTANT: with AMP/GradScaler, grads are scaled. Unscale before measuring/clipping,
                # otherwise grad_norm can appear to "explode" (artifact of scaling).
                scaler.unscale_(opt)

                params = list(model.parameters()) + list(triplet_proj.parameters()) + list(entail_head.parameters())
                gmin = float("inf")
                gmax = float("-inf")
                any_grad = False
                for p in params:
                    if p.grad is None:
                        continue
                    any_grad = True
                    gg = p.grad.detach()
                    if gg.is_sparse:
                        gg = gg.coalesce().values()
                    if gg.numel() == 0:
                        continue
                    gmin = min(gmin, float(gg.min().item()))
                    gmax = max(gmax, float(gg.max().item()))
                if not any_grad:
                    grad_norm = 0.0
                    gmin = 0.0
                    gmax = 0.0
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip_max_norm)).item())

                skip_thr = float(args.skip_grad_norm_threshold)
                do_skip = bool(any_grad and skip_thr > 0.0 and grad_norm > skip_thr)
                if do_skip:
                    skipped_updates += 1
                    ma_skipped.append(1)
                    log_f.write(
                        json.dumps(
                            {"type": "warn", "opt_step": int(opt_steps), "msg": "grad_norm_skip", "grad_norm": grad_norm, "threshold": skip_thr},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log_f.flush()
                    opt.zero_grad(set_to_none=True)
                    # Still update scaler to avoid getting "stuck" at a bad scale.
                    scaler.update()
                else:
                    ma_skipped.append(0)
                    # Update ratio diagnostics (trainable params only): ||Δθ|| / ||θ||.
                    # This is cheap for PEFT+heads and gives a stable "how big was the update" signal.
                    trainable = [p for p in params if p.requires_grad]
                    # Note: We avoid an expensive exact ||Δθ||. For PEFT+heads, we can
                    # approximate update ratio by using gradient norm and lr:
                    # update_ratio ≈ lr * ||g|| / ||θ|| (after unscale+clip).
                    with torch.no_grad():
                        p2 = 0.0
                        for p in trainable:
                            p2 += float((p.detach().float().pow(2)).sum().item())
                    param_norm = math.sqrt(max(p2, 0.0))
                    scaler.step(opt)
                    scaler.update()
                    lr0 = float(opt.param_groups[0]["lr"]) if opt.param_groups else 0.0
                    update_ratio = float((lr0 * float(grad_norm)) / (param_norm + 1e-12))
                    opt.zero_grad(set_to_none=True)
                    opt_steps += 1

                ma_gn.append(float(grad_norm))

                if opt_steps > 0 and (opt_steps % 50 == 0):
                    # Optional diagnostics: rolling stats.
                    mean_gn = float(sum(ma_gn) / max(1, len(ma_gn))) if ma_gn else None
                    skipped_pct = float(100.0 * (sum(ma_skipped) / max(1, len(ma_skipped)))) if ma_skipped else None
                    log_f.write(
                        json.dumps(
                            {
                                "type": "grad_diag",
                                "opt_step": int(opt_steps),
                                "rolling_mean_grad_norm_w50": mean_gn,
                                "pct_skipped_updates_w50": skipped_pct,
                                "skipped_updates_total": int(skipped_updates),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log_f.flush()

                if int(args.checkpoint_every) > 0 and (opt_steps % int(args.checkpoint_every) == 0):
                    _save_training_checkpoint(
                        output_dir=args.output_dir,
                        epoch=epoch,
                        step_i=step_i,
                        opt_steps=opt_steps,
                        best_val=best_val,
                        model=model,
                        triplet_proj=triplet_proj,
                        entail_head=entail_head,
                        optimizer=opt,
                        scaler=scaler,
                    )

                if int(args.fixed_eval_every) > 0 and (opt_steps % int(args.fixed_eval_every) == 0):
                    _log_fixed_eval(opt_steps)

                if args.eval_every and val_rows and (opt_steps % int(args.eval_every) == 0):
                    metrics = evaluate(val_rows)
                    if metrics:
                        log_f.write(json.dumps({"epoch": epoch, "opt_step": opt_steps, **{f"val_{k}": v for k, v in metrics.items()}}) + "\n")
                        log_f.flush()
                        cur = float(metrics.get("loss", 0.0))
                        if best_val is None or cur < best_val:
                            best_val = cur
                            os.makedirs(best_dir, exist_ok=True)
                            tokenizer.save_pretrained(best_dir)
                            model.save_pretrained(best_dir)
                            torch.save(triplet_proj.state_dict(), os.path.join(best_dir, "triplet_proj.pt"))
                            torch.save(entail_head.state_dict(), os.path.join(best_dir, "entail_head.pt"))

            log_f.write(
                json.dumps(
                    {
                        "type": "train_micro_batch",
                        "epoch": epoch,
                        "global_micro_batch": int(micro_batch_global["count"]),
                        "micro_step": int(step_i),
                        "batches_per_epoch": int(len(loader)),
                        "opt_step": int(opt_steps),
                        "grad_norm": float(grad_norm) if "grad_norm" in locals() else None,
                        "min_grad": float(gmin) if "gmin" in locals() else None,
                        "max_grad": float(gmax) if "gmax" in locals() else None,
                        "skipped_update": bool(do_skip) if "do_skip" in locals() else False,
                        "entail_computed": bool(compute_entail) if "compute_entail" in locals() else False,
                        "moving_avg_loss": float(sum(ma_loss) / max(1, len(ma_loss))) if ma_loss else None,
                        "moving_avg_loss_entail": float(sum(ma_ent) / max(1, len(ma_ent))) if ma_ent else None,
                        "pct_triplet_near_margin_w50": float(100.0 * (sum(ma_triplet_sat) / max(1, len(ma_triplet_sat)))) if ma_triplet_sat else None,
                        "pct_entail_gt_0p8_w50": float(100.0 * (sum(ma_ent_high) / max(1, len(ma_ent_high)))) if ma_ent_high else None,
                        # Raw objective terms (same micro-batch; entail null when skipped via entail_every or ablation).
                        "generation_loss": float(loss_gen.item()),
                        "entailment_loss": entail_loss_logged,
                        "triplet_loss": float(loss_tr.item()),
                        # Legacy aliases (loss_entail null when entail not computed this step).
                        "loss_gen": float(loss_gen.item()),
                        "loss_entail": entail_loss_logged,
                        "loss_triplet": float(loss_tr.item()),
                        # Weighted terms actually added into L before gradient_accumulation normalization (λ * schedule * L_*).
                        "weighted_generation": float(contrib_gen) if "contrib_gen" in locals() else None,
                        "weighted_entailment": float(contrib_ent) if "contrib_ent" in locals() else None,
                        "weighted_triplet": float(contrib_tri) if "contrib_tri" in locals() else None,
                        "schedule_lambda_entail": float(schedule_lambda_entail),
                        "schedule_lambda_triplet": float(schedule_lambda_triplet),
                        "loss_contrib_gen": float(contrib_gen) if "contrib_gen" in locals() else None,
                        "loss_contrib_entail": float(contrib_ent) if "contrib_ent" in locals() else None,
                        "loss_contrib_triplet": float(contrib_tri) if "contrib_tri" in locals() else None,
                        "triplet_contrib_frac": (
                            float(contrib_tri) / float(max(contrib_gen + contrib_ent + contrib_tri, 1e-12))
                            if "contrib_gen" in locals() and "contrib_ent" in locals() and "contrib_tri" in locals()
                            else None
                        ),
                        "scaler_scale": float(scaler.get_scale()) if device.type == "cuda" else None,
                        "lr": float(opt.param_groups[0]["lr"]) if opt.param_groups else None,
                        "update_ratio": float(update_ratio) if "update_ratio" in locals() else None,
                        "loss": float(loss.item() * ga),
                    }
                )
                + "\n"
            )
            log_f.flush()

            ma_loss.append(float(loss.item() * ga))
            ma_ent.append(float(loss_e.item()))
            ma_triplet_sat.append(1 if float(loss_tr.item()) >= float(margin) * 0.90 else 0)
            ma_ent_high.append(1 if float(loss_e.item()) > 0.8 else 0)
            if len(ma_ent_high) == win and (sum(ma_ent_high) / win) > 0.25:
                log_f.write(
                    json.dumps({"type": "warn", "opt_step": int(opt_steps), "msg": "entailment_loss_gt_0.8_frequent_w50"}, ensure_ascii=False)
                    + "\n"
                )
                log_f.flush()

        if step_i % ga != 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(triplet_proj.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        return total / max(n, 1)

    start_ep = 0
    start_step_i = 0
    start_opt_steps = 0
    if resume_state is not None:
        start_ep = int(resume_state["epoch"])
        start_step_i = int(resume_state["step_i"])
        start_opt_steps = int(resume_state["opt_steps"])

    for ep in range(start_ep, args.num_epochs):
        # If resuming mid-epoch, only the counters are restored; dataloader order may differ from the original run.
        avg = run_epoch(ep, start_step_i=start_step_i if ep == start_ep else 0, start_opt_steps=start_opt_steps if ep == start_ep else 0)
        print(f"epoch {ep} avg_loss={avg:.4f}")
        start_step_i = 0
        start_opt_steps = 0

    # Save full final checkpoint (M2) for Stage 3 reference/policy init.
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    tokenizer.save_pretrained(final_dir)
    model.save_pretrained(final_dir)
    triplet_proj.cpu()
    torch.save(triplet_proj.state_dict(), os.path.join(final_dir, "triplet_proj.pt"))
    entail_head.cpu()
    torch.save(entail_head.state_dict(), os.path.join(final_dir, "entail_head.pt"))
    with open(os.path.join(final_dir, "stage2_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "init_from": args.init_from,
                "m1_path": args.m1_path,
                "ablation": args.ablation,
                "lambda_entail": lam_e,
                "lambda_triplet": lam_t,
                "embedding_model": emb_name,
                "nli_teacher": "microsoft/deberta-large-mnli",
            },
            f,
            indent=2,
        )
    log_f.close()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()

