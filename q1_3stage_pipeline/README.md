# Q1-style 3-stage legal dialogue research pipeline (standalone)

This folder contains **only** the 3-stage pipeline we discussed:

- **Stage 1 (SFT)** → **Stage 2 (multi-objective)** → **Stage 3 (DPO)**
- Strict **train / validation / test** protocol (no test leakage)

## Directory layout

- `configs/`: default YAML (`pipeline_default.yaml`)
- `data/`: dataset sync + `prepare_splits.py`, `merge_train_val.py`
- `stage1_sft/`: masked causal LM SFT
- `stage2_multi_objective/`: \(L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}\)
- `stage3_dpo/`: TRL DPO
- `evaluation/`: `metrics.py`, `run_eval.py`, `stats.py`
- `ablation/`: Stage 2 ablation runner
- `logs/`: optional logs

## Recent code updates (May 2026)

- **Stage 3 (`stage3_dpo/train.py`)**: Loads **PEFT adapter** checkpoints (Stage 2 `best/` folders), optional **`--load-in-4bit`** for VRAM; **`DPOConfig`** uses `max_prompt_length` only when your TRL version supports it (avoids `TypeError` on newer TRL).
- **Stage 2**: Earlier updates include entailment caching / `entail_every`, loss scheduling, gradient clipping and diagnostics, lighter PEFT checkpoints, **`--resume`**. See `REPORT_3STAGE_PIPELINE.md`.
- **Orchestration**: `run_full_pipeline.py` and `run_full_pipeline.sh` for end-to-end runs.
- **Training outputs** (checkpoints, JSONL logs) stay under `q1_3stage_pipeline/logs/` and are **gitignored**; copy them with `rsync` when moving to another server.

## Quickstart

Run from the **repo root** (so relative paths work):

All commands below assume `python3` is available (on this machine `python` may not exist).

### Run the complete pipeline (one command)

This repository includes a single orchestrator script that runs the full strict pipeline:

**Stage 1 (SFT) → Stage 2 (multi-objective) → Stage 3 (DPO)**.

From the repo root:

```bash
python3 q1_3stage_pipeline/run_full_pipeline.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --seed 43 \
  --gpu 0
```

### Run the complete pipeline (one command, shell + background)

If you prefer a single `.sh` script (easy to run with `nohup`), use:

```bash
nohup bash q1_3stage_pipeline/run_full_pipeline.sh --gpu 0 --seed 43 > /dev/null 2>&1 &
```

Logs will be written under:
`q1_3stage_pipeline/logs/pipeline_runs/`.

Notes:
- This script does **not** re-implement training; it calls the existing stage entrypoints:
  - `q1_3stage_pipeline/stage1_sft/train.py`
  - `q1_3stage_pipeline/stage2_multi_objective/train.py`
  - `q1_3stage_pipeline/stage3_dpo/train.py`
- It will auto-create `q1_3stage_pipeline/data/final_train_dialogues.jsonl` (dialogue-level train+val) if missing.
- If you want Stage 2 to resume from `output_dir/checkpoints/latest.pt`, add `--resume-stage2`.

### Prerequisites (Hugging Face access + token)

You must have access to the base model repo:
`meta-llama/Meta-Llama-3.1-8B-Instruct`.

Set your Hugging Face token **in your shell** (do not write it into any file).

```bash
export HF_TOKEN="PASTE_YOUR_TOKEN_HERE"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Optional verification:

```bash
python3 -c "import os; print('HF_TOKEN' in os.environ, 'HUGGINGFACE_HUB_TOKEN' in os.environ)"
```

Alternative persistent login (writes to your user cache):

```bash
huggingface-cli login
```

### Offline vs online model loading

On some shared clusters, we run with offline mode to avoid network/gated-repo failures.
If you are on a new server and need to download models, run with:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 q1_3stage_pipeline/run_full_pipeline.py --seed 43 --gpu 0
```

### 0) Sync the dataset into this folder

```bash
python3 q1_3stage_pipeline/data/sync_dataset.py \
  --source-dir experiments/exp3_pretraining_finetuning/finetuning \
  --out-dir q1_3stage_pipeline/data/raw
```

### 1) Create splits (train/val/test)

```bash
python3 q1_3stage_pipeline/data/prepare_splits.py \
  --source q1_3stage_pipeline/data/raw/train.jsonl \
  --out-dir q1_3stage_pipeline/data/splits \
  --ratios 0.8 0.1 0.1 \
  --seed 42
```

### Using the NEW dialogue-level 70/10/20 split (recommended)

If you already created the dialogue-level split (NO pairs) with:
`data/create_70_10_20_split_dialogue_level.py`,
then use:

- `q1_3stage_pipeline/data/train_70_dialogues.jsonl`
- `q1_3stage_pipeline/data/val_10_dialogues.jsonl`
- `q1_3stage_pipeline/data/test_20_dialogues.jsonl`

The training code will flatten dialogues into (input, output) examples **in-memory**.

## STRICT experiment protocol (no leakage)

- **train / validation / test** splits are strict:
  - Train ONLY on train
  - Validation ONLY for tuning (hyperparams / early stopping / selecting β)
  - Test NEVER used until the very end
- After tuning is complete:
  - Create **final_train = train + validation**
  - Retrain **Stage 2 (M2)** and **Stage 3 (M3)** from scratch using `final_train`
  - Run evaluation **exactly once** on test

## Global formatting contract (must stay consistent)

All stages use the same strict prompt template:

```text
[USER]: {input}
[ASSISTANT]:
```

### 2) Stage 1 — SFT (M1)

```bash
python3 q1_3stage_pipeline/stage1_sft/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl q1_3stage_pipeline/data/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42 \
  --seed 42
```

### 3) Stage 2 — Multi-objective (M2)

Stage 2 is **initialized from M1** and trains:

\[
L = L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}
\]

- \(L_{entail}\): frozen DeBERTa-large MNLI teacher + KL distillation head (teacher forcing; no gradient through decoding)
- \(L_{triplet}\): dynamic hard negatives (model-gen + legal corruption + cross-sample) + SBERT filtering + hard mining

```bash
python3 q1_3stage_pipeline/stage2_multi_objective/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --init-from m1 \
  --m1-path q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42/final \
  --ablation full \
  --train-jsonl q1_3stage_pipeline/data/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_full_seed42 \
  --eval-every 50 \
  --seed 42
```

### 4) Stage 3 — DPO (M3)

Stage 3 runs DPO where:
- **chosen** = ground truth
- **rejected** = dynamic hard negatives (generated on-the-fly; not stored in the dataset)
- **reference model** = M2 (frozen)

```bash
python3 q1_3stage_pipeline/stage3_dpo/train.py \
  --m2-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_full_seed42/final \
  --train-jsonl q1_3stage_pipeline/data/train_70_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage3/M3_beta0.1_seed42 \
  --beta 0.1 \
  --seed 42
```

#### β sweep (required)

```bash
for beta in 0.1 0.5 1.0; do
  python3 q1_3stage_pipeline/stage3_dpo/train.py \
    --m2-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_full_seed42/final \
    --train-jsonl q1_3stage_pipeline/data/train_70_dialogues.jsonl \
    --output-dir "q1_3stage_pipeline/logs/checkpoints/stage3/M3_beta${beta}_seed42" \
    --beta "$beta" \
    --seed 42
done
```

### 5) Stage 2 ablations

```bash
python3 q1_3stage_pipeline/ablation/run_stage2_ablations.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl q1_3stage_pipeline/data/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/val_10_dialogues.jsonl \
  --m1-path q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42/final \
  --out-root q1_3stage_pipeline/logs/checkpoints/stage2_ablations
```

### 6) Evaluation helper (reference/candidate pairs)

You must generate model outputs on the **test split** first, then run `run_eval.py`.
`pred-jsonl` must be in the same order as `test-jsonl` and contain either `candidate` or `output`.

```bash
python3 q1_3stage_pipeline/evaluation/run_eval.py \
  --test-jsonl q1_3stage_pipeline/data/test_20_dialogues.jsonl \
  --pred-jsonl q1_3stage_pipeline/logs/preds.jsonl
```

`run_eval.py` reports automatic metrics (ROUGE/BLEU/METEOR/NLI) plus:
- statute correctness proxies from `statutes_cited`
- safety/refusal proxy rates

## Multiple seeds (required)

Run every reported setting with **3 random seeds** (example: 42, 43, 44) and report mean/std.

Example for Stage 1:

```bash
for seed in 42 43 44; do
  python3 q1_3stage_pipeline/stage1_sft/train.py \
    --config q1_3stage_pipeline/configs/pipeline_default.yaml \
    --train-jsonl q1_3stage_pipeline/data/train_70_dialogues.jsonl \
    --val-jsonl q1_3stage_pipeline/data/val_10_dialogues.jsonl \
    --output-dir "q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed${seed}" \
    --seed "$seed"
done
```

