# Research handoff: completed work, remaining work, and server resume

Give this file to anyone bringing the project up on a **new machine**. It records what finished, what did not, what to copy, and the exact commands to continue without redoing Stages 1–2.

---

## Checklist: new server (do in order)

1. **Install system tools:** `git`, `python3` (3.10+ recommended), `rsync` (optional, for copying from old host), NVIDIA driver + CUDA if you use GPU.
2. **Authenticate with GitHub** (pick one): SSH key added to GitHub, or HTTPS with a [Personal Access Token](https://github.com/settings/tokens) (`repo` scope) when `git` asks for a password.
3. **Clone the repo** (either URL; same `main`):
   - `git clone https://github.com/harshalDharpure/Legal_posco-3stages.git`
   - or `git clone https://github.com/harshalDharpure/Stage3-model.git`
4. **`cd` into the repo** and `git pull` whenever you switch machines so you have the latest scripts.
5. **Create a virtualenv** (do not copy `.venv` from the old machine):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r q1_3stage_pipeline/requirements.txt
   ```
   Install **PyTorch with the correct CUDA wheel** for your GPU from [pytorch.org](https://pytorch.org/) if the requirements file does not match your CUDA version.
6. **Copy large artifacts** from the old server with `rsync` or equivalent (see **section 4**): `data/` (especially `final_train_dialogues.jsonl`), M2 `final/`, Stage 3 folder with `checkpoint-200` and `preferences.jsonl`. Copy M1 `final/` too if you will run **test eval** or re-train Stage 2.
7. **Set `HF_TOKEN`** in the environment if `meta-llama/Meta-Llama-3.1-8B-Instruct` (or other models) are gated on your Hugging Face account.
8. **Resume Stage 3** (section 5) or **run test evaluation** (section 10). Use `tmux`/`screen`/`nohup` for long jobs.
9. **Commit and push** any new results you want preserved (`git add` / `git commit` / `git push`; see section 11).

---

## What is in Git vs what you must copy

| Item | In Git on `main`? | Action on new server |
|------|-------------------|------------------------|
| Training / eval **Python code**, configs, `RESEARCH_HANDOFF.md` | Yes | `git pull` |
| **`q1_3stage_pipeline/requirements.txt`** | Yes | `pip install -r ...` |
| **`final_train_dialogues.jsonl`** | Often **no** (`.gitignore`) | **Copy** from old host |
| **Model weights** (`*.safetensors`, etc. under `logs/checkpoints/`) | **No** (too large) | **Copy** `final/` trees and Stage 3 checkpoints |
| **Eval outputs** under `logs/evaluation/` | Only if someone committed them | Optional; regenerate with section 10 |

GitHub is **not** used to store multi‑GB checkpoints; use `rsync`, S3, or an internal file share.

---

## 1. Project shape (three stages)

| Stage | Role | Primary script |
|-------|------|------------------|
| **1** | Supervised fine-tuning (SFT) on dialogue data | `q1_3stage_pipeline/stage1_sft/train.py` |
| **2** | Multi-objective training (generation + entailment + triplet) on top of M1 | `q1_3stage_pipeline/stage2_multi_objective/train.py` |
| **3** | DPO alignment on top of M2 (TRL `DPOTrainer`) | `q1_3stage_pipeline/stage3_dpo/train.py` |

End-to-end orchestration (optional): `q1_3stage_pipeline/run_full_pipeline.py`  
Default path config: `q1_3stage_pipeline/configs/pipeline_default.yaml`

---

## 2. What is **completed** (as of this handoff)

### Stage 1 — **done**

- **Run name:** `M1_seed43_qlora` (seed **43** in this experiment line).
- **Evidence:** `q1_3stage_pipeline/logs/stage1_metrics/M1_seed43_qlora/summary.json` reports finished training through **epoch 3.0**, `train_loss`, `val_perplexity`, `train_runtime`.
- **Exported weights:** `q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed43_qlora/final/` (and intermediate checkpoints under the same parent).

### Stage 2 — **done** (M2 produced for downstream use)

- **Run directory:** `q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/`
- **Evidence:** `best/` and `final/` HF-style trees exist; `train_log.jsonl` covers the full training window for this run; `run_args.json` records hyperparameters and paths.
- **Stage 3 consumes:** `.../M2_fromM1_seed43_full_finaltrain/final/` as `--m2-path`.

### Repository and automation

- Code and **non-weight** run artifacts (metrics, configs, text logs, tokenizer JSON, etc.) are versioned under `q1_3stage_pipeline/logs/` per `.gitignore` rules (large `*.safetensors`, `optimizer.pt`, `latest.pt` remain local only unless you add other storage).
- Stage 3 **resume from checkpoint** is implemented: `--resume`, `--resume-from-checkpoint`, and pipeline flag `--resume-stage3` (see section 5).

---

## 3. What is **left** (not finished)

### Stage 3 (DPO) — **incomplete**

- **Output directory:** `q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1/`
- **Checkpoint on disk:** `checkpoint-200/` (Hugging Face trainer checkpoint).
- **Trainer schedule (from `checkpoint-200/trainer_state.json`):** `max_steps` = **466** for `num_train_epochs` = **1**, but training reached **global_step** = **200** only (~**43%** of one epoch in step terms).
- **No** `final/` export under Stage 3 yet (that is written when training runs to completion and `save_model` runs at the end of the script).
- **Past failures:** some pipeline logs show **`CUDA out of memory`** during Stage 3 on an ~80 GiB GPU with prior settings; the new server may need more headroom or different memory settings (see section 6).

**Goal on the new server:** resume Stage 3 from `checkpoint-200` until training finishes, then verify `.../stage3/.../final/` exists.

---

## 4. What to **copy** to the new server (cannot rely on Git alone)

### Must have for resume

1. **Repo:** `git clone` from GitHub, then `git pull` on branch `main` (same code as handoff with resume flags).
2. **Data (including gitignored files):**
   - `q1_3stage_pipeline/data/train_70_dialogues.jsonl`
   - `q1_3stage_pipeline/data/val_10_dialogues.jsonl`
   - `q1_3stage_pipeline/data/test_20_dialogues.jsonl` (for held-out evaluation later, if used)
   - **`q1_3stage_pipeline/data/final_train_dialogues.jsonl`** — required for Stage 3 strict path; often **not** in Git due to `.gitignore`
3. **Stage 1 `final/`** — only if you plan to re-run Stage 2; not required if you only resume Stage 3.
4. **Stage 2 `final/` (M2)** — **required** for Stage 3: entire folder  
   `q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final/`  
   including all **`.safetensors`** shards and index, configs, tokenizer files.
5. **Stage 3 partial run directory:**  
   `q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1/`  
   at minimum: **`checkpoint-200/`** (full tree), **`preferences.jsonl`**, and any small side files you already had there.

### Optional

- `q1_3stage_pipeline/logs/pipeline_runs/*.log` for debugging history.
- `splits_dialogue_level/` and root `README.md` / `report.md` for documentation context.

### Do **not** copy

- **`.venv/`** — recreate the environment on the new machine (see section 6).

### Practical copy command (example)

From the old host (adjust user, host, and paths):

```bash
rsync -avz --progress /path/to/Legal_posco-3stages/q1_3stage_pipeline/data/ user@NEW:/path/to/Legal_posco-3stages/q1_3stage_pipeline/data/
# M1 final (needed for Stage 1 test eval or to re-run Stage 2 from M1)
rsync -avz --progress /path/to/Legal_posco-3stages/q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed43_qlora/final/ user@NEW:/path/to/Legal_posco-3stages/q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed43_qlora/final/
rsync -avz --progress /path/to/Legal_posco-3stages/q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final/ user@NEW:/path/to/Legal_posco-3stages/q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final/
rsync -avz --progress /path/to/Legal_posco-3stages/q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1/ user@NEW:/path/to/Legal_posco-3stages/q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1/
```

---

## 5. **Resume** Stage 3 only (new server)

Run from the **repository root** (`Legal_posco-3stages/`). Use the **same** hyperparameters as the original Stage 3 run unless you intentionally start a new experiment.

### Canonical resume command

```bash
python3 -u q1_3stage_pipeline/stage3_dpo/train.py \
  --m2-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final \
  --train-jsonl q1_3stage_pipeline/data/final_train_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1 \
  --beta 0.1 \
  --lr 5e-6 \
  --epochs 1.0 \
  --batch-size 1 \
  --grad-accum 8 \
  --seed 43 \
  --resume
```

- **`--resume`** picks up the **latest** checkpoint under `--output-dir` (here, `checkpoint-200` if nothing newer exists).
- To pin a folder explicitly: add **`--resume-from-checkpoint q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1/checkpoint-200`** instead of `--resume`.

### Via pipeline (skip Stages 1–2, resume Stage 3)

```bash
python3 -u q1_3stage_pipeline/run_full_pipeline.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --skip-stage1 \
  --skip-stage2 \
  --resume-stage3
```

Ensure **`final_train_dialogues.jsonl`** exists at the path in the YAML (`final_train_path`) before running.

### Optional: resume **Stage 2** only (multi-objective)

If you ever need to continue Stage 2 from `output_dir/checkpoints/latest.pt` on the new machine, use **`--resume`** on `stage2_multi_objective/train.py`, or the orchestrator:

```bash
python3 -u q1_3stage_pipeline/run_full_pipeline.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --skip-stage1 \
  --resume-stage2
```

You must have copied the full Stage 2 **run directory** (including `latest.pt` if resuming), not only `final/`.

---

## 6. Environment and stability notes

1. **Python / CUDA:** Install PyTorch with CUDA matching the new GPU; reinstall `trl`, `peft`, `transformers`, `accelerate`, `bitsandbytes` (Stage 3 default uses **`paged_adamw_8bit`** in `DPOConfig`).
2. **Hugging Face:** If the base model is gated, set **`HF_TOKEN`** (or `huggingface-cli login`) on the new machine.
3. **Match training mode when resuming:** use the same **`--load-in-4bit`** (on/off) as the run that produced `checkpoint-200`. Mixing quantization settings with an existing checkpoint can fail to load.
4. **VRAM:** If OOM persists, try gradient checkpointing (on by default unless **`--no-grad-checkpoint`**), larger **`--grad-accum`**, or a GPU with more memory — after resume, the trainer continues optimizer state from disk; do not change batch semantics lightly without understanding checkpoint compatibility.

---

## 7. How to **verify** completion after Stage 3

- Directory exists:  
  `q1_3stage_pipeline/logs/checkpoints/stage3/M3_fromM2_seed43_beta0.1/final/`
- `trainer_state.json` in the last checkpoint shows **`global_step`** reached the scheduled **`max_steps`** (or training ended by epoch as configured).
- Script prints **`Saved .../final`** at the end of `stage3_dpo/train.py`.

---

## 8. Quick reference table (completed vs left)

| Item | Status |
|------|--------|
| Stage 1 (`M1_seed43_qlora`) | **Complete** |
| Stage 2 (`M2_fromM1_seed43_full_finaltrain` → `final/`) | **Complete** |
| Stage 3 DPO (`M3_fromM2_seed43_beta0.1`) | **Incomplete** — resume from `checkpoint-200` |
| Stage 3 `final/` export | **Missing** until training finishes |
| Full pipeline log “all stages green” | **Not guaranteed** until Stage 3 completes on a machine without OOM |

---

## 9. Repositories (code)

- **Primary:** `https://github.com/harshalDharpure/Legal_posco-3stages`
- **Mirror / experiment hub:** `https://github.com/harshalDharpure/Stage3-model`  
  Keep **`main`** in sync via `git pull` / `git push` depending on which repo you treat as source of truth.

---

## 10. **Test-set evaluation** (Stage 1 & 2) on another machine

**What is on GitHub:** evaluation **code** only — `q1_3stage_pipeline/evaluation/checkpoint_test_eval.py`, `run_stage1_stage2_test_eval.sh`, `stage2_test_eval.py` (shim), and `metrics.py`. You can `git clone` / `git pull` on the new server and run the same commands.

**What is not in Git (must copy or have locally):**

- **M1 weights:** `q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed43_qlora/final/` (full HF tree including `*.safetensors`).
- **M2 weights:** `q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final/`.
- **Test dialogues:** `q1_3stage_pipeline/data/test_20_dialogues.jsonl` (tracked in Git if present in repo; if missing, copy from your machine).

**Run both evaluations (from repo root, after `python -m venv .venv` + install deps):**

```bash
chmod +x q1_3stage_pipeline/evaluation/run_stage1_stage2_test_eval.sh   # once
./q1_3stage_pipeline/evaluation/run_stage1_stage2_test_eval.sh
```

By default the shell script uses **`$ROOT/.venv/bin/python`**. Override if needed:  
`PY=/path/to/python ./q1_3stage_pipeline/evaluation/run_stage1_stage2_test_eval.sh`

**Smoke test (first N flattened pairs only):** append to the command above, e.g.  
`./q1_3stage_pipeline/evaluation/run_stage1_stage2_test_eval.sh --max-examples 20`

**BERTScore:** append **`--bertscore`** to pass through to both runs (slow; downloads a large model).

**Notes:** `METEOR` may print as **0.0** until NLTK WordNet resources are installed; BLEU, ROUGE, NLI entailment score, and statute proxies still compute. The flattened test set from `test_20_dialogues.jsonl` is on the order of **~950** `(prompt, reference)` pairs — full eval can take **on the order of one hour per model** on a single GPU; use a persistent session.

**Alternative metric entrypoint:** if you already have a predictions JSONL aligned with the test file, you can use `q1_3stage_pipeline/evaluation/run_eval.py --test-jsonl ... --pred-jsonl ...` (see root `README.md`).

Or run `checkpoint_test_eval.py` once per stage with `--eval-name`, `--model-path`, and output paths under `q1_3stage_pipeline/logs/evaluation/` (see script docstring).

**Saving metrics to Git:** after a full run, add and push the generated files, for example:

```bash
git add q1_3stage_pipeline/logs/evaluation/
git status
git commit -m "Test eval results (M1/M2 on test split)"
git push origin main && git push stage3 main
```

Keep total size reasonable (GitHub warns above ~100 MB per file). Prediction JSONLs for ~950 pairs are typically a few MB.

---

## 11. **Git:** saving work and dual remotes

This project is often pushed to **two** remotes with the same `main` history:

| Remote | URL |
|--------|-----|
| `origin` | `https://github.com/harshalDharpure/Legal_posco-3stages` |
| `stage3` | `https://github.com/harshalDharpure/Stage3-model` |

On a **new** clone you only have `origin` unless you add the mirror:

```bash
git remote add stage3 https://github.com/harshalDharpure/Stage3-model.git
```

After local commits:

```bash
git push origin main
git push stage3 main
```

Use **SSH** remotes (`git@github.com:...`) or **HTTPS + PAT** as described in the checklist above.

---

*This document describes the state of the research line using **seed 43** and the checkpoint paths above. If you start a new seed or run name, update paths and this file accordingly.*
