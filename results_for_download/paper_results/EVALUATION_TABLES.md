# Paper-ready results tables

**Output folder:** `paper_results/` (repository root).

## How these numbers were produced

- **Test data:** `q1_3stage_pipeline/data/test_20_dialogues.jsonl`, flattened to SFT (prompt, reference) pairs via `DatasetBuilder.build_sft()`.
- **Decoding:** greedy (`do_sample=False`, `num_beams=1`), `max_new_tokens=96`, `max_input_tokens=512`, seed **43**.
- **NLI score:** script-computed entailment-style score over (reference, candidate) pairs (see `q1_3stage_pipeline/evaluation/metrics.py`).
- **Statute metrics:** overlap-style scores using `statutes_cited` fields when present (see `legal_metrics.py`).
- **METEOR / BERTScore:** reported as **0** in these runs (optional resources / flags off); do not interpret as model quality without enabling them.

## Important warnings before publishing

- Stage 1 (M1) test-set metrics below are from a short run (n_examples=4). For the paper, rerun full test eval for M1 to match n=950.

- Stage 2 (M2) test-set metrics are from a full run (n_examples=950).

- Stage 2 training losses are micro-batch logs; tail statistics are over the last 500 logged micro-batches.

## Table 1 — Held-out test automatic metrics

| Model | $n$ | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | R-1 | R-2 | R-L | NLI | Stat P/R/F1 | Harm. | Refusal | Avg ref len | Avg cand len | Len ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 (SFT) | 4 | 0.1323 | 0.0642 | 0.0334 | 0.0196 | 0.2833 | 0.0772 | 0.1799 | 0.6407 | 0.0000/0.0000/0.0000 | 0.0000 | 0.0000 | 87.00 | 52.50 | 0.6034 |
| M2 (multi-objective) | 950 | 0.1451 | 0.0618 | 0.0310 | 0.0173 | 0.2414 | 0.0553 | 0.1538 | 0.4322 | 0.5231/0.3970/0.4134 | 0.0053 | 0.0000 | 89.74 | 53.40 | 0.5951 |

## Table 2 — Training summaries (not test BLEU/ROUGE)

| Item | Value |
| --- | --- |
| Stage 1 final **train loss** (HF Trainer) | **1.016733** |
| Stage 1 **validation perplexity** | **5.353511** |
| Stage 1 epochs completed | **3.0** |
| Stage 2 tail **total loss** mean ± std (last 500 micro-batches) | **1.654286 ± 0.647091** |
| Stage 2 tail **generation loss** mean ± std | **1.480871 ± 0.590463** |
| Stage 2 tail **weighted entailment** mean ± std | **0.123964 ± 0.268322** |
| Stage 2 tail **weighted triplet** mean ± std | **0.049452 ± 0.040843** |
| Stage 2 last logged **optimizer step** / **global micro-batch** | **231** / **3723** |

## Table 3 — Stage 2 objective hyperparameters (from `run_args.json`)

| Key | Value |
| --- | --- |
| `seed` | `43` |
| `init_from` | `m1` |
| `ablation` | `full` |
| `lambda_entail` | `0.5` |
| `lambda_triplet` | `0.5` |
| `triplet_margin` | `0.3` |
| `embedding_model` | `sentence-transformers/all-mpnet-base-v2` |
| `nli_teacher` | `microsoft/deberta-large-mnli` |
| `eval_every` | `200` |
| `checkpoint_every` | `10` |
| `entail_every` | `2` |
| `entail_cache_size` | `4096` |

## Files in this folder

| File | Purpose |
| --- | --- |
| `EVALUATION_TABLES.md` | This document (paper-oriented narrative + tables). |
| `evaluation_testset_wide.csv` | One row per model; wide metric columns. |
| `evaluation_testset_long.csv` | Long/tidy format for plotting in R/Python. |
| `training_summary.json` | Raw Stage 1 summary + Stage 2 tail loss stats. |
| `RESULTS_bundle.json` | Single JSON with protocol, warnings, and embedded eval dicts. |
| `latex_table_test_metrics.tex` | LaTeX `booktabs` table snippet. |
