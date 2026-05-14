results_for_download/
=====================
Ready to zip and download (repo root on server).

Contents:
  evaluation/          — All files from q1_3stage_pipeline/logs/evaluation/
  paper_results/       — Paper-oriented tables (EVALUATION_TABLES.md, CSV, JSON, LaTeX)
  training_summaries/  — Stage 1 HF summary + Stage 2 run_args.json

Notes:
  • Stage 2 (M2) test eval: complete (950 pairs) — see evaluation/stage2_m2_test_results.*
  • Stage 1 (M1) test eval: check evaluation/stage1_m1_test_predictions.jsonl line count.
    If not 950, the run was interrupted; stage1_m1_test_results.json may still show n_examples=4 until a full run finishes.

Create a single archive from repo root:
  zip -r results_for_download.zip results_for_download
  # or
  tar -czvf results_for_download.tar.gz results_for_download
