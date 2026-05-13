#!/usr/bin/env python3
"""
One-command entrypoint to run the strict 3-stage pipeline:
  Stage 1 (SFT) -> Stage 2 (multi-objective) -> Stage 3 (DPO)

This script does NOT re-implement training logic; it orchestrates the existing
stage scripts with consistent paths and optional resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]  # .../Legal_posco-3stages
PIPE = REPO / "q1_3stage_pipeline"


def sh(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_final_train(train_path: Path, val_path: Path, out_path: Path) -> None:
    if out_path.is_file():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Concatenate train + val at dialogue-level to create a "final train" file.
    out_path.write_bytes(train_path.read_bytes() + b"\n" + val_path.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="q1_3stage_pipeline/configs/pipeline_default.yaml")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--gpu", default="", help="Optional: set CUDA_VISIBLE_DEVICES for stage2/stage3 (e.g. '1').")

    ap.add_argument("--run-name", default="", help="Optional suffix for output dirs.")

    # Stage toggles
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--skip-stage2", action="store_true")
    ap.add_argument("--skip-stage3", action="store_true")

    # Resume flags
    ap.add_argument("--resume-stage2", action="store_true", help="Resume Stage 2 from output_dir/checkpoints/latest.pt.")
    ap.add_argument(
        "--resume-stage3",
        action="store_true",
        help="Resume Stage 3 DPO from the latest HF checkpoint under the stage3 output dir (e.g. checkpoint-200).",
    )

    # Stage 2 runtime knobs (safe defaults)
    ap.add_argument("--stage2-entail-max-new-tokens", type=int, default=32)
    ap.add_argument("--stage2-entail-every", type=int, default=2)
    ap.add_argument("--stage2-entail-cache-size", type=int, default=4096)
    ap.add_argument("--stage2-checkpoint-every", type=int, default=10)
    ap.add_argument("--stage2-grad-accum", type=int, default=8)
    ap.add_argument("--stage2-grad-clip", type=float, default=1.0)
    ap.add_argument("--stage2-skip-grad-norm-threshold", type=float, default=0.0)
    ap.add_argument("--stage2-load-in-4bit", action="store_true", help="Load policy in 4-bit to fit on busy GPUs.")
    ap.add_argument("--stage2-nli-on-cpu", action="store_true", help="Run NLI teacher on CPU to save VRAM.")

    # Stage 3 knobs
    ap.add_argument("--stage3-beta", type=float, default=0.1)
    ap.add_argument("--stage3-lr", type=float, default=5e-6)
    ap.add_argument("--stage3-epochs", type=float, default=1.0)
    ap.add_argument("--stage3-batch-size", type=int, default=1)
    ap.add_argument("--stage3-grad-accum", type=int, default=8)

    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = load_yaml(cfg_path)

    paths = cfg.get("paths", {})
    ckpt_root = Path(paths.get("checkpoints_root", "q1_3stage_pipeline/logs/checkpoints"))
    if not ckpt_root.is_absolute():
        ckpt_root = REPO / ckpt_root

    data = cfg.get("data", {})
    train_path = Path(data["train_path"])
    val_path = Path(data["val_path"])
    final_train_path = Path(data.get("final_train_path", "q1_3stage_pipeline/data/final_train_dialogues.jsonl"))
    if not train_path.is_absolute():
        train_path = REPO / train_path
    if not val_path.is_absolute():
        val_path = REPO / val_path
    if not final_train_path.is_absolute():
        final_train_path = REPO / final_train_path

    ensure_final_train(train_path, val_path, final_train_path)

    suffix = f"_{args.run_name}" if args.run_name else ""

    m1_dir = ckpt_root / "stage1" / f"M1_seed{args.seed}_qlora{suffix}"
    m2_dir = ckpt_root / "stage2" / f"M2_fromM1_seed{args.seed}_full_finaltrain{suffix}"
    m3_dir = ckpt_root / "stage3" / f"M3_fromM2_seed{args.seed}_beta{args.stage3_beta}{suffix}"

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Avoid hub network failures in shared environments.
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # ---------------- Stage 1 ----------------
    if not args.skip_stage1:
        # Note: Stage 1 script defaults to full-finetune unless memory fallback is requested.
        sh(
            [
                sys.executable,
                "-u",
                str(PIPE / "stage1_sft" / "train.py"),
                "--config",
                str(cfg_path),
                "--train-jsonl",
                str(train_path),
                "--val-jsonl",
                str(val_path),
                "--output-dir",
                str(m1_dir),
                "--seed",
                str(args.seed),
            ],
            env=env,
        )

    # M1 final HF folder expected by Stage 2.
    m1_final = m1_dir / "final"
    if not m1_final.is_dir() and not args.skip_stage2:
        raise SystemExit(f"Stage1 final folder not found: {m1_final}")

    # ---------------- Stage 2 ----------------
    if not args.skip_stage2:
        stage2_cmd = [
            sys.executable,
            "-u",
            str(PIPE / "stage2_multi_objective" / "train.py"),
            "--config",
            str(cfg_path),
            "--init-from",
            "m1",
            "--m1-path",
            str(m1_final),
            "--ablation",
            "full",
            "--train-jsonl",
            str(final_train_path),
            "--val-jsonl",
            str(val_path),
            "--output-dir",
            str(m2_dir),
            "--eval-every",
            "200",
            "--seed",
            str(args.seed),
            "--num-epochs",
            "1",
            "--gen-max-new-tokens",
            "96",
            "--entail-max-new-tokens",
            str(args.stage2_entail_max_new_tokens),
            "--entail-every",
            str(args.stage2_entail_every),
            "--entail-cache-size",
            str(args.stage2_entail_cache_size),
            "--fixed-eval-every",
            "200",
            "--checkpoint-every",
            str(args.stage2_checkpoint_every),
            "--gradient-accumulation-steps",
            str(args.stage2_grad_accum),
            "--grad-clip-max-norm",
            str(args.stage2_grad_clip),
            "--skip-grad-norm-threshold",
            str(args.stage2_skip_grad_norm_threshold),
        ]
        if args.stage2_load_in_4bit:
            stage2_cmd.append("--load-in-4bit")
        if args.stage2_nli_on_cpu:
            stage2_cmd.append("--nli-on-cpu")
        if args.resume_stage2:
            stage2_cmd.append("--resume")

        sh(stage2_cmd, env=env)

    m2_final = m2_dir / "final"
    if not m2_final.is_dir() and not args.skip_stage3:
        raise SystemExit(f"Stage2 final folder not found: {m2_final}")

    # ---------------- Stage 3 ----------------
    if not args.skip_stage3:
        stage3_cmd = [
            sys.executable,
            "-u",
            str(PIPE / "stage3_dpo" / "train.py"),
            "--m2-path",
            str(m2_final),
            "--train-jsonl",
            str(final_train_path),
            "--output-dir",
            str(m3_dir),
            "--beta",
            str(args.stage3_beta),
            "--lr",
            str(args.stage3_lr),
            "--epochs",
            str(args.stage3_epochs),
            "--batch-size",
            str(args.stage3_batch_size),
            "--grad-accum",
            str(args.stage3_grad_accum),
            "--seed",
            str(args.seed),
        ]
        if args.resume_stage3:
            stage3_cmd.append("--resume")
        sh(stage3_cmd, env=env)

    # Write a tiny run manifest.
    manifest = {
        "config": str(cfg_path),
        "seed": args.seed,
        "stage1_dir": str(m1_dir),
        "stage2_dir": str(m2_dir),
        "stage3_dir": str(m3_dir),
        "final_train_jsonl": str(final_train_path),
        "val_jsonl": str(val_path),
    }
    (ckpt_root / "pipeline_last_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDone. Wrote manifest to:", ckpt_root / "pipeline_last_run.json")


if __name__ == "__main__":
    main()

