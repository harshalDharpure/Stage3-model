#!/usr/bin/env bash
# Greedy test-set eval for Stage 1 (M1) and Stage 2 (M2) checkpoints. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
TEST_JSONL="${TEST_JSONL:-q1_3stage_pipeline/data/test_20_dialogues.jsonl}"
REFS="${REFS:-q1_3stage_pipeline/logs/evaluation/test_flat_refs.jsonl}"
OUT="$ROOT/q1_3stage_pipeline/logs/evaluation"
mkdir -p "$OUT"

echo "=== Stage 1 (M1) ==="
"$PY" -u q1_3stage_pipeline/evaluation/checkpoint_test_eval.py \
  --eval-name stage1_m1_test \
  --model-path q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed43_qlora/final \
  --test-jsonl "$TEST_JSONL" \
  --refs-jsonl "$REFS" \
  --preds-jsonl q1_3stage_pipeline/logs/evaluation/stage1_m1_test_predictions.jsonl \
  --results-json q1_3stage_pipeline/logs/evaluation/stage1_m1_test_results.json \
  --results-md q1_3stage_pipeline/logs/evaluation/stage1_m1_test_results.md \
  "$@"

echo "=== Stage 2 (M2) ==="
"$PY" -u q1_3stage_pipeline/evaluation/checkpoint_test_eval.py \
  --eval-name stage2_m2_test \
  --model-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_seed43_full_finaltrain/final \
  --test-jsonl "$TEST_JSONL" \
  --refs-jsonl "$REFS" \
  --preds-jsonl q1_3stage_pipeline/logs/evaluation/stage2_m2_test_predictions.jsonl \
  --results-json q1_3stage_pipeline/logs/evaluation/stage2_m2_test_results.json \
  --results-md q1_3stage_pipeline/logs/evaluation/stage2_m2_test_results.md \
  "$@"

echo "Done. Results under $OUT (stage1_m1_* and stage2_m2_*)."

SUM="$OUT/stage1_stage2_test_summary.json"
"$PY" - <<PY
import json
from pathlib import Path
out = Path("$SUM")
p1 = Path("$OUT/stage1_m1_test_results.json")
p2 = Path("$OUT/stage2_m2_test_results.json")
merged = {
    "stage1_m1": json.loads(p1.read_text(encoding="utf-8")),
    "stage2_m2": json.loads(p2.read_text(encoding="utf-8")),
}
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote", out)
PY
