#!/usr/bin/env bash
set -euo pipefail

WEIGHTS=/home/danielseb/checkpoints/yolo/foia_yolo_yolov8s_1280/best.pt
INPUT=/home/danielseb/datasets/redaction_detection_benchmark/images

# 1. Hybrid backend (default) — redaction detection enabled
MINERU_REDACTION_WEIGHTS="$WEIGHTS" \
  uv run mineru -p "$INPUT" -o tmp/redaction_hybrid_test

# 2. Pipeline backend — redaction detection enabled
# MINERU_REDACTION_WEIGHTS="$WEIGHTS" \
#   uv run mineru -p "$INPUT" -o tmp/redaction_pipeline_test -b pipeline

# 3. No redaction weights — baseline regression (no [REDACTED] markers expected)
# uv run mineru -p "$INPUT" -o tmp/redaction_baseline_test