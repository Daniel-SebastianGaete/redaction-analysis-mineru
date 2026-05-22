#!/usr/bin/env bash
# Developer test script: invoke this fork's MinerU with redaction
# detection enabled, dispatching to a chosen family in the
# redaction-detection sibling repo.
#
# Required env vars for the adapter:
#   MINERU_REDACTION_FAMILY          yolo | mmdet | detectron | rtdetr | rfdetr
#   MINERU_REDACTION_CHECKPOINT      checkpoint path (.pt / .pth)
#   MINERU_REDACTION_DETECTION_ROOT  path to components/redaction-detection
# Family-specific (when applicable):
#   MINERU_REDACTION_MODEL_CONFIG    mmdet config .py, detectron .yaml, rtdetr .yml
#   MINERU_REDACTION_MODEL_TYPE      detectron: maskrcnn | mask2former
#   MINERU_REDACTION_RTDETR_ROOT     rtdetr: path to RT-DETRv2 vendor repo
# Optional:
#   MINERU_REDACTION_CONF            confidence threshold (default 0.5)

set -euo pipefail

# Edit these for your machine. Defaults assume the standard tesis layout.
CHECKPOINT="${MINERU_REDACTION_CHECKPOINT:-/home/danielseb/checkpoints/yolo/foia_yolo_yolov8s_1280/best.pt}"
DETECTION_ROOT="${MINERU_REDACTION_DETECTION_ROOT:-/home/danielseb/tesis/components/redaction-detection}"
INPUT="${INPUT:-/home/danielseb/datasets/redaction_detection_benchmark/images}"

export MINERU_REDACTION_FAMILY="${MINERU_REDACTION_FAMILY:-yolo}"
export MINERU_REDACTION_CHECKPOINT="$CHECKPOINT"
export MINERU_REDACTION_DETECTION_ROOT="$DETECTION_ROOT"

# 1. Hybrid backend (default)
uv run mineru -p "$INPUT" -o tmp/redaction_hybrid_test

# 2. Pipeline backend
# uv run mineru -p "$INPUT" -o tmp/redaction_pipeline_test -b pipeline

# 3. Baseline regression — disable redaction detection by unsetting FAMILY
# unset MINERU_REDACTION_FAMILY
# uv run mineru -p "$INPUT" -o tmp/redaction_baseline_test
