# About this fork

Fork of [opendatalab/MinerU](https://github.com/opendatalab/MinerU) with a
YOLO redaction detection model integrated into the document analysis
pipeline. Scanned documents with redactions are extracted with
`[REDACTED]` markers preserved in the markdown / content_list output.

Part of the thesis on redaction detection in documents — parent repo
[Daniel-SebastianGaete/tesis](https://github.com/Daniel-SebastianGaete/tesis)
orchestrates the full pipeline (OCR + redaction detection + redaction
classification).

## Branches

| Branch | Purpose | Base |
|---|---|---|
| `master` | Tracks upstream MinerU; no redaction code | upstream master |
| `redaction` | Tested redaction integration used in the thesis | upstream `a12610fb` + 3 commits |

`master` exists so future upstream syncs (`git fetch upstream &&
git merge upstream/master`) are fast-forward. The `redaction` branch is
the version actually used in the thesis pipeline.

The tag `v0.1-redaction-tested-on-a12610fb` pins the exact state used
for thesis results.

## Using the redaction branch

```bash
git clone https://github.com/Daniel-SebastianGaete/redaction-analysis-mineru.git
cd redaction-analysis-mineru
git checkout redaction
uv sync
MINERU_REDACTION_WEIGHTS=/path/to/yolo/best.pt uv run mineru -p input/ -o output/
```

When `MINERU_REDACTION_WEIGHTS` points at a YOLO weights file, the pipeline
detects redaction regions on each page, treats them as a first-class
block type, and renders them as `[REDACTED]` in markdown /
content_list output. When the env var is unset, MinerU runs unchanged.

`modified_mineru.sh` is the developer test script with concrete paths.

## Implementation

The integration adds redaction as a new `BlockType` / `ContentType` /
`CategoryId` (102), parallel to layout and formula detection.

Files touched:

- `mineru/model/redaction/yolo_redaction.py` (new) — YOLO model wrapper
  (`predict`, `batch_predict`)
- `mineru/utils/enum_class.py` — `BlockType.REDACTION`,
  `ContentType.REDACTION`, `CategoryId.Redaction = 102`
- `mineru/backend/pipeline/model_list.py` — `AtomicModel.RedactionDetection`
- Pipeline backend:
  - `model_init.py` — env-gated init in `MineruPipelineModel` and
    `MineruHybridModel`
  - `batch_analyze.py` — `batch_predict` call after layout detection
  - `pipeline_magic_model.py` — `get_redaction_blocks()`,
    `__fix_redaction_overlaps()` (drops text/title blocks more than
    70% covered by a redaction)
  - `model_json_to_middle_json.py` — surface redactions as blocks
  - `pipeline_middle_json_mkcontent.py` — render as `[REDACTED]`
  - `utils/model_utils.py` — skip redaction regions in layout routing
- Hybrid backend:
  - `hybrid_analyze.py` — `_run_redaction_detection` (handles both
    pipeline-driven and `vlm_ocr_enable=True` paths)
  - `hybrid_magic_model.py` — REDACTION block surfacing
  - `hybrid_model_output_to_middle_json.py` — include in page blocks
- VLM backend:
  - `vlm_middle_json_mkcontent.py` — render REDACTION spans / blocks /
    content_list entries
- Type membership:
  - `utils/block_sort.py`, `utils/span_block_fix.py` — include
    REDACTION in block-type checks

See `PLAN_A_integrated_redaction_model.md` for the original design.
`PLAN_B_postprocessing_overlay.md` is the rejected alternative,
retained for reference.

## Future upstream port

The redaction integration is based on upstream commit `a12610fb`. After
that commit, upstream made an architectural change:

- The `CategoryId` numeric class was removed from
  `mineru/utils/enum_class.py`
- `mineru/utils/block_sort.py` was deleted
- Numeric `category_id` checks (e.g. `category_id == 102`) were
  replaced with string `label` matching throughout the pipeline

Porting the redaction integration to upstream's current `master` tip
therefore requires:

1. Re-deciding how to identify redaction regions in the new label-based
   system (a new label string instead of `102`)
2. Re-wiring `model_utils`, `pipeline_magic_model`, and the mkcontent
   files to use upstream's current shape
3. Re-testing end-to-end against the existing checkpoints

This is deferred until the parent thesis orchestrator is complete and
there is an end-to-end benchmark to verify the port against.
