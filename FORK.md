# About this fork

Fork of [opendatalab/MinerU](https://github.com/opendatalab/MinerU) with a
swappable redaction-detection step integrated into the document analysis
pipeline. Scanned documents with redactions are extracted with `*****`
markers preserved in the markdown / content_list output, ready to be
classified downstream.

Part of the thesis on redaction detection in documents — parent repo
[Daniel-SebastianGaete/tesis](https://github.com/Daniel-SebastianGaete/tesis)
orchestrates the full pipeline (OCR + redaction detection + redaction
classification).

## Branches

| Branch | Purpose | Base |
|---|---|---|
| `master` | Tracks upstream MinerU; no redaction code | upstream master |
| `redaction` | Tested redaction integration used in the thesis | upstream `a12610fb` + redaction commits |

`master` exists so future upstream syncs (`git fetch upstream &&
git merge upstream/master`) are fast-forward. The `redaction` branch is
the version actually used in the thesis pipeline.

The tag `v0.1-redaction-tested-on-a12610fb` pins the YOLO-only state
that was used for the original thesis tests, before the multi-family
adapter refactor on this branch.

## Using the redaction branch

The redaction step is enabled by setting `MINERU_REDACTION_FAMILY` (and
related env vars) before invoking MinerU. The chosen detection family
is run via subprocess into the
[redaction-detection](https://github.com/Daniel-SebastianGaete/redaction-detection)
sibling repo, which holds the trained checkpoints and per-family
inference code.

```bash
git clone https://github.com/Daniel-SebastianGaete/redaction-analysis-mineru.git
cd redaction-analysis-mineru
git checkout redaction
uv sync

# Configure the redaction adapter
export MINERU_REDACTION_FAMILY=yolo                              # or mmdet | detectron | rtdetr | rfdetr
export MINERU_REDACTION_CHECKPOINT=/path/to/best.pt
export MINERU_REDACTION_DETECTION_ROOT=/path/to/redaction-detection
# Family-specific (where applicable):
# export MINERU_REDACTION_MODEL_CONFIG=/path/to/config.{py,yaml}  # mmdet, detectron, optional rtdetr
# export MINERU_REDACTION_MODEL_TYPE=mask2former                  # detectron only: maskrcnn | mask2former
# export MINERU_REDACTION_RTDETR_ROOT=/path/to/rtdetrv2_pytorch   # rtdetr only

uv run mineru -p input/ -o output/
```

When `MINERU_REDACTION_FAMILY` is unset, MinerU runs unchanged (no
redaction detection). When set, the adapter is loaded and called per
page batch, redactions are surfaced as first-class blocks, and the
output renders them as `*****`.

`modified_mineru.sh` is the developer test script with concrete paths.

## Implementation

The integration adds redaction as a new `BlockType` / `ContentType` /
`CategoryId` (102), parallel to layout and formula detection.

Detection itself is delegated to the redaction-detection sibling repo
via a subprocess adapter (`mineru/model/redaction/adapter.py`). The
adapter writes a batch of page images to a temp dir, synthesizes a
minimal COCO ground-truth JSON (predict scripts only need it for
filename → image_id mapping), invokes
`<detection-root>/scripts/run.sh models/<family>/predict.py`, reads the
resulting `coco_results.json`, and converts each bbox into MinerU's
`poly` quadrilateral. One subprocess call processes a whole page batch
so the model load amortizes across pages.

Files:

- `mineru/model/redaction/adapter.py` (new) — `RedactionDetectionAdapter`,
  `RedactionDetectionAdapter.from_env()` factory, batch subprocess logic.
- `mineru/utils/enum_class.py` — `BlockType.REDACTION`,
  `ContentType.REDACTION`, `CategoryId.Redaction = 102`.
- `mineru/backend/pipeline/model_list.py` — `AtomicModel.RedactionDetection`.
- Pipeline backend:
  - `model_init.py` — env-gated init in `MineruPipelineModel` and
    `MineruHybridModel` (checks `MINERU_REDACTION_FAMILY`).
  - `batch_analyze.py` — `batch_predict` call after layout detection.
  - `pipeline_magic_model.py` — `get_redaction_blocks()`,
    `__fix_redaction_overlaps()` (drops text/title blocks more than 70%
    covered by a redaction).
  - `model_json_to_middle_json.py` — surface redactions as blocks.
  - `pipeline_middle_json_mkcontent.py` — render as `*****`.
  - `utils/model_utils.py` — skip redaction regions in layout routing.
- Hybrid backend:
  - `hybrid_analyze.py` — `_run_redaction_detection` (handles both the
    pipeline-driven path and the `vlm_ocr_enable=True` path).
  - `hybrid_magic_model.py` — REDACTION block surfacing.
  - `hybrid_model_output_to_middle_json.py` — include in page blocks.
- VLM backend:
  - `vlm_middle_json_mkcontent.py` — render REDACTION spans / blocks /
    content_list entries.
- Type membership:
  - `utils/block_sort.py`, `utils/span_block_fix.py` — include REDACTION
    in block-type checks.

The original YOLO-only model wrapper at `mineru/model/redaction/yolo_redaction.py`
is no longer imported. It can be deleted; left in place for reference.

See `PLAN_A_integrated_redaction_model.md` for the original design.
`PLAN_B_postprocessing_overlay.md` is the rejected alternative, retained
for reference.

## Subprocess cost

Each `batch_predict` call spawns a subprocess that loads the chosen
detection model once and processes all pages in the batch. For a
multi-page document this amortizes well; single-image calls pay the full
model-load tax. The orchestrator should pass all pages of a document
together when possible.

The subprocess relies on `<detection-root>/scripts/run.sh` resolving the
correct per-family `.venv`. Run
`bash <detection-root>/scripts/setup_venvs.sh` before first use.

## Future upstream port

The redaction integration is based on upstream commit `a12610fb`. After
that commit, upstream made an architectural change:

- The `CategoryId` numeric class was removed from
  `mineru/utils/enum_class.py`
- `mineru/utils/block_sort.py` was deleted
- Numeric `category_id` checks (e.g. `category_id == 102`) were replaced
  with string `label` matching throughout the pipeline

Porting the redaction integration to upstream's current `master` tip
therefore requires:

1. Re-deciding how to identify redaction regions in the new label-based
   system (a new label string instead of `102`).
2. Re-wiring `model_utils`, `pipeline_magic_model`, and the mkcontent
   files to use upstream's current shape.
3. Re-testing end-to-end against the existing checkpoints.

This is deferred until the parent thesis orchestrator is complete and
there is an end-to-end benchmark to verify the port against.
