# Plan B: Post-Processing Overlay (Minimal Invasion)

This plan runs the YOLO redaction model as a separate pass after the MinerU pipeline has produced its intermediate blocks, then replaces overlapping text spans with `[REDACTED]`. Only 2-3 files are modified and no new category/block types are introduced.

---

## Overview

Instead of integrating the redaction model into the detection pipeline, this approach:

1. Runs your YOLO model on each page image independently
2. After MagicModel has extracted all blocks and spans, checks each text span against redaction bboxes
3. Replaces the content of overlapping text spans with `[REDACTED]`

The redaction detection runs inside `page_model_info_to_page_info()` in `model_json_to_middle_json.py`, after spans are created but before they're filled into blocks.

**Pros**: Minimal code changes (~2-3 files). No new category IDs or block types. Existing pipeline logic is untouched. Easy to enable/disable.

**Cons**: Redactions are not first-class pipeline objects — they don't appear in `content_list` output as their own type. Only works by overwriting existing text spans, so if the layout model missed a text region under a redaction, nothing will appear. Doesn't handle the case where a redaction covers an entire area that has no detected text block (the redaction won't appear at all in the output).

---

## Files to Modify

### 1. New file: `mineru/model/redaction/yolo_redaction.py`

Same model wrapper as Plan A (reuse that code). A simpler version since we only need `predict()`:

```python
from typing import List, Dict, Union
from ultralytics import YOLO
import numpy as np
from PIL import Image


class YOLORedactionModel:
    def __init__(self, weight: str, device: str = "cuda", imgsz: int = 1280, conf: float = 0.5):
        self.model = YOLO(weight)
        self.model.to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        """Returns list of redaction bboxes as [x_min, y_min, x_max, y_max]."""
        prediction = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False
        )[0]

        results = []
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return results

        for xyxy, conf_score, cls in zip(
            prediction.boxes.xyxy.cpu(),
            prediction.boxes.conf.cpu(),
            prediction.boxes.cls.cpu(),
        ):
            coords = list(map(int, xyxy.tolist()))
            results.append({
                "bbox": coords,  # [x_min, y_min, x_max, y_max]
                "score": round(float(conf_score.item()), 3),
                "class": int(cls.item()),  # 0=black, 1=white, 2=gray
            })
        return results
```

### 2. `mineru/backend/pipeline/model_json_to_middle_json.py`

This is the main integration point. Modify `page_model_info_to_page_info()` to optionally run redaction detection and overlay results on the text spans.

**a) Add imports at top of file:**

```python
import os
from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
```

(The boxbase import may already be there; the `os` import is for reading the env var.)

**b) Add a helper function to compute redaction overlays (before `page_model_info_to_page_info`):**

```python
# Singleton to avoid reloading the model on every page
_redaction_model = None

def _get_redaction_model():
    global _redaction_model
    if _redaction_model is not None:
        return _redaction_model

    weights_path = os.environ.get('MINERU_REDACTION_WEIGHTS', None)
    if not weights_path or not os.path.exists(weights_path):
        return None

    from mineru.model.redaction.yolo_redaction import YOLORedactionModel
    from mineru.utils.config_reader import get_device
    device = get_device()
    _redaction_model = YOLORedactionModel(weights_path, device=device)
    return _redaction_model


def apply_redaction_overlay(spans, page_pil_img, scale, overlap_threshold=0.5):
    """
    Run redaction detection on the page image and replace overlapping
    text spans with [REDACTED].

    Args:
        spans: list of span dicts with 'bbox', 'type', 'content' keys
        page_pil_img: PIL Image of the page
        scale: coordinate scale factor (YOLO coords are in image space,
               spans are in page space)
        overlap_threshold: minimum overlap ratio to consider a span redacted

    Returns:
        modified spans list (in-place), list of redaction bboxes (page coords)
    """
    model = _get_redaction_model()
    if model is None:
        return spans, []

    # Run detection on the full page image
    redaction_dets = model.predict(page_pil_img)
    if not redaction_dets:
        return spans, []

    # Convert redaction bboxes from image coords to page coords
    redaction_bboxes = []
    for det in redaction_dets:
        x_min, y_min, x_max, y_max = det['bbox']
        page_bbox = [
            int(x_min / scale),
            int(y_min / scale),
            int(x_max / scale),
            int(y_max / scale),
        ]
        redaction_bboxes.append(page_bbox)

    # Check each text span against redaction bboxes
    for span in spans:
        if span.get('type') != 'text':
            continue
        span_bbox = span['bbox']
        for red_bbox in redaction_bboxes:
            overlap = calculate_overlap_area_in_bbox1_area_ratio(span_bbox, red_bbox)
            if overlap > overlap_threshold:
                span['content'] = '[REDACTED]'
                break  # One redaction match is enough

    return spans, redaction_bboxes
```

**c) Call the overlay in `page_model_info_to_page_info()` (around line 55, after `spans = magic_model.get_all_spans()`):**

```python
    """获取所有的spans信息"""
    spans = magic_model.get_all_spans()

    # Redaction overlay: replace text spans overlapping with detected redactions
    spans, redaction_bboxes = apply_redaction_overlay(spans, page_pil_img, scale)
```

That's it for the basic overlay. Text spans whose bbox overlaps with a redaction bbox by >50% will have their content replaced with `[REDACTED]`.

### 3. (Optional) Handle redactions that don't overlap any text span

The overlay approach has a gap: if the layout model doesn't detect a text block under a redaction (which is likely — redacted text is blacked out and OCR won't find text there), the redaction won't appear in the output at all.

To fix this, inject synthetic `[REDACTED]` spans for redaction bboxes that didn't match any text span.

**In `apply_redaction_overlay()`, after the overlap check loop, add:**

```python
    # Track which redaction bboxes matched at least one span
    matched_redactions = set()
    for span in spans:
        if span.get('content') == '[REDACTED]':
            span_bbox = span['bbox']
            for i, red_bbox in enumerate(redaction_bboxes):
                overlap = calculate_overlap_area_in_bbox1_area_ratio(span_bbox, red_bbox)
                if overlap > overlap_threshold:
                    matched_redactions.add(i)

    # Inject synthetic spans for unmatched redactions
    for i, red_bbox in enumerate(redaction_bboxes):
        if i not in matched_redactions:
            spans.append({
                'bbox': red_bbox,
                'type': 'text',
                'content': '[REDACTED]',
                'score': 1.0,
            })
```

This ensures every detected redaction region produces a `[REDACTED]` marker in the output, even if no text block was detected underneath.

---

## Configuration

Same as Plan A — enable via environment variable:

```bash
export MINERU_REDACTION_WEIGHTS=/path/to/your/training_annotator_32.pt
```

When not set, the `_get_redaction_model()` function returns `None` and the overlay is skipped entirely — zero impact on existing behavior.

---

## Coordinate System Notes

The MinerU pipeline uses two coordinate systems:

1. **Image coordinates**: Used by YOLO models. These are pixel coordinates in the rendered page image (the PIL image passed to the model).
2. **Page coordinates**: Used internally by MagicModel and blocks/spans. These are the image coordinates divided by `scale`.

The `scale` factor is computed during page rendering (typically ~2.0-3.0x for higher resolution). Your YOLO model operates in image space, so you must divide by `scale` to convert to page coordinates before comparing with span bboxes.

This conversion is already handled in `apply_redaction_overlay()` above:
```python
page_bbox = [int(x / scale) for x in image_bbox]
```

---

## Testing

1. Process a FOIA document with known redactions — verify `[REDACTED]` replaces text content
2. Process a clean document — verify output is identical to baseline (no env var set)
3. Check edge cases:
   - Redaction covering an entire text block → should become `[REDACTED]`
   - Redaction partially overlapping a text block → depends on threshold (50% default)
   - Redaction in an area with no detected text → synthetic span injection (if step 3 is implemented)
4. Compare output with and without redaction model on the same document

---

## Limitations

1. **No dedicated block type**: Redactions are not visible as a separate entity in `content_list` mode — they appear as text blocks with `[REDACTED]` content. If you need structured redaction metadata (bbox, confidence, subtype), Plan A is better.

2. **Depends on layout detection quality**: If DocLayout-YOLO doesn't detect a region under the redaction, the overlay approach needs the synthetic span injection (step 3) to avoid silent gaps.

3. **No overlap resolution with layout model**: Redaction doesn't participate in MagicModel's overlap deduplication, so if a redaction overlaps a table or image, those won't be affected. Only text spans are replaced.

4. **Single model load timing**: The model is loaded lazily on first page. For large PDFs this is fine, but the first page will be slightly slower.

---

## Summary of touched files

| File | Change |
|------|--------|
| `mineru/model/redaction/yolo_redaction.py` | **New file** — YOLO model wrapper (simplified) |
| `mineru/backend/pipeline/model_json_to_middle_json.py` | Add `apply_redaction_overlay()` helper and call after span extraction |

---

## When to choose Plan B over Plan A

- You want a quick proof-of-concept before committing to full integration
- You only need markdown output (not content_list with structured redaction metadata)
- You want to minimize the risk of breaking existing pipeline behavior
- You plan to iterate on the redaction model and don't want to touch 8+ files each time

When you're confident the model works well and you need structured output, migrate to Plan A.
