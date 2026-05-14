# Plan A: Integrated Redaction Model (Full Pipeline Integration)

This plan adds a new YOLO redaction detection model as a first-class citizen in the MinerU pipeline, parallel to how DocLayout-YOLO layout detection and YOLOv8 formula detection work. Redaction detections get their own `CategoryId`, `BlockType`, and markdown rendering.

---

## Overview

Redaction bboxes from your YOLO model are injected into `images_layout_res` alongside layout detections. They flow through MagicModel, block preparation, span filling, and ultimately render as `[REDACTED]` in the markdown output.

**Pros**: Clean, idiomatic integration. Redactions participate in overlap resolution, sorting, and paragraph splitting like any other block type. Works with all output modes (markdown, content_list).

**Cons**: Touches ~8 files. Requires understanding the full pipeline. Overlap resolution with text blocks needs careful tuning.

---

## Files to Modify

### 1. `mineru/backend/pipeline/model_list.py` (line 1-9)

Add a new atomic model name for the redaction detector.

```python
class AtomicModel:
    Layout = "layout"
    MFD = "mfd"
    MFR = "mfr"
    OCR = "ocr"
    WirelessTable = "wireless_table"
    WiredTable = "wired_table"
    TableCls = "table_cls"
    ImgOrientationCls = "img_ori_cls"
    RedactionDetection = "redaction_detection"  # NEW
```

### 2. `mineru/utils/enum_class.py`

**a) Add CategoryId (line 83):**

```python
class CategoryId:
    ...
    ImageFootnote = 101
    Redaction = 102          # NEW - redacted region
```

**b) Add BlockType (line 17):**

```python
class BlockType:
    ...
    DISCARDED = 'discarded'
    REDACTION = 'redaction'  # NEW
    ...
```

**c) Add ContentType (line 40):**

```python
class ContentType:
    ...
    CODE = 'code'
    REDACTION = 'redaction'  # NEW
```

### 3. New file: `mineru/model/redaction/yolo_redaction.py`

Create a model wrapper similar to `mineru/model/layout/doclayoutyolo.py` (which wraps `doclayout_yolo.YOLOv10`). Since your model uses `ultralytics.YOLO`, adapt accordingly:

```python
from typing import List, Dict, Union
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from PIL import Image

# Mapping from your YOLO class IDs to a single "Redaction" category
# Your model outputs: 0=black, 1=white, 2=gray redactions
# All map to CategoryId.Redaction (102) for the MinerU pipeline


class YOLORedactionModel:
    def __init__(self, weight: str, device: str = "cuda", imgsz: int = 1280, conf: float = 0.5):
        self.model = YOLO(weight)
        self.model.to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def _parse_prediction(self, prediction) -> List[Dict]:
        layout_res = []
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            return layout_res

        for xyxy, conf, cls in zip(
            prediction.boxes.xyxy.cpu(),
            prediction.boxes.conf.cpu(),
            prediction.boxes.cls.cpu(),
        ):
            coords = list(map(int, xyxy.tolist()))
            xmin, ymin, xmax, ymax = coords
            layout_res.append({
                "category_id": 102,  # CategoryId.Redaction - all redaction classes map here
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
                "redaction_subtype": int(cls.item()),  # 0=black, 1=white, 2=gray (optional metadata)
            })
        return layout_res

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        prediction = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False
        )[0]
        return self._parse_prediction(prediction)

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,
    ) -> List[List[Dict]]:
        results = []
        with tqdm(total=len(images), desc="Redaction Predict") as pbar:
            for idx in range(0, len(images), batch_size):
                batch = images[idx : idx + batch_size]
                predictions = self.model.predict(
                    batch, imgsz=self.imgsz, conf=self.conf, verbose=False
                )
                for pred in predictions:
                    results.append(self._parse_prediction(pred))
                pbar.update(len(batch))
        return results
```

### 4. `mineru/backend/pipeline/model_init.py`

**a) Add import (after line 11):**

```python
from ...model.redaction.yolo_redaction import YOLORedactionModel
```

**b) Add init function (after `doclayout_yolo_model_init` at line 96):**

```python
def redaction_model_init(weight, device='cpu'):
    model = YOLORedactionModel(weight, device)
    return model
```

**c) Register in `atom_model_init()` (line 154-198) — add a new elif before the else:**

```python
    elif model_name == AtomicModel.RedactionDetection:
        atom_model = redaction_model_init(
            kwargs.get('redaction_weights'),
            kwargs.get('device')
        )
```

**d) Initialize in `MineruPipelineModel.__init__()` (after layout_model at ~line 246):**

Add a configuration flag so the feature is opt-in. Read the weights path from an environment variable or config:

```python
        # Initialize redaction detection model (opt-in)
        redaction_weights = os.environ.get('MINERU_REDACTION_WEIGHTS', None)
        if redaction_weights and os.path.exists(redaction_weights):
            self.redaction_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.RedactionDetection,
                redaction_weights=redaction_weights,
                device=self.device,
            )
            self.apply_redaction = True
        else:
            self.redaction_model = None
            self.apply_redaction = False
```

### 5. `mineru/backend/pipeline/batch_analyze.py`

**After the DocLayout-YOLO prediction (line 54), add redaction detection:**

```python
        # Redaction detection (if enabled)
        if hasattr(self.model, 'apply_redaction') and self.model.apply_redaction:
            images_redaction_res = self.model.redaction_model.batch_predict(
                pil_images, YOLO_LAYOUT_BASE_BATCH_SIZE
            )
            # Merge redaction detections into layout results
            for page_idx in range(len(pil_images)):
                images_layout_res[page_idx] += images_redaction_res[page_idx]
```

### 6. `mineru/utils/model_utils.py` — `get_res_list_from_layout_res()` (line 345)

Redaction detections should NOT be routed to OCR, table, or formula processing. They must be skipped in the categorization loop. Add a filter at line 354:

```python
    for i, res in enumerate(layout_res):
        category_id = int(res['category_id'])

        if category_id == 102:  # Redaction — skip, handled separately
            continue
        elif category_id in [13, 14]:  # Formula regions
            ...
```

### 7. `mineru/backend/pipeline/pipeline_magic_model.py`

**a) Add to the IoU dedup filter (line 106-117) — include Redaction in the list so overlapping redactions get deduped:**

```python
        layout_dets = list(filter(
            lambda x: x['category_id'] in [
                    CategoryId.Title,
                    CategoryId.Text,
                    CategoryId.ImageBody,
                    CategoryId.ImageCaption,
                    CategoryId.TableBody,
                    CategoryId.TableCaption,
                    CategoryId.TableFootnote,
                    CategoryId.InterlineEquation_Layout,
                    CategoryId.InterlineEquationNumber_Layout,
                    CategoryId.Redaction,  # NEW
                ], self.__page_model_info['layout_dets']
            )
        )
```

**b) Add a getter method (after `get_title_blocks()` at line 306):**

```python
    def get_redaction_blocks(self) -> list:
        blocks = self.__get_blocks_by_type(CategoryId.Redaction)
        return blocks
```

**c) Add redactions to `get_all_spans()` (line 319 — add to `allow_category_id_list`):**

```python
        allow_category_id_list = [
            CategoryId.ImageBody,
            CategoryId.TableBody,
            CategoryId.InlineEquation,
            CategoryId.InterlineEquation_YOLO,
            CategoryId.OcrText,
            CategoryId.Redaction,  # NEW
        ]
```

And add the span type mapping in the loop (after the OcrText elif at line 350):

```python
                elif category_id == CategoryId.Redaction:
                    span['content'] = '[REDACTED]'
                    span['type'] = ContentType.REDACTION
```

### 8. `mineru/backend/pipeline/model_json_to_middle_json.py` — `page_model_info_to_page_info()`

**a) Extract redaction blocks (after line 43):**

```python
    redaction_blocks = magic_model.get_redaction_blocks()
```

**b) Include redaction blocks in `prepare_block_bboxes()` calls (lines 106 and 117).**

This requires modifying `prepare_block_bboxes` in `block_pre_proc.py` to accept an additional parameter, OR you can add redaction blocks to the existing `text_blocks` list with a different type. The cleanest approach:

Add them as a separate add_bboxes call inside `prepare_block_bboxes` (see step 9).

Alternatively, add them after the call:

```python
    # After prepare_block_bboxes returns, add redaction blocks
    from mineru.utils.block_pre_proc import add_bboxes
    add_bboxes(redaction_blocks, BlockType.REDACTION, all_bboxes)
```

### 9. `mineru/utils/block_pre_proc.py` — `prepare_block_bboxes()`

**Option A (cleaner):** Add a new parameter:

```python
def prepare_block_bboxes(
    img_body_blocks, img_caption_blocks, img_footnote_blocks,
    table_body_blocks, table_caption_blocks, table_footnote_blocks,
    discarded_blocks, text_blocks, title_blocks,
    interline_equation_blocks, page_w, page_h,
    redaction_blocks=None,  # NEW optional parameter
):
    ...
    if redaction_blocks:
        add_bboxes(redaction_blocks, BlockType.REDACTION, all_bboxes)
    ...
```

**Option B (less invasive):** In `page_model_info_to_page_info()`, call `add_bboxes(redaction_blocks, BlockType.REDACTION, all_bboxes)` right after `prepare_block_bboxes` returns.

### 10. `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py`

**a) In `make_blocks_to_markdown()` (after the TABLE elif at line 81), add:**

```python
        elif para_type == BlockType.REDACTION:
            para_text = '[REDACTED]'
```

**b) In `make_blocks_to_content_list()` (line 182+), add:**

```python
    elif para_type == BlockType.REDACTION:
        para_content = {
            'type': ContentType.REDACTION,
            'text': '[REDACTED]',
        }
```

**c) In `merge_para_with_text()` (line 106), handle REDACTION spans:**

Add to the span_type handling (around line 123-132):

```python
            elif span_type == ContentType.REDACTION:
                content = '[REDACTED]'
```

---

## Overlap Resolution Strategy

Redaction regions will overlap with text/table/image regions detected by DocLayout-YOLO. The critical question is: **what happens when a redaction bbox overlaps a text bbox?**

### Recommended approach:

In `MagicModel.__fix_by_remove_high_iou_and_low_confidence()`, redactions with high IoU against text blocks should **replace** them (the redaction wins). This is already partially handled since both participate in the IoU dedup — the one with higher confidence survives.

However, you may want a more aggressive policy: **any text block with >70% overlap with a redaction should be removed**. Add a new fix method in MagicModel:

```python
def __fix_redaction_overlaps(self):
    """Remove text/title blocks that overlap significantly with redaction blocks."""
    need_remove = []
    layout_dets = self.__page_model_info['layout_dets']
    redactions = [x for x in layout_dets if x['category_id'] == CategoryId.Redaction]
    text_types = [CategoryId.Text, CategoryId.Title, CategoryId.OcrText]

    for det in layout_dets:
        if det['category_id'] not in text_types:
            continue
        for redaction in redactions:
            overlap = calculate_overlap_area_in_bbox1_area_ratio(det['bbox'], redaction['bbox'])
            if overlap > 0.7 and det not in need_remove:
                need_remove.append(det)
                break

    for det in need_remove:
        layout_dets.remove(det)
```

Call this in `MagicModel.__init__()` after `__fix_by_remove_high_iou_and_low_confidence()`.

---

## Configuration

Enable via environment variable:

```bash
export MINERU_REDACTION_WEIGHTS=/path/to/your/training_annotator_32.pt
```

When the env var is not set, the pipeline behaves exactly as before — zero impact on existing functionality.

---

## Testing

1. Process a FOIA document with known redactions and verify `[REDACTED]` appears in output
2. Process a clean document (no redactions) and verify output is unchanged
3. Check that redaction blocks are properly sorted (reading order) relative to surrounding text
4. Verify the content_list output mode includes redaction entries with correct bbox coordinates

---

## Summary of touched files

| File | Change |
|------|--------|
| `mineru/backend/pipeline/model_list.py` | Add `RedactionDetection` atomic model name |
| `mineru/utils/enum_class.py` | Add `CategoryId.Redaction`, `BlockType.REDACTION`, `ContentType.REDACTION` |
| `mineru/model/redaction/yolo_redaction.py` | **New file** — YOLO model wrapper |
| `mineru/backend/pipeline/model_init.py` | Register model init, add to `MineruPipelineModel` |
| `mineru/backend/pipeline/batch_analyze.py` | Run redaction inference, merge into layout_res |
| `mineru/utils/model_utils.py` | Skip redaction category in `get_res_list_from_layout_res` |
| `mineru/backend/pipeline/pipeline_magic_model.py` | Add getter, span handling, IoU dedup, overlap resolution |
| `mineru/backend/pipeline/model_json_to_middle_json.py` | Extract redaction blocks, add to block preparation |
| `mineru/utils/block_pre_proc.py` | Add redaction blocks to `prepare_block_bboxes` |
| `mineru/backend/pipeline/pipeline_middle_json_mkcontent.py` | Render `[REDACTED]` in markdown and content_list |
