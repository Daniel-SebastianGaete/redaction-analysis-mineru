"""Classic-CV detector for page-scale black redaction rectangles.

An alternate redaction source that coexists with the trained detector
families: it finds contiguous near-black regions covering a large
fraction of the page and shaped like solid rectangles. Its detections
are merged with the trained detector's output in
``RedactionDetectionAdapter`` (large rectangles win overlaps), and from
there flow through MinerU's normal redaction plumbing unchanged.

Configuration via environment variables (see ``CvParagraphDetector.from_env``):

    MINERU_REDACTION_CV_ENABLED            "1"/"true" to enable (default: off)
    MINERU_REDACTION_CV_MIN_AREA_FRACTION  min contour area as a fraction of
                                           page area (default 0.4)
    MINERU_REDACTION_CV_EXTENT_MIN         min filled-area / bbox-area ratio
                                           (default 0.9)
    MINERU_REDACTION_CV_DARKNESS_MAX       max mean grayscale intensity inside
                                           the bbox, 0=black (default 30)
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Binarization cutoff used to pick candidate dark pixels. Deliberately looser
# than DARKNESS_MAX: it only shapes the candidate mask; the mean-intensity
# check below is what enforces "black".
BINARIZE_THRESHOLD = 100

_TRUTHY = {"1", "true", "yes", "on"}


class CvParagraphDetector:
    """Find large solid black rectangles on a page image."""

    def __init__(
        self,
        min_page_area_fraction: float = 0.4,
        extent_min: float = 0.9,
        darkness_max: float = 30.0,
    ):
        self.min_page_area_fraction = min_page_area_fraction
        self.extent_min = extent_min
        self.darkness_max = darkness_max

    @classmethod
    def from_env(cls) -> Optional["CvParagraphDetector"]:
        """Build a detector from MINERU_REDACTION_CV_* env vars.

        Returns None when MINERU_REDACTION_CV_ENABLED is unset/falsy.
        Raises ValueError on non-numeric threshold overrides.
        """
        if os.getenv("MINERU_REDACTION_CV_ENABLED", "").strip().lower() not in _TRUTHY:
            return None

        def _float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                raise ValueError(f"{name} must be a float, got {raw!r}.")

        return cls(
            min_page_area_fraction=_float_env("MINERU_REDACTION_CV_MIN_AREA_FRACTION", 0.4),
            extent_min=_float_env("MINERU_REDACTION_CV_EXTENT_MIN", 0.9),
            darkness_max=_float_env("MINERU_REDACTION_CV_DARKNESS_MAX", 30.0),
        )

    def detect(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return (xmin, ymin, xmax, ymax) for each qualifying rectangle.

        Accepts an RGB (H, W, 3) or grayscale (H, W) uint8 array in the
        same pixel space MinerU analyzes the page in, so the returned
        boxes need no coordinate transform.
        """
        if image_rgb.ndim == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb

        page_h, page_w = gray.shape[:2]
        page_area = float(page_h * page_w)
        if page_area <= 0:
            return []
        min_area = self.min_page_area_fraction * page_area

        _, mask = cv2.threshold(gray, BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            extent = contour_area / float(w * h)
            if extent < self.extent_min:
                continue
            mean_intensity = float(gray[y:y + h, x:x + w].mean())
            if mean_intensity > self.darkness_max:
                continue
            rects.append((x, y, x + w, y + h))
        return rects


def merge_cv_rects(
    detections: List[Dict],
    rects: List[Tuple[int, int, int, int]],
    category_id: int,
    covered_ratio: float = 0.5,
) -> List[Dict]:
    """Merge CV rectangles into a trained detector's layout_res detections.

    Priority goes to the large rectangle: any trained detection whose box
    is covered by a CV rect over more than ``covered_ratio`` of its own
    area is dropped, then each rect is appended as a full-confidence
    detection in the same layout_res shape the adapter emits.
    """
    if not rects:
        return detections

    def _bbox_of(det: Dict) -> Tuple[float, float, float, float]:
        xs = det["poly"][0::2]
        ys = det["poly"][1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def _covered(det_bbox, rect) -> float:
        dx0, dy0, dx1, dy1 = det_bbox
        rx0, ry0, rx1, ry1 = rect
        ix = max(0.0, min(dx1, rx1) - max(dx0, rx0))
        iy = max(0.0, min(dy1, ry1) - max(dy0, ry0))
        det_area = max(0.0, dx1 - dx0) * max(0.0, dy1 - dy0)
        if det_area <= 0:
            return 0.0
        return (ix * iy) / det_area

    kept = [
        det for det in detections
        if all(_covered(_bbox_of(det), rect) <= covered_ratio for rect in rects)
    ]
    for (x0, y0, x1, y1) in rects:
        kept.append(
            {
                "category_id": category_id,
                "poly": [x0, y0, x1, y0, x1, y1, x0, y1],
                "score": 1.0,
                "redaction_subtype": 0,
                "redaction_source": "cv_paragraph",
            }
        )
    return kept
