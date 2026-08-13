"""Adopt line-scale redactions into the text block that owns their line.

The layout detector bounds text regions *around* solid black redaction
bars, so a redaction sitting at either end of a line has almost no area
overlap with the block that owns that line. Both gates deciding a
redaction's fate ask whether it is *inside* a text block —
``model_json_to_middle_json``'s standalone-block test and
``fill_spans_in_blocks``' attachment ratio — so such a redaction is
rejected by both, becomes its own ``BlockType.REDACTION`` block, and
reaches the orchestrator as a paragraph-scale redaction. Paragraph-kind
redactions skip the classifier and take a fixed label, so a wrapped
inline redaction silently loses its entity type.

Containment is the wrong question; adjacency is the answer the geometry
supports. This module asks whether a redaction *belongs to a line*, and
when it does, widens the host block's bbox to include it. The existing
containment tests then admit it with no further changes, and it renders
inline inside its sentence.

Two shapes are adopted:

- a redaction sharing a line's vertical band, horizontally adjacent to
  the block that owns it (a redaction at the start or end of a line);
- a redaction lying in the *gap between* two lines, within a line-pitch
  of one of them — the continuation fragment of a redaction that wraps
  across a line break often overlaps no line's band at all.

Rejected: anything taller than a line (a redaction genuinely covering a
block of text, which is what ``BlockType.REDACTION`` is for) and
anything outside the page's text column (stray detections in the
margins).

All thresholds are multiples of the page's own median text-line height,
so nothing is tuned to a particular document.

Configuration via environment variables:

    MINERU_REDACTION_ADOPT             "0"/"false"/"no"/"off" disables
                                       adoption (default: enabled)
    MINERU_REDACTION_ADOPT_MAX_HEIGHT  max redaction height, in line
                                       pitches, to count as line-scale
                                       (default 1.5)
    MINERU_REDACTION_ADOPT_VGAP        max vertical gap to the host
                                       block, in line pitches (default 1.0)
    MINERU_REDACTION_ADOPT_HGAP        max horizontal gap to the host
                                       block, in line pitches (default 1.0)
"""

import os
from typing import Dict, List, Optional, Sequence

_FALSEY = {"0", "false", "no", "off"}

# A widened host must not engulf a neighbouring block: if more than this
# fraction of another block's area would fall inside the widened bbox, the
# adoption is skipped rather than risk corrupting the page's layout.
NEIGHBOUR_ENGULF_MAX = 0.5


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a float, got {raw!r}.")


def is_enabled() -> bool:
    return os.getenv("MINERU_REDACTION_ADOPT", "").strip().lower() not in _FALSEY


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Gap between two 1-D intervals; 0 when they overlap."""
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _union(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _engulf_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner``'s area that falls inside ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if area <= 0:
        return 0.0
    return (ix * iy) / area


def adopt_redactions_into_lines(
    redaction_blocks: List[Dict],
    host_blocks: List[Dict],
    text_span_bboxes: Sequence[Sequence[float]],
) -> int:
    """Widen host blocks so line-scale redactions become inline spans.

    Mutates the ``bbox`` of the host blocks in place — call this *before*
    the block bboxes are collected into ``all_bboxes``, since that step
    copies them.

    Args:
      redaction_blocks: the page's redaction blocks (dicts with 'bbox').
      host_blocks: surviving text/title blocks (dicts with 'bbox').
      text_span_bboxes: bboxes of the page's OCR'd text spans, used to
        measure the line pitch and the text column.

    Returns:
      How many redactions were adopted.
    """
    if not is_enabled() or not redaction_blocks or not host_blocks:
        return 0
    if not text_span_bboxes:
        return 0

    pitch = _median([b[3] - b[1] for b in text_span_bboxes])
    if pitch <= 0:
        return 0
    col_x0 = min(b[0] for b in text_span_bboxes)
    col_x1 = max(b[2] for b in text_span_bboxes)

    max_height = _float_env("MINERU_REDACTION_ADOPT_MAX_HEIGHT", 1.5) * pitch
    v_tol = _float_env("MINERU_REDACTION_ADOPT_VGAP", 1.0) * pitch
    h_tol = _float_env("MINERU_REDACTION_ADOPT_HGAP", 1.0) * pitch

    adopted = 0
    for redaction in redaction_blocks:
        rb = redaction.get("bbox")
        if not rb:
            continue
        if (rb[3] - rb[1]) > max_height:
            continue  # covers more than a line — a genuine block redaction
        if rb[2] <= col_x0 or rb[0] >= col_x1:
            continue  # outside the text column (margin artefact)

        best: Optional[tuple] = None
        for block in host_blocks:
            hb = block.get("bbox")
            if not hb:
                continue
            v_gap = _gap(rb[1], rb[3], hb[1], hb[3])
            h_gap = _gap(rb[0], rb[2], hb[0], hb[2])
            if v_gap > v_tol or h_gap > h_tol:
                continue
            key = (v_gap, h_gap)
            if best is None or key < best[0]:
                best = (key, block)
        if best is None:
            continue

        host = best[1]
        widened = _union(host["bbox"], rb)
        if any(
            other is not host
            and other.get("bbox")
            and _engulf_ratio(other["bbox"], widened) > NEIGHBOUR_ENGULF_MAX
            for other in host_blocks
        ):
            continue  # would swallow a neighbouring block; leave it standalone

        host["bbox"] = widened
        adopted += 1

    return adopted
