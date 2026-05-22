"""
Swappable redaction-detection adapter.

Subprocess-invokes the chosen detection family from the
redaction-detection sibling repo and adapts its COCO-format predictions
into MinerU's layout_res / span shape (CategoryId.Redaction = 102, poly,
score, redaction_subtype).

Configuration via environment variables:

    MINERU_REDACTION_FAMILY         required: yolo | mmdet | detectron | rtdetr | rfdetr
    MINERU_REDACTION_CHECKPOINT     required: path to checkpoint weights
    MINERU_REDACTION_DETECTION_ROOT required: path to the redaction-detection repo root
    MINERU_REDACTION_MODEL_CONFIG   required for mmdet, detectron; optional for rtdetr
    MINERU_REDACTION_MODEL_TYPE     required for detectron: maskrcnn | mask2former
    MINERU_REDACTION_RTDETR_ROOT    required for rtdetr: path to the RT-DETRv2 vendor repo
    MINERU_REDACTION_CONF           optional: confidence threshold (default 0.5)

Implementation: for each call to batch_predict, the adapter writes images
to a temp directory, synthesizes a minimal COCO ground-truth JSON
(predict scripts use it only to map filenames to image_id), invokes
`<detection-root>/scripts/run.sh models/<family>/predict.py`, reads the
resulting coco_results.json, and converts each bbox into MinerU's poly
quadrilateral.

Cost: each batch_predict call launches a subprocess that loads the
chosen model once. For a multi-page document this amortizes well; for
single-image calls the model-load overhead dominates inference. Pass
all pages of a document together when possible.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from mineru.utils.enum_class import CategoryId

SUPPORTED_FAMILIES = {"yolo", "mmdet", "detectron", "rtdetr", "rfdetr"}


class RedactionDetectionAdapter:
    """Run the configured redaction-detection family via subprocess.

    Exposes the same `predict(image)` and `batch_predict(images)` shape as
    the previous in-process YOLO wrapper so existing MinerU call sites
    don't need to change beyond construction.
    """

    def __init__(
        self,
        family: str,
        checkpoint: str,
        detection_root: str,
        model_config: Optional[str] = None,
        model_type: Optional[str] = None,
        rtdetr_root: Optional[str] = None,
        conf: float = 0.5,
    ):
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported family '{family}'. Expected one of {sorted(SUPPORTED_FAMILIES)}."
            )
        if family in {"mmdet", "detectron"} and not model_config:
            raise ValueError(f"Family '{family}' requires MINERU_REDACTION_MODEL_CONFIG.")
        if family == "detectron" and model_type not in {"maskrcnn", "mask2former"}:
            raise ValueError(
                "Family 'detectron' requires MINERU_REDACTION_MODEL_TYPE in "
                "{maskrcnn, mask2former}."
            )
        if family == "rtdetr" and not rtdetr_root:
            raise ValueError("Family 'rtdetr' requires MINERU_REDACTION_RTDETR_ROOT.")

        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        detection_root_path = Path(detection_root).expanduser().resolve()
        run_sh = detection_root_path / "scripts" / "run.sh"
        if not run_sh.is_file():
            raise FileNotFoundError(
                f"redaction-detection run.sh not found at {run_sh}. "
                "Set MINERU_REDACTION_DETECTION_ROOT to the repo root."
            )

        self.family = family
        self.checkpoint = str(checkpoint_path.resolve())
        self.detection_root = detection_root_path
        self.run_sh = str(run_sh)
        self.model_config = str(Path(model_config).expanduser().resolve()) if model_config else None
        self.model_type = model_type
        self.rtdetr_root = str(Path(rtdetr_root).expanduser().resolve()) if rtdetr_root else None
        self.conf = conf

    @classmethod
    def from_env(cls) -> Optional["RedactionDetectionAdapter"]:
        """Build an adapter from MINERU_REDACTION_* env vars.

        Returns None if MINERU_REDACTION_FAMILY is unset (redaction
        detection disabled). Raises ValueError on partial / invalid
        configuration.
        """
        family = os.getenv("MINERU_REDACTION_FAMILY")
        if not family:
            return None

        checkpoint = os.getenv("MINERU_REDACTION_CHECKPOINT")
        detection_root = os.getenv("MINERU_REDACTION_DETECTION_ROOT")
        if not checkpoint or not detection_root:
            raise ValueError(
                "MINERU_REDACTION_FAMILY is set, but MINERU_REDACTION_CHECKPOINT or "
                "MINERU_REDACTION_DETECTION_ROOT is missing."
            )

        try:
            conf = float(os.getenv("MINERU_REDACTION_CONF", "0.5"))
        except ValueError:
            raise ValueError(
                f"MINERU_REDACTION_CONF must be a float, got {os.environ['MINERU_REDACTION_CONF']!r}."
            )

        return cls(
            family=family,
            checkpoint=checkpoint,
            detection_root=detection_root,
            model_config=os.getenv("MINERU_REDACTION_MODEL_CONFIG"),
            model_type=os.getenv("MINERU_REDACTION_MODEL_TYPE"),
            rtdetr_root=os.getenv("MINERU_REDACTION_RTDETR_ROOT"),
            conf=conf,
        )

    def predict(self, image: Union[np.ndarray, Image.Image]) -> List[Dict]:
        return self.batch_predict([image])[0]

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,  # accepted for API parity; one subprocess processes all
    ) -> List[List[Dict]]:
        if not images:
            return []

        with tempfile.TemporaryDirectory(prefix="mineru_redaction_") as workdir_str:
            workdir = Path(workdir_str)
            image_dir = workdir / "images"
            image_dir.mkdir()
            output_dir = workdir / "output"
            output_dir.mkdir()

            gt_images = []
            for idx, img in enumerate(images, start=1):
                pil = self._to_pil(img)
                fname = f"page_{idx:05d}.png"
                pil.save(image_dir / fname)
                gt_images.append(
                    {
                        "id": idx,
                        "file_name": fname,
                        "width": pil.width,
                        "height": pil.height,
                    }
                )

            gt_path = workdir / "gt.json"
            with open(gt_path, "w") as f:
                json.dump(
                    {"images": gt_images, "categories": [], "annotations": []}, f
                )

            cmd = self._build_cmd(
                gt_path=str(gt_path),
                image_dir=str(image_dir),
                output_dir=str(output_dir),
            )

            # Strip VIRTUAL_ENV so `uv run` inside redaction-detection's run.sh
            # discovers the correct per-family .venv instead of inheriting
            # MinerU's own venv path.
            sub_env = os.environ.copy()
            sub_env.pop("VIRTUAL_ENV", None)

            try:
                completed = subprocess.run(
                    cmd, env=sub_env, check=False, capture_output=True, text=True
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Failed to invoke redaction-detection subprocess: {exc}"
                ) from exc

            if completed.returncode != 0:
                raise RuntimeError(
                    "redaction-detection predict failed "
                    f"(exit {completed.returncode}, family={self.family}):\n"
                    f"stdout tail: {completed.stdout[-800:]}\n"
                    f"stderr tail: {completed.stderr[-800:]}"
                )

            preds_path = output_dir / "coco_results.json"
            if not preds_path.is_file():
                return [[] for _ in images]

            with open(preds_path) as f:
                preds = json.load(f)

        per_image: List[List[Dict]] = [[] for _ in images]
        for det in preds:
            image_id = det.get("image_id")
            if image_id is None:
                continue
            idx = int(image_id) - 1
            if not 0 <= idx < len(images):
                continue

            x, y, w, h = det["bbox"]
            xmin, ymin = int(round(x)), int(round(y))
            xmax, ymax = int(round(x + w)), int(round(y + h))

            per_image[idx].append(
                {
                    "category_id": CategoryId.Redaction,
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(det.get("score", 0.0)), 3),
                    "redaction_subtype": int(det.get("category_id", 0)),
                }
            )

        return per_image

    def _build_cmd(self, gt_path: str, image_dir: str, output_dir: str) -> List[str]:
        cmd = [
            "bash",
            self.run_sh,
            f"models/{self.family}/predict.py",
            "--checkpoint",
            self.checkpoint,
            "--gt-json",
            gt_path,
            "--image-dir",
            image_dir,
            "--output-dir",
            output_dir,
            "--conf",
            str(self.conf),
        ]
        if self.family == "mmdet":
            cmd.extend(["--config", self.model_config])
        elif self.family == "detectron":
            cmd.extend(["--model-type", self.model_type, "--config-file", self.model_config])
        elif self.family == "rtdetr":
            cmd.extend(["--rtdetr-root", self.rtdetr_root])
            if self.model_config:
                cmd.extend(["--config", self.model_config])
        return cmd

    @staticmethod
    def _to_pil(img: Union[np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img if img.mode == "RGB" else img.convert("RGB")
        if isinstance(img, np.ndarray):
            return Image.fromarray(img).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(img).__name__}")
