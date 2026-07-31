# Copyright (c) Opendatalab. All rights reserved.
"""Doctr-based drop-in text recognizer (DHiSS fine-tunes).

Env-gated replacement for the Paddle text recognizer inside the pipeline
backend's OCR (``PytorchPaddleOCR.text_recognizer``). Activated by
``MINERU_OCR_RECO_FAMILY=doctr``; see ``DoctrTextRecognizer.from_env``.

The doctr recognizers (vitstr_base / parseq / sar_resnet31) are *word*
recognizers: their vocabs contain no space character and ``max_length``
caps the output. MinerU's OCR feeds *line*-level crops (det boxes pass
through ``merge_det_boxes``), so each crop is first split into word
sub-crops at horizontal ink gaps, recognized in batch, and re-joined
with single spaces. The line score is the word-length-weighted mean of
word confidences, mirroring Paddle's per-character mean so the existing
``OcrConfidence.min_confidence`` gate keeps working.

Only the recognizer is swapped: detection, layout, reading order and the
redaction adapter are untouched, so span structure and redaction-marker
alignment in middle.json are identical to a Paddle-recognizer run.
"""

import os
import time

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from mineru.utils.config_reader import get_device

_INSTANCE_CACHE = {}


class DoctrTextRecognizer:
    """Drop-in for ``predict_rec.TextRecognizer``.

    Same contract: ``__call__(img_list, tqdm_enable=..., tqdm_desc=...)``
    with BGR numpy crops in, ``([(text, score), ...], elapse)`` out,
    results aligned with the input order.
    """

    def __init__(
        self,
        checkpoint: str,
        arch: str = "vitstr_base",
        vocab: str = "spanish",
        max_length: int = 50,
        batch_size: int = 64,
        gap_frac: float = 0.2,
        ink_tol: int = 1,
        device: str = None,
    ):
        from doctr.datasets import VOCABS
        from doctr.models import recognition_predictor
        from doctr.models import parseq, sar_resnet31, vitstr_base

        arch_builders = {
            "vitstr_base": vitstr_base,
            "parseq": parseq,
            "sar_resnet31": sar_resnet31,
        }
        if arch not in arch_builders:
            raise ValueError(
                f"Unsupported doctr reco arch: {arch!r}. "
                f"Expected one of {sorted(arch_builders)}."
            )
        if vocab not in VOCABS:
            raise ValueError(f"Unknown doctr vocab: {vocab!r}.")

        kwargs = {"pretrained": False, "pretrained_backbone": False, "vocab": VOCABS[vocab]}
        if max_length:
            kwargs["max_length"] = max_length
        model = arch_builders[arch](**kwargs)

        state_dict = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # strict=False mirrors the upstream DHiSS loader, but a partial
            # load silently degrades to random weights — make it loud.
            logger.warning(
                f"doctr reco checkpoint partial load: "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys."
            )

        self.device = device or get_device()
        self.batch_size = batch_size
        self.gap_frac = gap_frac
        self.ink_tol = ink_tol
        self.predictor = recognition_predictor(
            arch=model, pretrained=False, batch_size=batch_size
        ).eval()
        self.predictor.to(self.device)
        logger.info(
            f"doctr recognizer active: arch={arch} vocab={vocab} "
            f"max_length={max_length} checkpoint={checkpoint} device={self.device}"
        )

    @classmethod
    def from_env(cls):
        """Build (or reuse) an instance from MINERU_OCR_RECO_* env vars.

        Returns None unless ``MINERU_OCR_RECO_FAMILY`` is set. Instances are
        cached module-wide so per-language OCR atom models share one loaded
        network.
        """
        family = os.getenv("MINERU_OCR_RECO_FAMILY")
        if not family:
            return None
        if family != "doctr":
            raise ValueError(
                f"Unsupported MINERU_OCR_RECO_FAMILY: {family!r} (expected 'doctr')."
            )
        checkpoint = os.getenv("MINERU_OCR_RECO_CHECKPOINT")
        if not checkpoint:
            raise ValueError(
                "MINERU_OCR_RECO_FAMILY is set but MINERU_OCR_RECO_CHECKPOINT is not."
            )
        arch = os.getenv("MINERU_OCR_RECO_ARCH", "vitstr_base")
        vocab = os.getenv("MINERU_OCR_RECO_VOCAB", "spanish")
        max_length = int(os.getenv("MINERU_OCR_RECO_MAX_LENGTH", "50") or 0)
        batch_size = int(os.getenv("MINERU_OCR_RECO_BATCH", "64"))
        gap_frac = float(os.getenv("MINERU_OCR_RECO_GAP_FRAC", "0.2"))
        ink_tol = int(os.getenv("MINERU_OCR_RECO_INK_TOL", "1"))

        key = (checkpoint, arch, vocab, max_length, batch_size, gap_frac, ink_tol)
        if key not in _INSTANCE_CACHE:
            _INSTANCE_CACHE[key] = cls(
                checkpoint=checkpoint,
                arch=arch,
                vocab=vocab,
                max_length=max_length,
                batch_size=batch_size,
                gap_frac=gap_frac,
                ink_tol=ink_tol,
            )
        return _INSTANCE_CACHE[key]

    def _split_words(self, crop):
        """Split a BGR line crop into word sub-crops at horizontal ink gaps.

        Returns ``(clean_crop, segments)`` where segments is a list of
        (x0, x1) column ranges, left to right (empty when the crop has no
        ink), and clean_crop is the crop with underlines / horizontal rules
        whitened out — an underline bridges every word gap and corrupts the
        glyphs handed to the recognizer.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Underline / rule rows: near-full ink coverage (text rows measure
        # ≤ ~0.6 even in dense lines). Grow the mask one row each way to
        # catch the anti-aliased halo.
        row_cover = (binary > 0).mean(axis=1)
        rule_rows = row_cover > 0.8
        clean_crop = crop
        if rule_rows.any():
            grown = rule_rows.copy()
            grown[1:] |= rule_rows[:-1]
            grown[:-1] |= rule_rows[1:]
            grown &= row_cover > 0.3
            binary[grown] = 0
            clean_crop = crop.copy()
            clean_crop[grown] = 255

        # A column counts as ink only above ink_tol pixels, so isolated
        # speckle noise on historical scans can't bridge a word gap.
        ink_cols = (binary > 0).sum(axis=0) > self.ink_tol

        # Measured on 200-DPI typewritten FOIA pages (line height ~50px):
        # inter-character gaps run 1-9px, inter-word gaps 12-31px, so the
        # default gap_frac=0.2 puts the cut in the valley between modes.
        h = crop.shape[0]
        min_gap = max(3, int(round(self.gap_frac * h)))

        segments = []
        start = None
        gap = 0
        for x, has_ink in enumerate(ink_cols):
            if has_ink:
                if start is None:
                    start = x
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= min_gap:
                    segments.append((start, x - gap + 1))
                    start = None
        if start is not None:
            segments.append((start, len(ink_cols)))

        # Pad each segment by 1px and drop sub-2px noise slivers.
        w = crop.shape[1]
        return clean_crop, [
            (max(0, x0 - 1), min(w, x1 + 1))
            for x0, x1 in segments
            if x1 - x0 >= 2
        ]

    def __call__(self, img_list, tqdm_enable=False, tqdm_desc="OCR-rec Predict"):
        start_time = time.time()
        rec_res = [("", 0.0)] * len(img_list)

        # Flatten every line crop into word crops, remembering which line
        # each word belongs to, then recognize all words in one batched pass.
        word_crops = []
        owners = []
        for i, crop in enumerate(img_list):
            if crop is None or crop.size == 0:
                continue
            clean_crop, segments = self._split_words(crop)
            for x0, x1 in segments:
                word_crops.append(cv2.cvtColor(clean_crop[:, x0:x1], cv2.COLOR_BGR2RGB))
                owners.append(i)

        words_by_line = {}
        if word_crops:
            preds = []
            with torch.no_grad():
                with tqdm(
                    total=len(word_crops), desc=tqdm_desc, disable=not tqdm_enable
                ) as pbar:
                    for beg in range(0, len(word_crops), self.batch_size):
                        chunk = word_crops[beg:beg + self.batch_size]
                        preds.extend(self.predictor(chunk))
                        pbar.update(len(chunk))
            for owner, (text, conf) in zip(owners, preds):
                words_by_line.setdefault(owner, []).append((text, conf))

        for i, words in words_by_line.items():
            words = [(t, c) for t, c in words if t]
            if not words:
                continue
            text = " ".join(t for t, _ in words)
            # Word-length-weighted mean ≈ Paddle's per-character mean; a
            # plain min() would let one noisy word sink a healthy line
            # below OcrConfidence.min_confidence.
            total_len = sum(len(t) for t, _ in words)
            score = sum(len(t) * float(c) for t, c in words) / total_len
            rec_res[i] = (text, score)

        return rec_res, time.time() - start_time
