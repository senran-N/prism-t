"""Aspect-preserving Japanese line recognition with pinned NAR weights.

Model: muscgab/manga-ocr-nar-small, revision 81ff4f3313263d1da349946a9dd20d14e0fb0e66.
The graph returns IDs, not probabilities; do not manufacture confidence values.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Sequence

import numpy as np
from PIL import Image

from .models import ROOT, ASSET_DIGESTS, create_ort_session, resolve_ort_providers, run_ort_on_pinned_thread


_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def source_ocr_directory() -> Path:
    return Path(os.environ.get("PRISM_T_SOURCE_OCR_DIR", str(ROOT / "manga_ocr_nar_small")))


def prepare_source_patches(rgb: np.ndarray, budget: int, patch: int) -> dict[str, np.ndarray]:
    height, width = rgb.shape[:2]
    # The area bound brackets the largest aspect-preserving fit. Stop at
    # floating-point convergence, rather than selecting a DPI or iteration cap.
    low, high = 0., math.sqrt(budget * patch * patch / (height * width))
    while True:
        scale = (low + high) / 2
        if scale == low or scale == high:
            break
        if math.ceil(height * scale / patch) * math.ceil(width * scale / patch) <= budget:
            low = scale
        else:
            high = scale
    rows, columns = math.ceil(height * low / patch), math.ceil(width * low / patch)
    rw, rh = max(1, min(columns * patch, round(width * low))), max(1, min(rows * patch, round(height * low)))
    resized = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BICUBIC))
    border = np.concatenate((resized[0], resized[-1], resized[:, 0], resized[:, -1]))
    canvas = np.empty((rows * patch, columns * patch, 3), np.uint8)
    canvas[:] = np.median(border, axis=0).astype(np.uint8)
    canvas[:rh, :rw] = resized
    # CLIP channel normalization is part of this checkpoint's published input
    # contract, along with bicubic sampling and border-median padding.
    normalized = (canvas.astype(np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
    count = rows * columns
    features = np.zeros((budget, 3 * patch * patch), np.float32)
    features[:count] = normalized.reshape(rows, patch, columns, patch, 3).transpose(0, 2, 4, 1, 3).reshape(count, -1)
    index = np.arange(count)
    valid, row, column = np.zeros(budget, np.float32), np.zeros(budget, np.int32), np.zeros(budget, np.int32)
    valid[:count], row[:count], column[:count] = 1, index // columns, index % columns
    return dict(P=features, V=valid, R=row, C=column, H=np.array([rows], np.int32), W=np.array([columns], np.int32))


class JapaneseSourceOCR:
    def __init__(self, directory: Path, providers: Sequence[str] | None = None):
        for name, relative in (("source_ocr", "onnx/mangaocr_nar_small_fp32.onnx"), ("source_vocabulary", "nar_vocabulary_6817.json")):
            with (directory / relative).open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != ASSET_DIGESTS[name]:
                raise ValueError(f"{name}: installed bytes differ from the pinned source recognizer")
        self.session = create_ort_session(directory / "onnx/mangaocr_nar_small_fp32.onnx", providers)
        self._lock = threading.Lock()
        self.characters = json.loads((directory / "nar_vocabulary_6817.json").read_text(encoding="utf-8"))["characters"]
        self.budget, features = next(item.shape for item in self.session.get_inputs() if item.name == "P")
        self.patch = math.isqrt(features // 3)
        if features != 3 * self.patch * self.patch:
            raise ValueError("source OCR graph has a non-RGB patch contract")

    def recognize(self, crop: np.ndarray) -> tuple[str, bool]:
        feed = prepare_source_patches(crop, self.budget, self.patch)
        with self._lock:
            ids = run_ort_on_pinned_thread(lambda: self.session.run(None, feed))[0].reshape(-1).tolist()
        eos = len(self.characters)
        complete = eos in ids
        if complete:
            ids = ids[:ids.index(eos)]
        return "".join(self.characters[index] for index in ids), not complete


_source_ocr_lock = threading.Lock()


@lru_cache(maxsize=2)
def _source_ocr_for_providers(directory: Path, providers: tuple[str, ...]) -> JapaneseSourceOCR:
    return JapaneseSourceOCR(directory, providers)


def get_source_ocr() -> JapaneseSourceOCR:
    providers = tuple(resolve_ort_providers())
    with _source_ocr_lock:
        return _source_ocr_for_providers(source_ocr_directory(), providers)
