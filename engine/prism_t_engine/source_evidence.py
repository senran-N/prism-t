"""Independent recognizer observations, never a substitute for source pixels."""
from __future__ import annotations

import cv2
import numpy as np

from .ocr_tps import _get_rapidocr, _rapid_inference_lock
from .source_ocr import get_source_ocr


def rectify_source_line(rgb: np.ndarray, polygon: np.ndarray, vertical: bool) -> np.ndarray:
    """Keep upright glyphs while removing the observed line's rotation."""
    points = cv2.boxPoints(cv2.minAreaRect(np.asarray(polygon, np.float32)))
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    axes = edges / np.maximum(lengths[:, None], np.finfo(float).eps)
    reference = np.array([0., 1.]) if vertical else np.array([1., 0.])
    u = axes[np.argmax(np.abs(axes @ reference))]
    if u @ reference < 0:
        u = -u
    v = np.array([u[1], -u[0]]) if vertical else np.array([-u[1], u[0]])
    along, cross = points @ u, points @ v
    basis = np.column_stack((v, u)) if vertical else np.column_stack((u, v))
    origin = u * along.min() + v * cross.min()
    width, height = (np.ptp(cross), np.ptp(along)) if vertical else (np.ptp(along), np.ptp(cross))
    transform = np.column_stack((basis, origin)).astype(np.float32)
    return cv2.warpAffine(
        rgb, cv2.invertAffineTransform(transform),
        (max(1, int(np.ceil(width))), max(1, int(np.ceil(height)))),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def collect_source_readings(
    rgb: np.ndarray,
    polygons: list,
    vertical: bool,
    japanese: bool,
    format_profile: str | None = None,
) -> list[dict]:
    """Read owned lines without re-detection, confidence filtering or voting.

    Scores belong to their recognizer, not a shared correctness scale. Keeping
    conflicting and empty readings lets the vision model inspect the glyphs
    instead of inheriting an OCR winner or silently losing a difficult region.
    """
    return collect_page_source_readings(
        rgb,
        [(polygons, vertical, japanese)],
        format_profile=format_profile,
    )[0]


def collect_page_source_readings(
    rgb: np.ndarray,
    node_specs: list[tuple[list, bool, bool]],
    format_profile: str | None = None,
) -> list[list[dict]]:
    """Batch-collect independent recognizer observations across all nodes on a page.

    All horizontal CTC crops on the page are batched into a single `ocr.text_rec`
    call instead of one locked call per node.
    """
    per_node_readings: list[list[dict]] = [[] for _ in node_specs]
    per_node_crops: list[list[np.ndarray]] = []
    ctc_jobs: list[tuple[int, int, np.ndarray]] = []
    mocr_jobs: list[tuple[int, int, np.ndarray]] = []

    for node_idx, (polygons, vertical, japanese) in enumerate(node_specs):
        crops = [rectify_source_line(rgb, polygon, vertical) for polygon in polygons]
        per_node_crops.append(crops)
        if not crops:
            continue
        # Only invoke Japanese NAR OCR when Japanese script evidence is present,
        # or on explicit manga_rtl / unspecified direct unit-test calls; never on
        # non-Japanese comic_ltr or webtoon_vertical pages solely due to vertical aspect.
        use_mocr = japanese or (vertical and format_profile in (None, "manga_rtl"))
        if use_mocr:
            for line_idx, crop in enumerate(crops):
                mocr_jobs.append((node_idx, line_idx, crop))
        # CTC's input contract is a horizontal row of upright glyphs. Rotating a
        # vertical column would rotate each glyph and is not valid evidence.
        if not vertical:
            for line_idx, crop in enumerate(crops):
                ctc_jobs.append((node_idx, line_idx, crop))

    if mocr_jobs:
        mocr = get_source_ocr()
        if mocr is not None:
            for node_idx, line_idx, crop in mocr_jobs:
                text, truncated = mocr.recognize(crop)
                per_node_readings[node_idx].append(
                    dict(
                        line_index=line_idx,
                        recognizer="manga_ocr_nar_small",
                        text=text,
                        confidence=None,
                        truncated=truncated,
                    )
                )

    if ctc_jobs:
        ocr = _get_rapidocr()
        if ocr is not None:
            bgr_crops = [cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) for _, _, crop in ctc_jobs]
            with _rapid_inference_lock:
                results, _ = ocr.text_rec(bgr_crops)
            for (node_idx, line_idx, _), (text, probability) in zip(ctc_jobs, results):
                per_node_readings[node_idx].append(
                    dict(
                        line_index=line_idx,
                        recognizer="rapidocr_ctc",
                        text=str(text),
                        confidence=float(probability),
                    )
                )

    return per_node_readings
