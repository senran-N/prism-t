"""Compare every persisted page, using the exact canonical production pixels.

Differences and geometry changes are evidence for inspection, not glyph truth
or a quality score. Different input pixels or page sets are explicit errors.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image, ImageDraw


def decode(data: bytes, mode: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert(mode))


def pages(database: Path) -> dict:
    with sqlite3.connect(database) as connection:
        rows = connection.execute('SELECT filename,status,raw_png,dom_json FROM pages ORDER BY id').fetchall()
    result = {}
    for filename, status, raw, payload in rows:
        if filename in result:
            raise ValueError(f'ambiguous filename in comparison: {filename}')
        result[filename] = status, decode(raw, 'RGB'), json.loads(payload)
    return result


def clean_plate(raw: np.ndarray, dom: dict) -> tuple[np.ndarray, np.ndarray]:
    clean = raw.copy()
    support = np.zeros(raw.shape[:2], bool)
    for node in dom['nodes']:
        patches = node['patches']
        x, y, w, h = patches['roi']
        alpha = decode(base64.b64decode(patches['stroke_alpha_png_b64']), 'L') > 0
        patch = decode(base64.b64decode(patches['clean_z0_png_b64']), 'RGB')
        if alpha.shape != (h, w) or patch.shape != (h, w, 3):
            raise ValueError(f'{node["id"]}: invalid recorded patch dimensions')
        clean[y:y+h, x:x+w][alpha] = patch[alpha]
        support[y:y+h, x:x+w] |= alpha
    return clean, support


def compare(baseline: Path, candidate: Path, output: Path) -> None:
    previous, current = pages(baseline), pages(candidate)
    if not previous or previous.keys() != current.keys():
        raise ValueError(f'page sets differ: baseline={list(previous)}, candidate={list(current)}')
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for filename in previous:
        old_status, old_raw, old_dom = previous[filename]
        new_status, raw, dom = current[filename]
        if not np.array_equal(old_raw, raw):
            raise ValueError(f'{filename}: canonical production input pixels differ')
        directory = output / filename.replace('.', '_')
        directory.mkdir(exist_ok=True)
        old_clean, old_support = clean_plate(raw, old_dom)
        new_clean, support = clean_plate(raw, dom)
        delta = np.max(np.abs(new_clean.astype(int)-old_clean.astype(int)), axis=2)
        changes = np.broadcast_to((raw.mean(axis=2)*.5).astype(np.uint8)[..., None], raw.shape).copy()
        changes[support & ~old_support] = [0, 220, 100]
        changes[old_support & ~support] = [220, 50, 180]
        Image.fromarray(old_clean).save(directory / 'baseline_clean.png')
        Image.fromarray(new_clean).save(directory / 'candidate_clean.png')
        Image.fromarray(changes).save(directory / 'support_added_green_removed_purple.png')
        geometry = Image.fromarray(raw.copy())
        draw = ImageDraw.Draw(geometry)
        tracks = []
        for node in dom['nodes']:
            before = node['detector_quadrilaterals']
            after = node['source_quadrilaterals']
            if len(before) != len(after):
                raise ValueError(f'{filename}/{node["id"]}: geometry lineage count differs')
            for index, (native, completed) in enumerate(zip(before, after, strict=True)):
                a, b = np.asarray(native, np.float32), np.asarray(completed, np.float32)
                if np.array_equal(a, b):
                    continue
                tracks.append(dict(node_id=node['id'], track_index=index,
                                   before=native, after=completed,
                                   area_before=float(cv2.contourArea(a)),
                                   area_after=float(cv2.contourArea(b))))
                for polygon, color in ((native, 'cyan'), (completed, 'red')):
                    points = [tuple(point) for point in polygon]
                    draw.line(points+[points[0]], fill=color, width=1)
        geometry.save(directory / 'changed_tracks_before_cyan_after_red.png')
        old_nodes = {node['id']: node for node in old_dom['nodes']}
        style_changes = []
        unpaired_styles = []
        for node in dom['nodes']:
            old = old_nodes.get(node['id'])
            if old is None or old['source_quadrilaterals'] != node['source_quadrilaterals']:
                unpaired_styles.append(node['id'])
                continue
            before = dict(style=old['style'], fusion_mode=old['fusion_mode'])
            after = dict(style=node['style'], fusion_mode=node['fusion_mode'])
            if before != after:
                style_changes.append(dict(node_id=node['id'],before=before,after=after))
        report = dict(filename=filename, raw_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                      baseline_status=old_status, candidate_status=new_status,
                      baseline_nodes=len(old_dom['nodes']), candidate_nodes=len(dom['nodes']),
                      baseline_support_pixels=int(old_support.sum()), candidate_support_pixels=int(support.sum()),
                      support_added=int((support & ~old_support).sum()),
                      support_removed=int((old_support & ~support).sum()),
                      changed_clean_pixels=int((delta > 0).sum()),
                      changed_clean_linf=int(delta.max(initial=0)), changed_tracks=tracks,
                      style_changes=style_changes, unpaired_candidate_styles=unpaired_styles,
                      glyph_truth_available=False, meaning_accuracy_checked=False,
                      quality_pass=None)
        reports.append(report)
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(filename, f'{len(tracks)} changed tracks;', report['support_added'], 'added;',
              report['support_removed'], 'removed; quality unassessed', flush=True)
    (output / 'summary.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.out)
