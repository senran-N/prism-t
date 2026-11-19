"""Check restoration information flow on every persisted page and text ROI.

Neighbor support is taken from recorded masks, not manually annotated glyph
truth. Black/white interventions check dependence on excluded context; they do
not measure the quality of reconstructed artwork or translation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from prism_t_engine.codec import decode_image_b64
from prism_t_engine.restoration import restore_clean_plate_roi


def audit(database: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        pages = connection.execute('SELECT filename,raw_png,dom_json FROM pages ORDER BY id').fetchall()
    import prism_t_engine.restoration as implementation
    digest = hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    reports = []
    for filename, raw_png, payload in pages:
        raw = decode_image_b64(base64.b64encode(raw_png).decode(), mode='RGB')
        dom = json.loads(payload)
        owners = np.zeros(raw.shape[:2], np.int32)
        masks = []
        for node in dom['nodes']:
            x, y, w, h = node['patches']['roi']
            alpha = decode_image_b64(node['patches']['stroke_alpha_png_b64'], mode='L') > 0
            owners[y:y+h, x:x+w] += alpha
            masks.append(alpha)
        rows = []
        for node, active in zip(dom['nodes'], masks, strict=True):
            x, y, w, h = node['patches']['roi']
            foreign = owners[y:y+h, x:x+w] > active
            row = dict(node_id=node['id'], excluded_neighbor_pixels=int(foreign.sum()),
                       active_pixels=int(active.sum()), assessed=False)
            rows.append(row)
            if not foreign.any() or not active.any():
                continue
            source = raw[y:y+h, x:x+w]
            black, white = source.copy(), source.copy()
            # Overlapping target/neighbor pixels are hidden in both requests.
            black[foreign], white[foreign] = 0, 255
            first, path_a, _ = restore_clean_plate_roi(black, active.astype(np.float32), context_exclusion=foreign)
            second, path_b, _ = restore_clean_plate_roi(white, active.astype(np.float32), context_exclusion=foreign)
            row.update(assessed=True, black_path=path_a, white_path=path_b,
                       target_linf=int(np.max(np.abs(first[active].astype(int)-second[active].astype(int)), initial=0)),
                       black_protected_linf=int(np.max(np.abs(first[~active].astype(int)-black[~active].astype(int)), initial=0)),
                       white_protected_linf=int(np.max(np.abs(second[~active].astype(int)-white[~active].astype(int)), initial=0)))
            (output / 'in_progress.json').write_text(json.dumps(dict(filename=filename, rows=rows), indent=2))
            if path_a != path_b or any(row[key] for key in ('target_linf', 'black_protected_linf', 'white_protected_linf')):
                raise AssertionError(f'{filename}: excluded context affects {node["id"]}; see in_progress.json')
        report = dict(filename=filename, raw_png_sha256=hashlib.sha256(raw_png).hexdigest(),
                      restoration_sha256=digest, stage='actual stored geometry; excluded-context interventions',
                      glyph_truth_available=False, nodes=rows)
        reports.append(report)
        (output / 'summary.json').write_text(json.dumps(reports, indent=2))
        print(filename, sum(r['assessed'] for r in rows), 'assessed;',
              sum(not r['assessed'] for r in rows), 'without an active/neighbor pair', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.db, args.out)
