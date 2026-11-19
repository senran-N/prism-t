"""Recompute source typography for every recorded node, retaining cached Z0.

This isolates style/render changes; it is explicitly not a fresh visual analysis
or semantic translation run. No text, ink mask, or expected color is supplied.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from prism_t_engine.codec import decode_image_b64, encode_png_b64
from prism_t_engine.deep_fusion import extract_stroke_style_and_halo
from prism_t_engine.pipeline import typeset_and_render_page
from prism_t_engine.schemas import PageDOM, RenderRequest


def replay(source: Path, output: Path, render: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as connection:
        pages = connection.execute('SELECT id,filename,raw_png,dom_json,composite_png,status FROM pages ORDER BY id').fetchall()
    report = dict(stage='source typography replay; cached Z0 and recorded text',
                  source_database=str(source.resolve()), render_requested=render,
                  implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in Path(__file__).resolve().parents[1].joinpath('engine/prism_t_engine').glob('*.py')},
                  expected_pages=[p[1] for p in pages], pages=[], meaning_accuracy_checked=False,
                  font_size_recomputed=False)
    with sqlite3.connect(output / 'analysis.db') as destination:
        destination.execute('CREATE TABLE pages (id INTEGER PRIMARY KEY,filename TEXT,raw_png BLOB,dom_json TEXT,composite_png BLOB,status TEXT)')
        for page_id, filename, raw, payload, composite, status in pages:
            raw_b64 = base64.b64encode(raw).decode()
            rgb = decode_image_b64(raw_b64)
            dom = PageDOM.model_validate_json(payload)
            owners = np.zeros(rgb.shape[:2], bool)
            for node in dom.nodes:
                p = node.patches
                x,y,w,h = p.roi
                owners[y:y+h, x:x+w] |= decode_image_b64(p.source_observed_ink_png_b64, 'L') > 0
            rows = []
            for node in dom.nodes:
                p = node.patches
                x,y,w,h = p.roi
                observed = decode_image_b64(p.source_observed_ink_png_b64, 'L') > 0
                domain = decode_image_b64(p.source_domain_png_b64, 'L') > 0
                z2 = decode_image_b64(p.z2_occluder_mask_png_b64, 'L') > 0 if p.z2_occluder_mask_png_b64 else np.zeros((h,w),bool)
                substrate = ~owners[y:y+h,x:x+w] & ~domain & ~z2
                if node.bubble_polygon and node.bubble_kind not in ('floating_sfx','diegetic_surface'):
                    # Reproduce the production bubble substrate, including its
                    # current border exclusion; this is not a new mask proposal.
                    bubble = np.zeros((h,w), np.uint8)
                    polygon = (np.asarray(node.bubble_polygon)-[x,y]).astype(np.int32)
                    cv2.fillPoly(bubble, [polygon], 1)
                    inner = cv2.erode(bubble, np.ones((5,5),np.uint8))
                    if not inner.any():
                        inner = cv2.erode(bubble, np.ones((3,3),np.uint8))
                    substrate &= inner > 0
                old = node.style.model_dump()
                radii = distance_transform_edt(observed)
                radii = radii[radii > 0]
                stroke_w = float(2*np.median(radii)) if radii.size else 0.0
                stroke_mod = float(np.median(np.abs(2*radii-stroke_w))/stroke_w) if radii.size else 0.0
                style, fusion, exemplar = extract_stroke_style_and_halo(
                    rgb[y:y+h,x:x+w], observed.astype(np.float32), stroke_w,
                    stroke_mod, node.style.orientation, node.bubble_kind,
                    pigment_support=observed,
                    substrate_support=substrate,
                )
                style.source_font_size_px = node.style.source_font_size_px
                # Preserve independently established optical/occlusion contracts.
                if node.fusion_mode == 'alpha_blended':
                    style.fg_color_rgba = node.style.fg_color_rgba
                    style.blend_alpha = node.style.blend_alpha
                if node.fusion_mode not in ('alpha_blended', 'occluded_z2', 'halftone_backed', 'speedline_backed'):
                    node.fusion_mode = fusion
                node.style = style
                p.exemplar_patch_png_b64 = encode_png_b64(exemplar)
                rows.append(dict(node_id=node.id, before=old, after=style.model_dump()))
            rendered = render and status == 'translated'
            if rendered:
                response = typeset_and_render_page(RenderRequest(raw_image_png_b64=raw_b64, dom=dom))
                dom = response.dom
                composite = base64.b64decode(response.composite_png_b64)
                (output / (Path(filename).stem+'.png')).write_bytes(composite)
            destination.execute('INSERT INTO pages VALUES (?,?,?,?,?,?)',
                                (page_id, filename, raw, dom.model_dump_json(), composite, status))
            destination.commit()
            report['pages'].append(dict(filename=filename, rendered=rendered, nodes=rows,
                                        raw_png_sha256=hashlib.sha256(raw).hexdigest(),
                                        source_dom_sha256=hashlib.sha256(payload.encode()).hexdigest()))
            (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(filename, len(rows), 'styles recomputed;', 'rendered' if rendered else 'not rendered', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    replay(args.db, args.out, args.render)
