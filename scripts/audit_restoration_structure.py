"""Measure actual restored boundaries without rerunning the text detector.

These probes expose residual edge pairs, blur and directional discontinuities.
They do not provide hidden artwork truth or certify a translation/repair.
"""
from __future__ import annotations

import argparse
import base64
import io
import hashlib
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def decode(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)))


def gradients(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = rgb.astype(float)
    return (cv2.Scharr(image, cv2.CV_64F, 1, 0),
            cv2.Scharr(image, cv2.CV_64F, 0, 1))


def tensor(image: np.ndarray, domain: np.ndarray) -> dict | None:
    if not domain.any():
        return None
    gx, gy = gradients(image)
    jxx, jxy, jyy = (float(np.mean(a[domain])) for a in
                     (np.mean(gx*gx, axis=2), np.mean(gx*gy, axis=2), np.mean(gy*gy, axis=2)))
    total = jxx+jyy
    return dict(samples=int(domain.sum()), gradient_energy=total,
                coherence=float(np.hypot(jxx-jyy, 2*jxy)/total) if total else None,
                direction_radians=float(.5*np.arctan2(2*jxy, jxx-jyy)) if total else None)


def boundary_probe(raw: np.ndarray, clean: np.ndarray, active: np.ndarray,
                   unknown: np.ndarray) -> dict:
    if not active.any():
        return dict(assessed=False, reason='no recorded erasure support')
    inside = distance_transform_edt(active)
    outside = distance_transform_edt(~active)
    width = float(np.quantile(inside[active], .75))
    # Native derivative stencils require one observed lattice neighbor. This is
    # operator support, not a pixel-sized aesthetic or segmentation threshold.
    known = ~unknown
    stencil = np.ones((3, 3), np.uint8)
    interior = np.ones(active.shape, bool)
    interior[[0,-1],:] = False
    interior[:,[0,-1]] = False
    inner = (cv2.erode(active.astype(np.uint8), stencil) > 0) & interior
    outer = known & (outside <= width+float(outside[known].min())) if known.any() else known
    outer &= (cv2.erode(known.astype(np.uint8), stencil) > 0) & interior
    ring = ((active & (inside <= width)) | (~active & (outside <= width))) & interior
    signed=outside-inside
    nx,ny=cv2.Scharr(signed,cv2.CV_64F,1,0),cv2.Scharr(signed,cv2.CV_64F,0,1)
    if not ring.any():
        return dict(assessed=False,reason='no interior derivative stencil at the recorded boundary')
    length = np.hypot(nx, ny)
    nx = np.divide(nx, length, out=np.zeros_like(nx), where=length > 0)
    ny = np.divide(ny, length, out=np.zeros_like(ny), where=length > 0)
    observations = {}
    for label, image in (('raw', raw), ('clean', clean)):
        gx, gy = gradients(image)
        normal = np.mean(gx, axis=2)*nx+np.mean(gy, axis=2)*ny
        positive = float(np.mean(np.maximum(normal[ring], 0)**2))
        negative = float(np.mean(np.minimum(normal[ring], 0)**2))
        laplace = cv2.Laplacian(image.astype(float), cv2.CV_64F)
        high_inner = float(np.mean(laplace[inner]**2)) if inner.any() else None
        high_outer = float(np.mean(laplace[outer]**2)) if outer.any() else None
        observations[label] = dict(normal_positive_energy=positive,
            normal_negative_energy=negative, normal_paired_energy=min(positive, negative),
            interior_laplacian_energy=high_inner, exterior_laplacian_energy=high_outer,
            interior_to_exterior_high_frequency_ratio=high_inner/high_outer
                if high_inner is not None and high_outer else None,
            interior_tensor=tensor(image, inner), exterior_tensor=tensor(image, outer))
    return dict(assessed=True, stroke_radius_q75=width, ring_samples=int(ring.sum()),
                observations=observations, hidden_background_truth_available=False,
                quality_pass=None)


def audit(database: Path, output: Path, baseline: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        pages = connection.execute('SELECT filename,raw_png,dom_json FROM pages ORDER BY id').fetchall()
    reference={}
    if baseline is not None:
        with sqlite3.connect(baseline) as connection:
            reference={name:(pixels,json.loads(payload)) for name,pixels,payload in
                       connection.execute('SELECT filename,raw_png,dom_json FROM pages')}
        if set(reference)!={row[0] for row in pages}:
            raise ValueError('baseline and candidate must contain the same complete page set')
    reports = []
    for filename, raw_data, payload in pages:
        raw = np.array(Image.open(io.BytesIO(raw_data)).convert('RGB'))
        dom = json.loads(payload)
        source_nodes={}
        if baseline is not None:
            pixels,source_dom=reference[filename]
            if not np.array_equal(raw,np.array(Image.open(io.BytesIO(pixels)).convert('RGB'))):
                raise ValueError(f'{filename}: baseline source pixels differ')
            source_nodes={node['id']:node for node in source_dom['nodes']}
            if set(source_nodes)!={node['id'] for node in dom['nodes']}:
                raise ValueError(f'{filename}: node identities differ; do not force correspondence')
        rows = []
        for node in dom['nodes']:
            patches = node['patches']
            x,y,w,h = patches['roi']
            active = decode(base64.b64decode(patches['stroke_alpha_png_b64'])) > 0
            clean = decode(base64.b64decode(patches['clean_z0_png_b64']))
            # Older evidence has no quarantine; identify it explicitly instead
            # of presenting a guessed mask as a recorded reconstruction domain.
            unknown_data = patches.get('inpaint_unknown_png_b64')
            unknown = decode(base64.b64decode(unknown_data)) > 0 if unknown_data else active
            row=dict(node_id=node['id'], roi=patches['roi'],
                unknown_domain_recorded=bool(unknown_data),
                restoration_path=node['restoration_path'],
                **boundary_probe(raw[y:y+h,x:x+w], clean, active, unknown))
            if baseline is not None:
                original=source_nodes[node['id']]
                old=original['patches']
                if old['roi']!=patches['roi'] or original['source_quadrilaterals']!=node['source_quadrilaterals']:
                    raise ValueError(f'{filename}/{node["id"]}: source ROI or owned geometry changed')
                old_active=decode(base64.b64decode(old['stroke_alpha_png_b64']))>0
                old_unknown_data=old.get('inpaint_unknown_png_b64')
                old_unknown=decode(base64.b64decode(old_unknown_data))>0 if old_unknown_data else old_active
                shared_unknown=unknown|old_unknown|active|old_active
                native=raw[y:y+h,x:x+w]
                row['fixed_baseline_support_comparison']=dict(
                    baseline_database=str(baseline.resolve()),
                    baseline=boundary_probe(native,decode(base64.b64decode(old['clean_z0_png_b64'])),old_active,shared_unknown),
                    candidate=boundary_probe(native,clean,old_active,shared_unknown),
                    quality_pass=None)
            rows.append(row)
        reports.append(dict(filename=filename, nodes=rows, quality_pass=None,
                            source_database=str(database.resolve()),
                            raw_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            meaning_accuracy_checked=False))
        print(filename, len(rows), 'actual structure probes', flush=True)
    (output/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    audit(args.db,args.out,args.baseline)
