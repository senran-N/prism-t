"""Export independently checked artifacts from an actual persisted pipeline run.

No OCR/translation text, masks, or coordinates are supplied by this inspector.
"""
from __future__ import annotations

import argparse
import base64
import io
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw


def image(data):
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def export_node_ledger(directory: Path, raw: np.ndarray, composite: np.ndarray, node: dict) -> dict:
    directory.mkdir(parents=True,exist_ok=True)
    patches=node['patches']
    x,y,w,h=patches['roi']
    source=raw[y:y+h,x:x+w]
    arrays={'raw':source,'composite':composite[y:y+h,x:x+w]}
    for label,key in [('observed','source_observed_ink_png_b64'),('domain','source_domain_png_b64'),
                      ('refined','refined_stroke_png_b64'),('stroke_alpha','stroke_alpha_png_b64'),
                      ('clean_z0','clean_z0_png_b64')]:
        arrays[label]=np.array(Image.open(io.BytesIO(base64.b64decode(patches[key]))))
        if arrays[label].shape[:2] != (h,w):
            raise ValueError(f'{node["id"]}: {label} ledger dimensions differ from production ROI')
    if patches.get('inpaint_unknown_png_b64'):
        arrays['inpaint_unknown'] = np.array(Image.open(io.BytesIO(base64.b64decode(patches['inpaint_unknown_png_b64']))))
        if arrays['inpaint_unknown'].shape != (h,w):
            raise ValueError(f'{node["id"]}: reconstruction unknown dimensions differ from production ROI')
    for label in ('inferred_background','boundary_opacity'):
        if patches.get(label+'_png_b64'):
            arrays[label]=np.array(Image.open(io.BytesIO(base64.b64decode(patches[label+'_png_b64']))))
            if arrays[label].shape[:2]!=(h,w):
                raise ValueError(f'{node["id"]}: {label} dimensions differ from production ROI')
    digests={}
    for label,array in arrays.items():
        Image.fromarray(array).save(directory/f'{label}.png')
        digests[label]=hashlib.sha256(array.tobytes()).hexdigest()
    geometry=Image.fromarray(source.copy())
    draw=ImageDraw.Draw(geometry)
    for field,color in [('detector_quadrilaterals','cyan'),('source_quadrilaterals','red')]:
        for polygon in node[field]:
            points=[(a-x,b-y) for a,b in polygon]
            draw.line(points+[points[0]],fill=color,width=1)
    geometry.save(directory/'geometry.png')
    observed,domain,refined,alpha=[arrays[k]>0 for k in ('observed','domain','refined','stroke_alpha')]
    delta=np.max(np.abs(arrays['clean_z0'].astype(int)-source.astype(int)),axis=2)
    report=dict(node_id=node['id'],roi=patches['roi'],pixel_sha256=digests,
                detector_quadrilaterals=node['detector_quadrilaterals'],
                source_quadrilaterals=node['source_quadrilaterals'],
                source_text=node['source_text'],target_text=node['target_text'],
                observed_pixels=int(observed.sum()),domain_pixels=int(domain.sum()),
                refined_pixels=int(refined.sum()),final_support_pixels=int(alpha.sum()),
                observed_outside_domain=int((observed&~domain).sum()),
                observed_outside_final_support=int((observed&~alpha).sum()),
                refined_support_removed_later=int((refined&~alpha).sum()),
                support_added_later=int((alpha&~refined).sum()),
                clean_patch_changed_outside_support=int(((delta>0)&~alpha).sum()),
                glyph_truth_available=False,meaning_accuracy_checked=False)
    if 'inpaint_unknown' in arrays:
        unknown=arrays['inpaint_unknown']>0
        report.update(inpaint_unknown_pixels=int(unknown.sum()),
                      quarantine_outside_write_pixels=int((unknown&~alpha).sum()),
                      changed_quarantine_without_write_pixels=int(((delta>0)&unknown&~alpha).sum()))
    if 'boundary_opacity' in arrays:
        band=arrays['boundary_opacity']>0
        report.update(optically_unmixed_boundary_pixels=int(band.sum()),
                      boundary_changed_pixels=int((band&(delta>0)).sum()),
                      boundary_without_write_permission=int((band&~alpha).sum()),
                      boundary_opacity_levels=np.unique(arrays['boundary_opacity'][band]).tolist())
    (directory/'ledger.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def inspect(db_path: Path, output: Path, ledger: bool=False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT id,filename,dom_json,raw_png,composite_png,status FROM pages ORDER BY id").fetchall()
    summaries = []
    for page_id, filename, payload, raw_data, composite_data, status in rows:
        dom = json.loads(payload)
        raw, composite = image(raw_data), image(composite_data)
        support = np.zeros(raw.shape[:2], bool)
        clean = raw.copy()
        page_dir = output / f"page_{page_id:04d}"
        page_dir.mkdir(exist_ok=True)
        nodes = []
        node_ledgers=[]
        stored_nodes = dom.get("nodes", [])
        transcriptions = (dom.get("recognition_progress") or {}).get("results", {})
        source_parts = (dom.get("recognition_progress") or {}).get("line_parts", {})
        positive_sources = {key: value for key, value in transcriptions.items() if value["is_text"]}
        locked_ids = []
        unlocked_ids = []
        pending_source_ids = []
        if len(stored_nodes) != dom["metrics"]["total_nodes"]:
            raise ValueError(f"page {page_id}: node array and recorded count disagree")
        for node in stored_nodes:
            if ledger:
                node_ledgers.append(export_node_ledger(page_dir/'nodes'/node['id'],raw,composite,node))
            if node["id"] in positive_sources:
                source_record = positive_sources[node["id"]]
                order = source_record["line_order"]
                if sorted(order) != list(range(len(source_record["line_texts"]))):
                    raise ValueError(f"page {page_id}: invalid transcription reading order")
                expected = "\n".join(source_record["line_texts"][i] for i in order).strip()
                if status in {"translated", "reviewed"} and node["source_text"] != expected:
                    raise ValueError(f"page {page_id}, node {node['id']}: final source differs from saved independent transcription")
                if status in {"translated", "reviewed"}:
                    locked_ids.append(node["id"])
                else:
                    pending_source_ids.append(node["id"])
            else:
                unlocked_ids.append(node["id"])
            patches = node["patches"]
            x, y, w, h = patches["roi"]
            alpha = np.array(Image.open(io.BytesIO(base64.b64decode(patches["stroke_alpha_png_b64"]))).convert("L"))
            mask = alpha > 0
            support[y:y+h, x:x+w] |= mask
            z0 = image(base64.b64decode(patches["clean_z0_png_b64"]))
            roi = clean[y:y+h, x:x+w]
            roi[mask] = z0[mask]
            if patches["rendered_z1_png_b64"]:
                rgba = np.array(Image.open(io.BytesIO(base64.b64decode(patches["rendered_z1_png_b64"]))).convert("RGBA"))
                support[y:y+h, x:x+w] |= rgba[:, :, 3] > 0
            nodes.append({key:node.get(key) for key in ["id","bbox","source_lines","source_quadrilaterals","source_readings","ocr_text","source_text","target_text","target_segments","char_budget_max","layout","restoration_path","style","uncertainty"]})
        delta = np.abs(raw.astype(np.int16) - composite.astype(np.int16))
        protected = ~support
        report = {
            "page_id":page_id,"filename":filename,"status":status,
            "render_complete":dom["metrics"].get("render_ms",0)>0,"metrics":dom["metrics"],
            "independent_protected_pixels":int(protected.sum()),
            "independent_unmasked_linf":int(np.max(delta[protected], initial=0)),
            "independent_changed_protected_pixels":int(np.count_nonzero(np.any(delta > 0, axis=2) & protected)),
            "nodes":nodes,"rejected_regions":dom.get("rejected_regions",[]),
            "node_ledger":node_ledgers if ledger else None,
            "source_transcription": {
                "recorded_nodes": len(transcriptions),
                "partial_node_count": sum(node_id not in transcriptions for node_id in source_parts),
                "recorded_part_counts": {node_id:len(parts) for node_id,parts in source_parts.items()},
                "recorded_part_line_counts": {node_id:sum(len(part["line_texts"]) for part in parts.values()) for node_id,parts in source_parts.items()},
                "positive_nodes": len(positive_sources),
                "final_locked_ids": locked_ids,
                "final_unlocked_ids": unlocked_ids,
                "pending_source_ids": pending_source_ids,
                "rejected_after_positive_source": [r["id"] for r in dom.get("rejected_regions", []) if r["id"] in positive_sources],
                "meaning_accuracy_checked": False,
            },
        }
        audits = [n["uncertainty"] for n in nodes]
        report["source_ink_audit"] = {
            "assessed_nodes":sum(a.get("source_ink_residual_ratio") is not None for a in audits),
            "unassessed_nodes":sum(a.get("source_ink_residual_ratio") is None for a in audits),
            "automatic_repair_pixels":sum(a.get("automatic_repair_pixels",0) for a in audits),
            "ratio_stage":"before_automatic_repair",
        }
        (page_dir / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        Image.fromarray(raw).save(page_dir / "raw.png")
        Image.fromarray(clean).save(page_dir / "clean.png")
        Image.fromarray(composite).save(page_dir / "translated.png")
        Image.fromarray((support * 255).astype(np.uint8)).save(page_dir / "edit_support.png")
        Image.fromarray(np.clip(delta * 8,0,255).astype(np.uint8)).save(page_dir / "difference_x8.png")
        panels=[]
        final_label = "Translated" if report["render_complete"] else "Analysis preview (translation/render incomplete)"
        for label, arr in [("Original",raw),("Clean plate",clean),(final_label,composite)]:
            panel=Image.fromarray(arr);panel.thumbnail((640,900))
            canvas=Image.new("RGB",(640,930),"#f4f4f5")
            canvas.paste(panel,((640-panel.width)//2,30))
            ImageDraw.Draw(canvas).text((12,8),label,fill="black")
            panels.append(canvas)
        contact=Image.new("RGB",(1920,930),"white")
        for i,panel in enumerate(panels):contact.paste(panel,(640*i,0))
        contact.save(page_dir / "comparison.png")
        summaries.append({k:report[k] for k in report if k != "nodes"})
    (output / "summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--db",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument('--ledger',action='store_true',help='Require and export production geometry/observations/refinement/repair/composite evidence')
    args=parser.parse_args()
    inspect(args.db,args.out,args.ledger)
