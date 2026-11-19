"""Export the exact line evidence retained by a real pipeline run.

No additional OCR, target answers, hand-drawn boxes or acceptance labels.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from prism_t_engine.source_evidence import rectify_source_line


def inspect(database: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        pages = connection.execute("SELECT id,filename,dom_json,raw_png,status FROM pages ORDER BY id").fetchall()
    summary = []
    for page_id, filename, payload, raw_data, status in pages:
        dom = json.loads(payload)
        raw = np.array(Image.open(io.BytesIO(raw_data)).convert("RGB"))
        directory = output / f"page_{page_id:04d}"
        directory.mkdir(exist_ok=True)
        nodes = []
        for node in dom["nodes"] + dom.get("rejected_regions", []):
            readings = node.get("source_readings", [])
            lines = []
            polygons = node.get("source_quadrilaterals", [])
            for index, polygon in enumerate(polygons):
                points = np.array(polygon)
                vertical = node["style"]["orientation"] == "vertical" if "style" in node else np.var(points[:,1]) > np.var(points[:,0])
                crop = rectify_source_line(raw, polygon, vertical)
                name = f"{node['id']}_line_{index:03d}.png"
                Image.fromarray(crop).save(directory / name)
                lines.append({"line_index": index, "image": name, "polygon": polygon,
                              "readings": [r for r in readings if r["line_index"] == index]})
            nodes.append({"id": node["id"], "source_text": node.get("source_text", ""),
                          "target_text": node.get("target_text", ""), "reason": node.get("reason", ""), "lines": lines})
        report = {"page_id": page_id, "filename": filename, "status": status, "nodes": nodes,
                  "source_transcription": dom.get("recognition_progress")}
        (directory / "source_evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({"page_id": page_id, "filename": filename, "status": status,
                        "lines": sum(len(n["lines"]) for n in nodes),
                        "observations": sum(len(line["readings"]) for n in nodes for line in n["lines"])})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    inspect(args.db, args.out)
