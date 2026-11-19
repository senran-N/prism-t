"""Run the complete visual pipeline on every canonical recorded source page.

The output SQLite file is an analysis evidence bundle, not a resumable work
database. No recognition result, target text, geometry or mask is injected.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))

from prism_t_engine import pipeline
from prism_t_engine.schemas import AnalyzeRequest


def analyze(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as connection:
        inputs = connection.execute('SELECT id,filename,raw_png FROM pages ORDER BY id').fetchall()
    implementation = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path(pipeline.__file__).parent.glob('*.py')}
    report = dict(stage='full canonical-input visual analysis; no semantic translation',
                  source_database=str(source.resolve()), implementations=implementation,
                  expected_pages=[name for _, name, _ in inputs], pages=[],
                  glyph_truth_available=False, meaning_accuracy_checked=False)
    # Intentionally create a fresh bundle. A partial run cannot inherit old
    # page results and present them as newly analyzed evidence.
    with sqlite3.connect(output / 'analysis.db') as destination:
        destination.execute('CREATE TABLE pages (id INTEGER PRIMARY KEY, filename TEXT, raw_png BLOB, dom_json TEXT, composite_png BLOB, status TEXT)')
        for page_id, filename, raw in inputs:
            request = AnalyzeRequest(image_png_b64=base64.b64encode(raw).decode(), recognize_text=False)
            response = pipeline.analyze_and_restore_page(request)
            clean = base64.b64decode(response.clean_preview_png_b64)
            destination.execute('INSERT INTO pages VALUES (?,?,?,?,?,?)',
                                (page_id, filename, raw, response.dom.model_dump_json(), clean, 'analyzed'))
            destination.commit()
            row = dict(filename=filename, raw_png_sha256=hashlib.sha256(raw).hexdigest(),
                       nodes=len(response.dom.nodes), analyze_ms=response.dom.metrics.analyze_ms)
            report['pages'].append(row)
            (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(filename, len(response.dom.nodes), 'actual analyzed nodes', flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=Path('data/fixtures/canonical_pages.db'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.db, args.out)
