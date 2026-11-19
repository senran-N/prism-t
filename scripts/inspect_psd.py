"""Read exported PSDs independently of the Go encoder and compare composites.

Run with: uv run --with psd-tools --with numpy --with pillow python scripts/inspect_psd.py FILE.psd
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage


def inspect(path: Path):
    document = PSDImage.open(path)
    reference = np.array(Image.open(path.with_suffix(".png")).convert("RGB"))
    stored = np.array(document.topil().convert("RGB"))
    recomposed = np.array(document.composite(force=True).convert("RGB"))
    names = [layer.name for layer in document]
    report = {
        "path": str(path), "size": document.size, "layers_bottom_to_top": names,
        "stored_composite_linf": int(np.max(np.abs(stored.astype(int) - reference.astype(int)))),
        "recomposed_linf": int(np.max(np.abs(recomposed.astype(int) - reference.astype(int)))),
        "recomposed_changed_pixels": int(np.count_nonzero(np.any(recomposed != reference, axis=2))),
    }
    path.with_suffix(".inspection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    assert report["stored_composite_linf"] == 0
    assert names[-1] == "99_Foreground_Occluders_Z2"
    assert names[:2] == ["00_Original_Raw", "01_Restored_Clean_Z0"]
    # Photoshop's straight-alpha rounding may differ by a single quantization
    # unit, but larger deviations reveal an encoder/layer-compositing defect.
    assert report["recomposed_linf"] <= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    for path in parser.parse_args().files:
        inspect(path)
