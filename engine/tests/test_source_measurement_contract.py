import cv2
import numpy as np

from prism_t_engine import pipeline
from prism_t_engine.codec import encode_png_b64
from prism_t_engine.schemas import AnalyzeRequest


def test_cleanup_permission_changes_do_not_remeasure_the_source(monkeypatch):
    raw = np.full((120, 200, 3), 245, np.uint8)
    ink = np.zeros((70, 130), np.uint8)
    cv2.putText(ink, 'HI', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.7, 1, 4)
    raw[25:95, 30:160][ink > 0] = 20
    region = dict(bbox=[30, 25, 130, 70],
                  polygon=[[30, 25], [160, 25], [160, 95], [30, 95]],
                  orientation='horizontal', source_text='HI', ruby_pairs=[],
                  spine_points=[], ocr_entropy=0., ink_mask_roi=ink)
    recorded = []
    for partial in (False, True):
        def extract(rgb, coarse_mask):
            alpha = coarse_mask.astype(np.float32).copy()
            if partial:
                alpha[:, :alpha.shape[1]//2] = 0
            return alpha, 2., 0.
        monkeypatch.setattr(pipeline, 'build_stroke_trimap_and_alpha', extract)
        response = pipeline.analyze_and_restore_page(
            AnalyzeRequest(image_png_b64=encode_png_b64(raw)), [region],
        )
        node = response.dom.nodes[0]
        recorded.append((node.style.model_dump(), node.patches.exemplar_patch_png_b64))
    assert recorded[0] == recorded[1]
    assert recorded[0][0]['fg_color_rgba'][:3] == [20, 20, 20]
