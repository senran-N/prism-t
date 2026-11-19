from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from prism_t_engine.app import app
from prism_t_engine.codec import decode_image_b64, encode_png_b64
from prism_t_engine.ocr_tps import resolve_ort_providers


def test_fastapi_analyze_and_render_roundtrip(monkeypatch):
    monkeypatch.setenv("PRISM_DEVICE", "cpu")
    assert resolve_ort_providers() == ["CPUExecutionProvider"]

    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200
    health_data = r_health.json()
    assert health_data["status"] == "ok"
    assert health_data["device"] == "cpu"
    assert health_data["hardware"]["configured_device"] == "cpu"
    assert health_data["hardware"]["active_device"] == "cpu"
    assert health_data["ocr"]["resolved_providers"] == ["CPUExecutionProvider"]

    # Build a synthetic comic page with an enclosed speech bubble & crisp recognizable text
    page = np.full((380, 340, 3), 245, dtype=np.uint8)
    cv2.rectangle(page, (20, 20), (320, 360), (40, 40, 40), 3)
    cv2.ellipse(page, (170, 175), (105, 110), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (170, 175), (105, 110), 0, 0, 360, (15, 15, 15), 2)
    cv2.putText(page, "HELLO", (105, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA)

    b64 = encode_png_b64(page)
    r_ana = client.post("/v1/analyze", json={"image_png_b64": b64, "format_profile": "comic_ltr"})
    assert r_ana.status_code == 200
    ana_data = r_ana.json()
    assert ana_data["dom"]["width"] == 340
    assert ana_data["dom"]["height"] == 380
    assert len(ana_data["dom"]["nodes"]) >= 1
    assert ana_data["dom"]["metrics"]["bit_exact_unmasked_ratio"] == 1.0

    for node in ana_data["dom"]["nodes"]:
        node["target_text"] = "你好！"

    r_ren = client.post(
        "/v1/render",
        json={"raw_image_png_b64": b64, "dom": ana_data["dom"], "target_lang": "zh-CN"},
    )
    assert r_ren.status_code == 200
    ren_data = r_ren.json()
    assert ren_data["dom"]["metrics"]["bit_exact_unmasked_ratio"] == 1.0
    comp_rgb = decode_image_b64(ren_data["composite_png_b64"], mode="RGB")
    assert comp_rgb.shape == page.shape
    # Verify actual replacement occurred in composite image while border frame stayed untouched
    assert np.any(comp_rgb != page)
    assert np.array_equal(comp_rgb[:25, :, :], page[:25, :, :])


def test_two_stage_detect_layout_and_restore_page_with_exclude_and_cache_miss(monkeypatch):
    monkeypatch.setenv("PRISM_T_SKIP_WARMUP", "1")
    client = TestClient(app)

    # Two separated speech bubbles on a clean page
    page = np.full((460, 420, 3), 245, dtype=np.uint8)
    cv2.rectangle(page, (15, 15), (405, 445), (40, 40, 40), 3)
    cv2.ellipse(page, (210, 130), (130, 75), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (210, 130), (130, 75), 0, 0, 360, (15, 15, 15), 2)
    cv2.putText(page, "HELLO", (130, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA)

    cv2.ellipse(page, (210, 325), (130, 75), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (210, 325), (130, 75), 0, 0, 360, (15, 15, 15), 2)
    cv2.putText(page, "WORLD", (130, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 3, cv2.LINE_AA)

    b64 = encode_png_b64(page)
    r_det = client.post(
        "/v1/detect_layout",
        json={"image_png_b64": b64, "format_profile": "comic_ltr", "recognize_text": False},
    )
    assert r_det.status_code == 200
    det_data = r_det.json()
    token = det_data["layout_token"]
    assert isinstance(token, str) and len(token) >= 16
    det_nodes = det_data["dom"]["nodes"]
    assert len(det_nodes) >= 2
    assert all(n["patches"]["clean_z0_png_b64"] == "" for n in det_nodes)
    det_ids = [n["id"] for n in det_nodes]

    excluded = det_nodes[-1]
    excluded_id = excluded["id"]
    ex, ey, ew, eh = excluded["bbox"]

    r_rest = client.post(
        "/v1/restore_page",
        json={
            "layout_token": token,
            "format_profile": "comic_ltr",
            "exclude_node_ids": [excluded_id],
        },
    )
    assert r_rest.status_code == 200
    rest_data = r_rest.json()
    rest_ids = [n["id"] for n in rest_data["dom"]["nodes"]]
    assert excluded_id not in rest_ids
    assert rest_ids == [nid for nid in det_ids if nid != excluded_id]
    assert rest_data["dom"]["metrics"]["bit_exact_unmasked_ratio"] == 1.0
    assert rest_data["dom"]["metrics"]["unmasked_linf"] == 0

    clean_rgb = decode_image_b64(rest_data["clean_preview_png_b64"], mode="RGB")
    # Excluded node's pixels remain bit-exact to raw input!
    np.testing.assert_array_equal(clean_rgb[ey : ey + eh, ex : ex + ew], page[ey : ey + eh, ex : ex + ew])

    # Cache-miss fallback recomputes layout seamlessly with identical node IDs
    r_miss = client.post(
        "/v1/restore_page",
        json={
            "layout_token": "expired_or_missing_token",
            "image_png_b64": b64,
            "format_profile": "comic_ltr",
            "recognize_text": False,
        },
    )
    assert r_miss.status_code == 200
    miss_data = r_miss.json()
    assert [n["id"] for n in miss_data["dom"]["nodes"]] == det_ids

    # Expired token without image_png_b64 returns structured HTTP 410 instead of 500
    r_gone = client.post(
        "/v1/restore_page",
        json={"layout_token": "expired_or_missing_token", "image_png_b64": ""},
    )
    assert r_gone.status_code == 410
