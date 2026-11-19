from __future__ import annotations

import numpy as np

from prism_t_engine.codec import decode_image_b64, encode_png_b64
from prism_t_engine.pipeline import typeset_and_render_page
from prism_t_engine.schemas import PageDOM, RenderRequest, StyleSpec, TextNode


def test_manual_box_add_move_and_delete_preserve_linf_zero() -> None:
    rng = np.random.default_rng(7)
    raw = rng.integers(200, 245, size=(140, 160, 3), dtype=np.uint8)
    # Draw dark fake text strokes inside [20, 20, 50, 40]
    raw[30:48, 30:60] = 25
    raw_b64 = encode_png_b64(raw)

    # 1. User manually draws a brand-new box with empty patches
    manual_node = TextNode(
        id="manual_1",
        reading_order=0,
        bbox=[20, 20, 50, 40],
        source_text="",
        target_text="手动框",
        bubble_kind="speech_oval",
        restoration_path="lama_masked",
        style=StyleSpec(orientation="horizontal", font_category="gothic"),
    )
    dom = PageDOM(width=160, height=140, format_profile="manga_rtl", nodes=[manual_node])
    resp = typeset_and_render_page(
        RenderRequest(raw_image_png_b64=raw_b64, dom=dom, target_lang="zh-CN")
    )
    assert resp.dom.metrics.unmasked_linf == 0
    assert resp.dom.metrics.bit_exact_unmasked_ratio == 1.0
    assert resp.dom.nodes[0].patches.clean_z0_png_b64 != ""
    assert resp.dom.nodes[0].patches.roi[2] >= 50

    # 2. User drags/resizes that box to a new position outside old ROI
    moved_dom = resp.dom.model_copy(deep=True)
    moved_dom.nodes[0].bbox = [90, 70, 55, 45]
    moved_dom.nodes[0].bubble_polygon = []
    moved_resp = typeset_and_render_page(
        RenderRequest(raw_image_png_b64=raw_b64, dom=moved_dom, target_lang="zh-CN")
    )
    assert moved_resp.dom.metrics.unmasked_linf == 0
    assert moved_resp.dom.metrics.bit_exact_unmasked_ratio == 1.0
    nx0, ny0, nw, nh = moved_resp.dom.nodes[0].patches.roi
    assert nx0 <= 20 and ny0 <= 20 and (nx0 + nw) >= 145 and (ny0 + nh) >= 115

    # 3. User deletes the node -> composite must match raw bit-for-bit
    del_dom = moved_resp.dom.model_copy(deep=True)
    del_dom.nodes = []
    del_resp = typeset_and_render_page(
        RenderRequest(raw_image_png_b64=raw_b64, dom=del_dom, target_lang="zh-CN")
    )
    comp = decode_image_b64(del_resp.composite_png_b64, mode="RGB")
    assert int(np.max(np.abs(comp.astype(np.int16) - raw.astype(np.int16)))) == 0
