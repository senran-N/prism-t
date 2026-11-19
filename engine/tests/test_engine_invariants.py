from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from prism_t_engine.codec import decode_image_b64, encode_png_b64
from prism_t_engine.deep_fusion import (
    decompose_intrinsic_surface_shading,
    extract_stroke_style_and_halo,
    extract_z2_foreground_occluders,
    unmix_optical_alpha_layer,
)
from prism_t_engine.ocr_tps import rectify_curved_strip_tps, fit_centerline_spine
from prism_t_engine.pipeline import analyze_and_restore_page, harmonize_page_typography, typeset_and_render_page
from prism_t_engine.preprocess import (
    detect_format_profile,
    detect_seam_split_need,
    estimate_noise_sigma,
)
from prism_t_engine.restoration import (
    restore_clean_plate_roi,
    solve_anisotropic_tensor_inpaint,
)
from prism_t_engine.schemas import AnalyzeRequest, RenderRequest, RubyPair, SeamSlice, StyleSpec
from prism_t_engine.segmentation import (
    analyze_bubble_contour_and_tail,
    build_stroke_trimap_and_alpha,
    estimate_source_font_scale,
    separate_kanji_ruby_components,
)
from prism_t_engine.typesetter_sdf import (
    build_interior_sdf,
    conserve_optical_ink_mass,
    load_font,
    solve_sdf_knuth_plass_layout,
)


def test_donoho_noise_and_scale_invariant_format_detection():
    rng = np.random.default_rng(42)
    base = np.full((256, 256), 200.0, dtype=np.float32)
    noisy_4 = np.clip(base + rng.normal(0.0, 4.0, base.shape), 0, 255)
    noisy_10 = np.clip(base + rng.normal(0.0, 10.0, base.shape), 0, 255)

    s4 = estimate_noise_sigma(noisy_4)
    s10 = estimate_noise_sigma(noisy_10)
    assert 3.2 <= s4 <= 4.8
    assert 8.2 <= s10 <= 11.8

    # Monochrome vertical-column page => manga_rtl at both 1x and 2x DPI
    for scale in (1, 2):
        W, H = 400 * scale, 560 * scale
        page = np.full((H, W, 3), 248, dtype=np.uint8)
        for col_x in (320 * scale, 292 * scale, 120 * scale, 92 * scale):
            for row_y in range(60 * scale, 260 * scale, 22 * scale):
                cv2.rectangle(
                    page,
                    (col_x, row_y),
                    (col_x + 14 * scale, row_y + 14 * scale),
                    (15, 15, 15),
                    -1,
                )
        prof, is_col = detect_format_profile(page, "auto")
        assert prof == "manga_rtl"
        assert not is_col

    # Long-strip colorful webtoon => webtoon_vertical + seam split detection
    webtoon = np.full((1200, 400, 3), 245, dtype=np.uint8)
    cv2.rectangle(webtoon, (40, 100), (360, 1100), (60, 140, 230), -1)
    cv2.ellipse(webtoon, (200, 600), (110, 80), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(webtoon, (200, 600), (110, 80), 0, 0, 360, (10, 10, 10), 3)
    prof_wt, is_col_wt = detect_format_profile(webtoon, "auto")
    assert prof_wt == "webtoon_vertical"
    assert is_col_wt

    # Slice right through the speech bubble at y=600 => must request seam stitching!
    assert detect_seam_split_need(webtoon[:600], webtoon[600:]) is True
    # Slice inside pure white gutter at y=40 => must NOT request seam stitching!
    assert detect_seam_split_need(webtoon[:40], webtoon[40:90]) is False


def test_levin_closed_form_matting_preserves_hollow_glyph_interior():
    # Detailed fine artwork background (simulated eye iris / fine grid)
    H, W = 96, 96
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    art = np.zeros((H, W, 3), dtype=np.uint8)
    art[:, :, 0] = np.clip(180 + 45 * np.sin(xx * 0.35) * np.cos(yy * 0.35), 0, 255).astype(np.uint8)
    art[:, :, 1] = np.clip(175 + 45 * np.sin(xx * 0.35) * np.cos(yy * 0.35), 0, 255).astype(np.uint8)
    art[:, :, 2] = np.clip(190 + 40 * np.cos(xx * 0.25), 0, 255).astype(np.uint8)

    # Draw a HOLLOW dark-outlined ring glyph (radius=26, stroke thickness=3px)
    # Center [36:60, 36:60] is completely hollow, exposing raw underlying art!
    composite = art.copy()
    dist_c = np.hypot(xx - 48.0, yy - 48.0)
    ring_stroke = (dist_c >= 23.0) & (dist_c <= 26.0)
    composite[ring_stroke] = [12, 12, 14]

    coarse = ((dist_c >= 21.0) & (dist_c <= 28.0)).astype(np.uint8)
    alpha, sw, _ = build_stroke_trimap_and_alpha(composite, coarse_mask=coarse)

    # Hollow interior (dist < 18) MUST have alpha == 0.0!
    hollow_core = dist_c < 18.0
    assert float(np.max(alpha[hollow_core])) == 0.0
    assert 1.8 <= sw <= 6.0

    clean_roi, _, _, _ = restore_clean_plate_roi(composite, alpha)
    # Every single pixel inside the hollow interior MUST be bit-exact (L_inf == 0) to original art!
    assert np.array_equal(clean_roi[hollow_core], art[hollow_core])


def test_kanji_ruby_bic_separation_scale_invariance():
    for scale in (1, 2):
        H, W = 180 * scale, 100 * scale
        ink = np.zeros((H, W), dtype=np.uint8)
        # Main vertical kanji column: x in [28, 56]*scale
        for k in range(4):
            y0 = (20 + k * 34) * scale
            ink[y0 : y0 + 26 * scale, 28 * scale : 56 * scale] = 1
        # Companion right-hand furigana column: half width x in [62, 74]*scale
        for r in range(5):
            y0 = (24 + r * 24) * scale
            ink[y0 : y0 + 11 * scale, 62 * scale : 74 * scale] = 1

        base_m, ruby_m = separate_kanji_ruby_components(ink, vertical=True)
        # Partition completeness: no ink pixel lost
        assert np.array_equal((base_m | ruby_m), ink)
        # Purity: main column in base_m, companion narrow column in ruby_m
        assert np.sum(base_m[:, 28 * scale : 56 * scale]) == np.sum(ink[:, 28 * scale : 56 * scale])
        assert np.sum(ruby_m[:, 62 * scale : 74 * scale]) == np.sum(ink[:, 62 * scale : 74 * scale])


def test_2d_fft_halftone_crystal_restoration_sharpness():
    H, W = 112, 112
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    # Rotated 45-degree lithographic halftone screen (period ~ 6.5px)
    u = (xx + yy) / np.sqrt(2.0)
    v = (xx - yy) / np.sqrt(2.0)
    wave = np.cos(2.0 * np.pi * u / 6.5) * np.cos(2.0 * np.pi * v / 6.5)
    gt_plate = np.where(wave > 0.25, 35, 242).astype(np.uint8)
    gt_rgb = np.stack([gt_plate, gt_plate, gt_plate], axis=-1)

    # Paint opaque dark glyph strokes across the center
    damaged = gt_rgb.copy()
    stroke_mask = np.zeros((H, W), dtype=np.float32)
    cv2.line(stroke_mask, (34, 28), (78, 84), 1.0, thickness=4)
    cv2.line(stroke_mask, (76, 30), (36, 80), 1.0, thickness=4)
    damaged[stroke_mask > 0.1] = [5, 5, 5]

    restored, path, _, _ = restore_clean_plate_roi(damaged, stroke_mask)
    assert path == "fft_halftone_lattice"

    # Unmasked pixels bit-exact
    active = cv2.dilate((stroke_mask > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(restored[~active], damaged[~active])

    # Restored halftone inside stroke must preserve sharp periodic contrast (high std, low MAE vs ground truth)
    restored_vals = restored[active, 0].astype(np.float32)
    gt_vals = gt_rgb[active, 0].astype(np.float32)
    mae_fft = float(np.mean(np.abs(restored_vals - gt_vals)))
    flat_mean_mae = float(np.mean(np.abs(np.mean(gt_rgb[~active, 0]) - gt_vals)))
    assert float(np.std(restored_vals)) > 45.0  # Crisp dots, NOT blurry gray mud!
    assert mae_fft < flat_mean_mae * 0.65


def test_weickert_anisotropic_speedline_continuation_45deg():
    H, W = 96, 96
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    # Crisp 45-degree diagonal speedlines (constant along x - y = const)
    diag_coord = (xx - yy) / np.sqrt(2.0)
    lines = np.where(np.sin(2.0 * np.pi * diag_coord / 9.0) > 0.55, 25, 245).astype(np.uint8)
    gt_rgb = np.stack([lines, lines, lines], axis=-1)

    # Cut a stroke slit perpendicular to the speedlines
    mask = np.zeros((H, W), dtype=bool)
    slit_band = (np.abs((xx + yy) - 96.0) <= 2.2) & (xx >= 28) & (xx <= 68)
    mask[slit_band] = True
    corrupted = gt_rgb.copy()
    corrupted[mask] = 128

    recon, coh = solve_anisotropic_tensor_inpaint(corrupted, mask)
    assert coh > 0.60
    mae = float(np.mean(np.abs(recon[mask, 0] - gt_rgb[mask, 0].astype(np.float64))))
    assert mae < 22.0


def test_deep_fusion_unmixing_z2_occlusion_and_intrinsic_shading():
    # 1. Optical Inverse Alpha Unmixing over structured colorful background
    H, W = 96, 120
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:, :, 0] = np.clip(120 + 60 * np.sin(xx * 0.12), 0, 255).astype(np.uint8)
    bg[:, :, 1] = np.clip(140 + 55 * np.cos(yy * 0.15), 0, 255).astype(np.uint8)
    bg[:, :, 2] = np.clip(110 + 50 * np.sin((xx + yy) * 0.08), 0, 255).astype(np.uint8)

    fg_color = np.array([250.0, 245.0, 40.0], dtype=np.float32)
    alpha_gt = np.zeros((H, W), dtype=np.float32)
    cv2.rectangle(alpha_gt, (30, 28), (90, 68), 0.58, -1)
    blended = np.clip(
        np.rint(alpha_gt[:, :, None] * fg_color + (1.0 - alpha_gt[:, :, None]) * bg.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)

    is_semi, clean_B, _, _, med_a = unmix_optical_alpha_layer(blended, alpha_gt)
    assert is_semi is True
    assert 0.35 <= med_a <= 0.80
    raw_rmse = float(np.sqrt(np.mean((blended[alpha_gt > 0].astype(np.float32) - bg[alpha_gt > 0].astype(np.float32)) ** 2)))
    rec_rmse = float(np.sqrt(np.mean((clean_B[alpha_gt > 0].astype(np.float32) - bg[alpha_gt > 0].astype(np.float32)) ** 2)))
    assert rec_rmse < raw_rmse * 0.35

    # 2. End-to-End Z2 Foreground Occluder Preservation (Crimson Sword Ribbon crossing in front of SFX!)
    page = np.full((180, 220, 3), 244, dtype=np.uint8)
    # Dark SFX strokes at Z1
    sfx_mask = np.zeros((180, 220), dtype=np.uint8)
    cv2.rectangle(sfx_mask, (70, 55), (90, 125), 1, -1)
    cv2.rectangle(sfx_mask, (115, 55), (135, 125), 1, -1)
    page[sfx_mask > 0] = [18, 18, 22]

    # Foreground Z2 crimson ribbon entering from x=30 to x=190 right across both SFX strokes!
    z2_gt = np.zeros((180, 220), dtype=bool)
    z2_gt[84:94, 30:190] = True
    page[z2_gt] = [220, 30, 45]

    pre_reg = [
        {
            "bbox": [65, 50, 80, 80],
            "polygon": [[65.0, 50.0], [145.0, 50.0], [145.0, 130.0], [65.0, 130.0]],
            "orientation": "horizontal",
            "source_text": "ドーン",
            "ruby_pairs": [],
            "spine_points": [[65.0, 90.0], [145.0, 90.0]],
            "ocr_entropy": 0.05,
            "ink_mask_roi": (sfx_mask[50:130, 65:145] > 0).astype(np.uint8),
        }
    ]
    raw_b64 = encode_png_b64(page)
    ana = analyze_and_restore_page(AnalyzeRequest(image_png_b64=raw_b64, format_profile="manga_rtl"), pre_detected_regions=pre_reg)
    assert len(ana.dom.nodes) == 1
    assert ana.dom.nodes[0].fusion_mode == "occluded_z2"

    ana.dom.nodes[0].target_text = "轰隆"
    ren = typeset_and_render_page(RenderRequest(raw_image_png_b64=raw_b64, dom=ana.dom))
    final_rgb = decode_image_b64(ren.composite_png_b64, mode="RGB")

    # Every Z2 foreground crimson ribbon pixel inside the text region MUST be preserved intact on top of Z1!
    overlap_z2 = np.zeros_like(z2_gt)
    overlap_z2[85:93, 70:140] = True
    assert np.mean(np.all(final_rgb[overlap_z2] == [220, 30, 45], axis=-1)) > 0.92
    # And global untouched background bit-exactness == 1.0!
    assert ren.dom.metrics.bit_exact_unmasked_ratio == 1.0

    # 3. Intrinsic Shading Decomposition on Half-Shadowed Sign
    sign = np.full((60, 120, 3), 240, dtype=np.uint8)
    sign[:, 60:, :] = 120  # Right half in 50% shadow!
    albedo_0, shading_u8 = decompose_intrinsic_surface_shading(sign)
    assert min(albedo_0) >= 235
    assert float(np.mean(shading_u8[:, :55])) > 245.0
    assert 115.0 <= float(np.mean(shading_u8[:, 65:])) <= 135.0


def test_sdf_knuth_plass_kinsoku_ruby_and_brentq_ink_conservation():
    H, W = 180, 150
    bubble_poly = []
    for th in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        bubble_poly.append([75.0 + 62.0 * np.cos(th), 90.0 + 78.0 * np.sin(th)])
    sdf, pole = build_interior_sdf((H, W), (0, 0), bubble_poly)

    style_v = StyleSpec(orientation="vertical", font_category="gothic")
    rubies = [RubyPair(base="宿敌", ruby="挚友", char_start=4, char_end=6)]
    layout_v, alpha_v = solve_sdf_knuth_plass_layout(
        target_text="「笨蛋，宿敌啊！」",
        ruby_pairs=rubies,
        sdf=sdf,
        pole_xy=pole,
        style=style_v,
    )
    assert len(layout_v.lines) >= 1
    assert np.max(alpha_v) > 0.8
    # Verify vertical bracket mapping (`﹁` / `﹂`) and zero Kinsoku line-start violations
    joined = "".join(layout_v.lines)
    assert "﹁" in joined and "﹂" in joined
    for ln in layout_v.lines[1:]:
        assert ln[0] not in ("，", "！", "﹂")
    # Verify companion ruby glyphs were laid out
    assert any(g.is_ruby for g in layout_v.glyphs)

    # Brent's root-finding (`brentq`) Optical Ink-Mass Conservation test for thick display SFX
    dense_style = StyleSpec(
        orientation="horizontal",
        font_category="gothic",
        stroke_width_px=5.0,
        ink_density_orig=0.06,
    )
    flat_sdf = np.full((90, 180), 40.0, dtype=np.float32)
    _, solid_alpha = solve_sdf_knuth_plass_layout(
        target_text="轰隆隆震耳欲聋",
        ruby_pairs=[],
        sdf=flat_sdf,
        pole_xy=(90.0, 45.0),
        style=dense_style,
    )
    rho_before = float(np.mean(solid_alpha))
    assert rho_before > 0.06 * 1.35
    comp_alpha, comp_style = conserve_optical_ink_mass(solid_alpha, dense_style, over_artwork=True)
    rho_after = float(np.mean(comp_alpha))
    assert comp_style.hollow_mode in ("double_outline_hollow", "stroke_thinned")
    assert abs(rho_after - 0.06) < 0.02


def test_bookstein_tps_curved_strip_rectification():
    # Render text along a parabolic arc and verify TPS rectification flattens the baseline
    H, W = 90, 200
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    xs = np.linspace(25, 175, 9)
    for x in xs:
        y = int(round(28.0 + 24.0 * (((x - 100.0) / 75.0) ** 2)))
        cv2.circle(img, (int(round(x)), y), 6, (15, 15, 15), -1)

    mask = (img[:, :, 0] < 100).astype(np.uint8)
    spine = fit_centerline_spine(mask, vertical=False)
    rect = rectify_curved_strip_tps(img, spine, strip_thickness=28.0, vertical=False)
    # In rectified strip, dark dots should align horizontally near vertical midpoint!
    ry, _ = np.nonzero(rect[:, :, 0] < 100)
    assert ry.size > 20
    assert float(np.std(ry)) < 6.5


def test_page_font_consistency_and_subpixel_modulation():
    H, W = 520, 460
    page = np.full((H, W, 3), 246, dtype=np.uint8)
    # Two uniform-stroke speech ovals + one stepped L-shaped narration box
    cv2.ellipse(page, (130, 140), (85, 95), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (130, 140), (85, 95), 0, 0, 360, (12, 12, 12), 3)
    cv2.ellipse(page, (340, 150), (85, 95), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (340, 150), (85, 95), 0, 0, 360, (12, 12, 12), 3)

    # Stepped L-shaped rectangular narration box
    l_pts = np.array([[60, 300], [310, 300], [310, 375], [220, 375], [220, 445], [60, 445]], dtype=np.int32)
    cv2.fillPoly(page, [l_pts], (255, 255, 255))
    cv2.polylines(page, [l_pts], True, (12, 12, 12), 3)

    # Render uniform 2px dark rectangles/strokes inside each bubble (vertical lines to test sub-pixel modulation)
    pre_reg = []
    for bx, by, bw, bh, txt, ori in [
        (105, 85, 50, 110, "久等了", "vertical"),
        (315, 95, 50, 110, "走吧", "vertical"),
        (85, 320, 180, 40, "System Alert", "horizontal"),
    ]:
        for k in range(3):
            if ori == "vertical":
                cv2.rectangle(page, (bx + 8 + k * 14, by + 10), (bx + 10 + k * 14, by + bh - 10), (14, 14, 14), -1)
            else:
                cv2.rectangle(page, (bx + 10, by + 8 + k * 12), (bx + bw - 10, by + 10 + k * 12), (14, 14, 14), -1)
        ink_m = (page[by : by + bh, bx : bx + bw, 0] < 100).astype(np.uint8)
        pre_reg.append(
            {
                "bbox": [bx, by, bw, bh],
                "polygon": [[float(bx), float(by)], [float(bx + bw), float(by)], [float(bx + bw), float(by + bh)], [float(bx), float(by + bh)]],
                "orientation": ori,
                "source_text": txt,
                "ruby_pairs": [],
                "spine_points": [[float(bx + bw * 0.5), float(by)], [float(bx + bw * 0.5), float(by + bh)]],
                "ocr_entropy": 0.05,
                "ink_mask_roi": ink_m,
            }
        )

    ana = analyze_and_restore_page(AnalyzeRequest(image_png_b64=encode_png_b64(page), format_profile="manga_rtl"), pre_detected_regions=pre_reg)
    ovals = [n for n in ana.dom.nodes if n.bubble_kind == "speech_oval"]
    boxes = [n for n in ana.dom.nodes if n.bubble_kind == "narration_box"]
    assert len(ovals) == 2
    assert len(boxes) == 1
    assert all(n.style.font_category != "brush" for n in ovals)
    assert len({n.style.font_category for n in ovals}) == 1
    assert all(n.style.stroke_modulation < 0.25 for n in ovals)


def test_scale_anchored_sdf_typesetting_and_ribbon_bounds():
    H, W = 300, 220
    # Ellipse + narrow downward pointer tail at bottom
    poly = []
    for th in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        px = 110.0 + 78.0 * np.cos(th)
        py = 115.0 + 88.0 * np.sin(th)
        if abs(th - 0.5 * np.pi) < 0.14:
            py += 75.0
        poly.append([px, py])
    sdf, pole = build_interior_sdf((H, W), (0, 0), poly)

    style = StyleSpec(
        orientation="vertical",
        font_category="gothic",
        source_font_size_px=26.0,
    )
    layout, alpha = solve_sdf_knuth_plass_layout(
        target_text="久等了",
        ruby_pairs=[],
        sdf=sdf,
        pole_xy=pole,
        style=style,
        enclosed_bubble=True,
    )
    assert 22.0 <= layout.font_size <= 31.0
    # Zero rendered glyph pixels may overflow outside interior bubble mask (sdf <= 0)
    assert np.count_nonzero((alpha > 0.05) & (sdf <= 0.0)) == 0


def test_bubbleless_open_margin_and_bit_exact_preservation():
    H, W = 280, 320
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    page = np.zeros((H, W, 3), dtype=np.uint8)
    page[:, :, 0] = np.clip(185 + 25 * np.sin(xx * 0.04), 0, 255).astype(np.uint8)
    page[:, :, 1] = np.clip(195 + 20 * np.cos(yy * 0.04), 0, 255).astype(np.uint8)
    page[:, :, 2] = np.clip(205 + 15 * np.sin((xx + yy) * 0.03), 0, 255).astype(np.uint8)

    # Borderless text hugging x=8 (open page left margin!) + floating center text
    pil = Image.fromarray(page)
    draw = ImageDraw.Draw(pil)
    fnt = load_font("gothic", 20, needs_cjk=False)
    draw.text((8, 30), "Hello.", fill=(20, 20, 22), font=fnt)
    draw.text((130, 140), "Thank you.", fill=(20, 20, 22), font=fnt)
    page = np.asarray(pil, dtype=np.uint8)

    _, k1, _, _ = analyze_bubble_contour_and_tail(page, [6, 26, 70, 28])
    assert k1 == "floating_sfx"

    pre_reg = [
        {
            "bbox": [6, 26, 70, 28],
            "polygon": [[6.0, 26.0], [76.0, 26.0], [76.0, 54.0], [6.0, 54.0]],
            "orientation": "horizontal",
            "source_text": "Hello.",
            "ruby_pairs": [],
            "spine_points": [[6.0, 40.0], [76.0, 40.0]],
            "ocr_entropy": 0.05,
            "ink_mask_roi": (page[26:54, 6:76, 0] < 90).astype(np.uint8),
        },
        {
            "bbox": [128, 136, 115, 28],
            "polygon": [[128.0, 136.0], [243.0, 136.0], [243.0, 164.0], [128.0, 164.0]],
            "orientation": "horizontal",
            "source_text": "Thank you.",
            "ruby_pairs": [],
            "spine_points": [[128.0, 150.0], [243.0, 150.0]],
            "ocr_entropy": 0.05,
            "ink_mask_roi": (page[136:164, 128:243, 0] < 90).astype(np.uint8),
        },
    ]
    raw_b64 = encode_png_b64(page)
    ana = analyze_and_restore_page(AnalyzeRequest(image_png_b64=raw_b64, format_profile="comic_ltr"), pre_detected_regions=pre_reg)
    assert len(ana.dom.nodes) == 2
    for nd in ana.dom.nodes:
        assert nd.bubble_kind == "floating_sfx"
        assert nd.bubble_polygon == []
    ana.dom.nodes[0].target_text = "你好。"
    ana.dom.nodes[1].target_text = "谢谢。"
    ren = typeset_and_render_page(RenderRequest(raw_image_png_b64=raw_b64, dom=ana.dom))
    assert ren.dom.metrics.bit_exact_unmasked_ratio == 1.0
    for nd in ren.dom.nodes:
        assert nd.style.hollow_mode == "solid"


def test_single_column_packing_scale_and_harmonization():
    # Vertical single column with 5 glyphs packed into height=140px
    ink = np.zeros((140, 48), dtype=np.uint8)
    for k in range(5):
        ink[4 + k * 27 : 26 + k * 27, 4:44] = 1
    est = estimate_source_font_scale(ink, vertical=True, source_text="待たせたな")
    assert est <= (140.0 / 5.0) * 1.06

    # Peer harmonization pulls 1.75x outlier smoothly toward peer median
    meta = [
        {"bubble_kind": "speech_oval", "source_text": "foo", "style": StyleSpec(source_font_size_px=24.0, font_category="gothic")},
        {"bubble_kind": "speech_oval", "source_text": "bar", "style": StyleSpec(source_font_size_px=25.0, font_category="gothic")},
        {"bubble_kind": "speech_oval", "source_text": "baz", "style": StyleSpec(source_font_size_px=42.0, font_category="mincho")},
    ]
    harmonize_page_typography(meta)
    assert all(it["style"].font_category == "gothic" for it in meta)
    assert meta[2]["style"].source_font_size_px < 36.0


def test_thin_sepia_stroke_alpha_detection_and_solid_paper_non_hollow():
    # Low-contrast 1px sepia stroke on warm cream paper
    roi = np.full((50, 90, 3), [252, 244, 224], dtype=np.uint8)
    roi[24:25, 18:72] = [165, 130, 105]
    coarse = np.zeros((50, 90), dtype=np.uint8)
    coarse[23:26, 17:73] = 1
    alpha, _, _ = build_stroke_trimap_and_alpha(roi, coarse_mask=coarse)
    assert float(np.max(alpha[24:25, 18:72])) >= 0.45

    # Solid paper over_artwork=False preserves solid strokes (no double_outline_hollow)
    dense_style = StyleSpec(orientation="horizontal", font_category="gothic", stroke_width_px=5.0, ink_density_orig=0.05)
    flat_sdf = np.full((80, 180), 35.0, dtype=np.float32)
    _, solid_alpha = solve_sdf_knuth_plass_layout(
        target_text="第114话 新娘",
        ruby_pairs=[],
        sdf=flat_sdf,
        pole_xy=(90.0, 40.0),
        style=dense_style,
    )
    comp_alpha, out_st = conserve_optical_ink_mass(solid_alpha, dense_style, over_artwork=False)
    assert out_st.hollow_mode == "solid"
    assert np.allclose(comp_alpha, solid_alpha)


def test_slant_detection_enclosed_z2_and_real_ocr_pipeline():
    # 1. Non-zero italic slant angle estimation via second-order central moments
    roi = np.full((100, 120, 3), 250, dtype=np.uint8)
    clean_z0 = roi.copy()
    alpha = np.zeros((100, 120), dtype=np.float32)
    for x0 in (35, 65):
        cv2.line(alpha, (x0 + 12, 18), (x0 - 8, 82), 1.0, 5)
    roi[alpha > 0.5] = [15, 15, 15]
    spec, _, _ = extract_stroke_style_and_halo(
        roi_rgb=roi,
        stroke_alpha=alpha,
        stroke_w=5.0,
        stroke_mod=0.1,
        orientation="vertical",
        bubble_kind="speech_oval",
    )
    assert abs(spec.slant_deg) >= 8.0

    # 2. Enclosed speech bubble with foreground colored strand crossing text detects Z2 without catching bubble border
    bub_roi = np.full((120, 120, 3), 252, dtype=np.uint8)
    sub = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(sub, (60, 60), 54, 1, -1)
    cv2.circle(bub_roi, (60, 60), 55, (15, 15, 15), 2)
    b_alpha = np.zeros((120, 120), dtype=np.float32)
    b_alpha[35:85, 45:53] = 1.0
    b_alpha[35:85, 67:75] = 1.0
    bub_roi[b_alpha > 0.5] = [15, 15, 15]
    bub_roi[56:64, 15:105] = [215, 35, 50]  # Foreground crimson ribbon across bubble
    z2_enc, _ = extract_z2_foreground_occluders(bub_roi, b_alpha, enclosed_bubble=True, substrate_mask=sub)
    assert np.count_nonzero(z2_enc[56:64, 44:76] > 0) >= 80
    assert np.count_nonzero(z2_enc[:20, :] > 0) == 0

    # 3. Full OCR -> Analyze -> Render without pre_detected_regions (genuine zero-cheat detection test)
    page = np.full((340, 320, 3), 246, dtype=np.uint8)
    cv2.ellipse(page, (160, 170), (110, 95), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, (160, 170), (110, 95), 0, 0, 360, (12, 12, 12), 2)
    cv2.putText(page, "ATTACK", (88, 178), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (12, 12, 12), 3, cv2.LINE_AA)
    raw_b64 = encode_png_b64(page)
    ana_real = analyze_and_restore_page(AnalyzeRequest(image_png_b64=raw_b64, format_profile="comic_ltr"))
    assert len(ana_real.dom.nodes) >= 1
    assert any("ATTACK" in nd.source_text.upper() for nd in ana_real.dom.nodes)
