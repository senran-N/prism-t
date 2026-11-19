import cv2
import numpy as np

from prism_t_engine.deep_fusion import (
    apply_exemplar_stroke_shader,
    extract_stroke_style_and_halo,
    extract_z2_foreground_occluders,
)
from prism_t_engine.learned import refine_text_alpha


def _build_double_outlined_two_zone_crop():
    """Create a 3-layer subtitle (pink fill + white inner rim + black outer rim)
    crossing a top #000000 black margin and a bottom purple wall."""
    H, W = 200, 120
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[90:, :, :] = [138, 78, 178]  # purple wall bottom half; top half is pure black [0, 0, 0]

    core = np.zeros((H, W), dtype=np.uint8)
    for y0 in (24, 62, 108, 146):
        cv2.rectangle(core, (46, y0), (74, y0 + 22), 1, -1)
        cv2.rectangle(core, (52, y0 + 5), (68, y0 + 17), 0, -1)

    k_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    white_ring = (cv2.dilate(core, k_white) > 0) & (core == 0)
    black_ring = (cv2.dilate(core, k_black) > 0) & (cv2.dilate(core, k_white) == 0)

    rgb[black_ring] = [4, 2, 6]
    rgb[white_ring] = [254, 246, 252]
    rgb[core > 0] = [228, 54, 136]

    inner_support = (core > 0) | white_ring
    prob = np.zeros((H, W), dtype=np.float32)
    prob[inner_support] = 0.95
    prob[black_ring] = 0.20
    seed = np.zeros((H, W), dtype=np.uint8)
    seed[18:178, 38:82] = 1
    return rgb, core > 0, white_ring, black_ring, inner_support, prob, seed


def test_refine_text_alpha_recovers_outer_black_outline_on_multicolor_wall_without_margin_bleed():
    rgb, core, white_ring, black_ring, inner_support, prob, seed = _build_double_outlined_two_zone_crop()

    alpha = refine_text_alpha(rgb, prob, seed, observed_ink=inner_support.astype(np.uint8))
    erased = alpha > 0

    wall_black = black_ring.copy()
    wall_black[:94, :] = False
    black_recall = float(np.count_nonzero(erased & wall_black)) / float(max(1, np.count_nonzero(wall_black)))
    assert black_recall >= 0.90, f"expected >=90% outer black rim recovery on purple wall, got {black_recall:.3f}"

    pure_top_margin = np.zeros(rgb.shape[:2], dtype=bool)
    pure_top_margin[:86, :] = True
    pure_top_margin &= ~(core | white_ring | black_ring)
    margin_bleed = float(np.count_nonzero(erased & pure_top_margin)) / float(max(1, np.count_nonzero(pure_top_margin)))
    assert margin_bleed <= 0.01, f"outer rim recovery bled into top black margin: {margin_bleed:.4f}"

    pure_wall = np.zeros(rgb.shape[:2], dtype=bool)
    pure_wall[94:, :] = True
    pure_wall &= ~(core | white_ring | black_ring)
    wall_bleed = float(np.count_nonzero(erased & pure_wall)) / float(max(1, np.count_nonzero(pure_wall)))
    assert wall_bleed <= 0.02, f"outer rim recovery bled into purple wall: {wall_bleed:.4f}"


def test_z2_occluder_rejects_bimodal_two_zone_background():
    rgb, core, white_ring, black_ring, inner_support, prob, _ = _build_double_outlined_two_zone_crop()
    stroke_alpha = (inner_support | black_ring).astype(np.float32)

    z2_mask, _ = extract_z2_foreground_occluders(
        rgb,
        stroke_alpha,
        enclosed_bubble=False,
        substrate_mask=None,
        text_probability=prob,
    )
    assert int(np.count_nonzero(z2_mask)) == 0


def test_double_outlined_style_selects_inner_white_rim_not_muddy_gray():
    rgb, core, white_ring, black_ring, inner_support, _, _ = _build_double_outlined_two_zone_crop()
    # Focus on the lower purple-wall glyphs where all 3 layers (pink + white + black) are active
    sub_rgb = rgb[92:, :, :]
    sub_mask = (core | white_ring | black_ring)[92:, :]
    sub_bg = ~sub_mask

    style, _, _ = extract_stroke_style_and_halo(
        roi_rgb=sub_rgb,
        stroke_alpha=sub_mask.astype(np.float32),
        stroke_w=4.0,
        stroke_mod=0.1,
        orientation="vertical",
        bubble_kind="floating_sfx",
        pigment_support=sub_mask.astype(np.uint8),
        substrate_support=sub_bg.astype(np.uint8),
    )
    fg_rgb = np.array(style.fg_color_rgba[:3], dtype=float)
    assert np.linalg.norm(fg_rgb - np.array([228.0, 54.0, 136.0])) < 20.0, f"expected pink fill, got {style.fg_color_rgba}"
    out_rgb = np.array(style.outline_color_rgba[:3], dtype=float)
    assert style.outline_width_px > 0.0
    assert float(np.mean(out_rgb)) >= 210.0, f"expected inner white rim, got {style.outline_color_rgba}"
    assert style.outer_outline_width_px > 0.0, "expected detected outer black rim width > 0"
    assert float(np.mean(style.outer_outline_color_rgba[:3])) <= 35.0, (
        f"expected outer black rim, got {style.outer_outline_color_rgba}"
    )

    # Verify shader renders all 3 concentric layers (pink fill -> white inner -> black outer)
    glyph = np.zeros((80, 80), dtype=np.float32)
    cv2.rectangle(glyph, (30, 20), (50, 60), 1.0, -1)
    z1 = apply_exemplar_stroke_shader(glyph, None, style)
    assert z1[40, 40, 3] == 255
    assert np.linalg.norm(z1[40, 40, :3].astype(float) - fg_rgb) < 10.0
    # Inner white ring just outside glyph x=50
    inner_x = min(79, 50 + max(1, int(round(0.5 * style.outline_width_px))))
    assert float(np.mean(z1[40, inner_x, :3])) >= 200.0
    # Outer black ring outside white ring
    outer_x = min(79, 50 + int(round(style.outline_width_px + 0.5 * style.outer_outline_width_px + 0.5)))
    assert z1[40, outer_x, 3] > 60
    assert float(np.mean(z1[40, outer_x, :3])) <= 60.0


def test_enclosed_solid_heart_symbol_not_dropped_by_stroke_growth_bound():
    """A thick filled heart ♥ enclosed inside its outline must be erased completely,
    not pruned as a 'wide shirt' by bound_observed_stroke_growth."""
    H, W = 160, 100
    rgb = np.full((H, W, 3), [145, 90, 175], dtype=np.uint8)
    core = np.zeros((H, W), dtype=np.uint8)
    # Thin stroke glyphs at top
    cv2.putText(core, "A", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 1, 3)
    # Thick solid circular/heart blob at bottom (radius ~ 11px >> kanji stroke radius ~ 1.5px)
    cv2.circle(core, (50, 110), 11, 1, -1)
    k_w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    white_ring = (cv2.dilate(core, k_w) > 0) & (core == 0)
    black_ring = (cv2.dilate(core, k_b) > 0) & (cv2.dilate(core, k_w) == 0)
    rgb[black_ring] = [5, 2, 7]
    rgb[white_ring] = [252, 245, 250]
    rgb[core > 0] = [228, 56, 138]

    obs = ((core > 0) | white_ring | black_ring).astype(np.uint8)
    prob = np.zeros((H, W), dtype=np.float32)
    prob[core > 0] = 0.90
    prob[white_ring] = 0.70
    prob[black_ring] = 0.30
    seed = cv2.dilate(obs, np.ones((9, 9), np.uint8))

    alpha = refine_text_alpha(rgb, prob, seed, observed_ink=obs)
    heart_mask = np.zeros((H, W), dtype=bool)
    cv2.circle(heart_mask.view(np.uint8), (50, 110), 10, 1, -1)
    heart_recall = float(np.count_nonzero((alpha > 0) & heart_mask)) / float(np.count_nonzero(heart_mask))
    assert heart_recall >= 0.95, f"solid enclosed heart core was dropped: recall={heart_recall:.3f}"
