import cv2
import numpy as np

from prism_t_engine.deep_fusion import (
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
    out_rgb = np.array(style.outline_color_rgba[:3], dtype=float)
    assert style.outline_width_px > 0.0
    # Must select either the crisp inner white rim (~250) or crisp black rim (~0), NEVER a muddy gray midpoint (~128)
    lum = float(np.mean(out_rgb))
    assert lum >= 210.0 or lum <= 35.0, f"expected crisp rim color, got muddy gray {style.outline_color_rgba}"
