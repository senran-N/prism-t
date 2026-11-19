from __future__ import annotations

import time
import cv2
import numpy as np

from prism_t_engine.deep_fusion import (
    apply_exemplar_stroke_shader,
    composite_node_layers,
    extract_z2_foreground_occluders,
    unmix_optical_alpha_layer,
)
from prism_t_engine.schemas import StyleSpec
from prism_t_engine.typesetter_sdf import (
    _font_metrics,
    _text_advance,
    build_interior_sdf,
    conserve_optical_ink_mass,
    estimate_char_budget_from_sdf,
    load_font,
    solve_sdf_knuth_plass_layout,
)


def test_smooth_antialiased_outline_and_outer_feathering_preserves_linf_zero():
    H, W = 96, 96
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    # Smooth 2x-supersampled diagonal/circular glyph alpha (no binary stair-stepping)
    ss_yy, ss_xx = np.meshgrid(
        np.arange(H * 2, dtype=np.float32) * 0.5,
        np.arange(W * 2, dtype=np.float32) * 0.5,
        indexing="ij",
    )
    dist_c = np.hypot(ss_xx - 48.0, ss_yy - 48.0)
    up_ring = np.clip(1.0 - np.abs(dist_c - 22.0) + 1.5, 0.0, 1.0)
    glyph_alpha = cv2.resize(up_ring.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)

    style = StyleSpec(
        fg_color_rgba=[200, 17, 193, 255],
        outline_color_rgba=[250, 243, 252, 255],
        outline_width_px=2.5,
    )
    z1_rgba = apply_exemplar_stroke_shader(glyph_alpha, None, style)
    alpha_u8 = z1_rgba[:, :, 3]

    # Must preserve fractional glyph coverage everywhere and keep RGB=0 wherever alpha_u8 == 0
    assert np.all(alpha_u8 >= np.rint(glyph_alpha * 255.0).astype(np.uint8))
    assert np.all(z1_rgba[alpha_u8 == 0, :3] == 0)

    # Faint isolated pixel below 0.5/0.1 after bilinear upsampling must NOT fabricate phantom opaque patches
    faint_alpha = np.zeros((100, 100), dtype=np.float32)
    faint_alpha[50, 50] = 0.15
    z1_faint = apply_exemplar_stroke_shader(faint_alpha, None, StyleSpec(outline_width_px=2.0))
    assert int(z1_faint[44, 44, 3]) == 0
    assert np.all(z1_faint[z1_faint[:, :, 3] == 0, :3] == 0)
    assert int(z1_faint[50, 50, 3]) == int(round(0.15 * 255.0))

    # Outer anti-aliased feathering ramp must produce smooth intermediate alpha levels
    outer_ring = (alpha_u8 > 0) & (alpha_u8 < 255) & (glyph_alpha < 0.05)
    assert np.count_nonzero(outer_ring) > 30
    assert len(np.unique(alpha_u8[outer_ring])) >= 12

    # Bit-exact L_inf == 0 preservation outside rendered Z1 mask
    raw = np.stack(
        (
            np.clip(30 + xx * 1.5, 0, 255),
            np.clip(50 + yy * 1.2, 0, 255),
            np.clip(120 + (xx - yy) * 0.6, 0, 255),
        ),
        axis=-1,
    ).astype(np.uint8)
    out = composite_node_layers(
        raw_roi_rgb=raw,
        clean_z0_rgb=raw,
        stroke_alpha_orig=np.zeros((H, W), dtype=np.float32),
        rendered_z1_rgba=z1_rgba,
        z2_occluder_mask=None,
        shading_map_u8=None,
        style=style,
        fusion_mode="clean_interior",
    )
    untouched = alpha_u8 == 0
    assert np.any(untouched)
    assert np.array_equal(out[untouched], raw[untouched])

    # Tremor + outline path also succeeds cleanly without overflowing outside mask
    tremor_style = style.model_copy(update={"tremor_amplitude": 0.9, "stroke_width_px": 3.0})
    z1_tremor = apply_exemplar_stroke_shader(glyph_alpha, None, tremor_style)
    assert np.any(z1_tremor[:, :, 3] > 0)
    assert np.all(z1_tremor[:5, :5, 3] == 0)


def test_enclosed_bubble_and_narration_box_safe_interior_padding():
    font_sizes = []
    for scale in (1, 2):
        h, w = 80 * scale, 180 * scale
        poly = [
            [10.0 * scale, 10.0 * scale],
            [170.0 * scale, 10.0 * scale],
            [170.0 * scale, 70.0 * scale],
            [10.0 * scale, 70.0 * scale],
        ]
        sdf, pole = build_interior_sdf((h, w), (0, 0), poly)
        outline_w = 2.0 * scale
        style = StyleSpec(
            orientation="horizontal",
            font_category="gothic",
            source_font_size_px=22.0 * scale,
            outline_width_px=outline_w,
        )
        layout, glyph_alpha = solve_sdf_knuth_plass_layout(
            target_text="片刻之后……",
            ruby_pairs=[],
            sdf=sdf,
            pole_xy=pole,
            style=style,
            enclosed_bubble=True,
        )
        assert "".join(layout.lines) == "片刻之后……"
        font_sizes.append(layout.font_size)

        scaled_outline = outline_w * layout.font_size / float(style.source_font_size_px)
        z1 = apply_exemplar_stroke_shader(
            glyph_alpha,
            None,
            style.model_copy(update={"outline_width_px": scaled_outline}),
        )
        rendered = z1[:, :, 3] > 0
        assert np.any(rendered)
        # Neither glyphs nor their outer outline may collide with the dark box border (sdf <= 0)
        assert float(np.min(sdf[rendered])) > 1.5 * scale

    ratio = font_sizes[1] / max(font_sizes[0], 1e-6)
    assert 1.80 <= ratio <= 2.20


def test_rotated_diagonal_multiline_quadrilaterals_consistent_alignment():
    # Diagonal left-aligned block (-27 deg) with 4 full lines + 1 short tail line + 1 tiny punctuation box
    angle = np.deg2rad(-27.0)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-u[1], u[0]])
    origin = np.array([80.0, 180.0])
    axes = np.column_stack((u, v))

    def make_quad(u0: float, u1: float, v0: float, v1: float) -> list[list[float]]:
        pts = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ axes.T + origin
        return pts.tolist()

    quads = [
        make_quad(0.0, 230.0, 0.0, 20.0),
        make_quad(4.0, 245.0, 26.0, 46.0),
        make_quad(2.0, 260.0, 52.0, 72.0),
        make_quad(3.0, 255.0, 78.0, 98.0),
        make_quad(90.0, 145.0, 104.0, 120.0),   # short trailing word
        make_quad(148.0, 156.0, 106.0, 118.0),  # tiny punctuation fragment
    ]
    text = "你以为凭这点微末伎俩就能阻止塔尔坦吗！你们真是愚蠢！这场地震也是我引起的！"
    style = StyleSpec(orientation="horizontal", source_font_size_px=18.0)
    layout, alpha = solve_sdf_knuth_plass_layout(
        target_text=text,
        ruby_pairs=[],
        sdf=np.ones((320, 420), dtype=np.float32),
        pole_xy=(210.0, 160.0),
        style=style,
        enclosed_bubble=False,
        source_quadrilaterals=quads,
    )
    assert "".join(layout.lines) == text
    assert len(layout.lines) >= 3

    # Compute start-of-line projection along shared axis `u` and cross position along `v`
    starts_u = []
    cross_v = []
    idx = 0
    for ln in layout.lines:
        g0 = layout.glyphs[idx]
        idx += len(ln)
        p = np.array([g0.x + 0.5 * g0.w, g0.y + 0.5 * g0.h]) - origin
        starts_u.append(float(p @ u))
        cross_v.append(float(p @ v))

    # Lines must be strictly ordered along cross_axis `v`
    assert all(a < b for a, b in zip(cross_v, cross_v[1:]))
    # Line 1 must NOT be indented far to the right by the short tail line
    assert max(starts_u[:3]) - min(starts_u[:3]) < 12.0

    # All rendered ink must remain strictly inside the original block's [u_min, u_max] envelope
    yy, xx = np.nonzero(alpha > 0.05)
    pts_rel = np.column_stack((xx, yy)) - origin
    proj_u = pts_rel @ u
    assert float(proj_u.min()) >= -2.0
    assert float(proj_u.max()) <= 262.0


def test_vertical_ellipsis_and_symmetric_brackets_alignment_and_spacing():
    text = "（班上）……唔！"
    sdf, pole = build_interior_sdf((420, 120), (0, 0), [])
    layout, alpha = solve_sdf_knuth_plass_layout(
        target_text=text,
        ruby_pairs=[],
        sdf=sdf,
        pole_xy=pole,
        style=StyleSpec(orientation="vertical", source_font_size_px=28.0),
    )
    assert len(layout.lines) == 1
    g_open = next(g for g in layout.glyphs if g.char == "︵")
    g_ban = next(g for g in layout.glyphs if g.char == "班")
    g_shang = next(g for g in layout.glyphs if g.char == "上")
    g_close = next(g for g in layout.glyphs if g.char == "︶")
    g_dots = [g for g in layout.glyphs if g.char == "︙"]

    # Opening & closing vertical brackets must have symmetric height and non-overlapping spacing
    assert abs(g_open.h - g_close.h) < 1e-3
    assert g_open.y + g_open.h <= g_ban.y + 0.1
    assert g_shang.y + g_shang.h <= g_close.y + 0.1

    # Column centering: actual rendered ink X centroids for ︵, ︶, and ︙ sit on the column axis
    col_cx = g_ban.x + 0.5 * g_ban.w
    for g in (g_open, g_close, g_dots[0], g_dots[1]):
        y0 = max(0, int(np.floor(g.y)))
        y1 = min(alpha.shape[0], int(np.ceil(g.y + g.h)))
        _, xs = np.nonzero(alpha[y0:y1, :] > 0.25)
        assert xs.size > 0
        ink_cx = 0.5 * (float(xs.min()) + float(xs.max()))
        assert abs(ink_cx - col_cx) <= 1.8

    assert len(g_dots) == 2
    assert (g_dots[1].y - g_dots[0].y) <= layout.font_size * 0.76


def _slow_reference_budget(sdf: np.ndarray, scale: float, style: StyleSpec) -> int:
    font = load_font(style.font_category, scale, required_text="田", bold=style.weight_scale >= 1.10)
    ascent, descent = _font_metrics(font)
    step = float(ascent + descent)
    advance = _text_advance(font, "田")
    vertical = style.orientation == "vertical"
    support = sdf > float(style.outline_width_px)
    mask = support.T if vertical else support
    band = max(1, int(np.ceil(scale if vertical else step)))
    cell = float(scale if vertical else advance)
    if mask.shape[0] < band or cell <= 0:
        return 0
    capacities = []
    for phase in range(band):
        count = 0
        for start in range(phase, mask.shape[0] - band + 1, band):
            row = np.all(mask[start : start + band], axis=0)
            edges = np.diff(np.pad(row.astype(int), (1, 1)))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            count += int(np.floor(lengths / cell).sum())
        capacities.append(count)
    return max(capacities, default=0)


def test_vectorized_char_budget_and_fast_ink_mass_conservation():
    for ori in ("horizontal", "vertical"):
        for fs in (18.0, 27.0):
            poly = [[40.0, 30.0], [340.0, 40.0], [330.0, 260.0], [30.0, 240.0]]
            sdf, _ = build_interior_sdf((300, 380), (0, 0), poly)
            st = StyleSpec(orientation=ori, source_font_size_px=fs, outline_width_px=1.5)
            _, fast_cnt = estimate_char_budget_from_sdf(sdf, "", fs, st)
            ref_cnt = _slow_reference_budget(sdf, fs, st)
            assert fast_cnt == ref_cnt

    # Speed check on 400x400 SDF
    big_sdf, _ = build_interior_sdf((400, 400), (0, 0), [])
    st_big = StyleSpec(orientation="horizontal", source_font_size_px=24.0)
    estimate_char_budget_from_sdf(big_sdf, "", 24.0, st_big)  # warm font cache
    t0 = time.perf_counter()
    for _ in range(10):
        estimate_char_budget_from_sdf(big_sdf, "", 24.0, st_big)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 10.0
    assert elapsed_ms < 25.0

    # Sparse large-ROI conserve_optical_ink_mass matches target ink density accurately
    dense_style = StyleSpec(
        orientation="horizontal",
        font_category="gothic",
        stroke_width_px=5.0,
        ink_density_orig=0.018,
    )
    sparse_alpha = np.zeros((220, 280), dtype=np.float32)
    _, small_alpha = solve_sdf_knuth_plass_layout(
        target_text="轰隆隆震耳欲聋",
        ruby_pairs=[],
        sdf=np.full((90, 180), 40.0, dtype=np.float32),
        pole_xy=(90.0, 45.0),
        style=dense_style,
    )
    sparse_alpha[65:155, 50:230] = small_alpha
    comp_alpha, out_st = conserve_optical_ink_mass(sparse_alpha, dense_style, over_artwork=True)
    assert out_st.hollow_mode in ("double_outline_hollow", "stroke_thinned")
    assert abs(float(np.mean(comp_alpha)) - 0.018) < 0.008
    # Untouched margin remains exact zero
    assert np.all(comp_alpha[:55, :] == 0.0)

    # Already within tolerance -> skips Brent root-finding cleanly
    rho_now = float(np.mean(sparse_alpha))
    near_style = dense_style.model_copy(update={"ink_density_orig": rho_now * 0.98})
    unchanged_alpha, unchanged_st = conserve_optical_ink_mass(sparse_alpha, near_style, over_artwork=True)
    assert unchanged_st.hollow_mode == "solid"
    assert np.array_equal(unchanged_alpha, sparse_alpha)

    # Faint disconnected component (0 < alpha < 0.45) outside core bbox is preserved in hollow mode
    faint_box = np.zeros((220, 280), dtype=np.float32)
    faint_box[65:155, 50:230] = small_alpha
    faint_box[15:22, 15:22] = 0.25
    hollow_style = dense_style.model_copy(update={"hollow_mode": "double_outline_hollow", "ink_density_orig": 0.02})
    hollow_out, _ = conserve_optical_ink_mass(faint_box, hollow_style, over_artwork=True)
    assert np.allclose(hollow_out[15:22, 15:22], 0.25)


def test_unmix_and_z2_fast_paths_preserve_exact_behavior():
    # 1. Sparse semi-transparent text on large ROI unmixes accurately and leaves outer margin bit-exact
    H, W = 240, 280
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:, :, 0] = np.clip(110 + 55 * np.sin(xx * 0.08), 0, 255).astype(np.uint8)
    bg[:, :, 1] = np.clip(130 + 55 * np.cos(yy * 0.10), 0, 255).astype(np.uint8)
    bg[:, :, 2] = np.clip(120 + 45 * np.sin((xx + yy) * 0.07), 0, 255).astype(np.uint8)

    fg_color = np.array([248.0, 242.0, 38.0], dtype=np.float32)
    alpha_gt = np.zeros((H, W), dtype=np.float32)
    cv2.rectangle(alpha_gt, (95, 85), (175, 145), 0.56, -1)
    blended = np.clip(
        np.rint(alpha_gt[:, :, None] * fg_color + (1.0 - alpha_gt[:, :, None]) * bg.astype(np.float32)),
        0,
        255,
    ).astype(np.uint8)

    is_semi, clean_B, _, _, med_a = unmix_optical_alpha_layer(blended, alpha_gt)
    assert is_semi is True
    assert 0.35 <= med_a <= 0.80
    assert np.array_equal(clean_B[alpha_gt == 0], blended[alpha_gt == 0])
    rec_rmse = float(np.sqrt(np.mean((clean_B[alpha_gt > 0].astype(np.float32) - bg[alpha_gt > 0].astype(np.float32)) ** 2)))
    assert rec_rmse < 15.0

    # 2. Flat enclosed speech bubble early-exits cleanly with zero Z2 mask
    flat_bubble = np.full((140, 140, 3), 252, dtype=np.uint8)
    sub = np.zeros((140, 140), dtype=np.uint8)
    cv2.circle(sub, (70, 70), 60, 1, -1)
    b_alpha = np.zeros((140, 140), dtype=np.float32)
    b_alpha[45:95, 55:62] = 1.0
    b_alpha[45:95, 78:85] = 1.0
    flat_bubble[b_alpha > 0.5] = [18, 18, 20]
    z2_flat, amb_flat = extract_z2_foreground_occluders(
        flat_bubble, b_alpha, enclosed_bubble=True, substrate_mask=sub
    )
    assert not np.any(z2_flat > 0)
    assert amb_flat == 0.0
