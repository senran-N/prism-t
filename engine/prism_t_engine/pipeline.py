from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
import hashlib
import os
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import chi2

from .codec import decode_image_b64, encode_png_b64
from .learned import comic_text_evidence, learned_clean_plate_full_context, refine_text_alpha
from .deep_fusion import (
    apply_exemplar_stroke_shader,
    composite_node_layers,
    compute_diegetic_homography,
    decompose_intrinsic_surface_shading,
    extract_stroke_style_and_halo,
    extract_z2_foreground_occluders,
    unmix_optical_alpha_layer,
    warp_diegetic_layer_perspective_or_spine,
)
from .layout_dag import build_narrative_dag_and_ground_speakers
from .ocr_tps import detect_and_recognize_page_regions, recognize_region_crop
from .source_evidence import collect_page_source_readings, collect_source_readings
from .preprocess import (
    detect_format_profile,
    detect_spread_gutter,
    estimate_noise_sigma,
    estimate_substrate_noise,
    substrate_squared_distance,
    estimate_paper_tint,
    otsu_threshold_and_eta,
)
from .restoration import align_cross_edition_clean_plate, restore_clean_plate_roi, repair_observed_paper_residue, reconstruction_unknown, unmix_stroke_boundary
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectLayoutResponse,
    PageDOM,
    PageMetrics,
    PatchLayers,
    RenderRequest,
    RenderResponse,
    RestoreRequest,
    StyleSpec,
    TextNode,
    UncertaintySpec,
)
from .segmentation import (
    analyze_bubble_contour_and_tail,
    build_stroke_trimap_and_alpha,
    estimate_source_font_scale,
    separate_kanji_ruby_components,
)
from .typesetter_sdf import (
    build_interior_sdf,
    conserve_optical_ink_mass,
    estimate_char_budget_from_sdf,
    load_font,
    solve_sdf_knuth_plass_layout,
)

# Invalidate persisted raster results whenever any rendering implementation
# changes. Cache data stays in PageDOM; the operator has no business state.
_RENDER_IMPLEMENTATION = hashlib.sha256(b"".join(
    Path(__file__).with_name(name).read_bytes()
    for name in ("pipeline.py", "typesetter_sdf.py", "deep_fusion.py", "codec.py", "schemas.py")
)).digest()

log = logging.getLogger("uvicorn.error")
_font_metrics_lock = threading.Lock()

_LAYOUT_CACHE_LOCK = threading.Lock()
_LAYOUT_CACHE: OrderedDict[str, dict] = OrderedDict()
_LAYOUT_CACHE_MAXSIZE = 8


def _cache_layout_state(token: str, state: dict) -> None:
    with _LAYOUT_CACHE_LOCK:
        _LAYOUT_CACHE[token] = state
        _LAYOUT_CACHE.move_to_end(token)
        while len(_LAYOUT_CACHE) > _LAYOUT_CACHE_MAXSIZE:
            _LAYOUT_CACHE.popitem(last=False)


def _get_layout_state(token: str) -> dict | None:
    if not token:
        return None
    with _LAYOUT_CACHE_LOCK:
        state = _LAYOUT_CACHE.get(token)
        if state is not None:
            _LAYOUT_CACHE.move_to_end(token)
        return state


def _estimate_node_font_size(
    reg: dict,
    source_ink: np.ndarray,
    tx: int,
    ty: int,
    tw: int,
    th: int,
    rx0: int,
    ry0: int,
    rw: int,
    rh: int,
    src_fs_init: float,
    font_category: str = "gothic",
) -> float:
    bx_rel0 = max(0, tx - rx0)
    by_rel0 = max(0, ty - ry0)
    bx_rel1 = min(rw, bx_rel0 + tw)
    by_rel1 = min(rh, by_rel0 + th)
    active_ink = source_ink[by_rel0:by_rel1, bx_rel0:bx_rel1].astype(np.uint8)
    if np.count_nonzero(active_ink) >= 6:
        base_m, _ = separate_kanji_ruby_components(active_ink, vertical=(reg["orientation"] == "vertical"))
        est_fs = estimate_source_font_scale(
            base_m, vertical=(reg["orientation"] == "vertical"), source_text=reg["source_text"]
        )
        col_scale_px = float(reg.get("col_scale_px") or 0.0)
        if col_scale_px > 6.0 and len(reg.get("column_boxes") or []) >= 1:
            col_em = col_scale_px * 0.80
            if est_fs > 1.55 * col_em or est_fs < 0.45 * col_em:
                est_fs = col_em
            else:
                est_fs = 0.65 * est_fs + 0.35 * col_em
        out_fs = float(est_fs)
    else:
        col_scale_px = float(reg.get("col_scale_px") or 0.0)
        out_fs = float(col_scale_px * 0.80) if col_scale_px > 6.0 else float(src_fs_init)

    if reg["orientation"] == "horizontal" and reg.get("line_polygons"):
        line_heights = []
        for quad in reg["line_polygons"]:
            edges = np.roll(np.array(quad, float), -1, axis=0) - quad
            lengths = np.linalg.norm(edges, axis=1)
            along = int(np.argmax(np.abs(edges[:, 0]) / np.maximum(lengths, np.finfo(float).eps)))
            width, height = lengths[along], lengths[(along + 1) % 4]
            if width > height:
                line_heights.append(height)
        if line_heights:
            reference = "田" if any("\u3000" <= c <= "\u9fff" for c in reg["source_text"]) else "H"
            with _font_metrics_lock:
                metric = load_font(font_category, float(np.median(line_heights)), required_text=reference)
                cap = metric.getbbox(reference, anchor="ls")
            cap_h = max(1.0, float(cap[3] - cap[1]))
            out_fs = float(np.median(line_heights)) * metric.size / cap_h
    return out_fs


def _tag_low_confidence_candidates(items: list[dict], text_probability: np.ndarray | None) -> None:
    """Flag floating regions that are statistical outliers in peak probability or ink support."""
    if len(items) < 2:
        return
    probs = [float(m["_peak_prob"]) for m in items if m.get("_peak_prob") is not None]
    inks = [int(m["_ink_pixels"]) for m in items if int(m.get("_ink_pixels", 0)) > 0]
    font_sizes = [
        float(m["style"].source_font_size_px)
        for m in items
        if m.get("style") is not None and float(m["style"].source_font_size_px) > 0.0
    ]

    page_cut = float(otsu_threshold_and_eta(text_probability)[0]) if text_probability is not None else None
    med_p = float(np.median(probs)) if probs else None
    prob_fence = None
    if len(probs) >= 3:
        q1_p, q3_p = np.quantile(probs, [0.25, 0.75])
        prob_fence = max(0.0, float(q1_p - 1.5 * (q3_p - q1_p)))

    med_ink = float(np.median(inks)) if len(inks) >= 2 else None
    med_fs = float(np.median(font_sizes)) if font_sizes else 0.0

    for m in items:
        if m.get("bubble_kind") not in ("floating_sfx", "diegetic_surface"):
            continue
        peak_p = m.get("_peak_prob")
        ink_px = int(m.get("_ink_pixels", 0))
        char_cnt = int(m.get("_char_count", 0))
        ocr_ent = float(m["uncertainty"].ocr_entropy)

        low_prob = False
        if peak_p is not None and med_p is not None and len(probs) >= 3:
            fence = min(0.65 * med_p, prob_fence) if prob_fence is not None else 0.65 * med_p
            low_prob = bool(peak_p < fence or (page_cut is not None and peak_p <= page_cut))

        tiny_ink = False
        if med_ink is not None and ink_px < 0.15 * med_ink and (char_cnt <= 1 or ocr_ent > 0.35):
            tiny_ink = True
        elif med_fs > 0.0 and ink_px < 0.20 * (med_fs ** 2) and char_cnt <= 1:
            tiny_ink = True

        if low_prob or tiny_ink:
            m["_low_confidence_candidate"] = True
            if "low_confidence_candidate" not in m["uncertainty"].reasons:
                m["uncertainty"].reasons.append("low_confidence_candidate")


def _compute_layout_state(
    req: AnalyzeRequest, pre_detected_regions: list[dict] | None = None
) -> tuple[PageDOM, dict]:
    t0 = time.perf_counter()
    raw_rgb = decode_image_b64(req.image_png_b64, mode="RGB")
    H, W = raw_rgb.shape[:2]

    profile, is_color = detect_format_profile(raw_rgb, req.format_profile)
    paper_tint = estimate_paper_tint(raw_rgb)
    if req.is_spread or req.gutter_x is not None:
        is_spread = req.is_spread
        gutter_x = req.gutter_x if req.gutter_x is not None else W // 2
    else:
        is_spread, gutter_x = detect_spread_gutter(raw_rgb)

    default_vertical = profile == "manga_rtl"
    log.info("analyze %dx%d: profile=%s; loading text segmentation", W, H, profile)
    text_probability, detected_lines, detected_blocks = comic_text_evidence(raw_rgb) if pre_detected_regions is None else (None, None, None)
    log.info("analyze: segmentation ready after %.2fs; recognizing text", time.perf_counter() - t0)
    regions = (
        pre_detected_regions
        if pre_detected_regions is not None
        else detect_and_recognize_page_regions(raw_rgb, default_vertical=default_vertical, text_probability=text_probability, recognize_text=req.recognize_text,
                                               detected_lines=detected_lines, detected_blocks=detected_blocks)
    )
    log.info("analyze: OCR returned %d regions after %.2fs", len(regions), time.perf_counter() - t0)
    if req.format_profile == "auto" and regions:
        vertical_chars = sum(len(r["source_text"]) for r in regions if r["orientation"] == "vertical")
        horizontal_chars = sum(len(r["source_text"]) for r in regions if r["orientation"] != "vertical")
        has_kana = any("\u3040" <= ch <= "\u30ff" for r in regions for ch in r["source_text"])
        if profile != "webtoon_vertical" and has_kana and vertical_chars > horizontal_chars:
            profile = "manga_rtl"
            default_vertical = True

    # Pre-extract enclosing speech bubble contours and partition conjoined figure-8 multi-lobe bubbles via Voronoi!
    bubble_infos: list[list] = []
    for reg in regions:
        b_poly, b_kind, t_tip, t_dir = analyze_bubble_contour_and_tail(raw_rgb, reg["bbox"], reg.get("ink_mask_roi"))
        if b_poly and reg.get("ink_mask_roi") is not None:
            x, y, w, h = reg["bbox"]
            local_bubble = np.zeros((h, w), np.uint8)
            cv2.fillPoly(local_bubble, [np.rint(np.array(b_poly) - [x, y]).astype(np.int32)], 1)
            observed = reg["ink_mask_roi"] > 0
            if observed.any() and np.count_nonzero(observed & (local_bubble > 0)) < np.count_nonzero(observed) / 2:
                b_poly, b_kind, t_tip, t_dir = [], "floating_sfx", None, None
        if b_poly and text_probability is not None:
            xs, ys = np.array(b_poly).T
            x0, y0 = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
            x1, y1 = min(W, int(np.ceil(xs.max())) + 1), min(H, int(np.ceil(ys.max())) + 1)
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(mask, [np.rint(np.array(b_poly) - [x0, y0]).astype(np.int32)], 1)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
            probability = text_probability[y0:y1, x0:x1]
            cut, _ = otsu_threshold_and_eta(probability)
            observed_text = probability > cut
            radius = distance_transform_edt(observed_text)
            radius = float(np.median(radius[observed_text])) if observed_text.any() else 0
            background = mask & (distance_transform_edt(~observed_text) > radius)
            if background.any():
                lab = cv2.cvtColor(raw_rgb[y0:y1, x0:x1].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
                samples = lab[background]
                rgb_crop = raw_rgb[y0:y1, x0:x1]
                rgb_sigma = np.array([max(estimate_noise_sigma(rgb_crop[:, :, k]), 1 / np.sqrt(12)) for k in range(3)])
                center = np.median(rgb_crop[background], axis=0).astype(np.float32)
                probes = np.clip(np.vstack((center + np.eye(3), center - np.eye(3))), 0, 255).astype(np.float32)
                probe_lab = cv2.cvtColor((probes / 255).reshape(1, 6, 3), cv2.COLOR_RGB2LAB)[0]
                jacobian = (probe_lab[:3] - probe_lab[3:]).T / 2
                propagated = np.sqrt(np.sum((jacobian * rgb_sigma) ** 2, axis=1))
                sigma = np.maximum(propagated, [estimate_noise_sigma(lab[:, :, k]) for k in range(3)])
                sigma = np.maximum(sigma, np.finfo(np.float32).eps)
                design = np.column_stack((np.ones(len(samples)), samples[:, 0] - np.median(samples[:, 0])))
                prediction = design @ np.linalg.lstsq(design, samples[:, 1:], rcond=None)[0]
                deviations = np.sum(((samples[:, 1:] - prediction) / sigma[1:]) ** 2, axis=1)
                if np.quantile(deviations, 0.75) > chi2.ppf(0.75, 2):
                    b_poly, b_kind, t_tip, t_dir = [], "floating_sfx", None, None
        bubble_infos.append([b_poly, b_kind, t_tip, t_dir])

    n_reg = len(regions)
    regions = [dict(r) for r in regions]
    merged_away: set[int] = set()
    enclosed_indices = [
        i
        for i in range(n_reg)
        if bubble_infos[i][0] and bubble_infos[i][1] not in ("floating_sfx", "diegetic_surface")
    ]
    if len(enclosed_indices) >= 2:
        poly_bboxes = []
        for i in range(n_reg):
            bp = bubble_infos[i][0]
            if bp:
                pxs = [p[0] for p in bp]
                pys = [p[1] for p in bp]
                poly_bboxes.append((min(pxs), min(pys), max(pxs), max(pys)))
            else:
                poly_bboxes.append((0.0, 0.0, 0.0, 0.0))

        b_parent = {i: i for i in enclosed_indices}

        def b_find(k: int) -> int:
            while b_parent[k] != k:
                b_parent[k] = b_parent[b_parent[k]]
                k = b_parent[k]
            return k

        for a_i in range(len(enclosed_indices)):
            ia = enclosed_indices[a_i]
            ax0, ay0, ax1, ay1 = poly_bboxes[ia]
            a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
            for b_i in range(a_i + 1, len(enclosed_indices)):
                ib = enclosed_indices[b_i]
                bx0, by0, bx1, by1 = poly_bboxes[ib]
                b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
                iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
                ih = max(0.0, min(ay1, by1) - max(ay0, by0))
                if (iw * ih) / min(a_area, b_area) <= 0.45:
                    continue
                gx0 = max(0, int(np.floor(min(ax0, bx0))))
                gy0 = max(0, int(np.floor(min(ay0, by0))))
                gx1 = min(W, int(np.ceil(max(ax1, bx1))) + 1)
                gy1 = min(H, int(np.ceil(max(ay1, by1))) + 1)
                if gx1 - gx0 < 4 or gy1 - gy0 < 4:
                    continue
                ma_m = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
                mb_m = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
                cv2.fillPoly(ma_m, [np.array([[p[0] - gx0, p[1] - gy0] for p in bubble_infos[ia][0]], dtype=np.int32)], 1)
                cv2.fillPoly(mb_m, [np.array([[p[0] - gx0, p[1] - gy0] for p in bubble_infos[ib][0]], dtype=np.int32)], 1)
                denom_m = min(int(np.count_nonzero(ma_m)), int(np.count_nonzero(mb_m)))
                if denom_m > 0 and (np.count_nonzero(ma_m & mb_m) / float(denom_m)) > 0.50:
                    ra, rb = b_find(ia), b_find(ib)
                    if ra != rb:
                        b_parent[rb] = ra

        shared_clusters: dict[int, list[int]] = {}
        for i in enclosed_indices:
            shared_clusters.setdefault(b_find(i), []).append(i)

        for members in shared_clusters.values():
            if len(members) < 2:
                continue
            ux0 = max(0, int(np.floor(min(poly_bboxes[m][0] for m in members))) - 2)
            uy0 = max(0, int(np.floor(min(poly_bboxes[m][1] for m in members))) - 2)
            ux1 = min(W, int(np.ceil(max(poly_bboxes[m][2] for m in members))) + 2)
            uy1 = min(H, int(np.ceil(max(poly_bboxes[m][3] for m in members))) + 2)
            uh, uw = uy1 - uy0, ux1 - ux0
            if uh < 8 or uw < 8:
                continue
            union_mask = np.zeros((uh, uw), dtype=np.uint8)
            for m in members:
                pts_m = np.array([[p[0] - ux0, p[1] - uy0] for p in bubble_infos[m][0]], dtype=np.int32)
                cv2.fillPoly(union_mask, [pts_m], 1)

            sdf_u = distance_transform_edt(union_mask > 0)
            max_u = max(1.0, float(np.max(sdf_u)))
            sdf_s = cv2.GaussianBlur(
                sdf_u.astype(np.float32), (0, 0), max(1.2, 0.06 * max_u)
            ) * (union_mask > 0).astype(np.float32)

            lobe_peaks: dict[int, tuple[float, float, float]] = {}
            for m in members:
                mtx, mty, mtw, mth = regions[m]["bbox"]
                ry_a, ry_b = max(0, mty - uy0), min(uh, mty - uy0 + mth)
                rx_a, rx_b = max(0, mtx - ux0), min(uw, mtx - ux0 + mtw)
                sub_s = sdf_s[ry_a:ry_b, rx_a:rx_b]
                if sub_s.size > 0 and float(np.max(sub_s)) > 0.0:
                    flat_i = int(np.argmax(sub_s))
                    py_l, px_l = divmod(flat_i, sub_s.shape[1])
                    pk_x, pk_y = int(rx_a + px_l), int(ry_a + py_l)
                else:
                    pk_x = int(np.clip(round(mtx - ux0 + 0.5 * mtw), 0, uw - 1))
                    pk_y = int(np.clip(round(mty - uy0 + 0.5 * mth), 0, uh - 1))

                for _ in range(uh + uw):
                    cur_val = float(sdf_s[pk_y, pk_x])
                    y_lo, y_hi = max(0, pk_y - 1), min(uh, pk_y + 2)
                    x_lo, x_hi = max(0, pk_x - 1), min(uw, pk_x + 2)
                    patch_s = sdf_s[y_lo:y_hi, x_lo:x_hi]
                    best_f = int(np.argmax(patch_s))
                    dy_l, dx_l = divmod(best_f, patch_s.shape[1])
                    nx_l, ny_l = x_lo + dx_l, y_lo + dy_l
                    if float(sdf_s[ny_l, nx_l]) <= cur_val + 1e-6:
                        break
                    pk_x, pk_y = nx_l, ny_l
                pk_r = max(1.0, float(sdf_s[pk_y, pk_x]))
                lobe_peaks[m] = (float(pk_x), float(pk_y), pk_r)

            l_parent = {m: m for m in members}

            def l_find(k: int) -> int:
                while l_parent[k] != k:
                    l_parent[k] = l_parent[l_parent[k]]
                    k = l_parent[k]
                return k

            for i_a in range(len(members)):
                ma = members[i_a]
                pax, pay, ps_a = lobe_peaks[ma]
                for i_b in range(i_a + 1, len(members)):
                    mb = members[i_b]
                    pbx, pby, ps_b = lobe_peaks[mb]
                    peak_dist = float(np.hypot(pax - pbx, pay - pby))
                    ts = np.linspace(0.0, 1.0, 32)
                    xs_b = np.clip(np.rint(pax + ts * (pbx - pax)).astype(np.int32), 0, uw - 1)
                    ys_b = np.clip(np.rint(pay + ts * (pby - pay)).astype(np.int32), 0, uh - 1)
                    bridge_min = float(np.min(sdf_u[ys_b, xs_b]))
                    if peak_dist <= 0.42 * min(ps_a, ps_b) or (
                        bridge_min >= 0.85 * min(ps_a, ps_b) and bridge_min > 2.0
                    ):
                        ra, rb = l_find(ma), l_find(mb)
                        if ra != rb:
                            l_parent[rb] = ra

            lobe_groups: dict[int, list[int]] = {}
            for m in members:
                lobe_groups.setdefault(l_find(m), []).append(m)

            surviving_members: list[int] = []
            for grp in lobe_groups.values():
                rep = grp[0]
                surviving_members.append(rep)
                if len(grp) >= 2:
                    is_v = regions[rep]["orientation"] == "vertical"
                    grp_sorted = sorted(
                        grp,
                        key=lambda idx_m: (
                            -regions[idx_m]["bbox"][0] if is_v else regions[idx_m]["bbox"][1]
                        ),
                    )
                    gx0 = min(regions[idx_m]["bbox"][0] for idx_m in grp)
                    gy0 = min(regions[idx_m]["bbox"][1] for idx_m in grp)
                    gx1 = max(regions[idx_m]["bbox"][0] + regions[idx_m]["bbox"][2] for idx_m in grp)
                    gy1 = max(regions[idx_m]["bbox"][1] + regions[idx_m]["bbox"][3] for idx_m in grp)
                    gw, gh = gx1 - gx0, gy1 - gy0
                    comb_ink = np.zeros((gh, gw), dtype=np.uint8)
                    comb_pts = []
                    comb_rubies = []
                    comb_lines, comb_quads, comb_detectors = [], [], []
                    for idx_m in grp_sorted:
                        bx_m, by_m, bw_m, bh_m = regions[idx_m]["bbox"]
                        im_sub = regions[idx_m].get("ink_mask_roi")
                        if im_sub is not None and im_sub.shape == (bh_m, bw_m):
                            comb_ink[by_m - gy0 : by_m - gy0 + bh_m, bx_m - gx0 : bx_m - gx0 + bw_m] |= (
                                im_sub > 0
                            ).astype(np.uint8)
                        comb_pts.extend(regions[idx_m].get("polygon", []))
                        comb_rubies.extend(regions[idx_m].get("ruby_pairs", []))
                        comb_lines.extend(regions[idx_m].get("column_boxes", []))
                        comb_quads.extend(regions[idx_m].get("line_polygons", []))
                        comb_detectors.extend(regions[idx_m].get("detector_polygons", []))
                    sep = "" if is_v else " "
                    joined_txt = sep.join(regions[idx_m]["source_text"] for idx_m in grp_sorted if regions[idx_m]["source_text"])
                    if pre_detected_regions is None and req.recognize_text:
                        crop_m = raw_rgb[max(0, gy0 - 3) : min(H, gy1 + 3), max(0, gx0 - 3) : min(W, gx1 + 3)]
                        rec_t, rec_e = recognize_region_crop(crop_m, vertical=is_v, prefer_japanese=default_vertical)
                        if rec_t:
                            joined_txt = rec_t
                            regions[rep]["ocr_entropy"] = rec_e
                    regions[rep]["bbox"] = [gx0, gy0, gw, gh]
                    regions[rep]["ink_mask_roi"] = comb_ink
                    regions[rep]["source_text"] = joined_txt
                    regions[rep]["ruby_pairs"] = comb_rubies
                    regions[rep]["column_boxes"] = comb_lines
                    regions[rep]["line_polygons"] = comb_quads
                    regions[rep]["detector_polygons"] = comb_detectors
                    if len(comb_pts) >= 3:
                        hull_m = cv2.convexHull(np.array(comb_pts, dtype=np.float32))
                        regions[rep]["polygon"] = [[float(p[0][0]), float(p[0][1])] for p in hull_m]
                    for drop_m in grp[1:]:
                        merged_away.add(drop_m)

            if len(surviving_members) >= 2:
                yy_u, xx_u = np.meshgrid(np.arange(uh, dtype=np.float32), np.arange(uw, dtype=np.float32), indexing="ij")
                dist_stack = np.zeros((len(surviving_members), uh, uw), dtype=np.float32)
                for k_idx, m in enumerate(surviving_members):
                    pk_x, pk_y, pk_r = lobe_peaks[m]
                    dist_stack[k_idx] = np.hypot(xx_u - pk_x, yy_u - pk_y) / max(pk_r, 1.0)

                owner = np.argmin(dist_stack, axis=0)
                for k_idx, m in enumerate(surviving_members):
                    cell = ((union_mask > 0) & (owner == k_idx)).astype(np.uint8)
                    cnts, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if cnts:
                        best_c = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(best_c) >= 64:
                            ap = cv2.approxPolyDP(best_c, 0.75, True)
                            bubble_infos[m][0] = [[float(p[0][0] + ux0), float(p[0][1] + uy0)] for p in ap]
                    mtx_m, mty_m, mtw_m, mth_m = regions[m]["bbox"]
                    im_m = regions[m].get("ink_mask_roi")
                    if im_m is not None and im_m.shape == (mth_m, mtw_m):
                        gx_a, gy_a = max(0, mtx_m - ux0), max(0, mty_m - uy0)
                        gx_b, gy_b = min(uw, mtx_m + mtw_m - ux0), min(uh, mty_m + mth_m - uy0)
                        if gx_b > gx_a and gy_b > gy_a:
                            lx_a, ly_a = gx_a - (mtx_m - ux0), gy_a - (mty_m - uy0)
                            lx_b, ly_b = lx_a + (gx_b - gx_a), ly_a + (gy_b - gy_a)
                            im_new = np.zeros_like(im_m)
                            im_new[ly_a:ly_b, lx_a:lx_b] = im_m[ly_a:ly_b, lx_a:lx_b]
                            if np.any(im_new):
                                regions[m]["ink_mask_roi"] = im_new

    # Deduplicate tiny floating micro-fragments wholly enclosed inside an existing larger region
    # (e.g. a stray sub-character box spawned inside a multi-column vertical dialogue block).
    if pre_detected_regions is None and len(regions) >= 2:
        for i_sub in range(len(regions)):
            if i_sub in merged_away or bubble_infos[i_sub][1] != "floating_sfx":
                continue
            ax, ay, aw, ah = regions[i_sub]["bbox"]
            area_a = float(max(aw * ah, 1))
            for j_sup in range(len(regions)):
                if j_sup == i_sub or j_sup in merged_away:
                    continue
                bx, by, bw_j, bh_j = regions[j_sup]["bbox"]
                area_b = float(max(bw_j * bh_j, 1))
                if area_a >= 0.25 * area_b:
                    continue
                ix0, iy0 = max(ax, bx), max(ay, by)
                ix1, iy1 = min(ax + aw, bx + bw_j), min(ay + ah, by + bh_j)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = float((ix1 - ix0) * (iy1 - iy0))
                if inter / area_a >= 0.90:
                    im_a = regions[i_sub].get("ink_mask_roi")
                    im_b = regions[j_sup].get("ink_mask_roi")
                    if im_a is not None and im_b is not None and im_a.shape == (ah, aw) and im_b.shape == (bh_j, bw_j):
                        im_b[iy0 - by : iy1 - by, ix0 - bx : ix1 - bx] |= (
                            im_a[iy0 - ay : iy1 - ay, ix0 - ax : ix1 - ax] > 0
                        ).astype(np.uint8)
                    merged_away.add(i_sub)
                    break

    stroke_owners = np.zeros((H, W), np.int32)
    for region_index, region in enumerate(regions):
        x, y, width, height = region["bbox"]
        ink = region.get("ink_mask_roi")
        if region_index not in merged_away and ink is not None:
            stroke_owners[y:y + height, x:x + width] += ink > 0

    def _prepare_node_layout(item: tuple[int, dict]) -> dict | None:
        idx, reg = item
        tx, ty, tw, th = reg["bbox"]
        bubble_poly, bubble_kind, tail_tip, tail_dir = bubble_infos[idx]
        is_vert_reg = reg["orientation"] == "vertical"
        ink_sub = reg.get("ink_mask_roi")
        if ink_sub is not None and ink_sub.shape == (th, tw) and np.any(ink_sub > 0):
            bm_pre, _ = separate_kanji_ruby_components((ink_sub > 0).astype(np.uint8), vertical=is_vert_reg)
            src_fs_init = estimate_source_font_scale(bm_pre, vertical=is_vert_reg, source_text=reg["source_text"])
        else:
            src_fs_init = float(min(tw, th))

        xs_all = [float(tx), float(tx + tw)]
        ys_all = [float(ty), float(ty + th)]
        if bubble_poly:
            for px, py in bubble_poly:
                xs_all.append(px)
                ys_all.append(py)
            pad_x = max(4, int(round(min(tw, th) * 0.12)))
            pad_y = pad_x
        else:
            pad_x = max(14, int(round(max(src_fs_init * 1.30, tw * 0.28))))
            pad_y = max(12, int(round(max(src_fs_init * 0.90, th * 0.28))))

        rx0 = max(0, int(np.floor(min(xs_all))) - pad_x)
        ry0 = max(0, int(np.floor(min(ys_all))) - pad_y)
        rx1 = min(W, int(np.ceil(max(xs_all))) + pad_x)
        ry1 = min(H, int(np.ceil(max(ys_all))) + pad_y)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < 4 or rh < 4:
            return None

        roi_rgb = raw_rgb[ry0:ry1, rx0:rx1]
        coarse_roi = np.zeros((rh, rw), dtype=np.uint8)
        if ink_sub is not None and ink_sub.shape == (th, tw):
            coarse_roi[ty - ry0 : ty - ry0 + th, tx - rx0 : tx - rx0 + tw] = (ink_sub > 0).astype(np.uint8)
        else:
            roi_g0 = cv2.cvtColor(roi_rgb[ty - ry0 : ty - ry0 + th, tx - rx0 : tx - rx0 + tw], cv2.COLOR_RGB2GRAY).astype(np.float32)
            ink_threshold, _ = otsu_threshold_and_eta(roi_g0)
            df0 = np.mean(roi_g0 < ink_threshold) <= 0.55
            coarse_roi[ty - ry0 : ty - ry0 + th, tx - rx0 : tx - rx0 + tw] = (
                (roi_g0 < ink_threshold) if df0 else (roi_g0 > ink_threshold)
            ).astype(np.uint8)
        orig_ink_roi = coarse_roi.copy()

        bmask_safe: np.ndarray | None = None
        bmask_sub: np.ndarray | None = None
        bmask_inner: np.ndarray | None = None
        if bubble_poly and bubble_kind not in ("floating_sfx", "diegetic_surface"):
            bmask = np.zeros((rh, rw), dtype=np.uint8)
            bpts = np.array([[p[0] - rx0, p[1] - ry0] for p in bubble_poly], dtype=np.int32)
            cv2.fillPoly(bmask, [bpts], 1)
            bmask_inner = cv2.erode(bmask, np.ones((3, 3), np.uint8))
            bmask_sub = cv2.erode(bmask, np.ones((5, 5), np.uint8))
            if not np.any(bmask_sub):
                bmask_sub = bmask_inner
            wall_r = max(3, int(round(min(rw, rh) * 0.045)))
            ell_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wall_r + 1, 2 * wall_r + 1))
            bmask_safe = cv2.erode(bmask, ell_k)
            if not np.any(bmask_safe):
                bmask_safe = bmask_sub
            if np.any(bmask_safe):
                other_mask = np.zeros((rh, rw), dtype=np.uint8)
                for j_idx, j_reg in enumerate(regions):
                    if j_idx == idx or j_idx in merged_away:
                        continue
                    jx, jy, jw, jh = j_reg["bbox"]
                    jx0, jy0, jx1, jy1 = jx - rx0, jy - ry0, jx + jw - rx0, jy + jh - ry0
                    if jx1 <= 0 or jx0 >= rw or jy1 <= 0 or jy0 >= rh:
                        continue
                    j_ink = j_reg.get("ink_mask_roi")
                    if j_ink is not None and j_ink.shape == (jh, jw):
                        j_dil = cv2.dilate((j_ink > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
                        sx0, sy0 = max(0, jx0), max(0, jy0)
                        sx1, sy1 = min(rw, jx1), min(rh, jy1)
                        if sx1 > sx0 and sy1 > sy0:
                            other_mask[sy0:sy1, sx0:sx1] |= j_dil[
                                sy0 - jy0 : sy1 - jy0, sx0 - jx0 : sx1 - jx0
                            ]
                    else:
                        ox0, oy0 = max(0, jx0 - 3), max(0, jy0 - 3)
                        ox1, oy1 = min(rw, jx1 + 3), min(rh, jy1 + 3)
                        if ox1 > ox0 and oy1 > oy0:
                            other_mask[oy0:oy1, ox0:ox1] = 1

                roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                lobe_vals = roi_gray[bmask_safe > 0]
                t_lobe, eta_lobe = otsu_threshold_and_eta(lobe_vals)
                dark_fg = np.mean(lobe_vals < t_lobe) <= 0.50
                bg_part = (bmask_safe > 0) & ((roi_gray >= t_lobe) if dark_fg else (roi_gray <= t_lobe))
                sub_med = (
                    np.median(roi_rgb[bg_part].astype(np.float32), axis=0)
                    if np.any(bg_part)
                    else np.median(roi_rgb[bmask_safe > 0].astype(np.float32), axis=0)
                )
                channel_sigma = estimate_substrate_noise(roi_rgb, bg_part)
                background_samples = roi_rgb[bg_part] if bg_part.any() else np.empty((0, 3))
                is_uniform_sub = bool(
                    len(background_samples)
                    and np.quantile(substrate_squared_distance(background_samples, sub_med, channel_sigma), 0.75) <= chi2.ppf(0.75, 3)
                )

                lobe_ink = np.zeros((rh, rw), dtype=np.uint8)
                if is_uniform_sub:
                    lobe_dev = substrate_squared_distance(roi_rgb, sub_med, channel_sigma)
                    critical = chi2.isf(0.05 / max(1, np.count_nonzero(bmask_safe)), 3)
                    lobe_ink = (
                        (lobe_dev > critical) & (bmask_safe > 0) & (other_mask == 0)
                    ).astype(np.uint8)
                elif eta_lobe > 0:
                    lobe_ink = (
                        ((roi_gray < t_lobe) if dark_fg else (roi_gray > t_lobe))
                        & (bmask_safe > 0)
                        & (other_mask == 0)
                    ).astype(np.uint8)

                if np.any(lobe_ink > 0):
                    iy, ix = np.nonzero(lobe_ink > 0)
                    if iy.size >= 12:
                        nx0, ny0 = int(ix.min()) + rx0, int(iy.min()) + ry0
                        nx1, ny1 = int(ix.max()) + 1 + rx0, int(iy.max()) + 1 + ry0
                        if pre_detected_regions is None and req.recognize_text and (
                            (tx - nx0) > 6 or (nx1 - (tx + tw)) > 6 or (ty - ny0) > 6 or (ny1 - (ty + th)) > 6
                        ):
                            sub_med_u8 = np.clip(np.rint(sub_med), 0, 255).astype(np.uint8)
                            clean_lobe_canvas = np.broadcast_to(sub_med_u8, roi_rgb.shape).copy()
                            allowed_lobe = (bmask_sub > 0) & (other_mask == 0)
                            clean_lobe_canvas[allowed_lobe] = roi_rgb[allowed_lobe]
                            lobe_crop = clean_lobe_canvas[
                                max(0, ny0 - ry0 - 3) : min(rh, ny1 - ry0 + 3),
                                max(0, nx0 - rx0 - 3) : min(rw, nx1 - rx0 + 3),
                            ]
                            full_txt, full_ent = recognize_region_crop(
                                lobe_crop,
                                vertical=(reg["orientation"] == "vertical"),
                                prefer_japanese=default_vertical,
                            )
                            if full_txt and (
                                len(full_txt) >= len(reg["source_text"])
                                or full_ent < float(reg.get("ocr_entropy", 1.0))
                            ):
                                reg = dict(reg)
                                reg["source_text"] = full_txt
                                reg["ocr_entropy"] = full_ent
                        old_x1, old_y1 = tx + tw, ty + th
                        tx, ty = min(tx, nx0), min(ty, ny0)
                        tw = max(old_x1, nx1) - tx
                        th = max(old_y1, ny1) - ty
                        reg = dict(reg)
                        reg["bbox"] = [tx, ty, tw, th]
                        tip_pad = max(4, int(round(min(tw, th) * 0.14)))
                        expanded_box = np.zeros((rh, rw), dtype=np.uint8)
                        expanded_box[
                            max(0, ty - ry0 - tip_pad) : min(rh, ty - ry0 + th + tip_pad),
                            max(0, tx - rx0 - tip_pad) : min(rw, tx - rx0 + tw + tip_pad),
                        ] = 1
                        coarse_roi = orig_ink_roi | (expanded_box & lobe_ink)
                elif np.any(coarse_roi & bmask_safe):
                    coarse_roi = orig_ink_roi | (coarse_roi & bmask_safe)

        text_domain = coarse_roi.copy()
        for polygon in reg.get("line_polygons") or []:
            local = np.rint(np.asarray(polygon) - [rx0, ry0]).astype(np.int32)
            cv2.fillPoly(text_domain, [local], 1)

        source_ink = coarse_roi > 0
        init_style = StyleSpec(orientation=reg["orientation"])
        init_style.source_font_size_px = _estimate_node_font_size(
            reg, source_ink, tx, ty, tw, th, rx0, ry0, rw, rh, src_fs_init, init_style.font_category
        )
        sdf_poly = bubble_poly if bubble_poly else [
            [float(tx), float(ty)],
            [float(tx + tw), float(ty)],
            [float(tx + tw), float(ty + th)],
            [float(tx), float(ty + th)],
        ]
        sdf_roi, _ = build_interior_sdf((rh, rw), (rx0, ry0), sdf_poly)
        with _font_metrics_lock:
            b_min, b_max = estimate_char_budget_from_sdf(
                sdf_roi, reg["source_text"], source_font_size_px=init_style.source_font_size_px, style=init_style
            )

        peak_prob = None
        mean_ink_prob = None
        if text_probability is not None:
            tp_roi = text_probability[ry0:ry1, rx0:rx1]
            if np.any(text_domain > 0):
                peak_prob = float(np.max(tp_roi[text_domain > 0]))
            else:
                peak_prob = float(np.max(tp_roi))
            if np.any(coarse_roi > 0):
                mean_ink_prob = float(np.mean(tp_roi[coarse_roi > 0]))
            else:
                mean_ink_prob = 0.0

        ocr_ent = float(reg.get("ocr_entropy", 0.0))
        reasons: list[str] = []
        if ocr_ent > 0.45:
            reasons.append("high_ocr_entropy")
        if reg.get("ruby_pairs"):
            reasons.append("gikun_ruby_dual_semantics")
        unc_score = float(np.clip(0.45 * ocr_ent, 0.0, 1.0))

        return {
            "id": f"node_{idx + 1:03d}",
            "bbox": reg["bbox"],
            "polygon": reg["polygon"],
            "bubble_polygon": bubble_poly,
            "tail_tip": tail_tip,
            "tail_dir": tail_dir,
            "spine_points": reg["spine_points"],
            "source_lines": reg.get("column_boxes") or [],
            "source_quadrilaterals": reg.get("line_polygons") or [],
            "detector_quadrilaterals": reg.get("detector_polygons") or [],
            "source_readings": reg.get("source_readings", []) if pre_detected_regions is not None else [],
            "bubble_kind": bubble_kind,
            "source_text": reg["source_text"],
            "recognition_pending": not req.recognize_text,
            "ruby_pairs": reg["ruby_pairs"],
            "char_budget_min": b_min,
            "char_budget_max": b_max,
            "style": init_style,
            "patches": PatchLayers(roi=[rx0, ry0, rw, rh]),
            "uncertainty": UncertaintySpec(
                score=unc_score,
                ocr_entropy=ocr_ent,
                reasons=reasons,
                needs_review=bool(unc_score >= 0.38),
            ),
            "_idx": idx,
            "_reg": reg,
            "_roi": (rx0, ry0, rw, rh),
            "_coarse_roi": coarse_roi,
            "_orig_ink_roi": orig_ink_roi,
            "_bmask_safe": bmask_safe,
            "_bmask_sub": bmask_sub,
            "_bmask_inner": bmask_inner,
            "_text_domain": text_domain,
            "_src_fs_init": src_fs_init,
            "_sdf_roi": sdf_roi,
            "_peak_prob": peak_prob,
            "_mean_ink_prob": mean_ink_prob,
            "_ink_pixels": int(np.count_nonzero(coarse_roi)),
            "_bbox_area": float(tw * th),
            "_char_count": len([c for c in reg["source_text"].strip() if not c.isspace()]),
            "_low_confidence_candidate": False,
        }

    active_items = [(idx, reg) for idx, reg in enumerate(regions) if idx not in merged_away]
    if len(active_items) <= 1:
        stage1_meta = [res for res in (_prepare_node_layout(it) for it in active_items) if res is not None]
    else:
        max_workers = min(len(active_items), max(2, min(6, (os.cpu_count() or 4) // 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            stage1_meta = [res for res in pool.map(_prepare_node_layout, active_items) if res is not None]

    if pre_detected_regions is None and text_probability is not None and len(stage1_meta) >= 2:
        page_cut, _ = otsu_threshold_and_eta(text_probability)
        if page_cut > 0.0 and any((m.get("_peak_prob") or 0.0) > page_cut for m in stage1_meta):
            kept_stage1 = [
                m for m in stage1_meta
                if not (
                    m.get("bubble_kind") in ("floating_sfx", "diegetic_surface")
                    and (m.get("_peak_prob") or 0.0) <= page_cut
                    and (m.get("_mean_ink_prob") or 0.0) < 0.15 * page_cut
                )
            ]
            if len(kept_stage1) != len(stage1_meta) and kept_stage1:
                stage1_meta = kept_stage1
                stroke_owners = np.zeros((H, W), np.int32)
                for m in stage1_meta:
                    rx0_m, ry0_m, rw_m, rh_m = m["_roi"]
                    stroke_owners[ry0_m : ry0_m + rh_m, rx0_m : rx0_m + rw_m] += m["_orig_ink_roi"] > 0

    skip_source_ocr = (not req.recognize_text) and os.environ.get(
        "PRISM_T_SKIP_SOURCE_OCR", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if pre_detected_regions is None and stage1_meta and not skip_source_ocr:
        has_ja_script = default_vertical or any(
            "\u3040" <= ch <= "\u30ff" for m in stage1_meta for ch in m["source_text"]
        )
        specs = [
            (m["source_quadrilaterals"], m["style"].orientation == "vertical", has_ja_script)
            for m in stage1_meta
        ]
        all_readings = collect_page_source_readings(raw_rgb, specs, format_profile=profile)
        for m, rds in zip(stage1_meta, all_readings):
            m["source_readings"] = rds

    _tag_low_confidence_candidates(stage1_meta, text_probability)

    stage1_meta = build_narrative_dag_and_ground_speakers(
        raw_rgb, stage1_meta, profile=profile, is_spread=is_spread, gutter_x=gutter_x
    )

    layout_nodes = [
        TextNode(**{k: v for k, v in m.items() if not k.startswith("_")})
        for m in stage1_meta
    ]
    layout_ms = (time.perf_counter() - t0) * 1000.0
    layout_dom = PageDOM(
        schema_version=1,
        width=W,
        height=H,
        format_profile=profile,
        is_color=is_color,
        paper_tint_rgb=paper_tint,
        seam_slices=req.seam_slices,
        is_spread=is_spread,
        gutter_x=gutter_x,
        nodes=layout_nodes,
        metrics=PageMetrics(
            total_nodes=len(layout_nodes),
            needs_review_nodes=sum(1 for n in layout_nodes if n.uncertainty.needs_review),
            analyze_ms=layout_ms,
        ),
    )

    cached_state = {
        "raw_rgb": raw_rgb,
        "H": H,
        "W": W,
        "profile": profile,
        "is_color": is_color,
        "paper_tint": paper_tint,
        "seam_slices": req.seam_slices,
        "is_spread": is_spread,
        "gutter_x": gutter_x,
        "default_vertical": default_vertical,
        "recognize_text": req.recognize_text,
        "text_probability": text_probability,
        "stroke_owners": stroke_owners,
        "stage1_meta": stage1_meta,
    }
    return layout_dom, cached_state


def _restore_from_state(
    state: dict,
    reference_edition_png_b64: str | None = None,
    exclude_node_ids: list[str] | None = None,
) -> AnalyzeResponse:
    t0 = time.perf_counter()
    raw_rgb: np.ndarray = state["raw_rgb"]
    H: int = state["H"]
    W: int = state["W"]
    profile = state["profile"]
    is_color: bool = state["is_color"]
    paper_tint: list[int] = state["paper_tint"]
    seam_slices = state["seam_slices"]
    is_spread: bool = state["is_spread"]
    gutter_x: int | None = state["gutter_x"]
    text_probability: np.ndarray | None = state["text_probability"]
    stage1_meta: list[dict] = state["stage1_meta"]

    exclude_set = set(exclude_node_ids or [])
    if exclude_set:
        kept_meta = [m for m in stage1_meta if m["id"] not in exclude_set]
        stroke_owners = np.zeros((H, W), np.int32)
        for m in kept_meta:
            rx0, ry0, rw, rh = m["_roi"]
            stroke_owners[ry0 : ry0 + rh, rx0 : rx0 + rw] += m["_orig_ink_roi"] > 0
    else:
        kept_meta = stage1_meta
        stroke_owners = state["stroke_owners"]

    aligned_ref_full: np.ndarray | None = None
    ref_valid_mask: np.ndarray | None = None
    if reference_edition_png_b64:
        ref_rgb = decode_image_b64(reference_edition_png_b64, mode="RGB")
        ok_align, aligned_ref, valid_m = align_cross_edition_clean_plate(raw_rgb, ref_rgb)
        if ok_align:
            aligned_ref_full = aligned_ref
            ref_valid_mask = valid_m

    def _phase2a_prepare(m: dict) -> dict:
        reg = m["_reg"]
        tx, ty, tw, th = reg["bbox"]
        rx0, ry0, rw, rh = m["_roi"]
        roi_rgb = raw_rgb[ry0 : ry0 + rh, rx0 : rx0 + rw]
        coarse_roi = m["_coarse_roi"]
        orig_ink_roi = m["_orig_ink_roi"]
        bmask_safe = m["_bmask_safe"]
        bmask_sub = m["_bmask_sub"]
        text_domain = m["_text_domain"]
        bubble_kind = m["bubble_kind"]

        if text_probability is not None:
            stroke_alpha = refine_text_alpha(
                roi_rgb, text_probability[ry0 : ry0 + rh, rx0 : rx0 + rw], text_domain, observed_ink=coarse_roi
            )
            distances = distance_transform_edt(stroke_alpha > 0)
            radii = distances[distances > 0]
            stroke_w = float(2 * np.median(radii)) if radii.size else 1.0
            stroke_mod = float(np.median(np.abs(2 * radii - stroke_w)) / stroke_w) if radii.size else 0.0
        else:
            stroke_alpha, stroke_w, stroke_mod = build_stroke_trimap_and_alpha(roi_rgb, coarse_mask=coarse_roi)
        refined_stroke = stroke_alpha.copy()

        text_guard = np.zeros((rh, rw), dtype=np.uint8)
        col_boxes = reg.get("column_boxes") or []
        if bubble_kind == "floating_sfx" and len(col_boxes) >= 2:
            c_pad = max(3, int(round(max(stroke_w * 1.4, float(reg.get("col_scale_px", min(tw, th))) * 0.14))))
            for cb in col_boxes:
                cx0, cy0, cw_c, ch_c = cb
                gx0, gy0 = max(0, cx0 - rx0 - c_pad), max(0, cy0 - ry0 - c_pad)
                gx1, gy1 = min(rw, cx0 + cw_c - rx0 + c_pad), min(rh, cy0 + ch_c - ry0 + c_pad)
                if gx1 > gx0 and gy1 > gy0:
                    text_guard[gy0:gy1, gx0:gx1] = 1
        else:
            g_pad = max(5, int(round(max(stroke_w * 3.0, min(tw, th) * 0.16))))
            gx0, gy0 = max(0, tx - rx0 - g_pad), max(0, ty - ry0 - g_pad)
            gx1, gy1 = min(rw, tx - rx0 + tw + g_pad), min(rh, ty - ry0 + th + g_pad)
            text_guard[gy0:gy1, gx0:gx1] = 1
        if bmask_safe is not None and np.any(bmask_safe):
            orig_dil = cv2.dilate(coarse_roi, np.ones((5, 5), np.uint8))
            text_guard = (text_guard & bmask_safe) | orig_dil
        stroke_alpha = stroke_alpha * text_guard.astype(np.float32)

        enclosed = bubble_kind not in ("floating_sfx", "diegetic_surface")
        z2_mask, occ_amb = extract_z2_foreground_occluders(
            roi_rgb,
            stroke_alpha,
            enclosed_bubble=enclosed,
            substrate_mask=bmask_sub,
            text_probability=text_probability[ry0 : ry0 + rh, rx0 : rx0 + rw] if text_probability is not None else None,
        )

        if not enclosed:
            is_semi, unmixed_B, cont_alpha, unmix_fg, med_alpha = unmix_optical_alpha_layer(
                roi_rgb, stroke_alpha, observed_ink=coarse_roi,
            )
        else:
            is_semi, unmixed_B, cont_alpha, unmix_fg, med_alpha = False, roi_rgb, stroke_alpha, [12, 12, 14], 1.0

        ref_crop = None
        if aligned_ref_full is not None and ref_valid_mask is not None:
            if np.all(ref_valid_mask[ry0 : ry0 + rh, rx0 : rx0 + rw]):
                ref_crop = aligned_ref_full[ry0 : ry0 + rh, rx0 : rx0 + rw]

        inpaint_unknown = None
        needs_lama = False
        active_erasure = (stroke_alpha > 0) & (z2_mask == 0)
        if not (is_semi and ref_crop is None):
            inpaint_unknown = reconstruction_unknown(
                active_erasure, coarse_roi,
                stroke_owners[ry0 : ry0 + rh, rx0 : rx0 + rw] > orig_ink_roi,
            )
            if (not enclosed) and ref_crop is None and np.any(active_erasure):
                known_open = ~inpaint_unknown
                if np.any(known_open):
                    from .restoration import validation_collar
                    ring_open = validation_collar(active_erasure, known_open)
                    rp = roi_rgb[ring_open].astype(np.float64) if np.any(ring_open) else roi_rgb[known_open].astype(np.float64)
                    center_open = np.median(rp, axis=0)
                    sig_open = estimate_substrate_noise(roi_rgb, known_open)
                    null_d = substrate_squared_distance(rp, center_open, sig_open)
                    is_flat = all(np.quantile(null_d, q) <= chi2.ppf(q, 3) for q in (0.75, 0.95))
                    if not is_flat:
                        needs_lama = True

        return {
            "m": m,
            "roi": (rx0, ry0, rw, rh),
            "roi_rgb": roi_rgb,
            "stroke_alpha": stroke_alpha,
            "refined_stroke": refined_stroke,
            "z2_mask": z2_mask,
            "occ_amb": occ_amb,
            "enclosed": enclosed,
            "bmask_sub": bmask_sub,
            "is_semi": is_semi,
            "unmixed_B": unmixed_B,
            "cont_alpha": cont_alpha,
            "unmix_fg": unmix_fg,
            "med_alpha": med_alpha,
            "ref_crop": ref_crop,
            "inpaint_unknown": inpaint_unknown,
            "needs_lama": needs_lama,
        }

    if len(kept_meta) <= 1:
        phase2_contexts = [_phase2a_prepare(m) for m in kept_meta]
    else:
        max_workers = min(len(kept_meta), max(2, min(6, (os.cpu_count() or 4) // 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            phase2_contexts = list(pool.map(_phase2a_prepare, kept_meta))

    lama_full: np.ndarray | None = None
    if any(ctx["needs_lama"] for ctx in phase2_contexts):
        page_lama_mask = stroke_owners > 0
        page_z2_mask = np.zeros((H, W), dtype=bool)
        lama_target_mask = np.zeros((H, W), dtype=bool)
        for ctx in phase2_contexts:
            rx0, ry0, rw, rh = ctx["roi"]
            if ctx["z2_mask"] is not None:
                page_z2_mask[ry0 : ry0 + rh, rx0 : rx0 + rw] |= ctx["z2_mask"] > 0
            u_node = ctx["inpaint_unknown"]
            if u_node is not None:
                u_clean = u_node.copy()
                if ctx["z2_mask"] is not None:
                    u_clean &= ctx["z2_mask"] == 0
                if ctx["enclosed"] and ctx["bmask_sub"] is not None and np.any(ctx["bmask_sub"] > 0):
                    u_clean &= ctx["bmask_sub"] > 0
                page_lama_mask[ry0 : ry0 + rh, rx0 : rx0 + rw] |= u_clean
            if ctx["needs_lama"]:
                lama_target_mask[ry0 : ry0 + rh, rx0 : rx0 + rw] = True
        page_lama_mask &= ~page_z2_mask
        lama_full = learned_clean_plate_full_context(
            raw_rgb, page_lama_mask, active_mask=lama_target_mask
        )

    def _phase2c_finish(ctx: dict) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        m = ctx["m"]
        reg = m["_reg"]
        tx, ty, tw, th = reg["bbox"]
        rx0, ry0, rw, rh = ctx["roi"]
        roi_rgb = ctx["roi_rgb"]
        coarse_roi = m["_coarse_roi"]
        orig_ink_roi = m["_orig_ink_roi"]
        bmask_sub = ctx["bmask_sub"]
        text_domain = m["_text_domain"]
        stroke_alpha = ctx["stroke_alpha"]
        refined_stroke = ctx["refined_stroke"]
        z2_mask = ctx["z2_mask"]
        occ_amb = ctx["occ_amb"]
        enclosed = ctx["enclosed"]
        is_semi = ctx["is_semi"]
        unmixed_B = ctx["unmixed_B"]
        cont_alpha = ctx["cont_alpha"]
        unmix_fg = ctx["unmix_fg"]
        med_alpha = ctx["med_alpha"]
        ref_crop = ctx["ref_crop"]
        inpaint_unknown = ctx["inpaint_unknown"]

        if is_semi and ref_crop is None:
            clean_z0 = unmixed_B
            rest_path = "inverse_alpha_unmix"
            predicted = (1.0 - cont_alpha[:, :, None]) * clean_z0.astype(np.float32) + cont_alpha[:, :, None] * np.asarray(unmix_fg, np.float32)
            optical_support = cont_alpha > 0
            rest_res = float(np.sqrt(np.mean((predicted[optical_support] - roi_rgb[optical_support].astype(np.float32)) ** 2)) / 255.0)
            eff_alpha = cont_alpha
            inpaint_unknown_out = stroke_alpha > 0
            inferred_background = unmixed_B
            boundary_opacity = np.zeros(stroke_alpha.shape, np.float32)
        else:
            inpaint_unknown_out = inpaint_unknown
            pre_lama = lama_full[ry0 : ry0 + rh, rx0 : rx0 + rw] if (lama_full is not None and ctx["needs_lama"]) else None
            clean_z0, rest_path, rest_res, inferred_background = restore_clean_plate_roi(
                roi_rgb,
                stroke_alpha,
                z2_occluder_mask=z2_mask,
                aligned_ref_roi=ref_crop,
                substrate_mask=bmask_sub,
                context_exclusion=inpaint_unknown_out,
                precomputed_lama_roi=pre_lama,
            )
            eff_alpha = stroke_alpha
            foreign = stroke_owners[ry0 : ry0 + rh, rx0 : rx0 + rw] > orig_ink_roi
            protected = foreign | (z2_mask > 0)
            if enclosed and bmask_sub is not None:
                protected |= bmask_sub == 0
            clean_z0, eff_alpha, boundary_opacity = unmix_stroke_boundary(
                roi_rgb, clean_z0, eff_alpha, inferred_background, coarse_roi,
                inpaint_unknown_out, protected, rest_res,
            )

        audit_substrate = bmask_sub if enclosed and rest_path != "fft_halftone_lattice" else None
        clean_z0, eff_alpha, uncovered_ink, residual_ink, repaired_pixels = repair_observed_paper_residue(
            roi_rgb, clean_z0, eff_alpha, coarse_roi, audit_substrate, z2_mask, text_domain,
            context_exclusion=inpaint_unknown_out,
        )

        source_ink = coarse_roi > 0
        style_ink = source_ink
        if (not enclosed) and text_probability is not None:
            cand_style_ink = (eff_alpha > 0.1) & (z2_mask == 0)
            if int(np.count_nonzero(cand_style_ink)) >= 16 and np.any(~cand_style_ink):
                style_ink = cand_style_ink
        source_radii = distance_transform_edt(source_ink)[source_ink]
        source_width = float(2 * np.median(source_radii)) if source_radii.size else 0.0
        source_modulation = float(np.median(np.abs(2 * source_radii - source_width)) / source_width) if source_radii.size else 0.0
        foreign_owners = stroke_owners[ry0 : ry0 + rh, rx0 : rx0 + rw] > orig_ink_roi
        typography_substrate = (stroke_owners[ry0 : ry0 + rh, rx0 : rx0 + rw] == 0) & (text_domain == 0) & (z2_mask == 0)
        if enclosed and bmask_sub is not None:
            cand_sub = typography_substrate & (bmask_sub > 0)
            if np.any(cand_sub & (~style_ink)):
                typography_substrate = cand_sub
        if not np.any(typography_substrate & (~style_ink)):
            relaxed = (~style_ink) & (~foreign_owners) & (z2_mask == 0)
            if enclosed and bmask_sub is not None and np.any(relaxed & (bmask_sub > 0)):
                typography_substrate = relaxed & (bmask_sub > 0)
            elif np.any(relaxed):
                typography_substrate = relaxed
            else:
                typography_substrate = ~style_ink
        style_spec, fusion_mode, exemplar = extract_stroke_style_and_halo(
            roi_rgb=roi_rgb,
            stroke_alpha=style_ink.astype(np.float32),
            stroke_w=source_width,
            stroke_mod=source_modulation,
            orientation=reg["orientation"],
            bubble_kind=m["bubble_kind"],
            pigment_support=style_ink,
            substrate_support=typography_substrate,
        )
        if is_semi and ref_crop is None:
            fusion_mode = "alpha_blended"
            style_spec.blend_alpha = med_alpha
            style_spec.fg_color_rgba = unmix_fg + [255]
        elif np.any(z2_mask > 0):
            fusion_mode = "occluded_z2"
        elif rest_path == "fft_halftone_lattice" and fusion_mode == "clean_interior":
            fusion_mode = "halftone_backed"
        elif rest_path == "structure_tensor_line" and fusion_mode == "clean_interior":
            fusion_mode = "speedline_backed"

        shading_b64 = ""
        if fusion_mode == "diegetic_planar":
            _, shading_u8 = decompose_intrinsic_surface_shading(clean_z0)
            shading_b64 = encode_png_b64(shading_u8)

        if style_spec.font_category != "gothic" and reg["orientation"] == "horizontal" and reg.get("line_polygons"):
            style_spec.source_font_size_px = _estimate_node_font_size(
                reg, source_ink, tx, ty, tw, th, rx0, ry0, rw, rh, m["_src_fs_init"], style_spec.font_category
            )
        else:
            style_spec.source_font_size_px = float(m["style"].source_font_size_px)

        if style_spec.font_category != "gothic":
            with _font_metrics_lock:
                b_min, b_max = estimate_char_budget_from_sdf(
                    m["_sdf_roi"], reg["source_text"], source_font_size_px=style_spec.source_font_size_px, style=style_spec
                )
        else:
            b_min, b_max = m["char_budget_min"], m["char_budget_max"]

        reasons = list(m["uncertainty"].reasons)
        ocr_ent = float(m["uncertainty"].ocr_entropy)
        if occ_amb > 0.15 and "z2_foreground_interleaving" not in reasons:
            reasons.append("z2_foreground_interleaving")
        if rest_res > 0.18 and "complex_background_slit" not in reasons:
            reasons.append("complex_background_slit")
        if repaired_pixels and "observed_source_ink_automatically_repaired" not in reasons:
            reasons.append("observed_source_ink_automatically_repaired")

        unc_score = float(np.clip(0.45 * ocr_ent + 0.30 * occ_amb + 0.25 * rest_res, 0.0, 1.0))
        needs_rev = bool(unc_score >= 0.38 or "z2_foreground_interleaving" in reasons)

        patches = PatchLayers(
            roi=[rx0, ry0, rw, rh],
            clean_z0_png_b64=encode_png_b64(clean_z0),
            stroke_alpha_png_b64=encode_png_b64(np.clip(np.rint(eff_alpha * 255.0), 0, 255).astype(np.uint8)),
            source_observed_ink_png_b64=encode_png_b64((coarse_roi > 0).astype(np.uint8) * 255),
            source_domain_png_b64=encode_png_b64(text_domain.astype(np.uint8) * 255),
            refined_stroke_png_b64=encode_png_b64(np.clip(np.rint(refined_stroke * 255), 0, 255).astype(np.uint8)),
            inpaint_unknown_png_b64=encode_png_b64(inpaint_unknown_out.astype(np.uint8) * 255),
            inferred_background_png_b64=encode_png_b64(inferred_background),
            boundary_opacity_png_b64=encode_png_b64(np.rint(boundary_opacity * 255).astype(np.uint8)),
            z2_occluder_mask_png_b64=encode_png_b64(z2_mask) if np.any(z2_mask > 0) else "",
            shading_map_png_b64=shading_b64,
            exemplar_patch_png_b64=encode_png_b64(exemplar),
            rendered_z1_png_b64="",
        )

        node_dict = {
            "id": m["id"],
            "reading_order": m["reading_order"],
            "track": m["track"],
            "bbox": m["bbox"],
            "polygon": m["polygon"],
            "bubble_polygon": m["bubble_polygon"],
            "tail_tip": m["tail_tip"],
            "tail_dir": m["tail_dir"],
            "spine_points": m["spine_points"],
            "source_lines": m["source_lines"],
            "source_quadrilaterals": m["source_quadrilaterals"],
            "detector_quadrilaterals": m["detector_quadrilaterals"],
            "source_readings": m["source_readings"],
            "bubble_kind": m["bubble_kind"],
            "fusion_mode": fusion_mode,
            "restoration_path": rest_path,
            "speaker_cluster_id": m.get("speaker_cluster_id", ""),
            "speaker_appearance_b64": m.get("speaker_appearance_b64", ""),
            "source_text": m["source_text"],
            "recognition_pending": m["recognition_pending"],
            "ruby_pairs": m["ruby_pairs"],
            "char_budget_min": b_min,
            "char_budget_max": b_max,
            "style": style_spec,
            "patches": patches,
            "uncertainty": UncertaintySpec(
                score=unc_score,
                ocr_entropy=ocr_ent,
                occlusion_ambiguity=occ_amb,
                restoration_residual=rest_res,
                source_ink_uncovered_ratio=uncovered_ink,
                source_ink_residual_ratio=residual_ink,
                automatic_repair_pixels=repaired_pixels,
                reasons=reasons,
                needs_review=needs_rev,
            ),
        }
        return node_dict, (clean_z0, eff_alpha, z2_mask)

    if len(phase2_contexts) <= 1:
        finished = [_phase2c_finish(ctx) for ctx in phase2_contexts]
    else:
        max_workers = min(len(phase2_contexts), max(2, min(6, (os.cpu_count() or 4) // 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            finished = list(pool.map(_phase2c_finish, phase2_contexts))

    ordered_meta = [item for item, _ in finished]
    raw_layers = [layers for _, layers in finished]
    harmonize_page_typography(ordered_meta)

    clean_canvas = raw_rgb.copy()
    touched_global = np.zeros((H, W), dtype=bool)
    nodes: list[TextNode] = []
    halftone_cnt = 0
    fusion_cnt = 0
    review_cnt = 0

    for item, (c_z0, s_a, z2_m) in zip(ordered_meta, raw_layers):
        node = TextNode(**item)
        nodes.append(node)
        if node.restoration_path == "fft_halftone_lattice":
            halftone_cnt += 1
        if node.fusion_mode in ("alpha_blended", "occluded_z2", "diegetic_planar", "exemplar_stroke"):
            fusion_cnt += 1
        if node.uncertainty.needs_review:
            review_cnt += 1

        rx0, ry0, rw, rh = node.patches.roi
        erased = s_a > 0
        if z2_m is not None and np.any(z2_m > 0):
            erased = erased & (~(z2_m > 0))
        sub_canvas = clean_canvas[ry0 : ry0 + rh, rx0 : rx0 + rw]
        sub_canvas[erased] = c_z0[erased]
        touched_global[ry0 : ry0 + rh, rx0 : rx0 + rw] |= erased

    unmasked = ~touched_global
    bit_exact = (
        float(np.mean(np.all(clean_canvas[unmasked] == raw_rgb[unmasked], axis=-1)))
        if np.any(unmasked)
        else 1.0
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    dom = PageDOM(
        schema_version=1,
        width=W,
        height=H,
        format_profile=profile,
        is_color=is_color,
        paper_tint_rgb=paper_tint,
        seam_slices=seam_slices,
        is_spread=is_spread,
        gutter_x=gutter_x,
        nodes=nodes,
        metrics=PageMetrics(
            bit_exact_unmasked_ratio=bit_exact,
            unmasked_linf=int(np.max(np.abs(clean_canvas[unmasked].astype(np.int16) - raw_rgb[unmasked].astype(np.int16)), initial=0)),
            protected_pixels=int(np.count_nonzero(unmasked)),
            total_nodes=len(nodes),
            halftone_restored_nodes=halftone_cnt,
            deep_fusion_nodes=fusion_cnt,
            needs_review_nodes=review_cnt,
            analyze_ms=elapsed_ms,
        ),
    )
    return AnalyzeResponse(dom=dom, clean_preview_png_b64=encode_png_b64(clean_canvas))


class LayoutTokenExpiredError(Exception):
    """Raised when a layout_token is missing/evicted and no raw image was provided."""


def detect_page_layout(
    req: AnalyzeRequest, pre_detected_regions: list[dict] | None = None
) -> DetectLayoutResponse:
    layout_dom, cached_state = _compute_layout_state(req, pre_detected_regions=pre_detected_regions)
    token = uuid.uuid4().hex
    _cache_layout_state(token, cached_state)
    return DetectLayoutResponse(dom=layout_dom, layout_token=token)


def restore_page_from_layout(req: RestoreRequest) -> AnalyzeResponse:
    state = _get_layout_state(req.layout_token)
    if state is None:
        if not req.image_png_b64:
            raise LayoutTokenExpiredError("layout_token expired or missing and image_png_b64 is empty")
        analyze_req = AnalyzeRequest(
            image_png_b64=req.image_png_b64,
            reference_edition_png_b64=req.reference_edition_png_b64 or None,
            format_profile=req.format_profile,
            recognize_text=req.recognize_text,
            seam_slices=req.seam_slices,
            is_spread=req.is_spread,
            gutter_x=req.gutter_x,
        )
        _, state = _compute_layout_state(analyze_req)
    return _restore_from_state(
        state,
        reference_edition_png_b64=req.reference_edition_png_b64 or None,
        exclude_node_ids=req.exclude_node_ids,
    )


def analyze_and_restore_page(
    req: AnalyzeRequest, pre_detected_regions: list[dict] | None = None
) -> AnalyzeResponse:
    _, state = _compute_layout_state(req, pre_detected_regions=pre_detected_regions)
    return _restore_from_state(state, reference_edition_png_b64=req.reference_edition_png_b64)


def _cluster_by_rgb_distance(
    styles: list,
    attr: str,
    max_dist: float,
) -> list[list]:
    """Partition styles having non-None RGB attr into compact clusters (dist <= max_dist to medoid)."""
    with_rgb = [
        (st, np.array(getattr(st, attr)[:3], dtype=np.float32))
        for st in styles
        if getattr(st, attr, None) is not None and len(getattr(st, attr)) >= 3
    ]
    unassigned = list(range(len(with_rgb)))
    clusters: list[list] = []
    while unassigned:
        # Pick medoid with largest neighborhood inside max_dist for deterministic stability.
        best_idx = unassigned[0]
        best_nbrs: list[int] = [best_idx]
        for i in unassigned:
            ci = with_rgb[i][1]
            nbrs = [j for j in unassigned if float(np.linalg.norm(with_rgb[j][1] - ci)) <= max_dist]
            if len(nbrs) > len(best_nbrs):
                best_idx = i
                best_nbrs = nbrs
        clusters.append([with_rgb[j][0] for j in best_nbrs])
        nbr_set = set(best_nbrs)
        unassigned = [j for j in unassigned if j not in nbr_set]
    return clusters


def _unify_style_rgb(styles: list, attr: str) -> None:
    rgbs = np.array([getattr(st, attr)[:4] for st in styles], dtype=np.float32)
    med = [int(round(v)) for v in np.median(rgbs, axis=0)]
    for st in styles:
        orig = getattr(st, attr)
        alpha = int(orig[3]) if len(orig) >= 4 else (med[3] if len(med) >= 4 else 255)
        setattr(st, attr, [med[0], med[1], med[2], alpha])


def _cluster_floating_siblings(items: list[StyleSpec]) -> list[list[StyleSpec]]:
    """Cluster visually matching borderless floating dialogue peers without hardcoding text."""
    n = len(items)
    if n < 2:
        return []
    visited = [False] * n
    clusters: list[list[StyleSpec]] = []
    for i in range(n):
        if visited[i]:
            continue
        comp = [i]
        visited[i] = True
        queue = [i]
        while queue:
            u = queue.pop()
            su = items[u]
            ou = getattr(su, "orientation", "vertical")
            fu = float(getattr(su, "source_font_size_px", 0.0) or 0.0)
            cu = np.array((su.fg_color_rgba or [0, 0, 0])[:3], dtype=np.float32)
            for v in range(n):
                if visited[v]:
                    continue
                sv = items[v]
                if getattr(sv, "orientation", "vertical") != ou:
                    continue
                fv = float(getattr(sv, "source_font_size_px", 0.0) or 0.0)
                if fu > 4.0 and fv > 4.0:
                    ratio = min(fu, fv) / max(fu, fv)
                    if ratio < 0.60:
                        continue
                elif (fu > 4.0) != (fv > 4.0):
                    continue
                cv = np.array((sv.fg_color_rgba or [0, 0, 0])[:3], dtype=np.float32)
                if float(np.linalg.norm(cu - cv)) <= 26.0:
                    visited[v] = True
                    comp.append(v)
                    queue.append(v)
        if len(comp) >= 2:
            clusters.append([items[idx] for idx in comp])
    return clusters


def _harmonize_style_list(entries: list[tuple[str, StyleSpec]]) -> None:
    """
    Page-level typographic coherence across peer nodes:
      1) Unify font_category, dark/close fill pigment, and outline consistency by semantic role
         (dialogue oval / narration / floating borderless siblings).
      2) Pull in-range peer source_font_size_px strongly toward group median so peer bubbles
         stay balanced across the page while keeping explicit whispers/headlines distinct.
    """
    enclosed_groups: dict[str, list[StyleSpec]] = {}
    floating_pool: list[StyleSpec] = []
    for bk, st in entries:
        orient = str(getattr(st, "orientation", "vertical") or "vertical")
        if bk == "speech_oval":
            enclosed_groups.setdefault(f"dialogue_oval:{orient}", []).append(st)
        elif bk == "narration_box":
            enclosed_groups.setdefault(f"narration:{orient}", []).append(st)
        elif (
            bk == "floating_sfx"
            and float(getattr(st, "tremor_amplitude", 0.0) or 0.0) <= 0.05
            and float(getattr(st, "stroke_width_px", 0.0) or 0.0) < 4.5
        ):
            floating_pool.append(st)

    for grp in enclosed_groups.values():
        if len(grp) < 2:
            continue
        cat_counts = Counter(st.font_category for st in grp)
        consensus_cat = cat_counts.most_common(1)[0][0]
        for st in grp:
            st.font_category = consensus_cat

        scales = [float(st.source_font_size_px) for st in grp if float(st.source_font_size_px) > 4.0]
        if len(scales) >= 2:
            med_s = float(np.median(scales))
            for st in grp:
                s_i = float(st.source_font_size_px)
                if 0.55 * med_s <= s_i <= 1.85 * med_s:
                    st.source_font_size_px = 0.25 * s_i + 0.75 * med_s

        for col_cluster in _cluster_by_rgb_distance(grp, "fg_color_rgba", 28.0):
            if len(col_cluster) >= 2:
                _unify_style_rgb(col_cluster, "fg_color_rgba")

        zero_outlines = sum(1 for st in grp if float(getattr(st, "outline_width_px", 0.0) or 0.0) <= 0.05)
        outlined = [st for st in grp if float(getattr(st, "outline_width_px", 0.0) or 0.0) >= 0.8]
        if zero_outlines * 2 > len(grp):
            for st in grp:
                if 0.0 < float(getattr(st, "outline_width_px", 0.0) or 0.0) <= 1.5:
                    st.outline_width_px = 0.0
                    st.outline_color_rgba = [255, 255, 255, 255]
        elif len(outlined) * 2 > len(grp):
            for oc_cluster in _cluster_by_rgb_distance(outlined, "outline_color_rgba", 32.0):
                if len(oc_cluster) >= 2:
                    _unify_style_rgb(oc_cluster, "outline_color_rgba")
                    med_ow = float(np.median([float(st.outline_width_px) for st in oc_cluster]))
                    for st in oc_cluster:
                        st.outline_width_px = med_ow

    for sibling_grp in _cluster_floating_siblings(floating_pool):
        cat_counts = Counter(st.font_category for st in sibling_grp)
        consensus_cat = cat_counts.most_common(1)[0][0]
        for st in sibling_grp:
            st.font_category = consensus_cat

        scales = [float(st.source_font_size_px) for st in sibling_grp if float(st.source_font_size_px) > 4.0]
        if len(scales) >= 2:
            med_s = float(np.median(scales))
            for st in sibling_grp:
                if float(st.source_font_size_px) > 4.0:
                    st.source_font_size_px = med_s

        _unify_style_rgb(sibling_grp, "fg_color_rgba")

        outlined_sibs = [st for st in sibling_grp if float(getattr(st, "outline_width_px", 0.0) or 0.0) >= 0.8]
        if outlined_sibs:
            med_ow = float(np.median([float(st.outline_width_px) for st in outlined_sibs]))
            ref_with_col = [st for st in outlined_sibs if getattr(st, "outline_color_rgba", None) is not None]
            med_oc = None
            if ref_with_col:
                oc_arr = np.array([st.outline_color_rgba[:4] for st in ref_with_col], dtype=np.float32)
                med_oc = [int(round(v)) for v in np.median(oc_arr, axis=0)]
            for st in sibling_grp:
                st.outline_width_px = med_ow
                if med_oc is not None:
                    st.outline_color_rgba = list(med_oc)
            outer_sibs = [st for st in sibling_grp if float(getattr(st, "outer_outline_width_px", 0.0) or 0.0) >= 0.75]
            if outer_sibs:
                med_ow2 = float(np.median([float(st.outer_outline_width_px) for st in outer_sibs]))
                o_arr = np.array([st.outer_outline_color_rgba[:4] for st in outer_sibs], dtype=np.float32)
                med_oc2 = [int(round(v)) for v in np.median(o_arr, axis=0)]
                for st in sibling_grp:
                    st.outer_outline_width_px = med_ow2
                    st.outer_outline_color_rgba = list(med_oc2)


def harmonize_page_typography(ordered_meta: list[dict]) -> None:
    entries = [
        (str(item.get("bubble_kind", "") or ""), item["style"])
        for item in ordered_meta
        if item.get("style") is not None
    ]
    _harmonize_style_list(entries)


def harmonize_dom_nodes_typography(nodes: list) -> None:
    entries = [
        (str(getattr(node, "bubble_kind", "") or ""), node.style)
        for node in nodes
        if getattr(node, "style", None) is not None
    ]
    _harmonize_style_list(entries)


def _render_fingerprint(node: TextNode) -> str:
    payload = node.model_dump_json(exclude={
        "layout": True, "uncertainty": True,
        "patches": {"rendered_z1_png_b64", "render_fingerprint"},
    })
    return hashlib.sha256(_RENDER_IMPLEMENTATION + payload.encode()).hexdigest()


def _render_node_ink(node: TextNode, clean_z0: np.ndarray, stroke_alpha: np.ndarray) -> np.ndarray:
    if node.patches.rendered_z1_png_b64 and node.patches.render_fingerprint == _render_fingerprint(node):
        return decode_image_b64(node.patches.rendered_z1_png_b64, mode="RGBA")
    rx0,ry0,rw,rh = node.patches.roi
    exemplar = (
        decode_image_b64(node.patches.exemplar_patch_png_b64, mode="RGB")
        if node.patches.exemplar_patch_png_b64
        else np.zeros((16, 16, 3), dtype=np.uint8)
    )

    text_to_render = node.target_text if node.target_text.strip() else node.source_text
    rubies_to_render = node.target_ruby_pairs if node.target_text.strip() else node.ruby_pairs

    bx, by, bw_n, bh_n = node.bbox
    sdf_poly = node.bubble_polygon if node.bubble_polygon else [
        [float(bx), float(by)],
        [float(bx + bw_n), float(by)],
        [float(bx + bw_n), float(by + bh_n)],
        [float(bx), float(by + bh_n)],
    ]

    # For planar perspective signs, typeset in canonical upright space before perspective homography warp!
    enclosed = bool(node.bubble_polygon and node.bubble_kind not in ("floating_sfx", "diegetic_surface"))
    if node.fusion_mode == "diegetic_planar" and len(node.bubble_polygon) == 4:
        flat_sdf = np.ones((rh, rw), dtype=np.float32)
        padded = np.pad(flat_sdf, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        sdf_field = cv2.distanceTransform(padded.astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        pole_xy = (rw * 0.5, rh * 0.5)
    else:
        sdf_field, pole_xy = build_interior_sdf((rh, rw), (rx0, ry0), sdf_poly)
        if not node.bubble_polygon:
            bx, by, bw_n, bh_n = node.bbox
            pole_xy = (
                float(bx - rx0 + 0.5 * bw_n),
                float(by - ry0 + 0.5 * bh_n),
            )

    with _font_metrics_lock:
        layout, glyph_alpha = solve_sdf_knuth_plass_layout(
            target_text=text_to_render,
            target_segments=node.target_segments if node.target_text.strip() else [],
            ruby_pairs=rubies_to_render,
            sdf=sdf_field,
            pole_xy=pole_xy,
            style=node.style,
            enclosed_bubble=enclosed,
            source_lines=[[x - rx0, y - ry0, w, h] for x, y, w, h in node.source_lines] if not enclosed else [],
            source_quadrilaterals=[[[x-rx0, y-ry0] for x, y in quad] for quad in node.source_quadrilaterals] if not enclosed else [],
        )

    z0_gray = cv2.cvtColor(clean_z0, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sig_z0 = estimate_noise_sigma(z0_gray)
    z0_std = float(np.std(z0_gray))
    over_artwork = (node.bubble_kind == "floating_sfx") and (z0_std > max(4.5 * sig_z0, 10.0))
    comp_alpha, updated_style = conserve_optical_ink_mass(
        glyph_alpha, node.style, over_artwork=over_artwork
    )
    node.style = updated_style
    layout.ink_density_rendered = float(np.sum(comp_alpha)) / max(float(rh * rw), 1.0)
    node.layout = layout

    render_style = node.style
    if (
        (not enclosed)
        and node.fusion_mode not in ("diegetic_planar", "alpha_blended")
        and float(render_style.outline_width_px) < 0.25
        and np.any(comp_alpha > 0.35)
    ):
        fg_rgb = np.array(render_style.fg_color_rgba[:3], dtype=np.float32)
        bg_under = clean_z0[comp_alpha > 0.35].astype(np.float32)
        dists = np.linalg.norm(bg_under - fg_rgb[None, :], axis=1)
        if dists.size > 0 and float(np.percentile(dists, 15)) < 32.0:
            sub_px = clean_z0[stroke_alpha > 0.20] if np.any(stroke_alpha > 0.20) else clean_z0.reshape(-1, 3)
            sub_med = np.median(sub_px.astype(np.float32), axis=0)
            if float(np.linalg.norm(sub_med - fg_rgb)) >= 45.0:
                render_style = render_style.model_copy(deep=True)
                render_style.outline_width_px = 1.4
                render_style.outline_color_rgba = [int(np.clip(np.rint(c), 0, 255)) for c in sub_med] + [255]
                node.style = render_style

    # Source outline width is a font-space measurement. Scale the actual shader
    # by the chosen font size just as the layout's clearance calculation does.
    # Keep source style in the DOM so cached/edited renders do not scale twice.
    if render_style.source_font_size_px > 0:
        fs_ratio = layout.font_size / render_style.source_font_size_px
        render_style = render_style.model_copy(update={
            "outline_width_px": render_style.outline_width_px * fs_ratio,
            "outer_outline_width_px": float(getattr(render_style, "outer_outline_width_px", 0.0) or 0.0) * fs_ratio,
        })
    z1_rgba = apply_exemplar_stroke_shader(
        glyph_alpha=comp_alpha,
        exemplar_patch=exemplar if node.fusion_mode == "exemplar_stroke" else None,
        style=render_style,
    )

    # Apply 3D planar Homography or curved SFX spine TPS warp when physically grounded in scene
    if node.fusion_mode == "diegetic_planar" and len(node.bubble_polygon) == 4:
        quad_loc = np.array([[p[0] - rx0, p[1] - ry0] for p in node.bubble_polygon], dtype=np.float32)
        H_mat = compute_diegetic_homography(z1_rgba.shape[:2], quad_loc)
        node.homography_3x3 = [[float(v) for v in row] for row in H_mat.tolist()]
        z1_rgba = warp_diegetic_layer_perspective_or_spine(
            z1_rgba, (rh, rw), quad_local=quad_loc, vertical=(node.style.orientation == "vertical")
        )
    elif (
        node.bubble_kind == "floating_sfx"
        and not node.source_quadrilaterals
        and len(layout.lines) <= 1
        and len(node.spine_points) >= 4
    ):
        spine_loc = np.array([[p[0] - rx0, p[1] - ry0] for p in node.spine_points], dtype=np.float64)
        chord_vec = spine_loc[-1] - spine_loc[0]
        chord_len = float(np.hypot(chord_vec[0], chord_vec[1]))
        min_warp_len = max(44.0, 2.2 * max(float(node.style.source_font_size_px), float(layout.font_size)))
        if chord_len >= min_warp_len:
            # Measure perpendicular sagitta from straight axis; only warp genuinely curved/tilted single SFX strips
            unit_c = chord_vec / chord_len
            rel = spine_loc - spine_loc[0]
            perp = np.abs(rel[:, 0] * (-unit_c[1]) + rel[:, 1] * unit_c[0])
            axis_tilt = min(abs(unit_c[0]), abs(unit_c[1]))
            if axis_tilt >= 0.22 or (
                float(np.max(perp)) >= max(6.0, 0.12 * chord_len)
                and float(node.style.stroke_width_px) >= 3.2
            ):
                z1_rgba = warp_diegetic_layer_perspective_or_spine(
                    z1_rgba, (rh, rw), spine_local=spine_loc, vertical=(node.style.orientation == "vertical")
                )

    node.patches.rendered_z1_png_b64 = encode_png_b64(z1_rgba)
    node.patches.render_fingerprint = _render_fingerprint(node)
    return z1_rgba



def _ensure_node_patches_for_render(raw_rgb: np.ndarray, node: TextNode) -> None:
    """Materialize Z0 patches for user-drawn manual boxes, or expand ROI when a box is moved/resized.

    Strictly preserves L_inf == 0 outside the erased stroke mask and newly rendered Z1 pixels.
    """
    H, W = raw_rgb.shape[:2]
    if not node.bbox or len(node.bbox) != 4:
        return
    bx = max(0, min(W - 1, int(round(node.bbox[0]))))
    by = max(0, min(H - 1, int(round(node.bbox[1]))))
    bw = max(0, min(W - bx, int(round(node.bbox[2]))))
    bh = max(0, min(H - by, int(round(node.bbox[3]))))
    if bw < 6 or bh < 6:
        return
    node.bbox = [bx, by, bw, bh]

    pad = max(8, min(32, min(bw, bh) // 4))
    want_x0 = max(0, bx - pad)
    want_y0 = max(0, by - pad)
    want_x1 = min(W, bx + bw + pad)
    want_y1 = min(H, by + bh + pad)

    # Case 1: Brand-new user-drawn box (no Z0 patch yet) -> inpaint ONLY inside user's drawn box.
    if not node.patches.clean_z0_png_b64 or not node.patches.roi or len(node.patches.roi) != 4 or node.patches.roi[2] <= 0 or node.patches.roi[3] <= 0:
        rx0, ry0 = want_x0, want_y0
        rw, rh = want_x1 - want_x0, want_y1 - want_y0
        full_mask = np.zeros((H, W), dtype=bool)
        full_mask[by : by + bh, bx : bx + bw] = True

        clean_full: np.ndarray | None = None
        try:
            clean_full, _ = learned_clean_plate_full_context(raw_rgb, full_mask)
        except Exception:
            clean_full = None
        if clean_full is None:
            clean_roi = cv2.inpaint(
                raw_rgb[ry0 : ry0 + rh, rx0 : rx0 + rw],
                (full_mask[ry0 : ry0 + rh, rx0 : rx0 + rw].astype(np.uint8) * 255),
                3,
                cv2.INPAINT_TELEA,
            )
        else:
            clean_roi = clean_full[ry0 : ry0 + rh, rx0 : rx0 + rw].copy()
        alpha_u8 = full_mask[ry0 : ry0 + rh, rx0 : rx0 + rw].astype(np.uint8) * 255
        node.restoration_path = "lama_masked"

        node.patches.roi = [rx0, ry0, rw, rh]
        node.patches.clean_z0_png_b64 = encode_png_b64(clean_roi)
        node.patches.stroke_alpha_png_b64 = encode_png_b64(alpha_u8)
        node.patches.z2_occluder_mask_png_b64 = ""
        node.patches.shading_map_png_b64 = ""
        node.patches.rendered_z1_png_b64 = ""
        node.patches.render_fingerprint = ""
        return

    # Case 2: Existing node whose bbox was dragged or resized outside its original ROI.
    rx0, ry0, rw, rh = node.patches.roi
    rx1, ry1 = rx0 + rw, ry0 + rh
    if bx >= rx0 and by >= ry0 and (bx + bw) <= rx1 and (by + bh) <= ry1:
        return

    nx0 = min(rx0, want_x0)
    ny0 = min(ry0, want_y0)
    nx1 = max(rx1, want_x1)
    ny1 = max(ry1, want_y1)
    nw, nh = nx1 - nx0, ny1 - ny0

    new_clean = raw_rgb[ny0:ny1, nx0:nx1].copy()
    new_alpha = np.zeros((nh, nw), dtype=np.uint8)
    ox, oy = rx0 - nx0, ry0 - ny0

    old_clean = decode_image_b64(node.patches.clean_z0_png_b64, mode="RGB")
    if old_clean.shape[:2] == (rh, rw):
        new_clean[oy : oy + rh, ox : ox + rw] = old_clean
    if node.patches.stroke_alpha_png_b64:
        old_alpha = decode_image_b64(node.patches.stroke_alpha_png_b64, mode="L")
        if old_alpha.shape[:2] == (rh, rw):
            new_alpha[oy : oy + rh, ox : ox + rw] = old_alpha
    if node.patches.z2_occluder_mask_png_b64:
        old_z2 = decode_image_b64(node.patches.z2_occluder_mask_png_b64, mode="L")
        if old_z2.shape[:2] == (rh, rw):
            new_z2 = np.zeros((nh, nw), dtype=np.uint8)
            new_z2[oy : oy + rh, ox : ox + rw] = old_z2
            node.patches.z2_occluder_mask_png_b64 = encode_png_b64(new_z2)
    node.patches.shading_map_png_b64 = ""
    node.patches.roi = [nx0, ny0, nw, nh]
    node.patches.clean_z0_png_b64 = encode_png_b64(new_clean)
    node.patches.stroke_alpha_png_b64 = encode_png_b64(new_alpha)
    node.patches.rendered_z1_png_b64 = ""
    node.patches.render_fingerprint = ""


def typeset_and_render_page(req: RenderRequest) -> RenderResponse:
    t0 = time.perf_counter()
    raw_rgb = decode_image_b64(req.raw_image_png_b64, mode="RGB")
    H, W = raw_rgb.shape[:2]
    dom = req.dom.model_copy(deep=True)
    for node in dom.nodes:
        _ensure_node_patches_for_render(raw_rgb, node)
    any_valid_cache = any(
        bool(n.patches.rendered_z1_png_b64) and n.patches.render_fingerprint == _render_fingerprint(n)
        for n in dom.nodes
    )
    if not any_valid_cache:
        harmonize_dom_nodes_typography(dom.nodes)

    composite = raw_rgb.copy()
    modified_mask = np.zeros((H, W), dtype=bool)

    # Apply every Z0 patch before any Z1 text, and every Z2 occluder after all
    # text. Interleaving node-local Z0/Z1/Z2 erases previous nodes where ROIs
    # overlap, and makes an occluder depend on the arbitrary node loop order.
    # Decode each node's patch arrays ONCE and reuse across Z0 composition and Z1 rendering.
    foreground = np.zeros((H, W), dtype=bool)
    decoded_inputs: list[tuple[TextNode, int, int, int, int, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for node in dom.nodes:
        rx0, ry0, rw, rh = node.patches.roi
        if rw <= 0 or rh <= 0 or not node.patches.clean_z0_png_b64:
            continue
        alpha_u8 = (
            decode_image_b64(node.patches.stroke_alpha_png_b64, mode="L")
            if node.patches.stroke_alpha_png_b64
            else np.zeros((rh, rw), dtype=np.uint8)
        )
        erased = alpha_u8 > 0
        if node.patches.z2_occluder_mask_png_b64:
            z2 = decode_image_b64(node.patches.z2_occluder_mask_png_b64, mode="L")
            erased &= z2 == 0
            foreground[ry0 : ry0 + rh, rx0 : rx0 + rw] |= z2 > 127
        clean_z0 = decode_image_b64(node.patches.clean_z0_png_b64, mode="RGB")
        canvas_roi = composite[ry0 : ry0 + rh, rx0 : rx0 + rw]
        canvas_roi[erased] = clean_z0[erased]
        modified_mask[ry0 : ry0 + rh, rx0 : rx0 + rw] |= alpha_u8 > 0

        stroke_alpha = alpha_u8.astype(np.float32) / 255.0
        shading_u8 = (
            decode_image_b64(node.patches.shading_map_png_b64, mode="RGB")
            if node.patches.shading_map_png_b64
            else None
        )
        decoded_inputs.append((node, rx0, ry0, rw, rh, clean_z0, stroke_alpha, shading_u8))

    def _render_item(item: tuple[TextNode, int, int, int, int, np.ndarray, np.ndarray, np.ndarray | None]):
        node, rx0, ry0, rw, rh, clean_z0, stroke_alpha, shading_u8 = item
        z1_rgba = _render_node_ink(node, clean_z0, stroke_alpha)
        return (node, rx0, ry0, rw, rh, clean_z0, stroke_alpha, shading_u8, z1_rgba)

    if len(decoded_inputs) <= 1:
        prepared = [_render_item(it) for it in decoded_inputs]
    else:
        max_workers = min(len(decoded_inputs), max(2, min(6, (os.cpu_count() or 4) // 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            prepared = list(pool.map(_render_item, decoded_inputs))

    if not any_valid_cache:
        peer_groups: dict[tuple[str, str], list[int]] = {}
        for idx, item in enumerate(prepared):
            n = item[0]
            if n.bubble_kind in ("speech_oval", "narration_box"):
                peer_groups.setdefault((n.bubble_kind, str(n.style.orientation)), []).append(idx)
        for idxs in peer_groups.values():
            if len(idxs) < 3:
                continue
            fs_vals = [
                float(prepared[i][0].layout.font_size)
                for i in idxs
                if prepared[i][0].layout is not None and float(prepared[i][0].layout.font_size) > 0
            ]
            if len(fs_vals) < 3:
                continue
            med_fs = float(round(float(np.median(fs_vals))))
            for i in idxs:
                n, rx0, ry0, rw, rh, clean_z0, stroke_alpha, shading_u8, _ = prepared[i]
                if n.layout is None:
                    continue
                cur_fs = float(n.layout.font_size)
                if med_fs + 1.0 < cur_fs <= 1.50 * med_fs:
                    scale_up = 0.96 if n.style.orientation == "vertical" else 0.92
                    n.style.source_font_size_px = med_fs / scale_up
                    n.patches.render_fingerprint = ""
                    n.patches.rendered_z1_png_b64 = ""
                    new_z1 = _render_node_ink(n, clean_z0, stroke_alpha)
                    prepared[i] = (n, rx0, ry0, rw, rh, clean_z0, stroke_alpha, shading_u8, new_z1)

    for node, rx0, ry0, rw, rh, clean_z0, stroke_alpha, shading_u8, z1_rgba in prepared:
        raw_roi = composite[ry0 : ry0 + rh, rx0 : rx0 + rw].copy()
        merged_roi = composite_node_layers(
            raw_roi_rgb=raw_roi,
            clean_z0_rgb=clean_z0,
            stroke_alpha_orig=np.zeros_like(stroke_alpha),
            rendered_z1_rgba=z1_rgba,
            z2_occluder_mask=None,
            shading_map_u8=shading_u8,
            style=node.style,
            fusion_mode=node.fusion_mode,
        )
        composite[ry0 : ry0 + rh, rx0 : rx0 + rw] = merged_roi

        drawn = z1_rgba[:, :, 3] > 0
        modified_mask[ry0 : ry0 + rh, rx0 : rx0 + rw] |= drawn

    composite[foreground] = raw_rgb[foreground]
    untouched = ~modified_mask
    bit_exact = (
        float(np.mean(np.all(composite[untouched] == raw_rgb[untouched], axis=-1)))
        if np.any(untouched)
        else 1.0
    )
    dom.metrics.bit_exact_unmasked_ratio = bit_exact
    dom.metrics.unmasked_linf = int(np.max(np.abs(composite[untouched].astype(np.int16) - raw_rgb[untouched].astype(np.int16)), initial=0))
    dom.metrics.protected_pixels = int(np.count_nonzero(untouched))
    dom.metrics.total_nodes = len(dom.nodes)
    dom.metrics.halftone_restored_nodes = sum(n.restoration_path == "fft_halftone_lattice" for n in dom.nodes)
    dom.metrics.deep_fusion_nodes = sum(n.fusion_mode in ("alpha_blended", "occluded_z2", "diegetic_planar", "exemplar_stroke") for n in dom.nodes)
    dom.metrics.needs_review_nodes = sum(n.uncertainty.needs_review for n in dom.nodes)
    dom.metrics.render_ms = (time.perf_counter() - t0) * 1000.0

    # Lossless re-slicing for Webtoon Virtual Continuous Canvas!
    slice_b64s: list[str] = []
    if dom.seam_slices:
        for sl in dom.seam_slices:
            y0 = max(0, min(H, sl.y_offset))
            y1 = max(y0, min(H, sl.y_offset + sl.height))
            if y1 > y0:
                slice_b64s.append(encode_png_b64(composite[y0:y1, :, :]))

    return RenderResponse(
        dom=dom,
        composite_png_b64=encode_png_b64(composite),
        slice_pngs_b64=slice_b64s,
    )
