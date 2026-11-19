from __future__ import annotations

from itertools import combinations
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt
from .preprocess import estimate_noise_sigma, otsu_threshold_and_eta, estimate_substrate_noise
from scipy.stats import chi2
from .ocr_tps import solve_tps_weights, evaluate_tps_grid
from .schemas import BubbleKind, FusionMode, StyleSpec


def unmix_optical_alpha_layer(
    roi_rgb: np.ndarray, stroke_mask: np.ndarray,
    observed_ink: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, np.ndarray, list[int], float]:
    """
    Variational Color-Line Optical Unmixing for semi-transparent text / overlays:
        I(x) = alpha(x) * F + (1 - alpha(x)) * B(x)
    Detects genuine optical transparency (background gradient energy attenuated but transmitted
    through the glyph core) and solves the coupled sparse Tikhonov-Laplacian inverse problem:
        min_B sum_x || (1 - alpha(x)) B(x) - (I(x) - alpha(x) F) ||^2 + mu * || grad B ||^2
    avoiding 1/(1 - alpha) noise blowup near opaque regions.
    Returns:
      (is_semi_transparent, clean_B_uint8, continuous_alpha_map, fg_rgb_int, median_core_alpha)
    """
    h, w = roi_rgb.shape[:2]
    active = stroke_mask > 0.15
    if np.count_nonzero(active) < 12 or np.count_nonzero(~active) < 24:
        return False, roi_rgb.copy(), stroke_mask.astype(np.float32), [12, 12, 14], 1.0

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    sigma_n = estimate_noise_sigma(gray.astype(np.float32))

    # Erode active stroke mask to inspect pure stroke interior (away from anti-aliased 1px border)
    # Erasure support includes counters, the anti-alias collar and background
    # gaps. Background pixels in that collar transmit themselves perfectly;
    # regressing them as glyph cores fabricates optical transparency.
    ink = active if observed_ink is None else active & (observed_ink > 0)
    if np.count_nonzero(ink) < 6:
        foreground = np.median(roi_rgb[active].astype(np.float64), axis=0)
        return False, roi_rgb.copy(), stroke_mask.astype(np.float32), [int(c) for c in foreground], 1.0
    core = cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(core) < 6:
        core = ink

    # Crop morphological dilation & EDT to tight active bbox + 8px collar when safe_ext remains sufficient
    ay, ax = np.nonzero(active)
    cy0, cy1 = max(0, int(ay.min()) - 8), min(h, int(ay.max()) + 9)
    cx0, cx1 = max(0, int(ax.min()) - 8), min(w, int(ax.max()) + 9)
    active_c = active[cy0:cy1, cx0:cx1]
    active_c_u8 = active_c.astype(np.uint8)
    ring_c = (cv2.dilate(active_c_u8, np.ones((9, 9), np.uint8)) > 0) & (~active_c)
    if np.count_nonzero(ring_c) >= 12:
        bg_med = np.median(roi_rgb[cy0:cy1, cx0:cx1][ring_c].astype(np.float64), axis=0)
    else:
        bg_med = np.median(roi_rgb[~active].astype(np.float64), axis=0)

    core_px = roi_rgb[core].astype(np.float64)
    core_med = np.median(core_px, axis=0)
    contrast_vec = core_med - bg_med
    contrast_norm = float(np.linalg.norm(contrast_vec))
    if contrast_norm <= 4.0 * sigma_n:
        return False, roi_rgb.copy(), stroke_mask.astype(np.float32), [int(c) for c in core_med], 1.0

    # Exact Boundary Covariant Regression across nearest-neighbor (p_in, p_out) pairs:
    # By spatial continuity of underlying art B(p_in) ~ B(p_out) = I(p_out) across narrow border:
    #     I(p_in) = (1 - alpha_0) * I(p_out) + alpha_0 * F
    # When ink is opaque (alpha_0 = 1), slope tau = (1 - alpha_0) -> 0;
    # When layer is semi-transparent (0.10 <= alpha_0 <= 0.90), 0.10 <= tau <= 0.90 with positive R^2!
    safe_ext_c = ~(cv2.dilate(active_c_u8, np.ones((3, 3), np.uint8)) > 0)
    use_crop_edt = np.count_nonzero(safe_ext_c) >= 16
    if use_crop_edt:
        dist_to_ext_c, nearest_idx_c = distance_transform_edt(~safe_ext_c, return_indices=True)
        core_c = core[cy0:cy1, cx0:cx1]
        ink_c = ink[cy0:cy1, cx0:cx1]
        in_shell_c = core_c & (dist_to_ext_c <= 5.5)
        if np.count_nonzero(in_shell_c) < 8:
            in_shell_c = ink_c & (dist_to_ext_c <= 4.5)
        if np.count_nonzero(in_shell_c) < 8:
            return False, roi_rgb.copy(), stroke_mask.astype(np.float32), [int(c) for c in core_med], 1.0
        sy_c, sx_c = np.nonzero(in_shell_c)
        oy_c = nearest_idx_c[0, sy_c, sx_c]
        ox_c = nearest_idx_c[1, sy_c, sx_c]
        sy, sx = sy_c + cy0, sx_c + cx0
        oy, ox = oy_c + cy0, ox_c + cx0
        safe_ext = None
    else:
        safe_ext = ~(cv2.dilate(active.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if np.count_nonzero(safe_ext) < 16:
            safe_ext = ~active
        dist_to_ext, nearest_idx = distance_transform_edt(~safe_ext, return_indices=True)
        in_shell = core & (dist_to_ext <= 5.5)
        if np.count_nonzero(in_shell) < 8:
            in_shell = ink & (dist_to_ext <= 4.5)
        if np.count_nonzero(in_shell) < 8:
            return False, roi_rgb.copy(), stroke_mask.astype(np.float32), [int(c) for c in core_med], 1.0
        sy, sx = np.nonzero(in_shell)
        oy = nearest_idx[0, sy, sx]
        ox = nearest_idx[1, sy, sx]

    I_in = roi_rgb[sy, sx, :].astype(np.float64)    # (N_b, 3)
    I_out = roi_rgb[oy, ox, :].astype(np.float64)   # (N_b, 3)

    mean_in = np.mean(I_in, axis=0)
    mean_out = np.mean(I_out, axis=0)
    d_in = I_in - mean_in
    d_out = I_out - mean_out

    var_out = float(np.sum(d_out * d_out))
    out_std = float(np.sqrt(var_out / max(I_out.size, 1)))
    if out_std <= 3.0 * sigma_n:
        # Uniform background around stroke: cannot distinguish semi-transparent from solid tint,
        # and Laplace solid fill already restores uniform background identically.
        fg_int = [int(np.clip(np.rint(c), 0, 255)) for c in core_med]
        return False, roi_rgb.copy(), stroke_mask.astype(np.float32), fg_int, 1.0

    cov_io = float(np.sum(d_in * d_out))
    var_in = max(float(np.sum(d_in * d_in)), 1e-9)
    tau = cov_io / max(var_out, 1e-9)  # tau = 1 - alpha_0
    corr_r2 = (cov_io * cov_io) / max(var_out * var_in, 1e-9)

    alpha_0 = 1.0 - tau
    is_semi = (0.10 <= tau <= 0.88) and (0.12 <= alpha_0 <= 0.90) and (corr_r2 >= 0.25)
    if is_semi:
        # Opaque fill + a differently colored outline can correlate with its
        # surrounding artwork too. Compare that discrete-pigment explanation
        # with the actual optical regression instead of interpreting any
        # positive correlation as transmitted background light.
        from .learned import stroke_palette
        if use_crop_edt:
            rgb_sub = roi_rgb[cy0:cy1, cx0:cx1]
            pigments, pigment_labels_c = stroke_palette(rgb_sub, in_shell_c)
            assignments = pigment_labels_c[in_shell_c]
        else:
            pigments, pigment_labels = stroke_palette(roi_rgb, in_shell)
            assignments = pigment_labels[in_shell]
        counts = np.bincount(assignments,minlength=len(pigments))
        n = len(I_in)
        floor = 1/12  # variance of uint8 quantization
        opaque_errors = I_in-pigments[assignments]
        opaque_var = max(floor,float(np.mean(opaque_errors**2)))
        opaque_ll = np.sum(np.log(counts[assignments]/n)-1.5*np.log(2*np.pi*opaque_var)
                           -np.sum(opaque_errors**2,axis=1)/(2*opaque_var))
        regression_errors = d_in-tau*d_out
        regression_var = max(floor,float(np.mean(regression_errors**2)))
        regression_ll = -.5*n*3*np.log(2*np.pi*regression_var)-np.sum(regression_errors**2)/(2*regression_var)
        is_semi = 5*np.log(n)-2*regression_ll < 4*len(pigments)*np.log(n)-2*opaque_ll
    if not is_semi:
        fg_int = [int(np.clip(np.rint(c), 0, 255)) for c in core_med]
        return False, roi_rgb.copy(), stroke_mask.astype(np.float32), fg_int, 1.0

    F_est = (mean_in-tau*mean_out)/alpha_0
    # An optical fit must have a physically attainable background. A dark
    # fitted pigment at moderate opacity cannot produce nearly-white opaque
    # outlines: their inverse would require B > 255. Clipping that inverse
    # hid the contradiction and left whole original text columns in the plate.
    if use_crop_edt:
        noise = estimate_substrate_noise(roi_rgb[cy0:cy1, cx0:cx1], safe_ext_c)
    else:
        noise = estimate_substrate_noise(roi_rgb, safe_ext)
    minimum,maximum = alpha_0*F_est,alpha_0*F_est+tau*255
    physical_error = np.maximum(minimum-I_in,0)+np.maximum(I_in-maximum,0)
    impossible = np.sum((physical_error/noise)**2,axis=1)
    critical = chi2.isf(.05/(len(I_in)+1),3)
    foreground_error = np.maximum(-F_est,0)+np.maximum(F_est-255,0)
    if np.any(impossible > critical) or np.sum((alpha_0*foreground_error/noise)**2)>critical:
        fg_int = [int(np.clip(np.rint(c),0,255)) for c in core_med]
        return False,roi_rgb.copy(),stroke_mask.astype(np.float32),fg_int,1.0
    F_est = np.clip(F_est,0,255)

    # Construct spatially smooth opacity profile scaled to peak stroke_mask so background fluctuations
    # \Delta B(x) are NEVER absorbed into alpha(x)!
    peak_m = max(float(np.percentile(stroke_mask[active], 90)), 1e-3)
    norm_profile = np.clip(stroke_mask.astype(np.float64) / peak_m, 0.0, 1.0)
    alpha_map = np.where(active, np.clip(alpha_0 * norm_profile, 0.0, 0.94), 0.0)

    # Variational sparse solve for clean background B on active tight bbox:
    # ((1 - alpha)^2 + mu * L) B = (1 - alpha) * (I - alpha * F) + boundary coupling
    # Regularization mu scales strictly with sensor noise ratio so clean translucent regions invert algebraically!
    sy0, sy1 = int(ay.min()), int(ay.max()) + 1
    sx0, sx1 = int(ax.min()), int(ax.max()) + 1
    pad_I = np.pad(roi_rgb[sy0:sy1, sx0:sx1].astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    # Preserve exact full-ROI Dirichlet neighbors where active touches crop border
    if sy0 > 0:
        pad_I[0, 1:-1, :] = roi_rgb[sy0 - 1, sx0:sx1, :]
    if sy1 < h:
        pad_I[-1, 1:-1, :] = roi_rgb[sy1, sx0:sx1, :]
    if sx0 > 0:
        pad_I[1:-1, 0, :] = roi_rgb[sy0:sy1, sx0 - 1, :]
    if sx1 < w:
        pad_I[1:-1, -1, :] = roi_rgb[sy0:sy1, sx1, :]
    pad_a = np.pad(alpha_map[sy0:sy1, sx0:sx1], ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    pad_m = np.pad(active[sy0:sy1, sx0:sx1], ((1, 1), (1, 1)), mode="constant", constant_values=False)

    uy, ux = np.nonzero(pad_m)
    n_u = uy.size
    idx_map = np.full(pad_m.shape, -1, dtype=np.int32)
    idx_map[uy, ux] = np.arange(n_u, dtype=np.int32)

    one_minus_a = 1.0 - pad_a[uy, ux]
    data_w = one_minus_a * one_minus_a
    noise_ratio = float(np.clip((sigma_n / 32.0) ** 2, 1e-4, 0.05))
    mu_reg = max(1e-3, noise_ratio)

    rows_l = [np.arange(n_u, dtype=np.int32)]
    cols_l = [np.arange(n_u, dtype=np.int32)]
    vals_l = [data_w + 4.0 * mu_reg]
    rhs = one_minus_a[:, None] * (pad_I[uy, ux, :] - pad_a[uy, ux, None] * F_est[None, :])

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = uy + dy, ux + dx
        nid = idx_map[ny, nx]
        isu = nid >= 0
        if np.any(isu):
            ui = np.nonzero(isu)[0]
            rows_l.append(ui)
            cols_l.append(nid[isu])
            vals_l.append(np.full(ui.size, -mu_reg, dtype=np.float64))
        isk = ~isu
        if np.any(isk):
            ki = np.nonzero(isk)[0]
            rhs[ki, :] += mu_reg * pad_I[ny[isk], nx[isk], :]

    A_mat = sparse.csc_matrix(
        (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(n_u, n_u),
    )
    sol_B = spsolve(A_mat, rhs, permc_spec="MMD_AT_PLUS_A")
    pad_B = pad_I.copy()
    pad_B[uy, ux, :] = np.clip(sol_B, 0.0, 255.0)
    clean_B = roi_rgb.copy()
    restored_crop = np.clip(np.rint(pad_B[1:-1, 1:-1, :]), 0, 255).astype(np.uint8)
    act_sub = active[sy0:sy1, sx0:sx1]
    clean_B[sy0:sy1, sx0:sx1][act_sub] = restored_crop[act_sub]

    med_alpha = float(np.median(alpha_map[core])) if np.any(core) else float(np.median(alpha_map[active]))
    fg_int = [int(np.clip(np.rint(c), 0, 255)) for c in F_est]
    return True, clean_B, alpha_map.astype(np.float32), fg_int, float(np.clip(med_alpha, 0.15, 0.95))


def extract_z2_foreground_occluders(
    roi_rgb: np.ndarray,
    stroke_alpha: np.ndarray,
    enclosed_bubble: bool,
    substrate_mask: np.ndarray | None = None,
    text_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    2.5D Occlusion Ordering (`Z0 < Z1 < Z2`) via Chromatic Continuity & T-Junction Edge Tracing.
    Identifies foreground structures (hair strands, weapon blades, foreground speedlines/particles)
    that enter from outside the text stroke domain and continuously occlude the glyph (`Z2`).
    Returns:
      (z2_mask_uint8 [0 or 255], occlusion_ambiguity_score [0..1])
    """
    h, w = roi_rgb.shape[:2]
    if h < 8 or w < 8:
        return np.zeros((h, w), dtype=np.uint8), 0.0

    stroke_bin = stroke_alpha > 0.25
    if np.count_nonzero(stroke_bin) < 16:
        return np.zeros((h, w), dtype=np.uint8), 0.0

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma_n = estimate_noise_sigma(gray)

    # Convex hull of the text stroke domain: Z2 occluders enter from outside hull_bool and bridge across!
    ys, xs = np.nonzero(stroke_bin)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_bool = hull_mask > 0

    # Exterior anchor collar outside dilated text hull (scaled with ROI short side)
    hull_k = max(5, min(15, (min(h, w) // 10) | 1))
    dil_hull = cv2.dilate(hull_mask, np.ones((hull_k, hull_k), np.uint8)) > 0
    far_ext = ~dil_hull
    ext_band = ~hull_bool
    if enclosed_bubble and substrate_mask is not None and np.any(substrate_mask > 0):
        sub_b = substrate_mask > 0
        if np.count_nonzero(ext_band & sub_b) >= 12:
            ext_band = ext_band & sub_b
        if np.count_nonzero(far_ext & sub_b) >= 8:
            far_ext = far_ext & sub_b
    if np.count_nonzero(far_ext) < 8:
        far_ext = ext_band
    if np.count_nonzero(ext_band) < 12:
        return np.zeros((h, w), dtype=np.uint8), 0.0

    # 1. Estimate background substrate mode B in ext_band, and check far_ext BEFORE allocating full-ROI float32 arrays
    ext_px = roi_rgb[ext_band].astype(np.float32)
    bg_med = np.median(ext_px, axis=0)
    bg_mad = np.maximum(
        1.4826 * np.median(np.abs(ext_px - bg_med), axis=0),
        max(3.0 * sigma_n, 4.0),
    )
    # On open borderless artwork over multi-color CG (e.g. hair + sunset + shirt, or
    # a two-zone black-margin + wall split), exterior background variance is too
    # heterogeneous for global-segment Z2 separation—never mistake a second background
    # zone or the subtitle's own outline for a foreground Z2 occluder!
    # Check both MAD (50th pct) and Q75 so a 60/40 bimodal exterior (where MAD == 0)
    # is still detected as multi-zone heterogeneous.
    if not enclosed_bubble:
        roi_span = max(1.0, float(np.percentile(gray, 95) - np.percentile(gray, 5)))
        ext_abs = np.abs(ext_px - bg_med)
        bg_q75 = np.percentile(ext_abs, 75, axis=0)
        het_thresh = max(5.0 * sigma_n, 0.14 * roi_span)
        if float(np.median(bg_mad)) > het_thresh or float(np.median(bg_q75)) > 1.5 * het_thresh:
            return np.zeros((h, w), dtype=np.uint8), 0.0
    far_px = roi_rgb[far_ext].astype(np.float32)
    far_dist = np.sqrt(np.sum(((far_px - bg_med) / bg_mad) ** 2, axis=1))
    if np.count_nonzero(far_dist > 3.5) < 6:
        return np.zeros((h, w), dtype=np.uint8), 0.0

    img_f = roi_rgb.astype(np.float32)
    dist_to_bg = np.sqrt(np.sum(((img_f - bg_med) / bg_mad) ** 2, axis=2))

    # 2. Estimate pure glyph ink color F from stroke core pixels NOT matching external foreground colors!
    stroke_u8 = stroke_bin.astype(np.uint8)
    core = cv2.erode(stroke_u8, np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(core) < 8:
        core = stroke_bin
    # Stroke alpha includes white counters and the erasure collar. The pure
    # glyph is the component contrasting with the substrate, not the component
    # farthest from the external artwork (which selected paper on monochrome).
    core_contrast = np.linalg.norm(img_f[core] - bg_med, axis=1)
    contrast_cut, _ = otsu_threshold_and_eta(core_contrast)
    pure_core_pixels = img_f[core][core_contrast > contrast_cut]
    if pure_core_pixels.shape[0] < 4:
        pure_core_pixels = img_f[core]

    fg_med = np.median(pure_core_pixels, axis=0)
    fg_mad = np.maximum(
        1.4826 * np.median(np.abs(pure_core_pixels - fg_med), axis=0),
        max(3.0 * sigma_n, 4.0),
    )
    dist_to_glyph = np.sqrt(np.sum(((img_f - fg_med) / fg_mad) ** 2, axis=2))

    # 3. Exact Convex Color-Segment Exclusion:
    # Every anti-aliased edge pixel of text is a convex blend t*F + (1-t)*B (t in [0, 1])!
    # A genuine Z2 foreground occluder MUST lie strictly off the convex segment [B, F]!
    bf_vec = fg_med - bg_med
    bf_len2 = max(float(np.dot(bf_vec, bf_vec)), 1e-6)
    t_proj = np.clip(np.sum((img_f - bg_med[None, None, :]) * bf_vec[None, None, :], axis=2) / bf_len2, 0.0, 1.0)
    closest_blend = bg_med[None, None, :] + t_proj[:, :, None] * bf_vec[None, None, :]
    off_segment_dist = np.linalg.norm(img_f - closest_blend, axis=2)
    seg_tol = max(14.0, 4.0 * sigma_n)

    distinct_occluder = (dist_to_glyph > 3.5) & (dist_to_bg > 3.5) & (off_segment_dist > seg_tol)

    # T-junction edge-bounded ribbon test: continuous structure anchored in far_ext and penetrating hull interior
    inner_hull = cv2.erode(hull_mask, np.ones((3, 3), np.uint8)) > 0
    if not np.any(inner_hull):
        inner_hull = hull_bool
    num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
        distinct_occluder.astype(np.uint8), connectivity=8
    )
    stroke_near = cv2.dilate(stroke_u8, np.ones((3, 3), np.uint8)) > 0
    ring_k = max(5, min(11, (int(round(min(h, w) * 0.12)) | 1)))
    stroke_halo_ring = cv2.dilate(stroke_u8, np.ones((ring_k, ring_k), np.uint8)) > 0
    z2_mask = np.zeros((h, w), dtype=np.uint8)
    ambiguity = 0.0
    grad: np.ndarray | None = None
    for cid in range(1, num_cc):
        comp_area = int(stats[cid, cv2.CC_STAT_AREA])
        if comp_area < 10:
            continue
        comp = cc_labels == cid
        if (
            np.count_nonzero(comp & far_ext) >= max(6, int(0.15 * comp_area))
            and np.count_nonzero(comp & (~stroke_halo_ring)) >= max(6, int(0.18 * comp_area))
            and np.count_nonzero(comp & inner_hull) >= 4
        ):
            if enclosed_bubble and np.count_nonzero(comp & stroke_near) < 3:
                continue
            if grad is None:
                gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 16.0
                gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 16.0
                grad = np.hypot(gx, gy)
            comp_perim = (comp & hull_bool) & (
                cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
            )
            eval_zone = comp_perim if np.any(comp_perim) else (comp & hull_bool)
            comp_grad = float(np.percentile(grad[eval_zone], 75))
            if comp_grad >= 3.0 * sigma_n:
                z2_mask[comp & hull_bool] = 255
                ambiguity = max(ambiguity, 0.22)

    if text_probability is not None:
        text_cut, _ = otsu_threshold_and_eta(text_probability)
        # Independent text evidence wins over a global color continuity guess.
        # Otherwise original Japanese ink (and its immediate rim) can be composited above its translation.
        text_veto = cv2.dilate((text_probability > 0.5 * text_cut).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        z2_mask[text_veto] = 0
    return z2_mask, float(ambiguity)


def decompose_intrinsic_surface_shading(
    clean_z0_rgb: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """
    Intrinsic Surface Shading & Halftone Illumination Extraction (`B(x) = Albedo_0 * Shading(x)`).
    Given the restored clean surface `clean_z0_rgb` (which retains diagonal shadow screentones,
    vignette lighting, and stone/paper grain!), decomposes it into:
      - `albedo_0`: unshadowed diffuse surface pigment RGB
      - `shading_map_uint8`: normalized [0..255] per-channel transmittance map `255 * B(x) / Albedo_0`
    so that painted diegetic target text inherits 100% of surface shadows, screentones, and grain!
    """
    lum = cv2.cvtColor(clean_z0_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    q90 = float(np.percentile(lum, 90))
    bright = lum >= max(q90, 1.0)
    if np.any(bright):
        albedo = np.median(clean_z0_rgb[bright].astype(np.float32), axis=0)
    else:
        albedo = np.percentile(clean_z0_rgb.astype(np.float32), 90, axis=(0, 1))
    albedo = np.maximum(albedo, 16.0)

    ratio = np.clip(clean_z0_rgb.astype(np.float32) / albedo[None, None, :], 0.0, 1.15)
    # Encode [0.0 .. 1.0] shading transmittance linearly into uint8 [0 .. 255] (clamped lossless for <=1)
    shading_u8 = np.clip(np.rint(np.minimum(ratio, 1.0) * 255.0), 0, 255).astype(np.uint8)
    albedo_int = [int(np.clip(np.rint(c), 0, 255)) for c in albedo]
    return albedo_int, shading_u8


def extract_stroke_style_and_halo(
    roi_rgb: np.ndarray,
    stroke_alpha: np.ndarray,
    stroke_w: float,
    stroke_mod: float,
    orientation: str,
    bubble_kind: BubbleKind,
    pigment_support: np.ndarray | None = None,
    substrate_support: np.ndarray | None = None,
) -> tuple[StyleSpec, FusionMode, np.ndarray]:
    """
    Measure physical typography & paralinguistic parameters directly from raw glyph pixels:
      - Foreground ink color & interior texture/gradient exemplar
      - Outer sticker halo / outline width & color (via Euclidean outward distance ring profiling!)
      - Optical ink coverage density `ink_density_orig`
      - Slant angle (principal moment orientation of vertical/horizontal strokes)
      - Edge tremor / roughness amplitude
      - Exact FusionMode classification
    """
    h, w = roi_rgb.shape[:2]
    # The erasure envelope contains sampling collars and glyph counters. They
    # are restoration support, not evidence of fill pigment or stroke thickness.
    active = stroke_alpha > 0 if pigment_support is None else (pigment_support > 0) & (stroke_alpha > 0)
    total_area = max(float(h * w), 1.0)
    ink_density_orig = float(np.count_nonzero(active)) / total_area

    if not np.any(active):
        spec = StyleSpec(
            orientation="vertical" if orientation == "vertical" else "horizontal",
            stroke_width_px=stroke_w,
            stroke_modulation=stroke_mod,
            ink_density_orig=ink_density_orig,
        )
        return spec, "clean_interior", np.zeros((16, 16, 3), dtype=np.uint8)

    from .learned import prepare_lab_crop, stroke_palette, stroke_pigment_distance

    lab_cache = prepare_lab_crop(roi_rgb)
    pigments, pigment_labels = stroke_palette(roi_rgb, active, lab_cache=lab_cache)
    dist_in = distance_transform_edt(active)
    depths = np.array([np.median(dist_in[pigment_labels == j]) for j in range(len(pigments))])
    # Typography must not depend on whether background reconstruction succeeded.
    # Compare source pigments with actual visible source substrate, excluding
    # the complete erasure envelope and other nodes' unavailable pixels.
    radius = float(np.median(dist_in[active]))
    available = stroke_alpha == 0
    if substrate_support is not None:
        available &= substrate_support > 0
    if not available.any():
        raise ValueError("source typography has no observable substrate")
    distance = distance_transform_edt(~active)
    nearest = float(distance[available].min())
    collar = available & (distance <= nearest + radius)
    substrate_pigments, substrate_distance = stroke_pigment_distance(roi_rgb, collar, lab_cache=lab_cache)
    source_contrast = np.array([np.median(substrate_distance[pigment_labels == j])
                                for j in range(len(pigments))])
    noise_rgb = estimate_substrate_noise(roi_rgb, collar)
    noise = float(np.linalg.norm(noise_rgb))
    # A white outline may share its color with clothing elsewhere in the ROI.
    # Its local source boundary can still demonstrate an opaque pigment, even
    # when a pooled color model quite correctly contains that same white.
    _, nearest_visible = distance_transform_edt(~available, return_indices=True)
    local_delta = (roi_rgb.astype(float)-roi_rgb[tuple(nearest_visible)].astype(float))/(np.sqrt(2)*noise_rgb)
    local_contrast = np.sum(local_delta**2, axis=2)
    # Pooled artwork may contain the same ink color elsewhere. Compare each
    # source sample with its nearest available substrate too; two noisy observed
    # samples contribute twice the noise variance. Bonferroni covers both probes.
    local_source_contrast = np.array([
        np.median(local_contrast[pigment_labels == j]) for j in range(len(pigments))])
    critical = chi2.isf(.05/(2*len(pigments)),3)
    pigment_evidence = source_contrast > critical
    outline_evidence = pigment_evidence | np.array([
        np.quantile(local_contrast[pigment_labels == j], .75) > critical
        for j in range(len(pigments))])
    if collar.any() and bubble_kind not in ("floating_sfx","diegetic_surface"):
        substrate = roi_rgb[collar].astype(float)
        center = np.median(substrate,axis=0)
        q1,q3 = np.quantile(np.linalg.norm(substrate-center,axis=1),[.25,.75])
        outline_evidence &= np.linalg.norm(pigments-center,axis=1) > q3+1.5*(q3-q1)
    endpoints = np.ones(len(pigments),bool)
    pair_noise = np.sqrt(2.0) * noise_rgb
    mix_critical = chi2.isf(.05/max(1,len(pigments)),3)
    # Antialias colors mix ink with the visible substrate too. The pure paper
    # endpoint need not occur inside the observed ink mask; omitting it makes
    # the lightest sampled edge look like a second opaque outline pigment.
    for j,color in enumerate(pigments):
        others = np.concatenate((np.delete(pigments,j,axis=0), substrate_pigments))
        other_depths = np.concatenate((np.delete(depths,j), np.zeros(len(substrate_pigments))))
        for a,b in combinations(range(len(others)),2):
            # A boundary blends two layers, including at least one observed
            # ink. An arbitrary convex hull of artwork colors is not evidence
            # that a real source pigment is merely antialiasing.
            if a >= len(pigments)-1:
                continue
            if depths[j] > max(float(other_depths[a]), float(other_depths[b])):
                continue
            direction = others[a]-others[b]
            energy = float(direction @ direction)
            if energy == 0:
                continue
            amount = float((color-others[b]) @ direction/energy)
            projected = others[b]+np.clip(amount,0,1)*direction
            residual = np.linalg.norm(color-projected)
            sep = np.maximum(np.abs(color-projected)-1.0, 0.0)
            whitened_sq = float(np.sum((sep/pair_noise)**2))
            if 0 < amount < 1 and (residual <= noise or whitened_sq <= mix_critical):
                endpoints[j] = False
    candidates = pigment_evidence & endpoints
    if not candidates.any():
        # Weak/ambiguous source colors are ranked by their measured distinction
        # from visible substrate, never by palette order or a default ink color.
        # Use the spatial probe when the pooled palette cannot distinguish any
        # source endpoint. Taking the maximum of both everywhere overweights
        # antialias edges against distant, differently colored artwork.
        scores = np.where(endpoints, local_source_contrast, -np.inf) if endpoints.any() else local_source_contrast
    else:
        # Thickness times substrate-normalized contrast measures spatial ink
        # signal. Depth alone can prefer a thick pale edge over a thin ink core;
        # contrast alone can prefer a bright outer outline over its inner fill.
        scores = np.where(candidates, depths*np.sqrt(source_contrast), -np.inf)
    fill_id = max(range(len(pigments)), key=lambda j: (scores[j], source_contrast[j]))
    core = pigment_labels == fill_id
    fg_med = pigments[fill_id]
    fg_rgba = [int(c) for c in np.rint(fg_med)] + [255]
    outline_rgba = [255, 255, 255, 255]
    outline_w = 0.0
    internal_outlined = False
    pure_core = roi_rgb[core]
    if len(pigments) > 1:
        # Score candidate outlines on the immediate adjacency band around `core` so
        # double-outlined glyphs (fill + inner white rim + outer black rim) cleanly
        # select the true inner rim instead of averaging inner+outer into muddy gray.
        to_fill = distance_transform_edt(~core)
        q1_act, q3_act = np.quantile(dist_in[active], [0.25, 0.75])
        rim_fence = max(2.0, float(q3_act + 1.5 * (q3_act - q1_act)))
        near_ring = active & (~core) & (to_fill <= max(2.0, float(q3_act)))
        eval_mask = near_ring if int(np.count_nonzero(near_ring)) >= 12 else (active & (~core))
        pixels = roi_rgb[eval_mask].astype(float)
        scores = np.full(len(pigments), -np.inf)
        for j, color in enumerate(pigments):
            direction = color - fg_med
            length2 = float(direction @ direction)
            if (j == fill_id or length2 == 0 or not outline_evidence[j]
                    or not endpoints[j] or depths[j] > depths[fill_id]):
                continue
            amount = np.clip((pixels - fg_med) @ direction / length2, 0, 1)
            errors = np.linalg.norm(pixels - (fg_med + amount[:, None] * direction), axis=1)
            # Score all source samples. A median lets an intermediate edge
            # explain half the samples while ignoring the actual outer pigment.
            scores[j] = -float(np.mean(errors**2))
        rim_id = int(np.argmax(scores))
        rim = pigment_labels == rim_id
        if np.isfinite(scores[rim_id]):
            accept_rim = True
            if collar.any() and bubble_kind not in ("floating_sfx", "diegetic_surface"):
                # Inside enclosed bubbles, any candidate rim lying on the convex segment
                # [fg_med, substrate_median] is anti-aliased edge transition to paper, not an outline.
                sub_center = np.median(roi_rgb[collar].astype(float), axis=0)
                line_dir = sub_center - fg_med
                line_len2 = float(line_dir @ line_dir)
                if line_len2 > 0:
                    rim_color = pigments[rim_id]
                    t_sub = float((rim_color - fg_med) @ line_dir / line_len2)
                    if -0.05 <= t_sub <= 1.05:
                        proj_sub = fg_med + np.clip(t_sub, 0.0, 1.0) * line_dir
                        ortho_sep = np.maximum(np.abs(rim_color - proj_sub) - 1.0, 0.0)
                        ortho_sq = float(np.sum((ortho_sep / pair_noise) ** 2))
                        if ortho_sq <= chi2.isf(.05/max(1,len(pigments)), 2):
                            accept_rim = False
            if accept_rim:
                # Restrict profiling to the measured stroke thickness: disconnected
                # background-color islands cannot become a giant outline estimate.
                rim_distances = to_fill[rim & (to_fill <= rim_fence)]
                if rim_distances.size:
                    outline_w = float(np.median(rim_distances))
                    outline_rgba = [int(c) for c in np.rint(pigments[rim_id])] + [255]
                    internal_outlined = True

    core_radii = distance_transform_edt(core)[core]
    eff_sw = float(2 * np.median(core_radii)) if core_radii.size else float(stroke_w)
    # Never flip enclosed speech/narration/thought bubbles or thin outlined subtitles to cursive brush!
    tremor = (
        float(np.clip((stroke_mod - 0.34) * 1.6, 0.0, 2.2))
        if (bubble_kind in ("shout_spiky", "floating_sfx") and (not internal_outlined) and stroke_mod > 0.34)
        else 0.0
    )
    weight_scale = 1.0
    if bubble_kind == "shout_spiky":
        font_cat = "gothic"
        weight_scale = 1.16
    elif bubble_kind == "narration_box":
        font_cat = "mincho" if orientation == "vertical" else "gothic"
    elif bubble_kind == "thought_cloud":
        font_cat = "rounded"
    elif bubble_kind in ("floating_sfx", "diegetic_surface"):
        if (not internal_outlined) and stroke_mod > 0.30 and eff_sw >= 3.8:
            font_cat = "brush"
            weight_scale = 1.12
        else:
            font_cat = "gothic"
            weight_scale = 1.14 if eff_sw >= 3.6 else 1.0
    else:
        font_cat = "gothic"

    # Extract compact 32x32 vertical/spatial interior exemplar texture from pure core pixels;
    # erode 1px outer transition when thick enough so solid outlined fills stay pure.
    core_dist = distance_transform_edt(core)
    deep_core = core & (core_dist > 1.0)
    texture_pixels = roi_rgb[deep_core] if np.count_nonzero(deep_core) >= 8 else (
        pure_core if internal_outlined and pure_core.shape[0] >= 4 else roi_rgb[core]
    )
    core_pixels = (
        np.clip(np.rint(pure_core), 0, 255).astype(np.uint8)
        if internal_outlined and pure_core.shape[0] >= 4
        else roi_rgb[core]
    )
    exemplar = np.resize(core_pixels, (32 * 32, 3)).reshape(32, 32, 3).astype(np.uint8)

    # Determine FusionMode from physical geometry & interior texture variance:
    # solid-filled outlined dialogue uses exact SDF solid fill rather than stretching a 32x32 crop.
    core_std = float(np.mean(np.std(texture_pixels.astype(np.float32), axis=0)))
    if bubble_kind == "diegetic_surface":
        fusion: FusionMode = "diegetic_planar"
    elif bubble_kind == "floating_sfx" and (
        (eff_sw >= 3.5 and core_std > max(24.0, 4.5 * noise))
        or (eff_sw >= 5.5 and core_std > max(18.0, 3.5 * noise))
    ):
        fusion = "exemplar_stroke"
    else:
        fusion = "clean_interior"

    # A paragraph's central moment describes its arrangement, not font shear.
    # Estimate shear within connected glyph components and accept a direction
    # only when the median differs from zero under its sampling uncertainty.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
    shears = []
    for cid in range(1, len(stats)):
        x, y, cw, ch, area = stats[cid]
        if ch <= cw or area <= 1:
            continue
        component = (labels[y:y+ch, x:x+cw] == cid).astype(np.uint8)
        moment = cv2.moments(component)
        if moment["mu02"] > 0:
            shears.append(moment["mu11"] / moment["mu02"])
    slant_deg = 0.0
    if len(shears) > 1:
        shear = float(np.median(shears))
        mad = float(np.median(np.abs(np.asarray(shears) - shear)))
        error = mad / 0.6744897501960817 / np.sqrt(len(shears))
        # Universal Gaussian threshold controls multiplicity over glyph samples.
        if abs(shear) > error * np.sqrt(2 * np.log(len(shears))):
            slant_deg = float(np.degrees(np.arctan(shear)))

    spec = StyleSpec(
        orientation="vertical" if orientation == "vertical" else "horizontal",
        font_category=font_cat,
        stroke_width_px=eff_sw,
        stroke_modulation=float(stroke_mod),
        tremor_amplitude=tremor,
        slant_deg=slant_deg,
        fg_color_rgba=fg_rgba,
        outline_color_rgba=outline_rgba,
        outline_width_px=outline_w,
        blend_alpha=1.0,
        blend_mode="normal",
        ink_density_orig=ink_density_orig,
        hollow_mode="solid",
        weight_scale=weight_scale,
    )
    return spec, fusion, exemplar


def apply_exemplar_stroke_shader(
    glyph_alpha: np.ndarray,
    exemplar_patch: np.ndarray,
    style: StyleSpec,
) -> np.ndarray:
    """
    Zero-VAE SDF-Parameterized Exemplar Stroke Shader:
    Synthesizes an independent RGBA layer (`Z1`) around `glyph_alpha` reproducing:
      1) Outer sticker outline / halo (`outline_width_px`, `outline_color_rgba`) via continuous
         supersampled Euclidean SDF with smooth anti-aliased edge ramp & outer feathering rolloff
      2) Interior pigment/texture sampled from `exemplar_patch` (preserving screentone/gradient fill)
      3) Procedural high-frequency SDF edge perturbation (`tremor_amplitude`) when source strokes trembled
    Guarantees ZERO re-encoding of surrounding raw artwork (`L_inf == 0` where `alpha == 0`)!
    """
    h, w = glyph_alpha.shape[:2]
    out_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    nz_y, nz_x = np.nonzero(glyph_alpha > 1e-3)
    if nz_y.size == 0:
        return out_rgba

    halo_w = float(style.outline_width_px)
    tremor_amp = float(style.tremor_amplitude)
    # Pixel-grid Nyquist AA half-width (<= 0.5px) + scale-proportional outer visual feathering
    aa_half = 0.50 if halo_w > 0.25 else 0.0
    feather_span = (0.25 * halo_w) if halo_w > 0.25 else 0.0
    r_out = halo_w + aa_half + 0.5 * feather_span
    edge_width = max(2.0 * aa_half + feather_span, 1e-3)
    pad = int(np.ceil(max(r_out if halo_w > 0.25 else 0.0, 0.0) + max(tremor_amp, 0.0) + 1.0))
    y0 = max(0, int(nz_y.min()) - pad)
    y1 = min(h, int(nz_y.max()) + pad + 1)
    x0 = max(0, int(nz_x.min()) - pad)
    x1 = min(w, int(nz_x.max()) + pad + 1)
    ch, cw = y1 - y0, x1 - x0
    crop_alpha = glyph_alpha[y0:y1, x0:x1]

    if tremor_amp > 0.05:
        base_bin = crop_alpha >= 0.5
        if not np.any(base_bin):
            base_bin = crop_alpha > 0.1
        if not np.any(base_bin):
            fg_cov = np.clip(crop_alpha, 0.0, 1.0)
            total_cov = fg_cov.copy()
        else:
            d_in = distance_transform_edt(base_bin).astype(np.float32)
            d_out = distance_transform_edt(~base_bin).astype(np.float32)
            sdf = d_in - d_out
            yy, xx = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            # Scale-proportional deterministic spatial harmonic phase tied to stroke width
            k_freq = 2.5 / max(float(style.stroke_width_px), 2.0)
            wave = np.sin((0.55 * k_freq) * xx + (0.85 * k_freq) * yy) * np.cos(
                (0.75 * k_freq) * yy - (0.35 * k_freq) * xx
            )
            sdf = sdf + tremor_amp * wave
            fg_cov = np.clip(sdf + 0.5, 0.0, 1.0)
            if halo_w > 0.25:
                t_edge = np.clip((sdf + r_out) / edge_width, 0.0, 1.0)
                total_cov = t_edge * t_edge * (3.0 - 2.0 * t_edge)
            else:
                total_cov = fg_cov.copy()
    else:
        fg_cov = np.clip(crop_alpha, 0.0, 1.0)
        if halo_w > 0.25:
            # 2x supersampled continuous Euclidean SDF eliminates staircase aliasing on curved outlines
            ss = 2
            up_alpha = cv2.resize(fg_cov.astype(np.float32), (cw * ss, ch * ss), interpolation=cv2.INTER_LINEAR)
            up_bin = up_alpha >= 0.5
            if not np.any(up_bin):
                up_bin = up_alpha > 0.1
            if not np.any(up_bin):
                total_cov = fg_cov.copy()
            else:
                d_in_up = distance_transform_edt(up_bin).astype(np.float32) / float(ss)
                d_out_up = distance_transform_edt(~up_bin).astype(np.float32) / float(ss)
                # Zero-centered subpixel contour distance + scale-proportional smooth edge ramp
                half_px = 0.5 / float(ss)
                sdf_up = np.where(
                    up_bin,
                    np.maximum(d_in_up - half_px, 0.0),
                    -np.maximum(d_out_up - half_px, 0.0),
                )
                t_edge = np.clip((sdf_up + r_out) / edge_width, 0.0, 1.0)
                total_cov_up = t_edge * t_edge * (3.0 - 2.0 * t_edge)
                total_cov = cv2.resize(total_cov_up, (cw, ch), interpolation=cv2.INTER_AREA)
        else:
            total_cov = fg_cov.copy()

    # An outline adds coverage. Binarizing an antialiased glyph for its SDF
    # must never erase fractional fill samples at that boundary.
    total_cov = np.maximum(total_cov, fg_cov)

    # Build interior RGB fill from exemplar patch (tiled/interpolated smoothly along glyph vertical extent)
    if exemplar_patch is not None and exemplar_patch.size >= 12:
        fill_full = cv2.resize(exemplar_patch, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        fill_rgb = fill_full[y0:y1, x0:x1]
        # Blend 65% exemplar texture + 35% canonical median foreground color to stabilize legibility
        fg_base = np.array(style.fg_color_rgba[:3], dtype=np.float32)[None, None, :]
        fill_rgb = 0.65 * fill_rgb + 0.35 * fg_base
    else:
        fill_rgb = np.broadcast_to(
            np.array(style.fg_color_rgba[:3], dtype=np.float32), (ch, cw, 3)
        )

    halo_rgb = np.array(style.outline_color_rgba[:3], dtype=np.float32)[None, None, :]
    # Composite foreground fill over halo ring in premultiplied alpha space
    denom_a = np.maximum(total_cov, 1e-6)
    ring_only = np.maximum(total_cov - fg_cov, 0.0)
    comp_rgb = (fg_cov[:, :, None] * fill_rgb + ring_only[:, :, None] * halo_rgb) / denom_a[:, :, None]

    alpha_u8 = np.clip(np.rint(total_cov * 255.0 * float(style.blend_alpha)), 0, 255).astype(np.uint8)
    active_out = alpha_u8 > 0
    crop_out = out_rgba[y0:y1, x0:x1]
    crop_out[active_out, :3] = np.clip(np.rint(comp_rgb[active_out]), 0, 255).astype(np.uint8)
    crop_out[:, :, 3] = alpha_u8
    return out_rgba


def compute_diegetic_homography(flat_hw: tuple[int, int], quad_local: np.ndarray) -> np.ndarray:
    """Compute exact 4-point Perspective Homography mapping flat (fh, fw) typography onto quad_local."""
    fh, fw = flat_hw
    pts = np.asarray(quad_local, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]

    center = ordered.mean(axis=0, keepdims=True)
    inset_quad = center + 0.88 * (ordered - center)
    src_corners = np.array(
        [[0.0, 0.0], [float(max(fw - 1, 1)), 0.0], [float(max(fw - 1, 1)), float(max(fh - 1, 1))], [0.0, float(max(fh - 1, 1))]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_corners, inset_quad)


def warp_diegetic_layer_perspective_or_spine(
    flat_rgba: np.ndarray,
    roi_shape: tuple[int, int],
    quad_local: np.ndarray | None = None,
    spine_local: np.ndarray | None = None,
    vertical: bool = False,
) -> np.ndarray:
    """
    Map canonical flat-rendered typography RGBA onto a 3D perspective planar surface (via exact
    4-point Perspective Homography) or curved banner/cloth trajectory (via Bookstein TPS).
    """
    th, tw = roi_shape
    fh, fw = flat_rgba.shape[:2]
    if quad_local is not None and len(quad_local) == 4:
        H_mat = compute_diegetic_homography((fh, fw), quad_local)
        return cv2.warpPerspective(
            flat_rgba, H_mat, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

    if spine_local is not None and len(spine_local) >= 4:
        spine = np.asarray(spine_local, dtype=np.float64)
        tangents = np.gradient(spine, axis=0)
        tnorm = np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-6)
        tx, ty = tangents[:, 0] / tnorm, tangents[:, 1] / tnorm
        # Only apply TPS curve warp when spine exhibits genuine tilt or curvature
        curved = (vertical and np.max(np.abs(tx)) >= 0.06) or ((not vertical) and np.max(np.abs(ty)) >= 0.06)
        if curved:
            diffs = np.diff(spine, axis=0)
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
            total_len = max(float(arc[-1]), 4.0)
            nx, ny = (ty, -tx) if vertical else (-ty, tx)
            half_t = 0.45 * float(tw if vertical else fh)

            # Inverse mapping: destination (tw, th) -> canonical flat (fw, fh)
            dst_ctrl = []
            src_ctrl = []
            for i in range(len(spine)):
                u_along = (arc[i] / total_len) * float((fh - 1) if vertical else (fw - 1))
                for sign_off, u_cross in [(-1.0, 0.0), (1.0, float((fw - 1) if vertical else (fh - 1)))]:
                    qx = float(spine[i, 0]) + sign_off * half_t * float(nx[i])
                    qy = float(spine[i, 1]) + sign_off * half_t * float(ny[i])
                    dst_ctrl.append([qx, qy])
                    src_ctrl.append([u_cross, u_along] if vertical else [u_along, u_cross])

            W_tps, A_tps = solve_tps_weights(np.array(dst_ctrl), np.array(src_ctrl), reg=1e-2)
            xx, yy = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
            map_uv = evaluate_tps_grid(np.stack([xx, yy], axis=-1), np.array(dst_ctrl), W_tps, A_tps)
            return cv2.remap(
                flat_rgba,
                map_uv[:, :, 0],
                map_uv[:, :, 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

    if (fh, fw) != (th, tw):
        return cv2.resize(flat_rgba, (tw, th), interpolation=cv2.INTER_LINEAR)
    return flat_rgba


def composite_node_layers(
    raw_roi_rgb: np.ndarray,
    clean_z0_rgb: np.ndarray,
    stroke_alpha_orig: np.ndarray,
    rendered_z1_rgba: np.ndarray,
    z2_occluder_mask: np.ndarray | None,
    shading_map_u8: np.ndarray | None,
    style: StyleSpec,
    fusion_mode: FusionMode,
) -> np.ndarray:
    """
    Physically exact 4-layer re-synthesis (`Z0 -> Intrinsic Shading -> Z1 Blend -> Z2 Occluder`),
    strictly guaranteeing bit-exact raw preservation (`out == raw_roi_rgb`) at every pixel
    untouched by `stroke_alpha_orig` and `rendered_z1_rgba`!
    """
    # Start strictly from raw_roi_rgb; only apply clean_z0_rgb where original stroke was erased!
    erased = stroke_alpha_orig > 0
    if z2_occluder_mask is not None:
        erased = erased & (~(z2_occluder_mask > 0))

    canvas = raw_roi_rgb.copy()
    canvas[erased] = clean_z0_rgb[erased]

    z1_rgb = rendered_z1_rgba[:, :, :3].astype(np.float32)
    z1_a = (rendered_z1_rgba[:, :, 3].astype(np.float32) / 255.0)

    if np.any(z1_a > 1e-3):
        base_f = canvas.astype(np.float32)
        # Intrinsic surface shading modulation (for diegetic signs half-covered in shadow screentone!)
        if fusion_mode == "diegetic_planar" and shading_map_u8 is not None and shading_map_u8.shape == canvas.shape:
            shade = shading_map_u8.astype(np.float32) / 255.0
            fg_shaded = z1_rgb * shade
        else:
            fg_shaded = z1_rgb

        if style.blend_mode == "multiply":
            blended = (base_f * fg_shaded) / 255.0
        elif style.blend_mode == "screen":
            blended = 255.0 - ((255.0 - base_f) * (255.0 - fg_shaded)) / 255.0
        else:
            blended = fg_shaded

        comp = (1.0 - z1_a[:, :, None]) * base_f + z1_a[:, :, None] * blended
        write_mask = z1_a > 1e-3
        canvas[write_mask] = np.clip(np.rint(comp[write_mask]), 0, 255).astype(np.uint8)

    # Finally composite Z2 foreground occluders (hair, blades, speedlines) BACK ON TOP OF Z1!
    if z2_occluder_mask is not None and np.any(z2_occluder_mask > 0):
        z2_bool = z2_occluder_mask > 127
        canvas[z2_bool] = raw_roi_rgb[z2_bool]

    return canvas
