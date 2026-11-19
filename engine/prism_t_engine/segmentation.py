from __future__ import annotations

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from .preprocess import estimate_noise_sigma, otsu_threshold_and_eta
from .schemas import BubbleKind


def compute_stroke_width_transform(gray: np.ndarray, dark_on_light: bool = True) -> np.ndarray:
    """
    Vectorized Epshtein et al. (CVPR 2010) Stroke Width Transform (SWT).
    Returns per-pixel physical stroke width (float32, inf where no opposing edge exists).
    """
    h, w = gray.shape[:2]
    g = gray.astype(np.float32)
    sigma_n = estimate_noise_sigma(g)

    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0) / 16.0
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1) / 16.0
    mag = np.hypot(gx, gy)

    active_grad = mag[mag > 2.5 * sigma_n]
    if active_grad.size < 16:
        return np.full((h, w), np.inf, dtype=np.float32)
    edge_thresh, _ = otsu_threshold_and_eta(active_grad)
    edges = mag >= max(edge_thresh * 0.6, 3.0 * sigma_n)

    ys, xs = np.nonzero(edges)
    if xs.size == 0:
        return np.full((h, w), np.inf, dtype=np.float32)

    # Unit gradient pointing INTO the stroke interior
    sign = -1.0 if dark_on_light else 1.0
    inv_mag = 1.0 / np.maximum(mag[ys, xs], 1e-6)
    dx = sign * gx[ys, xs] * inv_mag
    dy = sign * gy[ys, xs] * inv_mag

    # Maximum plausible stroke width bounded by 1/3 of ROI short side (scale-invariant)
    max_step = max(6, min(h, w) // 3)
    swt = np.full((h, w), np.inf, dtype=np.float32)

    # Vectorized ray-marching across all edge pixels simultaneously
    alive = np.ones(xs.shape[0], dtype=bool)
    hit_step = np.zeros(xs.shape[0], dtype=np.int32)

    # Unit gradient lookup at every pixel for opposing-direction verification
    ux_map = np.zeros((h, w), dtype=np.float32)
    uy_map = np.zeros((h, w), dtype=np.float32)
    ux_map[ys, xs] = dx
    uy_map[ys, xs] = dy

    cos_opp = -np.cos(np.pi / 6.0)  # Exact opposing-normal cone (30 deg tolerance)
    for step in range(1, max_step + 1):
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        rx = np.rint(xs[idx] + dx[idx] * step).astype(np.int32)
        ry = np.rint(ys[idx] + dy[idx] * step).astype(np.int32)
        in_bounds = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)

        out_idx = idx[~in_bounds]
        alive[out_idx] = False

        valid_idx = idx[in_bounds]
        rx, ry = rx[in_bounds], ry[in_bounds]
        hit_edge = edges[ry, rx]
        if np.any(hit_edge):
            cand = valid_idx[hit_edge]
            cx, cy = rx[hit_edge], ry[hit_edge]
            dot = dx[cand] * ux_map[cy, cx] + dy[cand] * uy_map[cy, cx]
            valid_opp = dot <= cos_opp
            matched = cand[valid_opp]
            hit_step[matched] = step
            # Terminate all rays that struck an edge (valid or invalid)
            alive[cand] = False

    valid_rays = np.nonzero(hit_step > 0)[0]
    if valid_rays.size == 0:
        return swt

    # Pass 1: assign minimum ray length along each valid stroke chord
    lengths = hit_step[valid_rays].astype(np.float32)
    order = np.argsort(-lengths)  # Paint widest first so narrower chords overwrite cleanly
    for r_i in valid_rays[order]:
        L = int(hit_step[r_i])
        w_val = float(L)
        ts = np.arange(0, L + 1, dtype=np.float32)
        px = np.clip(np.rint(xs[r_i] + dx[r_i] * ts).astype(np.int32), 0, w - 1)
        py = np.clip(np.rint(ys[r_i] + dy[r_i] * ts).astype(np.int32), 0, h - 1)
        swt[py, px] = np.minimum(swt[py, px], w_val)

    # Pass 2: Epshtein median refinement to suppress corner diagonal overestimation
    for r_i in valid_rays:
        L = int(hit_step[r_i])
        ts = np.arange(0, L + 1, dtype=np.float32)
        px = np.clip(np.rint(xs[r_i] + dx[r_i] * ts).astype(np.int32), 0, w - 1)
        py = np.clip(np.rint(ys[r_i] + dy[r_i] * ts).astype(np.int32), 0, h - 1)
        med_w = float(np.median(swt[py, px]))
        swt[py, px] = np.minimum(swt[py, px], med_w)

    return swt


def closed_form_stroke_matting(
    roi_rgb: np.ndarray, trimap: np.ndarray
) -> np.ndarray:
    """
    Exact Levin et al. (TPAMI 2008) Closed-Form Matting Laplacian solver.
    Args:
        roi_rgb: uint8 (H, W, 3) image crop.
        trimap: int8/int32 (H, W) with:
            1  = definite foreground stroke core
            0  = definite background (including hollow glyph interiors!)
           -1  = unknown sub-pixel transition band
    Returns:
        alpha: float32 (H, W) in [0.0, 1.0], guaranteed 0 on trimap==0 and 1 on trimap==1.
    """
    h, w = roi_rgb.shape[:2]
    unknown = trimap < 0
    if not np.any(unknown):
        return (trimap > 0).astype(np.float32)
    if h < 3 or w < 3:
        return (trimap > 0).astype(np.float32)

    img = roi_rgb.astype(np.float64) / 255.0
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    sigma_n = estimate_noise_sigma(gray) / 255.0
    eps = max(sigma_n * sigma_n, 1e-5)

    n_pix = h * w
    idx_grid = np.arange(n_pix, dtype=np.int32).reshape(h, w)

    # Only 3x3 windows overlapping at least one unknown pixel contribute to unknown equations
    unk_dilated = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    win_mask = unk_dilated[1 : h - 1, 1 : w - 1]
    wy, wx = np.nonzero(win_mask)
    if wy.size == 0:
        return (trimap > 0).astype(np.float32)

    cy = wy + 1
    cx = wx + 1
    m_wins = cy.size

    # Gather 9-pixel neighborhoods: shape (M, 9, 3) and indices (M, 9)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    win_idx = np.empty((m_wins, 9), dtype=np.int32)
    win_rgb = np.empty((m_wins, 9, 3), dtype=np.float64)
    for k, (dy, dx) in enumerate(offsets):
        ny, nx = cy + dy, cx + dx
        win_idx[:, k] = idx_grid[ny, nx]
        win_rgb[:, k, :] = img[ny, nx, :]

    # Window mean (M, 1, 3) and regularized 3x3 covariance (M, 3, 3)
    mu = np.mean(win_rgb, axis=1, keepdims=True)
    centered = win_rgb - mu  # (M, 9, 3)
    cov = np.einsum("mni,mnj->mij", centered, centered) / 9.0
    cov += (eps / 9.0) * np.eye(3, dtype=np.float64)[None, :, :]

    # Solve (cov) * X = centered^T -> X shape (M, 3, 9)
    solved = np.linalg.solve(cov, np.swapaxes(centered, 1, 2))
    # Local 9x9 affinity W_k = (1/9) * (1 + centered @ solved)
    aff = (1.0 + np.einsum("mni,mik->mnk", centered, solved)) / 9.0
    # Local Laplacian block L_k = I_9 - aff
    lap_blocks = np.eye(9, dtype=np.float64)[None, :, :] - aff  # (M, 9, 9)

    rows = np.repeat(win_idx, 9, axis=1).ravel()
    cols = np.tile(win_idx, (1, 9)).ravel()
    vals = lap_blocks.ravel()
    L = sparse.csr_matrix((vals, (rows, cols)), shape=(n_pix, n_pix))

    known = (trimap >= 0).ravel()
    alpha_init = (trimap == 1).astype(np.float64).ravel()

    # Exact Dirichlet elimination on known nodes instead of arbitrary penalty weight!
    unk_flat = np.nonzero(~known)[0]
    kn_flat = np.nonzero(known)[0]
    if unk_flat.size == 0:
        return alpha_init.reshape(h, w).astype(np.float32)

    L_uu = L[unk_flat, :][:, unk_flat]
    L_uk = L[unk_flat, :][:, kn_flat]
    rhs = -L_uk.dot(alpha_init[kn_flat])
    # Tiny machine-precision diagonal safeguard for isolated singular components
    L_uu = L_uu + sparse.diags(np.full(unk_flat.size, 1e-9, dtype=np.float64))

    alpha_u = spsolve(L_uu.tocsc(), rhs)
    alpha_out = alpha_init.copy()
    alpha_out[unk_flat] = np.clip(alpha_u, 0.0, 1.0)
    return alpha_out.reshape(h, w).astype(np.float32)


def build_stroke_trimap_and_alpha(
    roi_rgb: np.ndarray, coarse_mask: np.ndarray | None = None
) -> tuple[np.ndarray, float, float]:
    """
    Construct a self-calibrated stroke Trimap (preserving hollow glyph interiors in Background!)
    and solve Closed-Form Alpha Matting.
    Returns (alpha_map [0..1], median_stroke_width_px, stroke_modulation_cv).
    """
    h, w = roi_rgb.shape[:2]
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma_n = estimate_noise_sigma(gray)
    rgb_f = roi_rgb.astype(np.float32)

    # Determine polarity (dark ink on bright substrate vs light ink on dark substrate) via Otsu partitions
    thresh_g, eta_g = otsu_threshold_and_eta(gray)
    fg_part = gray < thresh_g
    bg_part = ~fg_part
    if np.any(fg_part) and np.any(bg_part):
        gap_g = abs(float(np.median(gray[bg_part])) - float(np.median(gray[fg_part])))
    else:
        gap_g = float(np.percentile(gray, 90) - np.percentile(gray, 10))

    stroke_est = 2.0
    local_ring = np.zeros((h, w), dtype=bool)
    if coarse_mask is not None and np.any(coarse_mask > 0):
        c_bin = (coarse_mask > 0).astype(np.uint8)
        c_dist = cv2.distanceTransform(c_bin, cv2.DIST_L2, 3)
        pos_d = c_dist[c_dist > 0]
        if pos_d.size > 0:
            stroke_est = max(1.0, 2.0 * float(np.median(pos_d)))
        in_med = float(np.median(gray[c_bin > 0]))
        ring_r = max(2, int(round(max(stroke_est * 1.2, min(h, w) * 0.04))))
        local_ring = (
            cv2.dilate(c_bin, np.ones((2 * ring_r + 1, 2 * ring_r + 1), np.uint8)) > 0
        ) & (coarse_mask == 0)
        out_pixels = gray[local_ring] if np.any(local_ring) else gray[coarse_mask == 0]
        out_med = float(np.median(out_pixels)) if out_pixels.size > 0 else float(np.median(gray))
        dark_on_light = in_med <= out_med
    else:
        # Compare minority vs majority Otsu partition
        frac_dark = float(np.mean(gray < thresh_g))
        dark_on_light = frac_dark <= 0.55

    # Chromatic + luminance contrast relative to substrate (captures thin sepia/colored strokes)
    signal = (255.0 - gray) if dark_on_light else gray
    if coarse_mask is not None and np.any(coarse_mask > 0) and np.count_nonzero(local_ring) >= 8:
        bg_ref = local_ring
    else:
        bg_ref = (gray >= thresh_g) if dark_on_light else (gray < thresh_g)

    if np.any(bg_ref):
        bg_med_rgb = np.median(rgb_f[bg_ref], axis=0)
        bg_dists = np.linalg.norm(rgb_f[bg_ref] - bg_med_rgb[None, :], axis=1)
        bg_mad = 1.4826 * float(np.median(bg_dists))
        if bg_mad > max(5.0 * sigma_n, 0.16 * max(gap_g, 1.0)) and coarse_mask is not None and np.any(coarse_mask > 0):
            # Heterogeneous CG artwork behind borderless text: interpolate per-pixel local background
            # from unmasked exterior flanking pixels so sky/hair/outfit transitions never flip ink vs bg.
            env_pad = max(2, int(round(stroke_est * 1.1)))
            env_valid = (
                cv2.dilate((coarse_mask > 0).astype(np.uint8), np.ones((2 * env_pad + 1, 2 * env_pad + 1), np.uint8)) == 0
            )
            if np.count_nonzero(env_valid) < 16:
                env_valid = bg_ref
            v_f = env_valid.astype(np.float32)
            sig_env = max(2.0, stroke_est * 2.2)
            bv = cv2.GaussianBlur(v_f, (0, 0), sig_env)
            bc = cv2.GaussianBlur(rgb_f * v_f[:, :, None], (0, 0), sig_env)
            local_bg = np.broadcast_to(bg_med_rgb, rgb_f.shape).copy()
            ok1 = bv > 0.03
            if not np.all(ok1):
                bv2 = cv2.GaussianBlur(v_f, (0, 0), sig_env * 2.5)
                bc2 = cv2.GaussianBlur(rgb_f * v_f[:, :, None], (0, 0), sig_env * 2.5)
                ok2 = bv2 > 0.01
                local_bg[ok2] = bc2[ok2] / bv2[ok2, None]
            local_bg[ok1] = bc[ok1] / bv[ok1, None]
            local_bg_g = 0.299 * local_bg[:, :, 0] + 0.587 * local_bg[:, :, 1] + 0.114 * local_bg[:, :, 2]
            chroma_sig = np.linalg.norm(rgb_f - local_bg, axis=2)
            eff_sig = np.maximum(np.abs(gray - local_bg_g), 0.85 * chroma_sig)
            bg_mad = 1.4826 * float(np.median(eff_sig[env_valid]))
        else:
            chroma_sig = np.linalg.norm(rgb_f - bg_med_rgb[None, None, :], axis=2)
            bg_g_med = float(np.median(gray[bg_ref]))
            eff_sig = np.maximum(np.abs(gray - bg_g_med), 0.80 * chroma_sig)
    else:
        bg_mad = float("inf")
        eff_sig = signal

    if coarse_mask is not None and np.any(coarse_mask > 0):
        # Dilate coarse_mask proportionally to stroke scale to capture anti-aliased glyph & halo edges
        pad = max(2, int(round(stroke_est * 0.85)))
        search_zone = cv2.dilate((coarse_mask > 0).astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
        sub_vals = eff_sig[search_zone]
    else:
        search_zone = np.ones((h, w), dtype=bool)
        sub_vals = eff_sig.ravel()

    t_otsu, _ = otsu_threshold_and_eta(sub_vals)
    if np.isfinite(bg_mad) and bg_mad <= max(4.0 * sigma_n, 0.18 * max(t_otsu, 1.0)):
        ink_thresh = min(t_otsu, max(0.42 * t_otsu, 4.5 * bg_mad, 5.5 * sigma_n))
    else:
        ink_thresh = t_otsu
    raw_ink = (eff_sig >= ink_thresh) & search_zone

    # Estimate physical stroke width via SWT and continuous EDT ridge modulation (minus pixel-grid floor)
    swt = compute_stroke_width_transform(gray, dark_on_light=dark_on_light)
    valid_swt = swt[np.isfinite(swt) & raw_ink]
    dist_in = distance_transform_edt(raw_ink).astype(np.float32)
    if valid_swt.size >= 8:
        stroke_w = float(np.median(valid_swt))
    else:
        radii = dist_in[raw_ink]
        stroke_w = float(2.0 * np.percentile(radii, 75)) if radii.size > 0 else 2.5

    if np.any(raw_ink):
        dil_d = cv2.dilate(dist_in, np.ones((3, 3), np.uint8))
        ridge = raw_ink & (dist_in >= dil_d - 1e-4) & (dist_in > 0.45)
        widths_r = 2.0 * dist_in[ridge] if np.count_nonzero(ridge) >= 6 else 2.0 * dist_in[raw_ink]
        med_r = float(np.median(widths_r)) if widths_r.size > 0 else stroke_w
        quant_floor = 2.0 / float(np.sqrt(12.0))
        intrinsic_mad = max(0.0, 1.4826 * float(np.median(np.abs(widths_r - med_r))) - quant_floor)
        stroke_mod = float(intrinsic_mad / max(stroke_w, 1.0))
    else:
        stroke_mod = 0.0

    stroke_w = float(np.clip(stroke_w, 1.5, max(4.0, min(h, w) * 0.25)))
    # Transition band radius: proportional to stroke width & bounded by ROI short side
    max_band = max(2, int(np.ceil(min(h, w) * 0.08)))
    band_r = max(1, min(max_band, int(np.ceil(stroke_w * 0.25))))

    dist_to_bg = distance_transform_edt(raw_ink)
    dist_to_fg = distance_transform_edt(~raw_ink)

    trimap = np.full((h, w), -1, dtype=np.int8)
    # Definite foreground core: interior of stroke deeper than anti-aliasing band
    fg_core = raw_ink & (dist_to_bg > band_r)
    if not np.any(fg_core) and np.any(raw_ink):
        # Ultra-thin 1-2px strokes: local maxima of contrast inside raw_ink act as core seeds
        q65 = float(np.quantile(eff_sig[raw_ink], 0.65))
        fg_core = raw_ink & (eff_sig >= max(t_otsu, q65))
        if not np.any(fg_core):
            q75 = float(np.quantile(eff_sig[raw_ink], 0.75))
            fg_core = raw_ink & (eff_sig >= q75)
    trimap[fg_core] = 1

    # Definite background: outside band_r from raw_ink (Crucially includes hollow letter interiors!)
    bg_core = (~raw_ink) & (dist_to_fg > band_r)
    if coarse_mask is not None:
        bg_core = bg_core | (~search_zone)
    trimap[bg_core] = 0

    alpha = closed_form_stroke_matting(roi_rgb, trimap)
    return alpha, stroke_w, stroke_mod


def _fit_1d_gmm_bic(x: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Exact 1D 1-Gaussian vs 2-Gaussian EM with Bayesian Information Criterion (BIC) model selection.
    Returns (prefers_two_clusters, means [mu_small, mu_large], labels {0=small, 1=large}).
    """
    n = x.size
    if n < 6:
        return False, np.array([float(np.mean(x)), float(np.mean(x))]), np.zeros(n, dtype=np.int32)

    mu1 = float(np.mean(x))
    var1 = max(float(np.var(x)), 1e-4)
    ll1 = float(np.sum(-0.5 * np.log(2.0 * np.pi * var1) - 0.5 * ((x - mu1) ** 2) / var1))
    bic1 = 2.0 * np.log(n) - 2.0 * ll1  # 2 free params: (mu, var)

    # Initialize 2-GMM at 25th and 75th quantiles
    m = np.quantile(x, [0.25, 0.75]).astype(np.float64)
    if m[1] - m[0] < 1e-3:
        return False, np.array([mu1, mu1]), np.zeros(n, dtype=np.int32)
    v = np.array([var1 * 0.5, var1 * 0.5], dtype=np.float64)
    pi = np.array([0.35, 0.65], dtype=np.float64)

    for _ in range(40):
        # E-step
        p0 = pi[0] * np.exp(-0.5 * ((x - m[0]) ** 2) / v[0]) / np.sqrt(2.0 * np.pi * v[0])
        p1 = pi[1] * np.exp(-0.5 * ((x - m[1]) ** 2) / v[1]) / np.sqrt(2.0 * np.pi * v[1])
        denom = np.maximum(p0 + p1, 1e-30)
        g0 = p0 / denom
        g1 = 1.0 - g0
        # M-step
        n0, n1 = float(np.sum(g0)), float(np.sum(g1))
        if n0 < 1.5 or n1 < 1.5:
            return False, np.array([mu1, mu1]), np.zeros(n, dtype=np.int32)
        pi[0], pi[1] = n0 / n, n1 / n
        m[0] = float(np.sum(g0 * x) / n0)
        m[1] = float(np.sum(g1 * x) / n1)
        v[0] = max(float(np.sum(g0 * (x - m[0]) ** 2) / n0), 1e-3)
        v[1] = max(float(np.sum(g1 * (x - m[1]) ** 2) / n1), 1e-3)

    # Order so cluster 0 is smaller scale, cluster 1 is larger scale
    if m[0] > m[1]:
        m = m[::-1]
        v = v[::-1]
        pi = pi[::-1]

    p0 = pi[0] * np.exp(-0.5 * ((x - m[0]) ** 2) / v[0]) / np.sqrt(2.0 * np.pi * v[0])
    p1 = pi[1] * np.exp(-0.5 * ((x - m[1]) ** 2) / v[1]) / np.sqrt(2.0 * np.pi * v[1])
    ll2 = float(np.sum(np.log(np.maximum(p0 + p1, 1e-30))))
    bic2 = 5.0 * np.log(n) - 2.0 * ll2  # 5 free params: (pi0, m0, m1, v0, v1)

    labels = (p1 >= p0).astype(np.int32)
    return bool(bic2 < bic1), m, labels


def separate_kanji_ruby_components(
    binary_ink: np.ndarray, vertical: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale-invariant Kanji-Ruby (Furigana / ルビ) geometric separator using:
      1) Cross-axis projection column/row clustering, and
      2) BIC-selected 2-component log-scale Gaussian Mixture on track thickness & glyph size,
         verified by canonical typographical adjacency (ruby to the right of vertical columns,
         or above horizontal rows).
    Returns:
      (base_mask, ruby_mask) as uint8 masks partitioning binary_ink without losing any ink pixel.
    """
    ink = (binary_ink > 0).astype(np.uint8)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num_labels <= 4:
        return ink.copy(), np.zeros_like(ink)

    # Project ink onto cross-reading axis to isolate parallel columns (vertical) or rows (horizontal)
    # vertical=True => cross-axis is X (axis=0 sum); vertical=False => cross-axis is Y (axis=1 sum)
    proj = np.sum(ink, axis=0 if vertical else 1).astype(np.float64)
    active = proj > 0
    if not np.any(active):
        return ink.copy(), np.zeros_like(ink)

    # Connected 1D intervals along cross-axis
    padded = np.concatenate([[False], active, [False]])
    starts = np.nonzero(padded[1:] & ~padded[:-1])[0]
    ends = np.nonzero(~padded[1:] & padded[:-1])[0]  # exclusive
    spans = ends - starts
    if spans.size < 2:
        # Check component-level bimodal scale within merged span via log-dimension GMM
        comp_scales = (
            stats[1:, cv2.CC_STAT_WIDTH] if vertical else stats[1:, cv2.CC_STAT_HEIGHT]
        ).astype(np.float64)
        valid_c = stats[1:, cv2.CC_STAT_AREA] >= 3
        if np.count_nonzero(valid_c) < 6:
            return ink.copy(), np.zeros_like(ink)
        is_bimodal, means, c_labels = _fit_1d_gmm_bic(np.log(np.maximum(comp_scales[valid_c], 1.0)))
        ratio = float(np.exp(means[0] - means[1]))
        # Standard typographic ruby scale ratio in JIS X 4051 is ~0.5 (accept physical bimodal mode < exp(-0.45) ~ 0.64)
        if not is_bimodal or not (0.20 <= ratio <= np.exp(-0.45)):
            return ink.copy(), np.zeros_like(ink)
        valid_ids = np.nonzero(valid_c)[0] + 1
        small_ids = valid_ids[c_labels == 0]
        large_ids = valid_ids[c_labels == 1]
        # Verify spatial offset: small glyphs must sit systematically on the ruby side of nearest large glyph
        ruby_ids: list[int] = []
        large_pts = centroids[large_ids]
        exp_large = float(np.exp(means[1]))
        for sid in small_ids:
            sx, sy = centroids[sid]
            dists = np.hypot(large_pts[:, 0] - sx, large_pts[:, 1] - sy)
            nearest = int(np.argmin(dists))
            lx, ly = large_pts[nearest]
            if vertical:
                # Ruby sits immediately to the right (+x) of vertical column
                if 0.25 * exp_large <= (sx - lx) <= 1.6 * exp_large and abs(sy - ly) <= 2.0 * exp_large:
                    ruby_ids.append(int(sid))
            else:
                # Ruby sits immediately above (-y) horizontal line
                if 0.25 * exp_large <= (ly - sy) <= 1.6 * exp_large and abs(sx - lx) <= 2.0 * exp_large:
                    ruby_ids.append(int(sid))
        if not ruby_ids:
            return ink.copy(), np.zeros_like(ink)
        ruby_mask = np.isin(labels_im, ruby_ids).astype(np.uint8)
        base_mask = (ink & ~ruby_mask).astype(np.uint8)
        return base_mask, ruby_mask

    # Multi-track case: test 1D log-span bimodal GMM across tracks + typographic pairing
    if spans.size >= 2:
        log_spans = np.log(np.maximum(spans.astype(np.float64), 1.0))
        # For 2..5 tracks (e.g. 1 main column + 1 ruby column), BIC on N=2..5 has small-sample penalty,
        # so also pair adjacent narrow-wide tracks whose span ratio matches JIS X 4051 ruby geometry
        is_bimodal, means, t_labels = _fit_1d_gmm_bic(log_spans)
        ruby_intervals: list[tuple[int, int]] = []
        if is_bimodal and 0.20 <= np.exp(means[0] - means[1]) <= np.exp(-0.40):
            cand_ruby = set(np.nonzero(t_labels == 0)[0].tolist())
        else:
            # Direct adjacent track scale ratio test against median wide track
            max_span = float(np.max(spans))
            cand_ruby = {
                i for i, sp in enumerate(spans) if 0.20 * max_span <= sp <= 0.62 * max_span
            }

        for idx in cand_ruby:
            # Ruby track must tightly hug a wider base track on the canonical side:
            # vertical JP: ruby track i is immediately to the right of base track i-1
            # horizontal: ruby track i is immediately above base track i+1
            base_idx = idx - 1 if vertical else idx + 1
            if 0 <= base_idx < len(spans) and base_idx not in cand_ruby:
                base_w = float(spans[base_idx])
                gap = float(starts[idx] - ends[base_idx]) if vertical else float(starts[base_idx] - ends[idx])
                if 0.0 <= gap <= 0.85 * base_w and spans[idx] <= 0.65 * base_w:
                    ruby_intervals.append((int(starts[idx]), int(ends[idx])))

        if ruby_intervals:
            ruby_mask = np.zeros_like(ink)
            for s, e in ruby_intervals:
                if vertical:
                    ruby_mask[:, s:e] = ink[:, s:e]
                else:
                    ruby_mask[s:e, :] = ink[s:e, :]
            base_mask = (ink & ~ruby_mask).astype(np.uint8)
            return base_mask, ruby_mask

    return ink.copy(), np.zeros_like(ink)


def estimate_source_font_scale(
    base_mask: np.ndarray, vertical: bool, source_text: str = ""
) -> float:
    """
    Estimate physical em-square font size (in pixels) of source text directly from binary glyph ink
    after ruby stripping, via cross-axis projection tracks and track-to-track pitch.
    """
    if base_mask is None or base_mask.size == 0:
        return 20.0
    h, w = base_mask.shape[:2]
    ink = (base_mask > 0).astype(np.uint8)
    if np.count_nonzero(ink) < 4:
        return float(np.clip(min(h, w) * 0.8, 9.0, max(9.0, float(max(h, w)))))

    proj = np.sum(ink, axis=0 if vertical else 1) > 0
    # Bridge tiny <= 2px intra-glyph gaps along cross-axis
    bridged = cv2.morphologyEx(
        proj.astype(np.uint8).reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 3), np.uint8),
    ).ravel() > 0

    padded = np.concatenate([[False], bridged, [False]])
    starts = np.nonzero(padded[1:] & ~padded[:-1])[0]
    ends = np.nonzero(~padded[1:] & padded[:-1])[0]
    raw_spans = (ends - starts).astype(np.float64)
    keep = raw_spans >= 4.0
    starts, ends, spans = starts[keep], ends[keep], raw_spans[keep]

    em_est = 0.0
    clean_chars = [c for c in source_text.strip() if not c.isspace()]

    def _count_vertical_slots(chars: list[str]) -> int:
        slots = 0
        idx_c = 0
        while idx_c < len(chars):
            if chars[idx_c].isascii() and chars[idx_c].isalnum():
                run_start = idx_c
                while idx_c < len(chars) and chars[idx_c].isascii() and chars[idx_c].isalnum():
                    idx_c += 1
                run_len = idx_c - run_start
                slots += 1 if run_len <= 3 else run_len
            else:
                slots += 1
                idx_c += 1
        return slots

    if spans.size > 0:
        med_span = float(np.median(spans))
        if vertical:
            if spans.size == 1:
                row_has_ink = np.sum(ink, axis=1) > 0
                r_pad = np.concatenate([[False], row_has_ink, [False]])
                r_starts = np.nonzero(r_pad[1:] & ~r_pad[:-1])[0]
                r_ends = np.nonzero(~r_pad[1:] & r_pad[:-1])[0]
                gw_list: list[float] = []
                for rs, re_row in zip(r_starts, r_ends):
                    if re_row - rs >= 3:
                        cols_r = np.nonzero(np.sum(ink[rs:re_row, :], axis=0) > 0)[0]
                        if cols_r.size > 0:
                            gw_list.append(float(cols_r.max() - cols_r.min() + 1))
                if len(gw_list) >= 2:
                    em_est = float(np.median(gw_list)) / 0.94
                else:
                    em_est = med_span / 0.90
            else:
                em_est = med_span / 0.90

            if spans.size >= 2:
                centers = 0.5 * (starts.astype(np.float64) + ends.astype(np.float64))
                pitches = np.diff(centers)
                pos_p = pitches[pitches > 0]
                if pos_p.size > 0:
                    pitch_em = float(np.median(pos_p)) / 1.18
                    em_est = 0.65 * em_est + 0.35 * min(em_est * 1.25, pitch_em)
            elif len(clean_chars) >= 2:
                n_slots = _count_vertical_slots(clean_chars)
                if n_slots >= 2:
                    em_est = min(em_est, (float(h) / float(n_slots)) * 1.05, float(w) * 1.05)
        else:
            has_cjk = any(ord(c) >= 0x3000 for c in source_text)
            em_est = med_span / (0.88 if has_cjk else 0.76)
            if spans.size >= 2:
                centers = 0.5 * (starts.astype(np.float64) + ends.astype(np.float64))
                pitches = np.diff(centers)
                pos_p = pitches[pitches > 0]
                if pos_p.size > 0:
                    pitch_em = float(np.median(pos_p)) / 1.22
                    em_est = 0.65 * em_est + 0.35 * min(em_est * 1.25, pitch_em)
            elif has_cjk and len(clean_chars) >= 2:
                em_est = min(em_est, (float(w) / float(len(clean_chars))) * 1.06, float(h) * 1.05)
    else:
        num_cc, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        if num_cc > 1:
            dims = (
                stats[1:, cv2.CC_STAT_WIDTH] if vertical else stats[1:, cv2.CC_STAT_HEIGHT]
            ).astype(np.float64)
            valid = dims[dims >= 4.0]
            if valid.size > 0:
                em_est = float(np.median(valid)) / 0.85

    if em_est <= 1.0:
        em_est = float(min(h, w)) * 0.75
    return float(np.clip(em_est, 9.0, max(9.0, float(max(h, w)))))


def analyze_bubble_contour_and_tail(
    img_rgb: np.ndarray, text_bbox: list[int], text_mask: np.ndarray | None = None,
) -> tuple[list[list[float]], BubbleKind, list[float] | None, list[float] | None]:
    """
    Extract enclosing speech bubble interior contour (if any), classify contour geometry via
    Elliptic Fourier Descriptors & convexity structure, and locate the tail cusp & outward ray vector.
    Returns:
      (bubble_polygon, bubble_kind, tail_tip [x, y] | None, tail_dir [dx, dy] | None)
    """
    H, W = img_rgb.shape[:2]
    tx, ty, tw, th = text_bbox
    if tw <= 2 or th <= 2:
        return [], "floating_sfx", None, None
    footprint = np.ones((th,tw),np.uint8)
    if text_mask is not None and np.any(text_mask):
        yy,xx = np.nonzero(text_mask)
        footprint[:] = 0
        cv2.fillConvexPoly(footprint,cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)),1)

    span = max(tw, th)
    base_px = max(16, int(round(max(tw * 1.15, span * 0.95, min(H, W) * 0.12))))
    base_py = max(16, int(round(max(th * 1.15, span * 0.95, min(H, W) * 0.12))))

    def _extract_enclosed_mask(px_pad: int, py_pad: int, fixed_g_thresh: float | None = None):
        cx0, cy0 = max(0, tx - px_pad), max(0, ty - py_pad)
        cx1, cy1 = min(W, tx + tw + px_pad), min(H, ty + th + py_pad)
        sub_crop = img_rgb[cy0:cy1, cx0:cx1]
        sch, scw = sub_crop.shape[:2]
        s_gray = cv2.cvtColor(sub_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        s_sig = estimate_noise_sigma(s_gray)

        slx0, sly0 = tx - cx0, ty - cy0
        slx1, sly1 = slx0 + tw, sly0 + th
        in_g = s_gray[sly0:sly1, slx0:slx1]
        t_i, _ = otsu_threshold_and_eta(in_g)
        b_part = in_g >= t_i
        f_part = ~b_part
        if np.any(b_part) and np.any(f_part):
            gap_i = max(1.0, abs(float(np.median(in_g[b_part])) - float(np.median(in_g[f_part]))))
        else:
            gap_i = max(1.0, float(np.percentile(in_g, 90) - np.percentile(in_g, 10)))
        s_inner = b_part if np.mean(b_part) >= 0.35 else f_part
        if not np.any(s_inner):
            return None, cx0, cy0, cx1, cy1, sub_crop, 4.0 * s_sig, True, 0.0

        in_rgb = sub_crop[sly0:sly1, slx0:slx1].astype(np.float32)
        s_med = np.median(in_rgb[s_inner], axis=0)
        in_dists = np.linalg.norm(in_rgb[s_inner] - s_med[None, :], axis=1)
        raw_s_mad = 1.4826 * float(np.median(in_dists))
        raw_s_p85 = float(np.percentile(in_dists, 85))
        # Heterogeneous multi-colored artwork (hair/sky/outfit beneath borderless text) is NOT a bubble substrate
        if raw_s_mad > max(6.5 * s_sig, 0.16 * gap_i) and raw_s_p85 > max(14.0 * s_sig, 0.38 * gap_i):
            return None, cx0, cy0, cx1, cy1, sub_crop, 4.0 * s_sig, False, float(tw * th)
        # Cap substrate MAD proportionally to noise and local contrast span so periodic halftone screens never flood
        s_mad = np.clip(
            1.4826 * np.median(np.abs(in_rgb[s_inner] - s_med), axis=0),
            max(2.5 * s_sig, 0.02 * gap_i),
            max(4.5 * s_sig, 0.12 * gap_i),
        )

        sgx = cv2.Scharr(s_gray, cv2.CV_32F, 1, 0) / 16.0
        sgy = cv2.Scharr(s_gray, cv2.CV_32F, 0, 1) / 16.0
        s_grad = np.hypot(sgx, sgy)
        gt = fixed_g_thresh if fixed_g_thresh is not None else otsu_threshold_and_eta(s_grad)[0]

        s_mahal = np.sqrt(np.sum(((sub_crop.astype(np.float32) - s_med) / (s_mad + 1e-3)) ** 2, axis=2))
        s_pass = (s_mahal <= 3.5) & (s_grad <= max(gt, 4.0 * s_sig))

        s_bar = (~s_pass).astype(np.uint8)
        if text_mask is not None:
            # Clearing an entire rotated paragraph's AABB punched holes through
            # diagonal bubble walls. Clear only observed text strokes.
            local_bar = s_bar[sly0:sly1,slx0:slx1]
            local_bar[text_mask > 0] = 0
        s_bar = cv2.morphologyEx(s_bar, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, s_cc = cv2.connectedComponents((s_bar == 0).astype(np.uint8), connectivity=4)
        vote_labels = s_cc[sly0:sly1,slx0:slx1]
        votes = np.bincount(vote_labels.ravel())
        votes[0] = 0
        c_lbl = int(votes.argmax())
        if c_lbl == 0:
            return None, cx0, cy0, cx1, cy1, sub_crop, gt, True, 0.0

        bm = binary_fill_holes(s_cc == c_lbl).astype(np.uint8)
        a_bub = float(np.count_nonzero(bm))
        a_txt = float(tw * th)
        hf_res = np.abs(s_gray - cv2.GaussianBlur(s_gray, (5, 5), 1.2))
        txt_box_mask = np.zeros((sch, scw), dtype=np.uint8)
        txt_box_mask[sly0:sly1, slx0:slx1] = 1
        txt_dil = cv2.dilate(txt_box_mask, np.ones((7, 7), np.uint8)) > 0

        def _validate_candidate_bubble(cand_bm: np.ndarray) -> tuple[bool, bool]:
            crop_leak = (
                (cy0 > 0 and bool(np.any(cand_bm[0, :])))
                or (cy1 < H and bool(np.any(cand_bm[-1, :])))
                or (cx0 > 0 and bool(np.any(cand_bm[:, 0])))
                or (cx1 < W and bool(np.any(cand_bm[:, -1])))
            )
            if crop_leak:
                return False, True

            ca = float(np.count_nonzero(cand_bm))
            covered = cand_bm[sly0:sly1,slx0:slx1] > 0
            # A glyph's own gradient halo may form a closed island. The
            # enclosing interior must cover the space between glyphs too.
            ccov = float(np.mean(covered[footprint > 0]))
            # A tightly set paragraph can occupy nearly the whole bubble.
            # Enclosure and coverage, rather than an arbitrary extra 22% of
            # rectangle area, determine whether that interior is usable.
            if ccov < 0.88:
                return False, False

            non_txt = (cand_bm > 0) & (~txt_dil)
            n_non_txt = int(np.count_nonzero(non_txt))
            hf_p90 = float(np.percentile(hf_res[non_txt], 90)) if n_non_txt >= 16 else 0.0
            # Calibrate the residual in the same filter used for the test.
            # A high quantile cannot be compared directly with a one-sigma
            # MAD. The bright text substrate supplies observed paper residuals.
            substrate_hf = hf_res[sly0:sly1,slx0:slx1][s_inner]
            hf_q1,hf_q3 = np.quantile(substrate_hf,[.25,.75])
            if hf_p90 > max(hf_q3+1.5*(hf_q3-hf_q1), s_sig*np.sqrt(2*np.log(max(2,substrate_hf.size)))):
                return False, False
            # Reject complex multi-colored artwork (hair + sunset + uniform) accidentally trapped in gradient barriers
            if n_non_txt >= 24:
                bg_vals_rgb = sub_crop[non_txt].astype(np.float32)
                med_bg_nt = np.median(bg_vals_rgb, axis=0)
                dists_nt = np.linalg.norm(bg_vals_rgb - med_bg_nt[None, :], axis=1)
                mad_bg_nt = 1.4826 * float(np.median(dists_nt))
                p85_bg_nt = float(np.percentile(dists_nt, 85))
                if mad_bg_nt > max(6.0 * s_sig, 0.14 * gap_i) or p85_bg_nt > max(12.0 * s_sig, 0.35 * gap_i):
                    return False, False

            cnts_v, _ = cv2.findContours(cand_bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts_v:
                return False, False
            cnt_v = max(cnts_v, key=cv2.contourArea)
            hull_v = cv2.convexHull(cnt_v)
            sol_v = float(cv2.contourArea(cnt_v)) / max(float(cv2.contourArea(hull_v)), 1.0)

            page_touches = (
                int(cy0 == 0 and bool(np.any(cand_bm[0, :])))
                + int(cy1 == H and bool(np.any(cand_bm[-1, :])))
                + int(cx0 == 0 and bool(np.any(cand_bm[:, 0])))
                + int(cx1 == W and bool(np.any(cand_bm[:, -1])))
            )
            if page_touches >= 2:
                return False, False
            if page_touches == 1:
                if not (ca <= 4.5 * a_txt and sol_v >= 0.84 and hf_p90 <= max(8.5, 3.0 * s_sig)):
                    return False, False

            sdf_m = distance_transform_edt(cand_bm > 0)
            max_rm = max(float(np.max(sdf_m)), 1.0)
            tc_y = min(sch - 1, max(0, (sly0 + sly1) // 2))
            tc_x = min(scw - 1, max(0, (slx0 + slx1) // 2))
            tc_sdf = float(sdf_m[tc_y, tc_x])
            if ca > 5.5 * a_txt and (sol_v < 0.76 or tc_sdf < 0.34 * max_rm):
                return False, False

            rim_m = (cv2.dilate(cand_bm, np.ones((7, 7), np.uint8)) > 0) & (cand_bm == 0)
            if np.count_nonzero(rim_m) >= 8:
                rim_grad_p75 = float(np.percentile(s_grad[rim_m], 75))
                if rim_grad_p75 < max(6.0, 3.2 * s_sig):
                    return False, False

            return True, False

        valid_primary, leaked_p = _validate_candidate_bubble(bm)
        if valid_primary:
            return bm, cx0, cy0, cx1, cy1, sub_crop, gt, False, a_txt

        # Secondary pure gradient-barrier search inside enclosed regions
        gw_m = (s_grad > max(gt, 4.5 * s_sig)).astype(np.uint8)
        if text_mask is not None:
            gw_m[sly0:sly1,slx0:slx1][text_mask > 0] = 0
        gw_m = cv2.morphologyEx(gw_m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, gc_m = cv2.connectedComponents((gw_m == 0).astype(np.uint8), connectivity=4)
        vote_labels = gc_m[sly0:sly1,slx0:slx1]
        votes = np.bincount(vote_labels.ravel())
        votes[0] = 0
        gl = int(votes.argmax())
        if gl > 0:
            cbm = binary_fill_holes(gc_m == gl).astype(np.uint8)
            valid_sec, leaked_s = _validate_candidate_bubble(cbm)
            if valid_sec:
                return cbm, cx0, cy0, cx1, cy1, sub_crop, gt, False, a_txt
            leaked_p = leaked_p or leaked_s

        return None, cx0, cy0, cx1, cy1, sub_crop, gt, leaked_p, a_txt

    bubble_mask, x0, y0, x1, y1, crop, g_thresh, leaked_primary, area_text = _extract_enclosed_mask(base_px, base_py)
    # If primary crop clipped a long Webtoon pointer tail or stacked double-lobe bubble, retry once with expanded bounds
    # while holding g_thresh fixed to the primary local edge calibration!
    if bubble_mask is None and leaked_primary:
        exp_px = max(base_px, int(round(max(span * 1.35, min(H, W) * 0.28))))
        exp_py = max(base_py, int(round(max(span * 1.90, min(H, W) * 0.46))))
        if exp_px > base_px or exp_py > base_py:
            bm2, x0_2, y0_2, x1_2, y1_2, crop_2, _, _, _ = _extract_enclosed_mask(
                exp_px, exp_py, fixed_g_thresh=g_thresh
            )
            if bm2 is not None:
                bubble_mask, x0, y0, x1, y1, crop = bm2, x0_2, y0_2, x1_2, y1_2, crop_2

    ch, cw = crop.shape[:2]
    lx0, ly0 = tx - x0, ty - y0
    lx1, ly1 = lx0 + tw, ly0 + th

    if bubble_mask is None:
        # Check if enclosed in a planar quadrilateral sign/screen (diegetic surface) via Canny + approxPolyDP
        edges = cv2.Canny(crop, max(10.0, g_thresh * 0.5), max(30.0, g_thresh * 1.2))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            c_area = cv2.contourArea(cnt)
            if 1.1 * area_text <= c_area <= 0.85 * float(ch * cw):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    bx, by, bw, bh = cv2.boundingRect(approx)
                    if bx <= lx0 and by <= ly0 and bx + bw >= lx1 and by + bh >= ly1:
                        poly_pts = [[float(p[0][0] + x0), float(p[0][1] + y0)] for p in approx]
                        return poly_pts, "diegetic_surface", None, None
        return [], "floating_sfx", None, None

    # Fill internal text holes inside bubble_mask
    contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], "floating_sfx", None, None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 16:
        return [], "floating_sfx", None, None

    # Resample contour uniformly at M=128 arc-length points for Fourier & curvature analysis
    pts = cnt[:, 0, :].astype(np.float64)
    closed_pts = np.vstack([pts, pts[0]])
    seg_len = np.hypot(np.diff(closed_pts[:, 0]), np.diff(closed_pts[:, 1]))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum_len[-1]
    if total_len <= 1e-3:
        return [], "floating_sfx", None, None

    m_samples = 128
    s_grid = np.linspace(0.0, total_len, m_samples, endpoint=False)
    xs = np.interp(s_grid, cum_len, closed_pts[:, 0])
    ys = np.interp(s_grid, cum_len, closed_pts[:, 1])

    # Rectangularity test (narration box): area / minAreaRect_area
    rect = cv2.minAreaRect(cnt)
    rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
    rectangularity = float(cv2.contourArea(cnt)) / rect_area

    # Radial Fourier Descriptor energy around centroid
    cx_b, cy_b = float(np.mean(xs)), float(np.mean(ys))
    radii = np.hypot(xs - cx_b, ys - cy_b)
    r_mean = max(float(np.mean(radii)), 1.0)
    r_norm = (radii - r_mean) / r_mean
    spec = np.abs(np.fft.rfft(r_norm)) ** 2
    total_spec = max(float(np.sum(spec[1:])), 1e-9)
    # Harmonics k >= 7 capture spiky teeth / cloud scallops; k = 4 captures rectangular lobes
    high_freq_ratio = float(np.sum(spec[7:])) / total_spec
    high_rms = float(np.sqrt(np.sum(spec[7:]) / m_samples))

    # Signed discrete curvature kappa(s) along smoothed periodic contour
    # Periodic central differences
    dx = 0.5 * (np.roll(xs, -2) - np.roll(xs, 2))
    dy = 0.5 * (np.roll(ys, -2) - np.roll(ys, 2))
    ddx = np.roll(xs, -2) - 2.0 * xs + np.roll(xs, 2)
    ddy = np.roll(ys, -2) - 2.0 * ys + np.roll(ys, 2)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
    # Orient contour counter-clockwise in standard math coords (or normalize sign by mean curvature > 0)
    signed_area = 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
    orient = 1.0 if signed_area >= 0 else -1.0
    kappa = orient * (dx * ddy - dy * ddx) / denom  # > 0 at outward convex peaks, < 0 at inward concavities

    # Classify bubble semantics from exact geometric invariants
    hull = cv2.convexHull(cnt)
    solidity = float(cv2.contourArea(cnt)) / max(float(cv2.contourArea(hull)), 1.0)
    concavity_count = int(np.sum((kappa < -1.0 / r_mean) & (np.roll(kappa, 1) >= kappa) & (np.roll(kappa, -1) > kappa)))

    # Manhattan Rectilinearity R_4 = sum(ds * cos(4*theta)) / sum(ds): +1 for axis-aligned rects & L-boxes!
    seg_dx = np.roll(xs, -1) - xs
    seg_dy = np.roll(ys, -1) - ys
    seg_ds = np.hypot(seg_dx, seg_dy)
    theta_seg = np.arctan2(seg_dy, seg_dx)
    manhattan_r4 = float(np.sum(seg_ds * np.cos(4.0 * theta_seg)) / max(float(np.sum(seg_ds)), 1e-6))

    if (rectangularity >= 0.88 and solidity >= 0.92) or (manhattan_r4 >= 0.72 and solidity >= 0.72 and concavity_count <= 4):
        bubble_kind: BubbleKind = "narration_box"
    elif high_rms >= 0.045 and high_freq_ratio >= 0.30 and concavity_count >= 7:
        # Distinguish spiky shout (sharp outward cusps) vs scalloped cloud (rounded lobes, sharp inward cusps)
        pos_max = float(np.percentile(kappa, 96))
        neg_max = float(np.percentile(-kappa, 96))
        bubble_kind = "shout_spiky" if pos_max >= neg_max * 0.85 else "thought_cloud"
    else:
        bubble_kind = "speech_oval"

    # Tail-tip detection on speech bubbles: single dominant outward protrusion with high convex curvature
    tail_tip: list[float] | None = None
    tail_dir: list[float] | None = None
    if bubble_kind in ("speech_oval", "thought_cloud") and len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (ma, MA), angle_deg = ellipse
        a_semi, b_semi = max(ma * 0.5, 1.0), max(MA * 0.5, 1.0)
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Algebraic normalized radial distance from fitted body ellipse (> 1 means protruding outside ellipse)
        ux = (xs - ex) * cos_t + (ys - ey) * sin_t
        uy = -(xs - ex) * sin_t + (ys - ey) * cos_t
        ell_r = np.hypot(ux / a_semi, uy / b_semi)

        protrusion = ell_r - 1.0
        # Require protrusion significantly exceeding radial RMS fluctuation and positive outward curvature
        rms_prot = float(np.sqrt(np.mean(np.minimum(protrusion, 0.0) ** 2))) + 0.04
        cand = (protrusion > max(0.12, 2.5 * rms_prot)) & (kappa > 1.5 / r_mean)
        if np.any(cand):
            score = protrusion * np.maximum(kappa, 0.0)
            score[~cand] = -1.0
            best_i = int(np.argmax(score))
            tip_x, tip_y = xs[best_i], ys[best_i]
            # Bisector ray vector from local base chord mid-point through tip
            span = max(3, m_samples // 24)
            base_x = 0.5 * (xs[(best_i - span) % m_samples] + xs[(best_i + span) % m_samples])
            base_y = 0.5 * (ys[(best_i - span) % m_samples] + ys[(best_i + span) % m_samples])
            vx, vy = tip_x - base_x, tip_y - base_y
            vnorm = float(np.hypot(vx, vy))
            if vnorm > 1e-3:
                tail_tip = [float(tip_x + x0), float(tip_y + y0)]
                tail_dir = [float(vx / vnorm), float(vy / vnorm)]

    # Simplify polygon slightly using Douglas-Peucker at sub-pixel tolerance (0.5px) for compact lossless storage
    approx_poly = cv2.approxPolyDP(cnt, 0.75, True)
    bubble_poly = [[float(p[0][0] + x0), float(p[0][1] + y0)] for p in approx_poly]
    return bubble_poly, bubble_kind, tail_tip, tail_dir
