from __future__ import annotations

import cv2
import numpy as np
from scipy.special import ndtri
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from .schemas import FormatProfile, ResolvedFormatProfile

# Gaussian quantile constant Phi^{-1}(0.75) = 0.6744897501960817 (exact mathematical constant)
_GAUSSIAN_MAD_SCALE = 0.6744897501960817


def estimate_substrate_noise(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Estimate channel uncertainty using only observable substrate pairs.

    The page's lower-quartile Haar MAD can be zero on clipped/JPEG paper while
    its observed paper still varies. Adjacent known-pixel differences provide
    an independent local scale: Q75(|X-Y|)/(sqrt(2)*Phi^-1(7/8)).
    Each uint8 measurement represents a unit quantization interval. Use the
    upper endpoint of the difference interval rather than treating rounded
    zeros/ones as exact continuous measurements.
    A single object boundary cannot inflate this quartile into global variation.
    """
    h, w = known.shape
    values = rgb.astype(float)
    sigma = np.full(3, 1 / np.sqrt(12))

    # Every statistic must exclude the unavailable pixels. Computing the
    # page-wide Haar scale here let hidden text change restoration weights.
    eh, ew = h // 2 * 2, w // 2 * 2
    complete = known[:eh:2, :ew:2] & known[1:eh:2, :ew:2] & known[:eh:2, 1:ew:2] & known[1:eh:2, 1:ew:2]
    if complete.any():
        hh = np.abs(
            0.5 * (values[:eh:2, :ew:2] - values[:eh:2, 1:ew:2] - values[1:eh:2, :ew:2] + values[1:eh:2, 1:ew:2])
        )
        ch, cw = complete.shape
        side = max(1, int(np.ceil(np.sqrt(min(ch, cw)))))
        bh, bw = ch // side, cw // side
        tile_mads_batches: list[np.ndarray] = []
        partial_mads: list[np.ndarray] = []
        if bh > 0 and bw > 0:
            comp_grid = (
                complete[: bh * side, : bw * side]
                .reshape(bh, side, bw, side)
                .transpose(0, 2, 1, 3)
                .reshape(bh * bw, side * side)
            )
            hh_grid = (
                hh[: bh * side, : bw * side]
                .reshape(bh, side, bw, side, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(bh * bw, side * side, 3)
            )
            full_mask = comp_grid.all(axis=1)
            if full_mask.any():
                tile_mads_batches.append(np.median(hh_grid[full_mask], axis=1))
            for idx_t in np.flatnonzero(~full_mask & comp_grid.any(axis=1)):
                partial_mads.append(np.median(hh_grid[idx_t][comp_grid[idx_t]], axis=0))
        if bw * side < cw:
            for y in range(0, bh * side, side):
                available = complete[y : y + side, bw * side :]
                if available.any():
                    partial_mads.append(np.median(hh[y : y + side, bw * side :][available], axis=0))
        if bh * side < ch:
            for x in range(0, cw, side):
                available = complete[bh * side :, x : x + side]
                if available.any():
                    partial_mads.append(np.median(hh[bh * side :, x : x + side][available], axis=0))
        if partial_mads:
            tile_mads_batches.append(np.asarray(partial_mads, dtype=np.float64))
        if tile_mads_batches:
            tile_mads = (
                tile_mads_batches[0] if len(tile_mads_batches) == 1 else np.vstack(tile_mads_batches)
            )
            sigma = np.maximum(sigma, np.quantile(tile_mads, 0.25, axis=0) / _GAUSSIAN_MAD_SCALE)

    if rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3:
        hist = np.zeros((3, 256), dtype=np.int64)
        rgb_i16 = rgb.astype(np.int16)
        for dy, dx in ((0, 1), (1, 0)):
            hh_p, ww_p = h - dy, w - dx
            pairs = known[:hh_p, :ww_p] & known[dy:, dx:]
            if pairs.any():
                diff = np.abs(rgb_i16[:hh_p, :ww_p] - rgb_i16[dy:, dx:])[pairs]
                for c in range(3):
                    hist[c] += np.bincount(diff[:, c], minlength=256)
        total_pairs = int(hist[0].sum())
        if total_pairs > 0:
            pos = 0.75 * (total_pairs - 1)
            lo_i = int(np.floor(pos))
            hi_i = int(np.ceil(pos))
            frac = pos - lo_i
            q75 = np.empty(3, dtype=np.float64)
            for c in range(3):
                cdf = np.cumsum(hist[c])
                v_lo, v_hi = np.searchsorted(cdf, [lo_i + 1, hi_i + 1])
                q75[c] = float(v_lo) + frac * (float(v_hi) - float(v_lo))
            local = (q75 + 1.0) / (np.sqrt(2.0) * ndtri(0.875))
            sigma = np.maximum(sigma, local)
    else:
        differences = []
        for dy, dx in ((0, 1), (1, 0)):
            hh_p, ww_p = h - dy, w - dx
            pairs = known[:hh_p, :ww_p] & known[dy:, dx:]
            if pairs.any():
                differences.append(np.abs(values[:hh_p, :ww_p] - values[dy:, dx:])[pairs])
        if differences:
            local = (np.quantile(np.concatenate(differences), 0.75, axis=0) + 1) / (np.sqrt(2) * ndtri(0.875))
            sigma = np.maximum(sigma, local)
    return sigma


def substrate_squared_distance(values: np.ndarray, center: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Distance between quantization intervals in native uint8 code units.

    Both an observed sample and its measured median represent intervals of
    half a code unit on either side. Their minimum separation is |x-m|-1,
    clipped at zero. This is sensor quantization, not an image-size tolerance.
    """
    separation = np.maximum(np.abs(values.astype(float)-center)-1,0)
    return np.sum((separation/sigma)**2,axis=-1)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Scale- & texture-invariant Donoho-Johnstone 2D Haar diagonal wavelet noise estimator.
    Uses non-overlapping Haar coefficients and the lower quartile of local MADs.
    Tile side sqrt(short edge) balances within-tile samples and tile count;
    no fixed pixel-sized patch is tied to the input resolution.
    """
    g = gray.astype(np.float32)
    height, width = (g.shape[0] // 2) * 2, (g.shape[1] // 2) * 2
    if height == 0 or width == 0:
        return 0.0
    g = g[:height, :width]
    hh = np.abs(0.5 * (g[::2, ::2] - g[::2, 1::2] - g[1::2, ::2] + g[1::2, 1::2]))
    th, tw = hh.shape
    side = max(1, int(np.ceil(np.sqrt(min(th, tw)))))
    bh, bw = th // side, tw // side
    cropped = hh[: bh * side, : bw * side].reshape(bh, side, bw, side)
    tile_meds = np.median(cropped, axis=(1, 3))
    return float(np.quantile(tile_meds, 0.25) / _GAUSSIAN_MAD_SCALE)


def otsu_threshold_and_eta(values: np.ndarray) -> tuple[float, float]:
    """Compute exact Otsu threshold and global separability eta = sigma_B^2 / sigma_T^2 in [0, 1]."""
    flat = values.ravel().astype(np.float64)
    if flat.size < 2:
        return 0.0, 0.0
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax - vmin <= 1e-9:
        return vmin, 0.0
    bins = min(256, max(16, int(np.sqrt(flat.size))))
    hist, bin_edges = np.histogram(flat, bins=bins, range=(vmin, vmax))
    prob = hist.astype(np.float64) / flat.size
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_t2 = float(np.sum(prob * (centers - mu_t) ** 2))
    if sigma_t2 <= 1e-12:
        return vmin, 0.0

    valid = (omega > 0.0) & (omega < 1.0)
    sigma_b2 = np.zeros_like(omega)
    sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / (
        omega[valid] * (1.0 - omega[valid])
    )
    best_idx = int(np.argmax(sigma_b2))
    eta = float(sigma_b2[best_idx] / sigma_t2)
    return float(centers[best_idx]), eta


def is_color_page(img_rgb: np.ndarray) -> bool:
    """Self-calibrating color vs monochrome test comparing CIE Lab chroma variance to sensor noise floor."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_chan = lab[:, :, 1] - 128.0
    b_chan = lab[:, :, 2] - 128.0
    # Unit-spaced uint8 Lab quantization contributes variance 1/12. Whiten each
    # chroma axis by its own estimated noise, then test the Rayleigh zero-chroma
    # null. Bonferroni controls page-wide false alarms without independence.
    quantization_sigma = 1.0 / np.sqrt(12.0)
    sigma_a = max(estimate_noise_sigma(a_chan), quantization_sigma)
    sigma_b = max(estimate_noise_sigma(b_chan), quantization_sigma)
    radius_squared = (a_chan / sigma_a) ** 2 + (b_chan / sigma_b) ** 2
    family_false_alarm = 1e-4
    critical_squared = -2.0 * np.log(family_false_alarm / radius_squared.size)
    return bool(np.max(radius_squared) > critical_squared)


def estimate_paper_tint(img_rgb: np.ndarray) -> list[int]:
    """Extract paper substrate albedo mode from the bright Otsu partition."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    thresh, _ = otsu_threshold_and_eta(gray)
    substrate = gray >= thresh
    if not np.any(substrate):
        return [255, 255, 255]
    # Mode estimation via upper-half luminance median within substrate partition
    sub_gray = gray[substrate]
    bright_cut = np.median(sub_gray)
    paper_pixels = img_rgb[gray >= bright_cut]
    med = np.median(paper_pixels, axis=0)
    return [int(np.clip(np.rint(c), 0, 255)) for c in med]


def _mst_reading_anisotropy(gray: np.ndarray) -> float:
    """
    Scale-invariant text reading-axis anisotropy via Euclidean Minimum Spanning Tree (EMST)
    over bimodal ink connected components. Returns log((E_y + eps) / (E_x + eps)):
      > 0 => predominantly vertical columns (Japanese/Traditional CJK RTL)
      < 0 => predominantly horizontal rows (LTR / Korean Webtoon)
    """
    h, w = gray.shape[:2]
    short_edge = min(h, w)
    # Block window proportional to page geometry (odd integer)
    win = max(3, int(round(short_edge / 32.0)) | 1)
    sigma_n = estimate_noise_sigma(gray)
    local_mean = cv2.GaussianBlur(gray.astype(np.float32), (win, win), 0)
    diff = local_mean - gray.astype(np.float32)
    thresh, eta = otsu_threshold_and_eta(diff[diff > 3.0 * sigma_n]) if np.any(diff > 3.0 * sigma_n) else (0.0, 0.0)
    if thresh <= 0.0 or eta < 0.1:
        return 0.0

    ink = (diff >= thresh).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if num_labels <= 6:
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    aspect = np.maximum(widths, heights) / np.maximum(1.0, np.minimum(widths, heights))

    # Self-calibrating glyph candidate filter: compact components around median component scale
    page_area = float(h * w)
    compact = (areas > 4) & (areas < page_area * 0.01) & (aspect < 4.0)
    if np.count_nonzero(compact) < 6:
        return 0.0

    med_diag = float(np.median(np.hypot(widths[compact], heights[compact])))
    glyph_mask = compact & (
        np.abs(np.log(np.hypot(widths, heights) / max(med_diag, 1.0))) < np.log(2.5)
    )
    pts = centroids[1:][glyph_mask]
    if len(pts) < 6:
        return 0.0
    if len(pts) > 600:
        # Deterministic stride subsample to bound O(N^2) pairwise graph
        step = max(1, len(pts) // 600)
        pts = pts[::step]

    dist_mat = squareform(pdist(pts, metric="euclidean"))
    mst = minimum_spanning_tree(dist_mat).tocoo()
    edge_lengths = mst.data
    if edge_lengths.size < 4:
        return 0.0

    # Keep intra-line edges (below upper Tukey fence Q3 + 1.5 IQR of MST edge lengths)
    q1, q3 = np.quantile(edge_lengths, [0.25, 0.75])
    fence = q3 + 1.5 * max(q3 - q1, 1e-3)
    intra = edge_lengths <= fence
    if np.count_nonzero(intra) < 4:
        intra = np.ones_like(edge_lengths, dtype=bool)

    dx = np.abs(pts[mst.row[intra], 0] - pts[mst.col[intra], 0])
    dy = np.abs(pts[mst.row[intra], 1] - pts[mst.col[intra], 1])
    # Direction cosines along vertical vs horizontal axes
    lengths = np.hypot(dx, dy) + 1e-9
    cos_vert = dy / lengths
    cos_horiz = dx / lengths
    return float(np.log((np.mean(cos_vert) + 1e-6) / (np.mean(cos_horiz) + 1e-6)))


def detect_format_profile(
    img_rgb: np.ndarray, requested: FormatProfile = "auto"
) -> tuple[ResolvedFormatProfile, bool]:
    """Infer comic layout profile and chromaticity without resolution-dependent magic thresholds."""
    color_flag = is_color_page(img_rgb)
    if requested != "auto":
        return requested, color_flag

    h, w = img_rgb.shape[:2]
    # Standard tankobon trim ratio is ~sqrt(2) (B6/A5 ~ 1.414), US comic is ~1.52;
    # long-strip webtoons exceed two stacked standard frames (> 2 * 1/sqrt(2) * sqrt(2) = 2.0)
    if h > 2.0 * w:
        return "webtoon_vertical", color_flag

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    anisotropy = _mst_reading_anisotropy(gray)
    if anisotropy > 0.05:
        return "manga_rtl", color_flag
    if anisotropy < -0.05 and color_flag:
        return "comic_ltr", color_flag

    return ("comic_ltr" if color_flag else "manga_rtl"), color_flag


def detect_seam_split_need(top_rgb: np.ndarray, bottom_rgb: np.ndarray) -> bool:
    """
    Non-parametric empirical seam continuity test:
    Checks whether the boundary between top_rgb and bottom_rgb cuts across non-gutter structure
    whose cross-seam row transition matches intra-slice row transitions.
    """
    if top_rgb.shape[1] != bottom_rgb.shape[1] or top_rgb.shape[0] < 8 or bottom_rgb.shape[0] < 8:
        return False

    # Use within-slice observations directly; no DPI-dependent 32-row collar.
    top_band = top_rgb.astype(np.float32)
    bot_band = bottom_rgb.astype(np.float32)

    sigma_n = max(
        estimate_noise_sigma(cv2.cvtColor(top_band.astype(np.uint8), cv2.COLOR_RGB2GRAY)),
        estimate_noise_sigma(cv2.cvtColor(bot_band.astype(np.uint8), cv2.COLOR_RGB2GRAY)),
    )

    # 1. Is the seam sitting inside a uniform gutter? Compare boundary row spatial std to noise floor.
    seam_top = top_band[-1]
    seam_bot = bot_band[0]
    spatial_std = 0.5 * (float(np.std(seam_top, axis=0).mean()) + float(np.std(seam_bot, axis=0).mean()))
    if spatial_std <= 3.0 * sigma_n:
        # Pure flat gutter (white/black margin) — safe to process independently without stitching
        return False

    # 2. Compare cross-seam L1 step against empirical distribution of internal adjacent-row steps
    intra_top = np.mean(np.abs(np.diff(top_band, axis=0)), axis=(1, 2))
    intra_bot = np.mean(np.abs(np.diff(bot_band, axis=0)), axis=(1, 2))
    intra_steps = np.concatenate([intra_top, intra_bot])
    cross_step = float(np.mean(np.abs(seam_top - seam_bot)))

    q95_intra = float(np.quantile(intra_steps, 0.95))
    med_intra = float(np.median(intra_steps))
    mad_intra = 1.4826 * float(np.median(np.abs(intra_steps - med_intra))) + sigma_n
    # Continuous panel/bubble across seam if cross_step falls within empirical intra-slice step distribution
    return bool(cross_step <= max(q95_intra, med_intra + 3.0 * mad_intra))


def detect_spread_gutter(img_rgb: np.ndarray) -> tuple[bool, int | None]:
    """
    Detect double-page spread (`見開き`) via page aspect geometry (w > h, i.e., landscape orientation
    corresponding to two portrait trim pages side by side) and minimum column total-variation gutter search.
    """
    h, w = img_rgb.shape[:2]
    # Single portrait comic/manga page has w < h; two-page spread has w > h (typically ~2 / sqrt(2) = 1.414)
    if w <= h:
        return False, None
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Search central 10% around geometric midpoint w/2 for minimum vertical total variation
    half = w // 2
    radius = max(2, w // 20)
    left, right = max(0, half - radius), min(w, half + radius)
    tv_col = np.mean(np.abs(np.diff(gray[:, left:right], axis=0)), axis=0)
    gutter_x = left + int(np.argmin(tv_col))
    return True, gutter_x
