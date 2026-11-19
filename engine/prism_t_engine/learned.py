"""Learned comic text probability maps and native FFT, masked-only restoration."""
from __future__ import annotations

import contextlib
import cv2
import numpy as np
from functools import lru_cache
import threading
from scipy.ndimage import distance_transform_edt, maximum, maximum_filter
from scipy.stats import chi2, fisher_exact

from .models import create_ort_session, lama_model, model_path, resolve_ort_providers, run_ort_on_pinned_thread
from .preprocess import otsu_threshold_and_eta, estimate_noise_sigma, estimate_substrate_noise, substrate_squared_distance

_lama_inference_lock = threading.Lock()


def stroke_palette(
    rgb: np.ndarray,
    support: np.ndarray,
    *,
    lab_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """BIC-select a Lab pigment mixture; return RGB medians and pixel labels.

    Quantization noise bounds the covariance on flat digital ink. Components are
    initialized deterministically at farthest observed colors, without a list of
    expected text/outline colors. Candidate modes represent fill, outline,
    anti-alias mixtures, and accidentally included background; gradients remain
    within a mode's residual variance rather than becoming unlimited pigments.
    """
    if not np.any(support):
        return np.empty((0, 3)), np.full(support.shape, -1, np.int32)
    if lab_cache is not None:
        lab, sigma, _ = lab_cache
        noise = float(sigma[0])
    else:
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        noise = max(estimate_noise_sigma(lab[:, :, 0]), 100 / (255 * np.sqrt(12)))
    is_uint8 = rgb.dtype == np.uint8
    if is_uint8:
        sy, sx = np.nonzero(support)
        rgb_s = rgb[sy, sx]
        packed = (
            (rgb_s[:, 0].astype(np.uint32) << 16)
            | (rgb_s[:, 1].astype(np.uint32) << 8)
            | rgb_s[:, 2].astype(np.uint32)
        )
        _, first_idx, inverse, counts = np.unique(packed, return_index=True, return_inverse=True, return_counts=True)
        colors = lab[sy[first_idx], sx[first_idx]].astype(np.float64)
        n, dims = len(sy), 3
    else:
        values = lab[support].astype(np.float64)
        # Fit distinct observed colors with multiplicities instead of repeated pixels.
        colors, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
        n, dims = len(values), values.shape[1]
    variance_floor = noise ** 2
    total_weight = float(n)
    weighted_colors = colors * counts[:, None]
    best = None
    pigment_roles = ("fill", "inner_outline", "outer_outline", "anti_alias", "background")
    colors_sq = np.sum(colors * colors, axis=1)
    max_k = min(len(pigment_roles), len(colors))
    for k in range(1, max_k + 1):
        if k == 1:
            labels = np.zeros(len(colors), dtype=np.int32)
            centers = np.array([weighted_colors.sum(axis=0) / total_weight])
        else:
            centers = [weighted_colors.sum(axis=0) / total_weight]
            min_d = np.sum((colors - centers[0]) ** 2, axis=1)
            while len(centers) < k:
                centers.append(colors[int(np.argmax(min_d * counts))])
                if len(centers) < k:
                    d_new = np.sum((colors - centers[-1]) ** 2, axis=1)
                    min_d = np.minimum(min_d, d_new)
            centers = np.array(centers)
            previous = None
            for _ in range(100):
                centers_sq = np.sum(centers * centers, axis=1)
                dists = colors_sq[:, None] + centers_sq[None, :] - 2.0 * (colors @ centers.T)
                labels = np.argmin(dists, axis=1).astype(np.int32)
                if previous is not None and np.array_equal(previous, labels):
                    break
                previous = labels
                c_sums = np.bincount(labels, weights=counts, minlength=k)
                for dim in range(dims):
                    w_sums = np.bincount(labels, weights=weighted_colors[:, dim], minlength=k)
                    nonzero = c_sums > 0
                    centers[nonzero, dim] = w_sums[nonzero] / c_sums[nonzero]
        residual = colors - centers[labels]
        variance = max(variance_floor, float(np.sum(counts[:, None] * residual ** 2) / (n * dims)))
        weights = np.bincount(labels, weights=counts, minlength=k) / n
        ll = np.sum(counts * (np.log(np.maximum(weights[labels], np.finfo(float).tiny))
                    - dims / 2 * np.log(2 * np.pi * variance)
                    - np.sum(residual ** 2, axis=1) / (2 * variance)))
        bic = (k * dims + k) * np.log(max(n, 2)) - 2 * ll
        if best is None or bic < best[0]:
            best = bic, labels[inverse]
    if is_uint8:
        best_labels = best[1]
        unique_clusters = np.unique(best_labels)
        pigments = np.array([np.median(rgb_s[best_labels == j], axis=0) for j in unique_clusters])
        remap = np.empty(int(unique_clusters[-1]) + 1, dtype=np.int32)
        remap[unique_clusters] = np.arange(len(unique_clusters), dtype=np.int32)
        labels = np.full(support.shape, -1, dtype=np.int32)
        labels[sy, sx] = remap[best_labels]
        return pigments, labels
    labels = np.full(support.shape, -1, np.int32)
    labels[support] = best[1]
    pigments = np.array([np.median(rgb[labels == j], axis=0) for j in np.unique(best[1])])
    # Relabel occupied clusters densely (empty Lloyd components are irrelevant).
    for new, old in enumerate(np.unique(best[1])):
        labels[support & (labels == old)] = new
    return pigments, labels


def prepare_lab_crop(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute float32 CIE Lab, 3-channel noise floor, and whitened Lab once per crop."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    sigma = np.array(
        [max(estimate_noise_sigma(lab[:, :, k]), 100 / (255 * np.sqrt(12))) for k in range(3)],
        dtype=np.float64,
    )
    return lab, sigma, lab.astype(np.float64) / sigma


def normalized_lab_crop(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB crop to noise-normalized CIE Lab once for reuse."""
    return prepare_lab_crop(rgb)[2]


def stroke_pigment_distance(rgb: np.ndarray, support: np.ndarray,
                            reference: np.ndarray | None = None,
                            *,
                            normalized: np.ndarray | None = None,
                            lab_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray,np.ndarray]:
    """Whiten each observed Lab pigment by its own color covariance.

    Antialias mixtures and scan variation are directional in color space.
    A common spherical distance from a few median colors clips those tails.
    Covariance eigenvalues are bounded by the observed channel noise and Lab
    quantization scale; no expected lettering or outline color is supplied.
    With a reference, only pigments enriched relative to that visible sample
    survive; common artwork colors cannot anchor an entirely missed glyph.
    """
    if lab_cache is None and normalized is None:
        lab_cache = prepare_lab_crop(rgb)
    pigments, labels = stroke_palette(rgb, support, lab_cache=lab_cache)
    if normalized is None:
        normalized = lab_cache[2] if lab_cache is not None else normalized_lab_crop(rgb)
    distance=np.full(support.shape,np.inf)
    retained=[]
    for index in range(len(pigments)):
        samples=normalized[labels==index]
        center=np.mean(samples,axis=0)
        residual=samples-center
        covariance=residual.T@residual/len(samples)
        eigenvalues,basis=np.linalg.eigh(covariance)
        whitening=basis/np.sqrt(np.maximum(eigenvalues,1.))
        delta=(normalized-center)@whitening
        squared=np.sum(delta*delta,axis=2)
        if reference is not None:
            compatible=squared<=chi2.ppf(.75,3)
            inside=int(np.count_nonzero(compatible&support))
            outside=int(np.count_nonzero(compatible&reference))
            table=[[inside,int(support.sum())-inside],[outside,int(reference.sum())-outside]]
            if fisher_exact(table,alternative='greater').pvalue>=.05/max(1,len(pigments)):
                continue
        retained.append(pigments[index])
        distance=np.minimum(distance,squared)
    return np.array(retained).reshape(-1,3),distance


_COMIC_DETECTOR_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _comic_detector_for_providers(providers: tuple[str, ...]):
    session = create_ort_session(model_path("comic_text"), providers)
    return session, threading.Lock()


def comic_detector():
    providers = tuple(resolve_ort_providers())
    with _COMIC_DETECTOR_LOAD_LOCK:
        return _comic_detector_for_providers(providers)


def comic_text_evidence(rgb: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[list[float]]]:
    """Use the checkpoint's glyph AND DB line heads in one inference.

    DB's learned threshold map and the observed probability Otsu partition
    determine line contours. Glyph components intersecting each shrunken contour
    recover its full extent, without a fixed pixel unclip or a second detector.
    """
    session, lock = comic_detector()
    mh, mw = 1024, 1024  # fixed tensor dimensions of this pinned checkpoint
    h, w = rgb.shape[:2]
    # Sliding windows preserve glyph resolution on long strips. Tile geometry is
    # dictated by the model receptive field and source aspect, rather than DPI.
    source_h = max(w, int(round(w * mh / mw)))
    starts = list(range(0, max(1, h - source_h + 1), max(1, source_h // 2)))
    starts.append(max(0, h - source_h))
    result = np.zeros((h, w), np.float32)
    line_probability = np.zeros((h, w), np.float32)
    line_threshold = np.ones((h, w), np.float32)
    block_boxes, block_scores = [], []
    for y in sorted(set(starts)):
        crop = rgb[y:min(h, y + source_h)]
        ch, cw = crop.shape[:2]
        scale = min(mw / cw, mh / ch)
        sw, sh = int(round(cw * scale)), int(round(ch * scale))
        canvas = np.full((mh, mw, 3), 114, np.uint8)  # checkpoint's training letterbox value
        canvas[:sh, :sw] = cv2.resize(crop, (sw, sh))
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0)
        try:
            with lock:
                segmentation, detection, blocks = run_ort_on_pinned_thread(
                    lambda: session.run(["seg", "det", "blk"], {"images": tensor})
                )
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            cpu_session, cpu_lock = _comic_detector_for_providers(("CPUExecutionProvider",))
            with cpu_lock:
                segmentation, detection, blocks = run_ort_on_pinned_thread(
                    lambda: cpu_session.run(["seg", "det", "blk"], {"images": tensor})
                )
        # These are the pinned checkpoint's published operating points, not
        # image-resolution heuristics. Coordinates remain in native pixels.
        scores = blocks[0, :, 4] * np.max(blocks[0, :, 5:], axis=1)
        for block, confidence in zip(blocks[0][scores > .4], scores[scores > .4]):
            cx, cy, bw, bh = block[:4] / scale
            block_boxes.append([float(cx-bw/2), float(cy-bh/2+y), float(bw), float(bh)])
            block_scores.append(float(confidence))
        probability = segmentation[0, 0]
        ph, pw = probability.shape
        probability = probability[:int(round(sh * ph / mh)), :int(round(sw * pw / mw))]
        probability = cv2.resize(probability, (cw, ch))
        result[y:y + ch] = np.maximum(result[y:y + ch], probability)
        dh, dw = detection.shape[2:]
        maps = detection[0, :, :int(round(sh * dh / mh)), :int(round(sw * dw / mw))]
        local_p = cv2.resize(maps[0], (cw, ch))
        local_t = cv2.resize(maps[1], (cw, ch))
        take = local_p > line_probability[y:y + ch]
        line_probability[y:y + ch][take] = local_p[take]
        line_threshold[y:y + ch][take] = local_t[take]
    keep = cv2.dnn.NMSBoxes(block_boxes, block_scores, .4, .35)
    blocks = [block_boxes[int(i)] for i in np.asarray(keep).reshape(-1)]
    return result, text_line_polygons(result, line_probability, line_threshold), blocks


def comic_text_probability(rgb: np.ndarray) -> np.ndarray:
    return comic_text_evidence(rgb)[0]


def text_line_polygons(ink_probability, line_probability, learned_threshold):
    ink_cut, _ = otsu_threshold_and_eta(ink_probability)
    line_cut, _ = otsu_threshold_and_eta(line_probability)
    _, glyphs = cv2.connectedComponents((ink_probability > ink_cut).astype(np.uint8))
    binary = (line_probability > np.maximum(line_cut, learned_threshold)).astype(np.uint8)
    count, lines, stats, _ = cv2.connectedComponentsWithStats(binary)
    polygons = []
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        local = lines[y:y+h, x:x+w] == i
        yy, xx = np.nonzero(local)
        rect = cv2.minAreaRect(np.column_stack((xx, yy)).astype(np.float32))
        if min(rect[1]) <= 0:
            continue
        radius = min(rect[1]) / 2
        pad = int(np.ceil(radius))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(lines.shape[1], x+w+pad), min(lines.shape[0], y+h+pad)
        hit = lines[y0:y1, x0:x1] == i
        local_glyphs = glyphs[y0:y1, x0:x1]
        ids = np.unique(local_glyphs[hit])
        ids = ids[ids > 0]
        if not ids.size:
            continue
        # Some neighboring letters touch in the glyph map. Include only the
        # parts owned by this line rather than an entire connected paragraph.
        collar = distance_transform_edt(~hit) <= radius
        full = np.isin(local_glyphs, ids) & collar
        yy, xx = np.nonzero(full | hit)
        polygons.append(cv2.boxPoints(cv2.minAreaRect(np.column_stack((xx+x0, yy+y0)).astype(np.float32))))
    return polygons


def bound_observed_stroke_growth(rgb: np.ndarray, observed: np.ndarray,
                                anchors: np.ndarray, reference: np.ndarray,
                                contrast: np.ndarray,
                                *,
                                lab_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Separate supported pigment ribbons before granting component ownership.

    A white outline may meet a white shirt. Its component size says nothing
    about ownership. Measure the radius fence on neural-anchored observed
    ridges, then partition the enriched pigments and stop at wide medial
    regions. The rejected source samples remain forbidden during collar growth.
    """
    _, difference = stroke_pigment_distance(rgb, anchors, reference, lab_cache=lab_cache)
    critical = chi2.isf(.05/max(1, int(observed.sum())), 3)
    qualified = observed & (difference <= critical) & ((contrast > critical) | anchors)
    radii = distance_transform_edt(observed)
    ridges = observed & (radii >= maximum_filter(radii, size=3))
    # Some incomplete neural seeds lie only on a stroke boundary. Project them
    # onto observed ridges rather than estimating width from clipped seeds.
    if not np.any(ridges & anchors):
        _, nearest = distance_transform_edt(~ridges, return_indices=True)
        anchor_radii = radii[tuple(nearest[:, anchors])]
    else:
        anchor_radii = radii[ridges & anchors]
    q1, q3 = np.quantile(anchor_radii, [.25, .75])
    radius_limit = q3 + 1.5*(q3-q1)
    pigments, labels = stroke_palette(rgb, qualified, lab_cache=lab_cache)
    permitted = qualified.copy()
    for index in range(len(pigments)):
        mode = labels == index
        radius = distance_transform_edt(mode)
        wide = mode & (radius > radius_limit)
        count, regions = cv2.connectedComponents(wide.astype(np.uint8), connectivity=8)
        # A letter intersection has a locally larger radius, but that excess
        # occupies less than a measured stroke radius. Demand extended support
        # for a width change, then restore the measured boundary radius around
        # that wide interior. This rejects a shirt, including its thin edge,
        # without clipping isolated T/X junctions or trusting filled neural tubes.
        extent = maximum(distance_transform_edt(wide), labels=regions, index=np.arange(count))
        broad_ids = np.flatnonzero(extent > radius_limit)
        broad_ids = broad_ids[broad_ids != 0]
        if broad_ids.size:
            # A solid enclosed glyph symbol (e.g. a filled heart ♥ surrounded by its
            # outline inside `observed`) is wider than a kanji stroke, yet its own
            # perimeter touches ONLY other ink layers inside `observed` (not `~observed`).
            # Comparing perimeter-on-exterior / total-perimeter is strictly scale-invariant.
            _, mode_cc = cv2.connectedComponents(mode.astype(np.uint8), connectivity=8)
            obs_boundary = radii <= 1.5
            mode_boundary = distance_transform_edt(mode) <= 1.5
            keep_reject = []
            for bid in broad_ids:
                cc_ids = np.unique(mode_cc[regions == bid])
                cc_ids = cc_ids[cc_ids != 0]
                parent_edge = np.isin(mode_cc, cc_ids) & mode_boundary
                if not np.any(parent_edge) or float(np.mean(obs_boundary[parent_edge])) > 0.15:
                    keep_reject.append(int(bid))
            if keep_reject:
                broad = np.isin(regions, keep_reject)
                permitted &= ~mode | (distance_transform_edt(~broad) > radius_limit)
    # A rejected solid width cannot regrow. Off-palette samples still get an
    # independent optical-mixture assessment below: real antialias edges need
    # not match any solid pigment, even when contrast extraction observed them.
    return permitted, qualified & ~permitted


def _bh_fdr_threshold(p_values: np.ndarray, q: float = 0.05) -> float:
    """Return Benjamini-Hochberg p-value cutoff at FDR level q (or -1.0 if none pass)."""
    m = p_values.size
    if m == 0:
        return -1.0
    p_sorted = np.sort(p_values)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    passing = np.flatnonzero(p_sorted <= (ranks / float(m)) * q)
    if passing.size == 0:
        return -1.0
    return float(min(q, p_sorted[int(passing[-1])]))


def refine_text_alpha(rgb: np.ndarray, probability: np.ndarray, seed: np.ndarray,
                      observed_ink: np.ndarray | None = None) -> np.ndarray:
    """Recover pigment and outline around neural glyph seeds at measured scale.

    Any expanded stroke/outline is recorded in alpha itself. Later restoration
    and compositing never silently enlarge that footprint.
    """
    cut, _ = otsu_threshold_and_eta(probability)
    support = probability > cut
    _, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    domain = seed > 0
    background = None
    uncertainty = None
    delta = None
    contrast = None
    ids = np.unique(labels[domain])
    ids = ids[ids != 0]
    # A probability component may connect several paragraphs or nearby art.
    # Geometry bounds this node; merely touching a seed does not grant ownership
    # of that entire page-wide component.
    selected = np.isin(labels, ids) & domain
    rejected_observed = np.zeros(domain.shape, bool)
    lab_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    flat_substrate = False
    flat_center = None
    flat_sigma = None
    observed_selected = np.zeros(domain.shape, bool)
    cached_pigment_dist: tuple[np.ndarray, np.ndarray] | None = None

    if observed_ink is not None:
        observed = (observed_ink > 0) & domain
        substrate = domain & ~observed
        if np.count_nonzero(substrate) >= 8:
            samples = rgb[substrate].astype(float)
            center = np.median(samples, axis=0)
            sigma = estimate_substrate_noise(rgb, substrate)
            noise_radius = substrate_squared_distance(samples, center, sigma)
            flat_substrate = bool(
                np.quantile(noise_radius, 0.75) <= chi2.ppf(0.75, 3)
                and np.quantile(noise_radius, 0.95) <= chi2.ppf(0.95, 3)
            )
            if flat_substrate:
                flat_center = center
                flat_sigma = sigma
                background = np.broadcast_to(center[None, None, :], rgb.shape)
                uncertainty = sigma
                delta = (rgb.astype(float) - background) / uncertainty
                contrast = np.sum(delta * delta, axis=2)

        lab_cache = prepare_lab_crop(rgb)
        if flat_substrate:
            # Uniform paper independently anchors missed glyphs; skip full-ROI Laplace solve.
            selected = observed.copy()
            observed_selected = observed
        else:
            if np.any(~domain):
                from .restoration import solve_laplace_dirichlet_fast
                background = solve_laplace_dirichlet_fast(rgb, domain)
                uncertainty = estimate_substrate_noise(rgb, ~domain)
                delta = (rgb.astype(float) - background) / uncertainty
                contrast = np.sum(delta * delta, axis=2)
            anchored_pixels = observed & selected
            permitted = observed
            if anchored_pixels.any() and contrast is not None:
                permitted, rejected_observed = bound_observed_stroke_growth(
                    rgb, observed, anchored_pixels, ~domain, contrast, lab_cache=lab_cache,
                )
            # Guard textured/illustrated scenes so observed highlights on hair or
            # collars cannot chain indefinitely beyond neural stroke proximity.
            if selected.any():
                if anchored_pixels.any():
                    anch_radii = distance_transform_edt(anchored_pixels)[anchored_pixels]
                    aq1, aq3 = np.quantile(anch_radii, [0.25, 0.75])
                    anchor_fence = float(aq3 + 1.5 * (aq3 - aq1))
                else:
                    anchor_fence = 1.0
                dist_from_neural = distance_transform_edt(~selected)
                near_neural = (dist_from_neural <= 3.0 * max(anchor_fence, 1.0)) | (probability > 0.25 * cut)
            else:
                near_neural = probability > 0.25 * cut
            permitted = permitted & near_neural
            _, ink_labels = cv2.connectedComponents(permitted.astype(np.uint8), connectivity=8)
            anchored = np.unique(ink_labels[selected])
            # The probability map detects text occupancy, including blank counters
            # and inter-character paper. It selects observed stroke components;
            # its filled paragraph tubes are never themselves erasure geometry.
            observed_selected = np.isin(ink_labels, anchored[anchored != 0])
            if observed_selected.any():
                if contrast is not None:
                    # Entire weak glyphs can be absent from the neural tube.
                    # A majority of the observed component must agree with an
                    # enriched source pigment and differ from visible background;
                    # one touching pixel cannot claim a connected clothing seam.
                    _, localized = stroke_pigment_distance(rgb, observed_selected, ~domain, lab_cache=lab_cache)
                    critical = chi2.isf(0.05 / max(1, int(observed.sum())), 3)
                    qualified = permitted & (localized <= critical) & (contrast > critical)
                    sizes = np.bincount(ink_labels.ravel())
                    hits = np.bincount(ink_labels.ravel(), weights=qualified.ravel(), minlength=len(sizes))
                    recovered_ids = np.flatnonzero(hits > sizes / 2)
                    observed_selected |= np.isin(ink_labels, recovered_ids[recovered_ids != 0])
                # Independent contrast extraction can miss whole glyphs on a
                # changing background. Recover neural pixels with the measured
                # ink palette instead of discarding those glyphs or accepting a
                # whole probability tube as foreground.
                pigments_os, difference = stroke_pigment_distance(rgb, observed_selected, lab_cache=lab_cache)
                cached_pigment_dist = (pigments_os, difference)
                critical = chi2.isf(0.05 / max(1, int(selected.sum())), 3)
                recovered = selected & (difference <= critical)
                if contrast is not None:
                    recovered &= contrast > critical
                recovered &= ~rejected_observed
                selected = observed_selected | recovered
            else:
                selected = permitted & selected
    if not selected.any():
        return np.zeros(probability.shape, np.float32)
    # Estimate the collar from observed strokes. Neural recovery may already
    # include flat counters/background; feeding that enlarged support back
    # into its own radius estimate amplifies a segmentation mistake.
    measured = observed_selected if observed_ink is not None and observed_selected.any() else selected
    radii = distance_transform_edt(measured)[measured]
    q1, q3 = np.quantile(radii, [0.25, 0.75])
    # Tukey's upper fence limits an outline collar using the observed stroke
    # radii, rather than a DPI-dependent dilation kernel.
    collar = q3 + 1.5 * (q3 - q1)
    outside = distance_transform_edt(~selected)
    if lab_cache is None:
        lab_cache = prepare_lab_crop(rgb)
    if cached_pigment_dist is not None and (measured is observed_selected):
        pigments, color_distance = cached_pigment_dist
    else:
        pigments, color_distance = stroke_pigment_distance(rgb, measured, lab_cache=lab_cache)
    candidates = outside <= collar
    critical = chi2.isf(0.05 / max(1, int(candidates.sum())), 3)
    extension = candidates & (color_distance <= critical)
    if contrast is not None:
        # White outlines and shadow-colored pigments also occur in artwork.
        # Color similarity alone is insufficient evidence of added lettering.
        extension &= contrast > critical
    edge_background = edge_delta = edge_noise = edge_contrast = None
    edge_candidates = candidates & ~domain
    if background is not None:
        # Assess uncertain pixels beyond the detector independently. Enlarging
        # its entire background hole changes core selection and can turn artwork
        # into lettering. Hide only measured strokes and their candidate collar;
        # their exterior is genuine observed context for this separate estimate.
        if edge_candidates.any() and np.any(~candidates):
            edge_flat = False
            if flat_substrate and flat_center is not None and flat_sigma is not None:
                ext_samples = rgb[~candidates]
                if ext_samples.shape[0] >= 8:
                    ext_dev = substrate_squared_distance(ext_samples, flat_center, flat_sigma)
                    if np.quantile(ext_dev, 0.75) <= chi2.ppf(0.75, 3):
                        edge_flat = True
            if edge_flat:
                edge_background = background
                edge_noise = uncertainty
                edge_delta = delta
                edge_contrast = contrast
            else:
                from .restoration import solve_laplace_dirichlet_fast
                edge_background = solve_laplace_dirichlet_fast(rgb, candidates)
                edge_noise = estimate_substrate_noise(rgb, ~candidates)
                edge_delta = (rgb.astype(float) - edge_background) / edge_noise
                edge_contrast = np.sum(edge_delta ** 2, axis=2)
            extension |= edge_candidates & (color_distance <= critical) & (edge_contrast > critical)
        # A glow or anti-aliased edge has neither the solid ink color nor the
        # background color: it lies between them. Comparing only to a solid
        # pigment leaves bright letter-shaped boundary samples visible, which
        # every restorer then propagates back into the erased glyph.
        dom_candidates = domain & (outside <= collar) & ~selected
        alpha_bonf = 0.05 / max(1, int(dom_candidates.sum()))
        proj_critical = chi2.isf(alpha_bonf, 1)
        orth_critical = chi2.isf(alpha_bonf, 2)
        best_proj_dom = np.zeros(domain.shape, dtype=np.float64)
        best_orth_dom = np.full(domain.shape, np.inf, dtype=np.float64)
        best_proj_edge = np.zeros(domain.shape, dtype=np.float64)
        best_orth_edge = np.full(domain.shape, np.inf, dtype=np.float64)
        for pigment in pigments:
            direction = (pigment - background) / uncertainty
            energy = np.sum(direction ** 2, axis=2)
            fraction = np.clip(np.sum(delta * direction, axis=2) / np.maximum(energy, np.finfo(float).tiny), 0, 1)
            residual = np.sum((delta - fraction[:, :, None] * direction) ** 2, axis=2)
            proj_e = (fraction ** 2) * energy
            extension |= dom_candidates & (proj_e > proj_critical) & (residual <= orth_critical)
            take_dom = residual <= chi2.isf(0.01, 2)
            best_proj_dom = np.where(take_dom & (proj_e > best_proj_dom), proj_e, best_proj_dom)
            best_orth_dom = np.minimum(best_orth_dom, residual)
            if edge_background is not None:
                direction_e = (pigment - edge_background) / edge_noise
                energy_e = np.sum(direction_e ** 2, axis=2)
                fraction_e = np.clip(np.sum(edge_delta * direction_e, axis=2) / np.maximum(energy_e, np.finfo(float).tiny), 0, 1)
                residual_e = np.sum((edge_delta - fraction_e[:, :, None] * direction_e) ** 2, axis=2)
                proj_ee = (fraction_e ** 2) * energy_e
                extension |= edge_candidates & (proj_ee > proj_critical) & (residual_e <= orth_critical)
                take_edge = residual_e <= chi2.isf(0.01, 2)
                best_proj_edge = np.where(take_edge & (proj_ee > best_proj_edge), proj_ee, best_proj_edge)
                best_orth_edge = np.minimum(best_orth_edge, residual_e)
    extension &= ~rejected_observed
    # Every added pixel needs pigment or optical-mixture evidence. Distance
    # alone cannot identify lettering: blind dilation and hole filling erase
    # artwork between strokes and then make that expanded support grow again.
    # Keep only qualified collar islands attached to the source segmentation.
    _, grown_labels = cv2.connectedComponents((selected | extension).astype(np.uint8), connectivity=8)
    attached = np.unique(grown_labels[selected])
    strong_attached = np.isin(grown_labels, attached[attached != 0])

    # Spatially-connected hysteresis extension on the immediate adjacency
    # shell (proportional to measured stroke radius) using Benjamini-Hochberg
    # FDR (q=0.05) to eliminate anti-alias / JPEG ringing speckle confetti.
    if contrast is not None and background is not None and len(pigments) > 0:
        fringe_r = min(float(collar), max(float(np.sqrt(2.0)), 0.5 * float(q3)))
        shell_domain = domain | edge_candidates
        shell_mask = (
            (~strong_attached)
            & (~rejected_observed)
            & shell_domain
            & (distance_transform_edt(~strong_attached) <= fringe_r)
        )
        if shell_mask.any():
            sy, sx = np.nonzero(shell_mask)
            in_dom = domain[sy, sx]
            pig_match = color_distance[sy, sx] <= chi2.isf(0.01, 3)
            bg_stat = np.where(in_dom, contrast[sy, sx], edge_contrast[sy, sx] if edge_background is not None else 0.0)
            mix_stat = np.where(in_dom, best_proj_dom[sy, sx], best_proj_edge[sy, sx] if edge_background is not None else 0.0)
            p_vals = np.ones(len(sy), dtype=np.float64)
            if pig_match.any():
                p_vals[pig_match] = np.minimum(p_vals[pig_match], chi2.sf(bg_stat[pig_match], 3))
            has_mix = mix_stat > 0.0
            if has_mix.any():
                p_vals[has_mix] = np.minimum(p_vals[has_mix], chi2.sf(mix_stat[has_mix], 1))
            p_cut = _bh_fdr_threshold(p_vals, q=0.05)
            if p_cut > 0.0:
                fdr_pass = np.zeros(domain.shape, dtype=bool)
                fdr_pass[sy, sx] = p_vals <= p_cut
                _, h_labels = cv2.connectedComponents((strong_attached | fdr_pass).astype(np.uint8), connectivity=8)
                h_attached = np.unique(h_labels[strong_attached])
                strong_attached = np.isin(h_labels, h_attached[h_attached != 0])

    strong_attached = _recover_outer_outline_shell(
        rgb,
        strong_attached,
        rejected_observed,
        collar=float(collar),
        q1=float(q1),
        q3=float(q3),
        core_pigments=pigments,
        lab_cache=lab_cache,
    )

    return strong_attached.astype(np.float32)


def _recover_outer_outline_shell(
    rgb: np.ndarray,
    strong_attached: np.ndarray,
    rejected_observed: np.ndarray,
    *,
    collar: float,
    q1: float,
    q3: float,
    core_pigments: np.ndarray,
    lab_cache: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Recover a contrasting outer outline shell (e.g. black outer rim on double-outlined
    subtitles) wrapping around already-anchored strokes without swallowing adjacent artwork.

    A genuine outer rim must:
      1. Contrast significantly with the Laplace-interpolated exterior background outside the shell;
      2. Wrap a substantial fraction of the contrasting perimeter of `strong_attached` (topological
         encirclement, distinguishing a glyph outline from a single touching sleeve or line);
      3. Remain bounded by the measured stroke scale (`collar`) and attached to `strong_attached`.
    """
    rim_r = max(float(collar), 2.0 * float(q3))
    if rim_r < 1.5 or not strong_attached.any():
        return strong_attached

    outside_sa = distance_transform_edt(~strong_attached)
    rim_band = (outside_sa > 0.0) & (outside_sa <= rim_r) & (~rejected_observed)
    rim_hull = strong_attached | rim_band
    if not rim_band.any() or int(np.count_nonzero(~rim_hull)) < 16:
        return strong_attached

    from .restoration import solve_laplace_dirichlet_fast

    shell_bg = solve_laplace_dirichlet_fast(rgb, rim_hull)
    shell_noise = estimate_substrate_noise(rgb, ~rim_hull)
    rgb_f = rgb.astype(np.float64)
    shell_delta = (rgb_f - shell_bg) / shell_noise
    shell_contrast = np.sum(shell_delta * shell_delta, axis=2)

    # Also compare against the nearest genuinely observed exterior pixel outside `rim_hull`
    # so harmonic diffusion across a two-zone border (e.g. #000000 margin meeting a bright wall)
    # never creates phantom contrast inside the flat dark margin!
    _, nearest_ext = distance_transform_edt(rim_hull, return_indices=True)
    nearest_ext_rgb = rgb_f[nearest_ext[0], nearest_ext[1]]
    ext_pair_noise = np.sqrt(2.0) * shell_noise
    nearest_delta = (rgb_f - nearest_ext_rgb) / ext_pair_noise
    nearest_contrast = np.sum(nearest_delta * nearest_delta, axis=2)
    eff_contrast = np.minimum(shell_contrast, nearest_contrast)

    ring_r = min(rim_r, max(2.5, float(q1)))
    ring1 = (outside_sa > 0.0) & (outside_sa <= ring_r) & (~rejected_observed)
    ring1_n = int(np.count_nonzero(ring1))
    if ring1_n < 16:
        return strong_attached

    ring1_crit = chi2.isf(0.05 / ring1_n, 3)
    ring1_contrast = ring1 & (eff_contrast > ring1_crit)
    if int(np.count_nonzero(ring1_contrast)) < 12:
        return strong_attached

    rim_pigments, rim_labels = stroke_palette(rgb, ring1_contrast, lab_cache=lab_cache)
    if len(rim_pigments) == 0:
        return strong_attached

    normalized_lab = lab_cache[2]
    band_n = int(np.count_nonzero(rim_band))
    band_crit = chi2.isf(0.05 / max(1, band_n), 3)
    band_proj_crit = chi2.isf(0.05 / max(1, band_n), 1)
    band_orth_crit = chi2.isf(0.05 / max(1, band_n), 2)
    loose_lab_crit = chi2.isf(0.01, 3)
    loose_orth_crit = chi2.isf(0.01, 2)

    confirmed_mask = np.zeros(strong_attached.shape, dtype=bool)
    confirmed_pigments: list[np.ndarray] = []
    confirmed_d2: list[np.ndarray] = []

    for j, pig_rgb in enumerate(rim_pigments):
        cluster_m = rim_labels == j
        if int(np.count_nonzero(cluster_m)) < 8:
            continue
        # Local exterior background must genuinely differ from this outer-rim pigment.
        bg_sep = (
            (np.sum(((shell_bg - pig_rgb[None, None, :]) / shell_noise) ** 2, axis=2) > loose_lab_crit)
            & (np.sum(((nearest_ext_rgb - pig_rgb[None, None, :]) / ext_pair_noise) ** 2, axis=2) > loose_lab_crit)
        )
        testable = ring1 & bg_sep
        testable_n = int(np.count_nonzero(testable))
        if testable_n < max(16, int(0.10 * ring1_n)):
            continue

        samples = normalized_lab[cluster_m]
        center = np.mean(samples, axis=0)
        residual = samples - center
        cov = residual.T @ residual / len(samples)
        evals, basis = np.linalg.eigh(cov)
        whiten = basis / np.sqrt(np.maximum(evals, 1.0))
        d_lab = (normalized_lab - center) @ whiten
        d2_j = np.sum(d_lab * d_lab, axis=2)

        # Evaluate optical blend between this outer-rim pigment and local exterior background.
        dir_bg = (pig_rgb[None, None, :] - shell_bg) / shell_noise
        en_bg = np.sum(dir_bg * dir_bg, axis=2)
        frac_bg = np.clip(np.sum(shell_delta * dir_bg, axis=2) / np.maximum(en_bg, np.finfo(float).tiny), 0.0, 1.0)
        res_bg = np.sum((shell_delta - frac_bg[:, :, None] * dir_bg) ** 2, axis=2)
        blend_bg_match = (frac_bg >= 0.25) & (res_bg <= loose_orth_crit) & (((frac_bg ** 2) * en_bg) > loose_lab_crit)

        # Also recognize inner-to-outer outline anti-alias blends (e.g. white inner rim -> black outer rim).
        inner_blend_match = np.zeros(strong_attached.shape, dtype=bool)
        for cp in core_pigments:
            d_cp = (pig_rgb - cp) / shell_noise
            en_cp = float(np.dot(d_cp, d_cp))
            if en_cp <= loose_lab_crit:
                continue
            rel_cp = (rgb_f - cp[None, None, :]) / shell_noise
            t_cp = np.clip(np.sum(rel_cp * d_cp[None, None, :], axis=2) / en_cp, 0.0, 1.0)
            r_cp = np.sum((rel_cp - t_cp[:, :, None] * d_cp[None, None, :]) ** 2, axis=2)
            inner_blend_match |= (t_cp >= 0.25) & (r_cp <= loose_orth_crit)

        wrap_hits = testable & ((d2_j <= loose_lab_crit) | blend_bg_match | inner_blend_match)
        wrap_ratio = float(np.count_nonzero(wrap_hits)) / float(testable_n)
        if wrap_ratio < 0.45:
            continue

        # Confirmed outer-outline pigment! Accept solid matches + optical transitions inside rim_band
        # only where the pixel genuinely contrasts with both Laplace and nearest observed exterior.
        proj_e_bg = (frac_bg ** 2) * en_bg
        cand_j = rim_band & (nearest_contrast > band_crit) & (
            ((d2_j <= band_crit) & (eff_contrast > band_crit))
            | ((proj_e_bg > band_proj_crit) & (res_bg <= band_orth_crit))
            | (inner_blend_match & (eff_contrast > band_crit))
        )
        confirmed_mask |= cand_j
        confirmed_pigments.append(pig_rgb)
        confirmed_d2.append(d2_j)

    if not confirmed_mask.any():
        return strong_attached

    _, grown_labels = cv2.connectedComponents((strong_attached | confirmed_mask).astype(np.uint8), connectivity=8)
    attached = np.unique(grown_labels[strong_attached])
    expanded = np.isin(grown_labels, attached[attached != 0])

    # 1-step BH-FDR anti-alias fringe outside the newly expanded outer outline so no faint
    # dark/colored outer halo remains on the inpainting boundary.
    fringe_r = min(rim_r, max(float(np.sqrt(2.0)), 0.5 * float(q3)))
    fringe_mask = (
        (~expanded)
        & (~rejected_observed)
        & rim_band
        & (nearest_contrast > loose_lab_crit)
        & (distance_transform_edt(~expanded) <= fringe_r)
    )
    if fringe_mask.any():
        fy, fx = np.nonzero(fringe_mask)
        p_vals = np.ones(len(fy), dtype=np.float64)
        bg_stat = eff_contrast[fy, fx]
        for pig_rgb, d2_j in zip(confirmed_pigments, confirmed_d2):
            pm = d2_j[fy, fx] <= loose_lab_crit
            if pm.any():
                p_vals[pm] = np.minimum(p_vals[pm], chi2.sf(bg_stat[pm], 3))
            d_bg = (pig_rgb[None, :] - shell_bg[fy, fx]) / shell_noise
            en = np.sum(d_bg * d_bg, axis=1)
            dl = shell_delta[fy, fx]
            fr = np.clip(np.sum(dl * d_bg, axis=1) / np.maximum(en, np.finfo(float).tiny), 0.0, 1.0)
            rs = np.sum((dl - fr[:, None] * d_bg) ** 2, axis=1)
            ok = (rs <= loose_orth_crit) & (fr > 0.0)
            if ok.any():
                pe = (fr[ok] ** 2) * en[ok]
                p_vals[ok] = np.minimum(p_vals[ok], chi2.sf(pe, 1))
        p_cut = _bh_fdr_threshold(p_vals, q=0.05)
        if p_cut > 0.0:
            fdr_pass = np.zeros(strong_attached.shape, dtype=bool)
            fdr_pass[fy, fx] = p_vals <= p_cut
            _, h_labels = cv2.connectedComponents((expanded | fdr_pass).astype(np.uint8), connectivity=8)
            h_attached = np.unique(h_labels[expanded])
            expanded = np.isin(h_labels, h_attached[h_attached != 0])

    return expanded


def missing_text_polygons(probability: np.ndarray, existing: list[np.ndarray], threshold: float | None = None,
                          join_components: bool = True, evidence_floor: float | None = None) -> list[np.ndarray]:
    """Recover neural text components missed by the OCR detector.

    Join aligned glyphs using the EMST edge-length IQR fence. No fixed
    pixel gap is used; block geometry follows this page's detected glyph spacing.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Delaunay
    from scipy.sparse import coo_matrix

    cut = otsu_threshold_and_eta(probability)[0] if threshold is None else threshold
    binary = (probability > cut).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return []
    points = centers[1:]
    coverage = np.zeros(probability.shape, np.uint8)
    for poly in existing:
        cv2.fillPoly(coverage, [np.rint(poly).astype(np.int32)], 1)
    covered = np.bincount(labels.ravel(), weights=coverage.ravel(), minlength=n)
    peaks = maximum(probability, labels=labels, index=np.arange(n))
    # Restrict pairwise work to foreground components not already accounted for.
    unseen = []
    for i in range(len(points)):
        if covered[i+1] < stats[i+1, cv2.CC_STAT_AREA] / 2 and (evidence_floor is None or peaks[i+1] >= evidence_floor):
            unseen.append(i + 1)
    if not unseen:
        return []
    if not join_components:
        polygons = []
        for cid in unseen:
            x, y, w, h, _ = stats[cid]
            yy, xx = np.nonzero(labels[y:y+h, x:x+w] == cid)
            rect = cv2.minAreaRect(np.column_stack((xx+x,yy+y)).astype(np.float32))
            if min(rect[1]) > 0:
                polygons.append(cv2.boxPoints(rect))
        return polygons
    points = centers[unseen]
    # An EMST is contained in the Delaunay triangulation. Construct O(N) edges
    # rather than an O(N^2) dense distance matrix on a detailed page.
    if len(points) == 1:
        yy, xx = np.nonzero(labels == unseen[0])
        rect = cv2.minAreaRect(np.column_stack((xx, yy)).astype(np.float32))
        return [cv2.boxPoints(rect)] if min(rect[1]) > 0 else []
    if len(points) == 2 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        order = np.argsort(points[:, np.argmax(np.ptp(points, axis=0))])
        edges = np.column_stack((order[:-1], order[1:]))
    else:
        triangles = Delaunay(points).simplices
        edges = np.unique(np.sort(np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, [0, 2]])), axis=1), axis=0)
    distances = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    graph = coo_matrix((np.tile(distances,2),(np.concatenate((edges[:,0],edges[:,1])),np.concatenate((edges[:,1],edges[:,0])))),shape=(len(points),len(points))).tocsr()
    mst = minimum_spanning_tree(graph).tocoo()
    lengths = mst.data
    q1, q3 = np.quantile(lengths, [.25, .75])
    threshold = q3 + 1.5 * (q3-q1)
    parent = np.arange(len(unseen))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, d in zip(mst.row, mst.col, lengths):
        ca, cb = unseen[a], unseen[b]
        ax, ay, aw, ah, _ = stats[ca]
        bx, by, bw, bh, _ = stats[cb]
        overlap_x = min(ax+aw,bx+bw)-max(ax,bx)
        overlap_y = min(ay+ah,by+bh)-max(ay,by)
        aligned = overlap_x > min(aw,bw)/2 or overlap_y > min(ah,bh)/2
        if d <= threshold and aligned:
            parent[find(a)] = find(b)
    groups = {}
    for i, cid in enumerate(unseen):
        groups.setdefault(find(i), []).append(cid)
    polygons = []
    for ids in groups.values():
        yy, xx = np.nonzero(np.isin(labels, ids))
        hull = cv2.convexHull(np.column_stack((xx, yy)).astype(np.float32))
        rect = cv2.minAreaRect(hull)
        if min(rect[1]) > 0:
            polygons.append(cv2.boxPoints(rect))
    return polygons


_LAMA_FP16_SAFE: bool | None = None


@lru_cache(maxsize=4)
def _lama_tile_window(th: int = 512, tw: int = 512) -> np.ndarray:
    interior = np.ones((th + 2, tw + 2), bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    return distance_transform_edt(interior)[1:-1, 1:-1].astype(np.float32)


def _lama_supports_fp16(net, device: str) -> bool:
    global _LAMA_FP16_SAFE
    if device != "cuda":
        return False
    if _LAMA_FP16_SAFE is not None:
        return _LAMA_FP16_SAFE
    # big-lama.pt uses FourierUnit (aten::fft_rfftn) which runs in experimental
    # ComplexHalf under FP16 autocast, causing NaN overflows on sharp edges and
    # 1-LSB non-determinism. Fall back cleanly to deterministic TF32/FP32.
    try:
        graph_str = str(getattr(net, "inlined_graph", "") or getattr(net, "code", ""))
        if "fft_" in graph_str:
            _LAMA_FP16_SAFE = False
            return False
        import torch
        gen = torch.Generator(device=device).manual_seed(0)
        probe_x = torch.rand((1, 3, 512, 512), generator=gen, dtype=torch.float32, device=device)
        probe_m = torch.zeros((1, 1, 512, 512), dtype=torch.float32, device=device)
        probe_m[:, :, 200:260, 200:260] = 1.0
        xm = probe_x * (1.0 - probe_m)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                y1 = net(xm, probe_m)
                y2 = net(xm, probe_m)
        _LAMA_FP16_SAFE = bool(torch.isfinite(y1).all().item() and torch.equal(y1, y2))
    except Exception:
        _LAMA_FP16_SAFE = False
    return _LAMA_FP16_SAFE


def _run_lama_tiles(
    net,
    device: str,
    padded_rgb: np.ndarray,
    padded_mask: np.ndarray,
    active_tiles: list[tuple[int, int]],
    th: int = 512,
    tw: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    import gc
    import logging
    import torch
    from .models import _LAMA_LOAD_LOCK, _lama_model_for_device

    ph, pw = padded_mask.shape
    accum = np.zeros((ph, pw, 3), dtype=np.float32)
    weights = np.zeros((ph, pw), dtype=np.float32)
    if not active_tiles:
        return accum, weights

    window = _lama_tile_window(th, tw)
    batch_size = 2 if device == "cuda" else 1
    cpu_fallback: tuple[object, str] | None = None

    def _infer_batch(cur_net, cur_dev: str, img_np: np.ndarray, mask_np: np.ndarray, allow_fp16: bool) -> np.ndarray:
        mask_t = None
        img_t = None
        try:
            mask_t = torch.from_numpy(mask_np).to(cur_dev)
            img_t = torch.from_numpy(img_np).to(cur_dev) * (1.0 - mask_t)
            if allow_fp16 and cur_dev == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred = cur_net(img_t, mask_t)
                if not torch.isfinite(pred).all():
                    raise RuntimeError("non-finite FP16 LaMa prediction")
            else:
                pred = cur_net(img_t, mask_t)
                if not torch.isfinite(pred).all():
                    raise RuntimeError("non-finite LaMa prediction")
            return pred.float().detach().cpu().numpy().transpose(0, 2, 3, 1) * np.float32(255.0)
        finally:
            del img_t, mask_t

    with (_lama_inference_lock if device == "cuda" else contextlib.nullcontext()):
        use_fp16 = _lama_supports_fp16(net, device)
        with torch.inference_mode():
            for start in range(0, len(active_tiles), batch_size):
                batch_coords = active_tiles[start:start + batch_size]
                img_batch = np.stack(
                    [padded_rgb[y:y + th, x:x + tw].transpose(2, 0, 1) for y, x in batch_coords],
                    axis=0,
                ).astype(np.float32) / np.float32(255.0)
                mask_batch = np.stack(
                    [padded_mask[y:y + th, x:x + tw][None] for y, x in batch_coords],
                    axis=0,
                ).astype(np.float32)
                pred_np: np.ndarray | None = None
                cuda_err: str | None = None
                try:
                    pred_np = _infer_batch(net, device, img_batch, mask_batch, use_fp16)
                except Exception as exc:
                    if device != "cuda":
                        raise
                    cuda_err = str(exc)

                if cuda_err is not None:
                    # Exit the except frame before retrying so traceback references to
                    # failed CUDA input/scratch tensors are completely released.
                    logging.getLogger("uvicorn.error").warning(
                        "LaMa CUDA inference failed (%s); clearing cache and falling back", cuda_err
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    tile_preds = []
                    for i_tile in range(len(batch_coords)):
                        im1 = img_batch[i_tile:i_tile + 1]
                        mk1 = mask_batch[i_tile:i_tile + 1]
                        tile_pred: np.ndarray | None = None
                        need_cpu = (len(batch_coords) == 1) and (not use_fp16)
                        if not need_cpu:
                            try:
                                tile_pred = _infer_batch(net, device, im1, mk1, False)[0]
                            except Exception:
                                need_cpu = True
                        if need_cpu:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if cpu_fallback is None:
                                with _LAMA_LOAD_LOCK:
                                    cpu_fallback = _lama_model_for_device("cpu")
                            cpu_net, cpu_dev = cpu_fallback
                            tile_pred = _infer_batch(cpu_net, cpu_dev, im1, mk1, False)[0]
                        tile_preds.append(tile_pred)
                    pred_np = np.stack(tile_preds, axis=0)
                for idx_b, (y, x) in enumerate(batch_coords):
                    accum[y:y + th, x:x + tw] += pred_np[idx_b] * window[:, :, None]
                    weights[y:y + th, x:x + tw] += window
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return accum, weights


def learned_clean_plate(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa sees context; only mask pixels are written back at native resolution.

    Overlapping tiles, reflection padding, and weighted reconstruction avoid
    squeezing a long text column into a square or resampling the original art.
    """
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return rgb.copy()
    net, device = lama_model()
    th, tw = 512, 512  # checkpoint's training crop size; no source image resize
    h, w = mask_bool.shape
    ph, pw = max(h, th), max(w, tw)
    if h < ph or w < pw:
        padded = np.pad(rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
        # Reflection repeats source pixels, including the glyphs being removed.
        # Their masks must follow the identical reflection; otherwise unmasked
        # copies outside a small ROI become evidence for reconstructing old text.
        padded_mask = np.pad(mask_bool, ((0, ph - h), (0, pw - w)), mode="reflect")
    else:
        padded = rgb
        padded_mask = mask_bool
    ys = sorted(set(list(range(0, ph - th + 1, max(1, th // 2))) + [ph - th]))
    xs = sorted(set(list(range(0, pw - tw + 1, max(1, tw // 2))) + [pw - tw]))
    active_tiles = [
        (y, x)
        for y in ys
        for x in xs
        if padded_mask[y:y + th, x:x + tw].any()
    ]
    accum, weights = _run_lama_tiles(net, device, padded, padded_mask, active_tiles, th, tw)
    reconstructed = accum[:h, :w] / np.maximum(weights[:h, :w, None], np.finfo(np.float32).tiny)
    out = rgb.copy()
    out[mask_bool] = np.clip(np.rint(reconstructed[mask_bool]), 0, 255).astype(np.uint8)
    return out


def learned_clean_plate_full_context(
    full_rgb: np.ndarray,
    full_mask: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    *,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run native-resolution 512x512 tiled LaMa with full-page context.

    All overlapping/neighboring strokes in `full_mask` are hidden from LaMa's
    context; only pixels inside `full_mask` (within `roi` if specified) are
    written back, preserving bit-exact `L_inf == 0` outside `full_mask`.
    """
    mask_bool = full_mask > 0
    h, w = mask_bool.shape
    if roi is not None:
        rx0, ry0, rw, rh = roi
        rx0 = max(0, int(rx0))
        ry0 = max(0, int(ry0))
        rx1 = min(w, rx0 + max(0, int(rw)))
        ry1 = min(h, ry0 + max(0, int(rh)))
        roi_rgb = full_rgb[ry0:ry1, rx0:rx1]
        roi_mask = mask_bool[ry0:ry1, rx0:rx1]
        if not np.any(roi_mask):
            return roi_rgb.copy()
    else:
        if not np.any(mask_bool):
            return full_rgb.copy()
        if active_mask is not None and not np.any(active_mask):
            return full_rgb.copy()

    net, device = lama_model()
    th, tw = 512, 512
    ph, pw = max(h, th), max(w, tw)
    if h < ph or w < pw:
        padded = np.pad(full_rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
        padded_mask = np.pad(mask_bool, ((0, ph - h), (0, pw - w)), mode="reflect")
        padded_active = (
            np.pad(active_mask > 0, ((0, ph - h), (0, pw - w)), mode="constant", constant_values=False)
            if active_mask is not None
            else padded_mask
        )
    else:
        padded = full_rgb
        padded_mask = mask_bool
        padded_active = (active_mask > 0) if active_mask is not None else padded_mask

    ys = sorted(set(list(range(0, ph - th + 1, max(1, th // 2))) + [ph - th]))
    xs = sorted(set(list(range(0, pw - tw + 1, max(1, tw // 2))) + [pw - tw]))

    if roi is not None:
        active_tiles = [
            (y, x)
            for y in ys
            for x in xs
            if (y < ry1 and y + th > ry0 and x < rx1 and x + tw > rx0)
            and padded_mask[y:y + th, x:x + tw].any()
        ]
    else:
        active_tiles = [
            (y, x)
            for y in ys
            for x in xs
            if padded_active[y:y + th, x:x + tw].any()
            and padded_mask[y:y + th, x:x + tw].any()
        ]

    accum, weights = _run_lama_tiles(net, device, padded, padded_mask, active_tiles, th, tw)
    if roi is not None:
        sub_w = weights[ry0:ry1, rx0:rx1]
        sub_a = accum[ry0:ry1, rx0:rx1]
        write_m = roi_mask & (sub_w > 0)
        out = roi_rgb.copy()
        if np.any(write_m):
            recon = sub_a[write_m] / np.maximum(sub_w[write_m, None], np.finfo(np.float32).tiny)
            out[write_m] = np.clip(np.rint(recon), 0, 255).astype(np.uint8)
        return out

    write_m = mask_bool & (weights[:h, :w] > 0)
    out = full_rgb.copy()
    if np.any(write_m):
        recon = accum[:h, :w][write_m] / np.maximum(weights[:h, :w][write_m, None], np.finfo(np.float32).tiny)
        out[write_m] = np.clip(np.rint(recon), 0, 255).astype(np.uint8)
    return out
