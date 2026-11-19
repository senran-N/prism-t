from __future__ import annotations

import cv2
import logging
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.ndimage import distance_transform_edt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import chi2
from .preprocess import estimate_noise_sigma, estimate_substrate_noise, substrate_squared_distance
from .schemas import RestorationPath

SPECTRAL_FALSE_ALARM = 1e-6


def repair_observed_paper_residue(
    raw: np.ndarray, clean: np.ndarray, alpha: np.ndarray,
    observed: np.ndarray, substrate: np.ndarray | None,
    occluders: np.ndarray | None = None,
    source_domain: np.ndarray | None = None,
    context_exclusion: np.ndarray | None = None,
) -> tuple[np.ndarray,np.ndarray,float | None,float | None,int]:
    """Audit observed glyph ink and repair it only on a measured flat substrate.

    This detects unchanged source strokes, not translation correctness or the
    truth of hidden artwork. No unobserved region is treated as missing text.
    """
    evidence = observed > 0
    if occluders is not None:
        evidence &= occluders == 0
    if substrate is None or not evidence.any():
        return clean,alpha,None,None,0
    known = (substrate > 0) & ~evidence & (alpha == 0)
    if context_exclusion is not None:
        known &= ~context_exclusion
    if not known.any():
        return clean,alpha,None,None,0
    samples = raw[known].astype(float)
    center = np.median(samples,axis=0)
    sigma = estimate_substrate_noise(raw,known)
    deviations = substrate_squared_distance(samples,center,sigma)
    if np.quantile(deviations,.75) > chi2.ppf(.75,3):
        return clean,alpha,None,None,0
    # Bonferroni bounds false foreground detections over this observed domain;
    # this statistical operating point is independent of input resolution.
    # A paper model has authority only inside the observed paper substrate.
    # Detector stroke evidence may also touch a wall or neighboring artwork;
    # it cannot extend a constant-paper repair beyond that measured domain.
    source_support = evidence & (substrate>0)
    if source_domain is not None:
        # Geometry is an independent check for glyph/anti-alias samples missed
        # by the initial stroke extractor; flat-paper evidence still determines
        # which of those raw pixels are ink.
        source_support |= (source_domain>0)&(substrate>0)
    tested = source_support | ((alpha > 0) & (substrate > 0))
    critical = chi2.isf(0.05 / max(1, np.count_nonzero(tested)), 3)
    fringe_crit = chi2.isf(0.01, 3)
    raw_dev = substrate_squared_distance(raw, center, sigma)
    source_ink = source_support & (raw_dev > critical)
    if occluders is not None:
        source_ink &= occluders == 0
    total = np.count_nonzero(source_ink)
    if not total:
        return clean, alpha, 0.0, 0.0, 0
    uncovered = float(np.count_nonzero(source_ink & (alpha == 0)) / total)
    clean_dev = substrate_squared_distance(clean, center, sigma)
    changed_background = clean_dev > critical
    residual = source_ink & changed_background
    residue_before = float(np.count_nonzero(residual) / total)
    result, recorded = clean.copy(), alpha.copy()

    # On enclosed flat paper where source_domain is supplied, include an immediate
    # hysteresis fringe (proportional to measured stroke radius) connected to strong
    # source_ink (using chi2.isf(0.01, 3)) so faint anti-aliased letter ghosting on
    # aged paper is cleanly repaired without touching unrelated pixels.
    fringe_ink = np.zeros(source_ink.shape, dtype=bool)
    if source_domain is not None:
        fringe_domain = (source_support | ((alpha > 0) & (substrate > 0))) & (raw_dev > fringe_crit)
        if occluders is not None:
            fringe_domain &= occluders == 0
        ink_radii = distance_transform_edt(source_ink)[source_ink]
        ink_q3 = float(np.quantile(ink_radii, 0.75)) if ink_radii.size else 1.0
        fringe_r = max(float(np.sqrt(2.0)), 0.5 * ink_q3)
        near_ink = distance_transform_edt(~source_ink) <= fringe_r
        fringe_cand = fringe_domain & near_ink & (~source_ink)
        if fringe_cand.any():
            _, f_labels = cv2.connectedComponents((source_ink | fringe_cand).astype(np.uint8), connectivity=8)
            ink_ids = np.unique(f_labels[source_ink])
            fringe_ink = np.isin(f_labels, ink_ids[ink_ids != 0]) & fringe_cand

    changed_for_repair = (clean_dev > fringe_crit) if source_domain is not None else changed_background
    # A learned restorer can generate a dark continuation beside the original
    # strokes. On measured paper, audit its entire recorded erasure footprint
    # inside that substrate, not just positions that used to be dark in raw.
    repair = (source_ink | fringe_ink | ((alpha > 0) & (substrate > 0))) & changed_for_repair
    if occluders is not None:
        repair &= occluders == 0
    if repair.any():
        recorded[repair] = 1
        # The flat-paper model passed on observable pixels; applying its
        # measured median avoids asking LaMa to reconstruct source lettering.
        result[repair] = np.clip(np.rint(center), 0, 255).astype(np.uint8)
    return result, recorded, uncovered, residue_before, int(np.count_nonzero(repair))


def fisher_g_survival(g: float, frequencies: int) -> float:
    """Fisher's exact maximum-periodogram tail for independent exponential bins.

    P(G >= g) = sum_k (-1)^(k-1) C(n,k) (1-k*g)^(n-1).
    Log arithmetic avoids overflowing binomial coefficients. The caller reports
    this as a white-noise null test, not a calibrated natural-image probability.
    """
    if frequencies < 2 or g <= 1.0 / frequencies:
        return 1.0
    if g >= 1.0:
        return 0.0
    k = np.arange(1, min(frequencies, int(np.floor(1.0 / g))) + 1, dtype=np.float64)
    k = k[k * g < 1.0]
    logs = gammaln(frequencies + 1) - gammaln(k + 1) - gammaln(frequencies - k + 1)
    logs += (frequencies - 1) * np.log1p(-k * g)
    signs = np.where(k.astype(np.int64) % 2, 1.0, -1.0)
    log_tail, sign = logsumexp(logs, b=signs, return_sign=True)
    return float(np.clip(sign * np.exp(log_tail), 0.0, 1.0))


def validation_collar(active: np.ndarray, known: np.ndarray) -> np.ndarray:
    radii = distance_transform_edt(active)[active]
    exterior_distance = distance_transform_edt(~active)
    thickness = max(float(np.quantile(radii, 0.75)), float(np.min(exterior_distance[known])))
    return known & (exterior_distance <= thickness)


def reconstruction_unknown(stroke_support: np.ndarray, observed: np.ndarray,
                           neighbor_support: np.ndarray) -> np.ndarray:
    """Separate unavailable reconstruction context from permission to write.

    The measured source stroke radius sets the quarantine extent. Suspected
    antialias/outline remnants cannot supply Dirichlet data or solver validation
    truth, even when they have not earned permission to change source pixels.
    """
    active = stroke_support > 0
    measured = active & (observed > 0)
    if not active.any():
        return neighbor_support.copy()
    radii = distance_transform_edt(measured)[measured]
    if not radii.size:
        radii = distance_transform_edt(active)[active]
    q1, q3 = np.quantile(radii, [.25, .75])
    radius = q3 + 1.5*(q3-q1)
    return (distance_transform_edt(~active) <= radius) | neighbor_support


def normalized_prediction_error(prediction: np.ndarray, truth: np.ndarray, holdout: np.ndarray) -> float:
    difference = prediction[holdout].astype(np.float64) - truth[holdout].astype(np.float64)
    return float(np.sqrt(np.mean(difference * difference)) / 255.0)


def select_context_restoration(rgb: np.ndarray, active: np.ndarray, known: np.ndarray,
                               context_exclusion: np.ndarray | None = None,
                               allow_learned: bool = True) -> tuple[np.ndarray, str, float, np.ndarray]:
    """Select one solver using observable color and edge prediction errors.

    Hide a collar measured from stroke EDT, reconstruct it together with the
    hole, and score only observable pixels/stencils. Spatial candidate mixtures
    were rejected on real artwork: collar accuracy did not prevent blurry or
    discontinuous hidden structures. No unobservable quality is certified.
    """
    unknown = active if context_exclusion is None else active | context_exclusion
    known = known & ~unknown
    holdout = validation_collar(active, known)
    validation_mask = unknown | holdout
    gradient_domain = holdout & (cv2.erode(known.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    derivative_x,derivative_y=cv2.getDerivKernels(1,0,cv2.FILTER_SCHARR,normalize=False)
    gradient_gain=float(np.sum((derivative_y@derivative_x.T)**2))
    truth=rgb.astype(float)
    truth_gradients=[cv2.Scharr(truth,cv2.CV_64F,dx,dy) for dx,dy in ((1,0),(0,1))]
    def validation_loss(prediction):
        domain=gradient_domain
        predicted=prediction.astype(float)
        errors=np.mean((predicted-truth)**2,axis=2)
        # Derivatives are evaluated only where their complete raw stencil is
        # observable. Whitening by the exact stencil gain puts color and edge
        # errors in the same sensor-noise units, without an aesthetic weight.
        for gradient,(dx,dy) in zip(truth_gradients,((1,0),(0,1))):
            delta=cv2.Scharr(predicted,cv2.CV_64F,dx,dy)-gradient
            errors[domain] += np.mean(delta[domain]**2,axis=1)/gradient_gain
        errors[domain] /= 3
        return errors[holdout]
    solvers = {
        "solid_fill": lambda image, mask: solve_laplace_dirichlet(image, mask),
        "structure_tensor_line": lambda image, mask: solve_anisotropic_tensor_inpaint(image, mask)[0],
        "narrow_slit_pde": lambda image, mask: solve_biharmonic_slit(image.astype(np.float64), mask),
    }
    if allow_learned:
        from .learned import learned_clean_plate
        solvers["lama_masked"] = learned_clean_plate
    losses = []
    for name, solve in solvers.items():
        logging.getLogger("uvicorn.error").info("restoration candidate %s: %d masked pixels", name, np.count_nonzero(validation_mask))
        candidate = solve(rgb, validation_mask)
        losses.append(validation_loss(candidate))
    losses = np.asarray(losses)
    best=int(np.argmin(losses.mean(axis=1)))
    logging.getLogger('uvicorn.error').info('restoration held-out color/edge RMSE: %s',
        {name:float(np.sqrt(losses[i].mean())) for i,name in enumerate(solvers)})
    selected=list(solvers)[best]
    candidate=solvers[selected](rgb,unknown)
    result=rgb.copy()
    result[active]=np.clip(np.rint(candidate[active]),0,255).astype(np.uint8)
    background=np.clip(np.rint(candidate),0,255).astype(np.uint8)
    return result,selected,float(np.sqrt(losses[best].mean())/255),background


def unmix_stroke_boundary(raw: np.ndarray, clean: np.ndarray, alpha: np.ndarray,
                          background: np.ndarray, observed: np.ndarray,
                          unknown: np.ndarray, forbidden: np.ndarray,
                          residual: float) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Remove optically supported edge mixtures without replacing their texture.

    Background inference and pigment evidence are separate inputs. Only the
    current stroke's quarantined neighborhood is examined; another node or Z2
    never earns write permission. The returned boundary opacity is recorded
    separately from the combined erasure support.
    """
    from .learned import _bh_fdr_threshold, prepare_lab_crop, stroke_palette
    active=alpha>0
    evidence=active & (observed>0) & ~forbidden
    band=unknown & ~active & ~forbidden
    opacity=np.zeros(alpha.shape,np.float32)
    if not evidence.any() or not band.any():
        return clean,alpha,opacity
    # Restrict the unknown domain to this stroke's measured neighborhood;
    # neighbor-text exclusions are not candidate erasure regions.
    radius_samples=distance_transform_edt(evidence)[evidence]
    q1,q3=np.quantile(radius_samples,[.25,.75])
    band &= distance_transform_edt(~active)<=q3+1.5*(q3-q1)
    known=~unknown & ~forbidden
    noise=estimate_substrate_noise(raw,known)
    lab_cache = prepare_lab_crop(raw)
    pigments,_=stroke_palette(raw,evidence,lab_cache=lab_cache)
    ys,xs=np.nonzero(band)
    if not len(ys):
        return clean,alpha,opacity
    global_res = np.maximum(255.0 * float(residual), 0.0)
    if global_res > 0.0 and known.any():
        known_f = known.astype(float)
        err_sum = np.where(known[..., None], np.abs(raw.astype(float) - background), 0.0)
        w_map = known_f
        k3 = np.ones((3, 3), dtype=np.float64)
        for _ in range(2):
            w_blur = cv2.filter2D(w_map, -1, k3, borderType=cv2.BORDER_REPLICATE)
            e_blur = np.stack(
                [cv2.filter2D(err_sum[..., c], -1, k3, borderType=cv2.BORDER_REPLICATE) for c in range(3)],
                axis=2,
            )
            err_sum = np.where(known[..., None] & (w_blur[..., None] > 0), e_blur / np.maximum(w_blur[..., None], 1e-6), 0.0)
            w_map = known_f
        _, nn_idx = distance_transform_edt(~known, return_indices=True)
        local_res = np.minimum(global_res, err_sum[nn_idx[0][ys, xs], nn_idx[1][ys, xs]])
    else:
        local_res = np.full((len(ys), 3), global_res, dtype=float)
    sigma=np.sqrt(noise[None, :]**2 + local_res**2)
    image=raw[ys,xs].astype(float)
    base=background[ys,xs].astype(float)
    signal=(image-base)/sigma
    # A background-only pixel must first be rejected; otherwise artwork that
    # already agrees with the prediction would be altered by an arbitrary fit.
    # Use Benjamini-Hochberg FDR (q=0.05) on the immediately-adjacent connected
    # boundary shell (scaled from measured stroke radius) so continuous
    # anti-aliased halos are cleanly unmixed.
    alpha_bonf=.05/(len(ys)*max(len(pigments),1))
    critical=chi2.isf(alpha_bonf,3)
    crit_1d=chi2.isf(alpha_bonf,1)
    crit_2d=chi2.isf(alpha_bonf,2)
    sig_energy = np.sum(signal**2,axis=1)
    detectable = sig_energy > critical
    shell_r = max(float(np.sqrt(2.0)), 0.5 * float(q3))
    adj_shell = distance_transform_edt(~active)[ys, xs] <= shell_r
    if np.any(adj_shell):
        p_adj = chi2.sf(sig_energy[adj_shell], 3)
        p_cut = _bh_fdr_threshold(p_adj, q=0.05)
        if p_cut > 0.0:
            detectable[adj_shell] |= (p_adj <= p_cut)
    best_error=np.full(len(ys),np.inf)
    best_alpha=np.zeros(len(ys))
    best_recovered=image.copy()
    max_gain = 4.0
    for pigment in pigments:
        direction=(pigment-base)/sigma
        information=np.sum(direction**2,axis=1)
        amount=np.divide(np.sum(signal*direction,axis=1),information,
                         out=np.zeros(len(ys)),where=information>0)
        error=np.sum((signal-amount[:,None]*direction)**2,axis=1)
        crit_1d_eff = np.where(adj_shell, chi2.isf(0.05, 1), crit_1d)
        uncertainty=np.sqrt(np.divide(crit_1d_eff,information,
                                      out=np.full(len(ys),np.inf),where=information>0))
        stable=(amount>uncertainty)&(amount+uncertainty<1)
        valid=detectable & stable & (error<=crit_2d)
        # Clamp extreme 1/(1 - alpha) amplification when alpha > 0.75 or JPEG ringing dominates
        denom = 1.0 - amount[:, None]
        inv_gain = np.where(denom > 1e-6, np.minimum(1.0 / np.maximum(denom, 1e-6), max_gain), 0.0)
        recovered = base + inv_gain * (image - amount[:, None] * pigment - denom * base)
        valid &= np.all((recovered>=0)&(recovered<=255),axis=1)
        take=valid & (error<best_error)
        best_error[take]=error[take]
        best_alpha[take]=amount[take]
        best_recovered[take]=recovered[take]

    # Decoupled luma/chroma (BT.601 YCbCr) mixture model on the immediate
    # stroke adjacency shell: JPEG 4:2:0 transforms, subsamples, and quantizes
    # Y, Cb, Cr as three independent channels, so a stroke-adjacent sky/art
    # pixel can shift monotonically toward the stroke's measured (Y, Cb, Cr)
    # pigment envelope at different rates per channel (alpha_Y != alpha_Cb != alpha_Cr).
    rem_mask = detectable & adj_shell & (best_alpha == 0)
    if np.any(rem_mask) and len(pigments) > 0:
        M_ycc = np.array(
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
            dtype=np.float64,
        )
        M_rgb = np.linalg.inv(M_ycc)
        rem_idx = np.flatnonzero(rem_mask)
        img_r = image[rem_idx]
        base_r = base[rem_idx]
        sig_r = sigma[rem_idx]
        ycc_sigma = np.maximum(np.sqrt((sig_r ** 2) @ (M_ycc.T ** 2)), 1e-6)
        img_ycc = img_r @ M_ycc.T
        base_ycc = base_r @ M_ycc.T
        delta_ycc = img_ycc - base_ycc
        pig_ycc = pigments @ M_ycc.T  # (K, 3)

        # Stroke + local background per-channel (Y, Cb, Cr) convex envelope
        pig_lo = np.min(pig_ycc, axis=0)[None, :]  # (1, 3)
        pig_hi = np.max(pig_ycc, axis=0)[None, :]  # (1, 3)
        env_lo = np.minimum(base_ycc, pig_lo)
        env_hi = np.maximum(base_ycc, pig_hi)
        z_1d = float(np.sqrt(chi2.isf(0.05, 1)))
        tol = z_1d * ycc_sigma

        in_envelope = np.all((img_ycc >= env_lo - tol) & (img_ycc <= env_hi + tol), axis=1)
        # Measure each channel's fractional shift from base_ycc toward the stroke envelope endpoint
        pos_span = np.maximum(pig_hi - base_ycc, 0.0)
        neg_span = np.maximum(base_ycc - pig_lo, 0.0)
        alpha_pos = np.divide(np.maximum(delta_ycc, 0.0), pos_span, out=np.zeros_like(delta_ycc), where=pos_span > tol)
        alpha_neg = np.divide(np.maximum(-delta_ycc, 0.0), neg_span, out=np.zeros_like(delta_ycc), where=neg_span > tol)
        alpha_ch = np.clip(np.where(delta_ycc >= 0.0, alpha_pos, alpha_neg), 0.0, 1.0)
        ch_sig = np.abs(delta_ycc) > tol
        moved_toward_stroke = np.any(ch_sig & (alpha_ch > 0.0), axis=1)

        valid_ycc = in_envelope & moved_toward_stroke
        if np.any(valid_ycc):
            # Restore channels that shifted toward the stroke back to isolated base_ycc
            # while retaining orthogonal/unshifted luma or chroma variation.
            rec_ycc = np.where(ch_sig & (alpha_ch > 0.0), base_ycc, img_ycc)
            rec_rgb = rec_ycc @ M_rgb.T
            valid_ycc &= np.all((rec_rgb >= -1.0) & (rec_rgb <= 256.0), axis=1)
            eff_a = np.max(np.where(ch_sig, alpha_ch, 0.0), axis=1)
            valid_ycc &= eff_a > 0.0
            if np.any(valid_ycc):
                dst_idx = rem_idx[valid_ycc]
                best_alpha[dst_idx] = eff_a[valid_ycc]
                best_recovered[dst_idx] = np.clip(rec_rgb[valid_ycc], 0.0, 255.0)

    opacity[ys,xs]=(np.rint(best_alpha*255)/255).astype(np.float32)
    # Optical agreement alone cannot acquire a disconnected same-color object.
    _,labels=cv2.connectedComponents((active|(opacity>0)).astype(np.uint8),connectivity=8)
    owned=np.unique(labels[active]);owned=owned[owned>0]
    opacity[~np.isin(labels,owned)]=0
    accepted=opacity[ys,xs]>0
    out=clean.copy()
    out[ys[accepted],xs[accepted]]=np.rint(best_recovered[accepted]).astype(np.uint8)
    support=alpha.copy()
    support[opacity>0]=opacity[opacity>0]
    return out,support,opacity


def solve_laplace_dirichlet(img_float: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    r"""
    Exact discrete Laplace equation (\Delta u = 0) inside mask_bool with exact Dirichlet
    boundary conditions u = img_float on ~mask_bool and natural zero flux at
    the actual image edge. Missing neighbors are not invented observations.
    Vectorized across channels.
    """
    full_h, full_w = mask_bool.shape[:2]
    out = img_float.copy().astype(np.float64)
    full_ys, full_xs = np.nonzero(mask_bool)
    n_unk = full_ys.size
    if n_unk == 0:
        return out
    if n_unk == mask_bool.size:
        raise ValueError("harmonic reconstruction requires observable Dirichlet pixels")

    # Solve on the 1-pixel-padded tight bbox: preserves every Dirichlet neighbor
    # and clamps at 0/full only when unknown pixels touch the real image edge.
    y0 = max(0, int(full_ys.min()) - 1)
    y1 = min(full_h, int(full_ys.max()) + 2)
    x0 = max(0, int(full_xs.min()) - 1)
    x1 = min(full_w, int(full_xs.max()) + 2)
    h, w = y1 - y0, x1 - x0
    ys = full_ys - y0
    xs = full_xs - x0

    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(n_unk, dtype=np.int32)

    rows_list = [np.arange(n_unk, dtype=np.int32)]
    cols_list = [np.arange(n_unk, dtype=np.int32)]
    degree = np.zeros(n_unk, dtype=np.float64)
    vals_list = [degree]

    is_2d = out.ndim == 2
    chans = 1 if is_2d else out.shape[2]
    work = out[:, :, None] if is_2d else out
    sub_work = work[y0:y1, x0:x1, :]
    rhs = np.zeros((n_unk, chans), dtype=np.float64)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        centers = np.flatnonzero(inside)
        ny, nx = ny[inside], nx[inside]
        degree[centers] += 1
        n_idx = idx_map[ny, nx]
        is_unk = n_idx >= 0
        if np.any(is_unk):
            u_i = centers[is_unk]
            rows_list.append(u_i)
            cols_list.append(n_idx[is_unk])
            vals_list.append(np.full(u_i.size, -1.0, dtype=np.float64))
        is_kn = ~is_unk
        if np.any(is_kn):
            k_i = centers[is_kn]
            rhs[k_i, :] += sub_work[ny[is_kn], nx[is_kn], :]

    A = sparse.csc_matrix(
        (np.concatenate(vals_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(n_unk, n_unk),
    )
    sol = spsolve(A, rhs, permc_spec="MMD_AT_PLUS_A")
    if sol.ndim == 1:
        sol = sol[:, None]
    work[full_ys, full_xs, :] = sol
    return work[:, :, 0] if is_2d else work


def solve_laplace_dirichlet_fast(img_float: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Two-level coarse-exact + C++ Jacobi harmonic solver for large background probes.

    Matches `solve_laplace_dirichlet` exactly for <= 4096 unknowns and runs ~15-30x
    faster on large text envelopes (100k+ pixels) with sub-pixel mean error (~0.2 RGB).
    """
    full_h, full_w = mask_bool.shape[:2]
    out = img_float.copy().astype(np.float64)
    full_ys, full_xs = np.nonzero(mask_bool)
    n_unk = full_ys.size
    if n_unk == 0:
        return out
    if n_unk == mask_bool.size:
        raise ValueError("harmonic reconstruction requires observable Dirichlet pixels")
    if n_unk <= 4096:
        return solve_laplace_dirichlet(img_float, mask_bool)

    y0 = max(0, int(full_ys.min()) - 1)
    y1 = min(full_h, int(full_ys.max()) + 2)
    x0 = max(0, int(full_xs.min()) - 1)
    x1 = min(full_w, int(full_xs.max()) + 2)
    h, w = y1 - y0, x1 - x0
    is_2d = out.ndim == 2
    work = out[:, :, None] if is_2d else out
    u = work[y0:y1, x0:x1, :].astype(np.float32, copy=True)
    m = mask_bool[y0:y1, x0:x1]

    # Pick coarse downsample factor so the coarse exact solve stays <= ~4096 unknowns.
    scale = 4 if n_unk > 12000 and min(h, w) >= 24 else 2
    ch, cw = max(4, (h + scale - 1) // scale), max(4, (w + scale - 1) // scale)
    kn = (~m).astype(np.float32)
    kn_c = cv2.resize(kn, (cw, ch), interpolation=cv2.INTER_AREA)
    m_c = kn_c <= 1e-6
    if m_c.any() and (~m_c).any():
        num_c = cv2.resize(u * kn[:, :, None], (cw, ch), interpolation=cv2.INTER_AREA)
        if num_c.ndim == 2:
            num_c = num_c[:, :, None]
        u_c = np.where(kn_c[:, :, None] > 1e-6, num_c / np.maximum(kn_c[:, :, None], 1e-6), 0.0)
        sol_c = solve_laplace_dirichlet_fast(u_c.astype(np.float64), m_c).astype(np.float32)
        up = cv2.resize(sol_c, (w, h), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[:, :, None]
        u[m] = up[m]

    kern = np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], dtype=np.float32)
    known_u = u.copy()
    m3 = m[:, :, None]
    for _ in range(20):
        avg = cv2.filter2D(u, -1, kern, borderType=cv2.BORDER_REPLICATE)
        if avg.ndim == 2:
            avg = avg[:, :, None]
        u = np.where(m3, avg, known_u)
    work[y0:y1, x0:x1, :] = u.astype(np.float64)
    return work[:, :, 0] if is_2d else work


def solve_biharmonic_slit(img_float: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    r"""
    Minimize squared native-grid Laplacian on the hole and its known collar.
    The image edge has natural zero-flux boundary, not copied hidden RGB.
    Exact observed Dirichlet values make L_unknown full column rank; no ridge
    term is needed to make the solution unique or shrink its color values.
    """
    full_h, full_w = mask_bool.shape[:2]
    if not np.any(mask_bool):
        return img_float.copy()

    if mask_bool.all():
        raise ValueError("biharmonic reconstruction requires observable Dirichlet pixels")

    # Solve strictly on the 2-pixel-padded tight bounding box of mask_bool:
    # the 1-ring collar and 4-neighbor Laplacian stencil never read beyond 2 px,
    # while boundary clamping at 0/full preserves the exact zero-flux edge.
    mask_ys, mask_xs = np.nonzero(mask_bool)
    y0 = max(0, int(mask_ys.min()) - 2)
    y1 = min(full_h, int(mask_ys.max()) + 3)
    x0 = max(0, int(mask_xs.min()) - 2)
    x1 = min(full_w, int(mask_xs.max()) + 3)

    sub_mask = mask_bool[y0:y1, x0:x1]
    h, w = sub_mask.shape[:2]

    # Laplacian stencil evaluated on unknown pixels AND their 1-ring known collar (enforcing normal slope continuity!)
    collar = cv2.dilate(sub_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0
    eq_ys, eq_xs = np.nonzero(collar)
    m_eq = eq_ys.size
    n_pix = h * w
    flat_idx = np.arange(n_pix, dtype=np.int32).reshape(h, w)

    rows = []
    cols = []
    vals = []
    center_id = flat_idx[eq_ys, eq_xs]
    eq_id = np.arange(m_eq, dtype=np.int32)
    rows.append(eq_id)
    cols.append(center_id)
    degree = np.zeros(m_eq, dtype=np.float64)
    vals.append(degree)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = eq_ys + dy, eq_xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        degree[inside] += 1
        rows.append(eq_id[inside])
        cols.append(flat_idx[ny[inside], nx[inside]])
        vals.append(np.full(np.count_nonzero(inside), -1.0, dtype=np.float64))

    L_op = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(m_eq, n_pix),
    )

    unk_flat = np.nonzero(sub_mask.ravel())[0]
    kn_flat = np.nonzero(~sub_mask.ravel())[0]
    is_2d = img_float.ndim == 2
    sub_values = img_float[y0:y1, x0:x1].astype(np.float64).copy()
    work = (sub_values[:, :, None] if is_2d else sub_values).reshape(n_pix, -1)

    L_u = L_op[:, unk_flat]
    L_k = L_op[:, kn_flat]
    # Minimize || L_u u_u + L_k u_k ||_2^2 => (L_u^T L_u) u_u = -L_u^T (L_k u_k)
    lhs = L_u.T @ L_u
    rhs = -(L_u.T @ (L_k @ work[kn_flat, :]))
    sol = spsolve(lhs.tocsc(), rhs, permc_spec="MMD_ATA")
    if sol.ndim == 1:
        sol = sol[:, None]
    work[unk_flat, :] = sol
    sub_restored = work.reshape((h, w) if is_2d else (h, w, -1))
    out = img_float.astype(np.float64).copy()
    out[y0:y1, x0:x1] = sub_restored
    return out


def solve_anisotropic_tensor_inpaint(
    img_rgb: np.ndarray, mask_bool: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Directional speedline / hatching continuation via double-angle harmonic orientation extension
    and elliptic anisotropic diffusion PDE: -div(D(theta) grad u) = 0.
    Returns (restored_rgb_float, mean_coherence_score).
    """
    h, w = mask_bool.shape[:2]
    if not mask_bool.any():
        return img_rgb.astype(np.float64),0.
    if mask_bool.all():
        raise ValueError("directional reconstruction requires observable Dirichlet pixels")
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0) / 16.0
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1) / 16.0

    # Scharr has one-pixel stencil support. Exclude every derivative whose
    # stencil touches unknown RGB; no arbitrary additional guard is required.
    guard = cv2.dilate(mask_bool.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gx[guard] = 0.0
    gy[guard] = 0.0

    radius = float(np.quantile(distance_transform_edt(mask_bool)[mask_bool],.75)) if mask_bool.any() else 1.
    j_xx = cv2.GaussianBlur(gx * gx, (0, 0), radius)
    j_xy = cv2.GaussianBlur(gx * gy, (0, 0), radius)
    j_yy = cv2.GaussianBlur(gy * gy, (0, 0), radius)

    # Eigenvalues of 2x2 structure tensor J: lambda_{1,2} = 0.5 * (tr +- sqrt((j_xx - j_yy)^2 + 4 j_xy^2))
    tr = j_xx + j_yy
    det_disc = np.hypot(j_xx - j_yy, 2.0 * j_xy)
    lam1 = 0.5 * (tr + det_disc)
    lam2 = 0.5 * (tr - det_disc)
    epsilon = np.finfo(float).eps
    coherence = ((lam1 - lam2) / np.maximum(lam1 + lam2,epsilon)) ** 2

    ring = validation_collar(mask_bool,~guard) if np.any(~guard) and mask_bool.any() else ~guard
    mean_coh = float(np.mean(coherence[ring])) if np.any(ring) else 0.0

    # Tangent direction e_parallel perpendicular to dominant gradient:
    # Double-angle vector of gradient is (j_xx - j_yy, 2*j_xy) = R * (cos 2*phi_g, sin 2*phi_g);
    # Tangent angle theta = phi_g + pi/2 => (cos 2*theta, sin 2*theta) = -(cos 2*phi_g, sin 2*phi_g)!
    norm_d = np.maximum(det_disc, epsilon)
    c2 = -(j_xx - j_yy) / norm_d
    s2 = -(2.0 * j_xy) / norm_d

    # Anisotropic diffusion tensor D = t t^T + kappa_perp * n n^T
    # Observed directional energy and propagated Scharr noise determine the
    # transverse conductivity. Flat regions become isotropic; coherent lines
    # become directional without a fixed diffusion-strength parameter.
    noise_rgb = estimate_substrate_noise(img_rgb, ~guard)
    scharr_x, scharr_y = cv2.getDerivKernels(1, 0, cv2.FILTER_SCHARR, normalize=False)
    noise_energy = float(np.dot(np.square([.299, .587, .114]), noise_rgb ** 2)
                         * np.sum(np.square(scharr_y @ scharr_x.T)) / 16 ** 2)
    ratio = (lam2 + noise_energy) / np.maximum(lam1 + noise_energy, epsilon)
    # Harmonically extend [c2, s2, ratio] across guard in one shared matrix solve.
    if np.any(~guard):
        ext = solve_laplace_dirichlet(np.stack((c2, s2, ratio), axis=-1), guard)
        c2_ext, s2_ext, kappa_perp = ext[:, :, 0], ext[:, :, 1], ext[:, :, 2]
    else:
        c2_ext, s2_ext, kappa_perp = c2, s2, ratio
    theta = 0.5 * np.arctan2(s2_ext, c2_ext)
    tx, ty = np.cos(theta), np.sin(theta)
    d_xx = tx * tx + kappa_perp * (ty * ty)
    d_yy = ty * ty + kappa_perp * (tx * tx)
    d_xy = (1.0 - kappa_perp) * tx * ty

    # Each undirected grid edge receives the average endpoint conductivity.
    # Missing exterior edges implement natural zero flux. This keeps the
    # stiffness symmetric and never declares copied unknown RGB observable.
    restored=img_rgb.astype(np.float64).copy()
    unk_ys, unk_xs = np.nonzero(mask_bool)
    n_unk = unk_ys.size
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[unk_ys, unk_xs] = np.arange(n_unk, dtype=np.int32)

    rows_l = []
    cols_l = []
    vals_l = []
    rhs = np.zeros((n_unk, 3), dtype=np.float64)
    diag = np.zeros(n_unk, dtype=np.float64)

    def add_coupling(dy: int, dx: int, weight: np.ndarray):
        nonlocal diag
        ny, nx = unk_ys + dy, unk_xs + dx
        inside=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        centers=np.flatnonzero(inside)
        ny,nx=ny[inside],nx[inside]
        edge_weight=.5*(weight[unk_ys[inside],unk_xs[inside]]+weight[ny,nx])
        diag[centers] += edge_weight
        n_id = idx_map[ny,nx]
        is_u = n_id >= 0
        if np.any(is_u):
            ui = centers[is_u]
            rows_l.append(ui)
            cols_l.append(n_id[is_u])
            vals_l.append(-edge_weight[is_u])
        is_k = ~is_u
        if np.any(is_k):
            ki = centers[is_k]
            rhs[ki, :] += edge_weight[is_k, None] * restored[ny[is_k], nx[is_k], :]

    # Monotone nine-point tensor approximation. Diagonal edges supply mixed
    # diffusion and subtract |D_xy| from the axial coefficients. Tensors that
    # are not diagonally dominant require clipping those axial coefficients;
    # this is an approximation, not an exact arbitrary-anisotropy stencil.
    abs_c = np.abs(d_xy)
    w_horizontal=np.maximum(d_xx-abs_c,epsilon)
    w_vertical=np.maximum(d_yy-abs_c,epsilon)
    w_se_nw = np.maximum(d_xy, 0.0)
    w_ne_sw = np.maximum(-d_xy, 0.0)

    add_coupling(0, 1, w_horizontal)
    add_coupling(0, -1, w_horizontal)
    add_coupling(1, 0, w_vertical)
    add_coupling(-1, 0, w_vertical)
    add_coupling(1, 1, w_se_nw)
    add_coupling(-1, -1, w_se_nw)
    add_coupling(-1, 1, w_ne_sw)
    add_coupling(1, -1, w_ne_sw)

    rows_l.append(np.arange(n_unk, dtype=np.int32))
    cols_l.append(np.arange(n_unk, dtype=np.int32))
    vals_l.append(diag)

    A = sparse.csc_matrix(
        (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(n_unk, n_unk),
    )
    sol = spsolve(A, rhs, permc_spec="MMD_AT_PLUS_A")
    restored[unk_ys, unk_xs, :] = np.clip(sol, 0.0, 255.0)
    return restored, mean_coh


def detect_and_synthesize_halftone_fft(
    roi_rgb: np.ndarray, mask_bool: np.ndarray
) -> tuple[bool, np.ndarray]:
    """
    Fisher's exact spectral crest test + Sub-pixel Reciprocal Lattice Harmonic Least-Squares Synthesis.
    Fits continuous-frequency trigonometric crystal basis EXCLUSIVELY on known unmasked pixels (~mask_bool)
    to eliminate mask-induced spectral leakage and reconstruct razor-sharp periodic screentones.
    """
    h, w = mask_bool.shape[:2]
    if h < 16 or w < 16:
        return False, roi_rgb.astype(np.float64)

    known = ~mask_bool
    if np.count_nonzero(known) < max(64, int(0.20 * h * w)):
        return False, roi_rgb.astype(np.float64)

    img_f = roi_rgb.astype(np.float64)
    # 1. Smooth low-frequency substrate/tone envelope across mask via exact Laplace Dirichlet
    k_blur = max(7, (min(h, w) // 5) | 1)
    init_rgb = solve_laplace_dirichlet(img_f, mask_bool)
    low_pass = cv2.GaussianBlur(init_rgb, (k_blur, k_blur), 0)

    residual_rgb = img_f - low_pass
    gray_res = 0.299 * residual_rgb[:, :, 0] + 0.587 * residual_rgb[:, :, 1] + 0.114 * residual_rgb[:, :, 2]

    # 2. 2D Hann-windowed FFT power spectrum over zero-masked residual
    obs = np.where(known, gray_res, 0.0)
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.fft.fftshift(np.fft.fft2(obs * win))
    power = np.abs(spec) ** 2

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    fy_bins = yy - cy
    fx_bins = xx - cx
    rad_norm = np.hypot(fy_bins / float(h), fx_bins / float(w))

    # Every resolvable non-DC frequency up to pixel-grid Nyquist is eligible.
    # No fixed 12..48 pixel screentone period window.
    band = (rad_norm > 0) & (rad_norm <= 0.5) & ((fy_bins > 0) | ((fy_bins == 0) & (fx_bins > 0)))
    if np.count_nonzero(band) < 16:
        return False, img_f

    band_power = power[band]
    med_floor = max(float(np.median(band_power)) / np.log(2.0), 1e-9)  # Exponential mean from median
    spectral_energy = float(np.sum(band_power))
    peak_g = float(np.max(band_power)) / max(spectral_energy, np.finfo(float).tiny)
    if fisher_g_survival(peak_g, band_power.size) > SPECTRAL_FALSE_ALARM:
        return False, img_f

    # 3. Extract significant reciprocal lattice harmonic peaks with sub-pixel parabolic log-amplitude interpolation
    peak_Suppress = power.copy()
    peak_Suppress[~band] = 0.0
    freqs_uv: list[tuple[float, float]] = []
    max_peaks = 12
    sup_r = max(2, min(h, w) // 24)

    for _ in range(max_peaks):
        idx = int(np.argmax(peak_Suppress))
        py, px = divmod(idx, w)
        val = peak_Suppress[py, px]
        if fisher_g_survival(float(val) / max(spectral_energy, np.finfo(float).tiny), band_power.size) > SPECTRAL_FALSE_ALARM:
            break

        # Exact vertex of parabola g(t) through (-1, lp[0]), (0, lp[1]), (+1, lp[2]):
        # t* = (lp[0] - lp[2]) / (2 * denom) = 0.5 * (lp[2] - lp[0]) / (-denom)
        dy_sub, dx_sub = 0.0, 0.0
        if 1 <= py < h - 1 and 1 <= px < w - 1:
            lp = np.log(np.maximum(power[py - 1 : py + 2, px - 1 : px + 2], 1e-12))
            denom_y = lp[0, 1] - 2.0 * lp[1, 1] + lp[2, 1]
            denom_x = lp[1, 0] - 2.0 * lp[1, 1] + lp[1, 2]
            if denom_y < -1e-6:
                dy_sub = float(np.clip(0.5 * (lp[2, 1] - lp[0, 1]) / (-denom_y), -0.5, 0.5))
            if denom_x < -1e-6:
                dx_sub = float(np.clip(0.5 * (lp[1, 2] - lp[1, 0]) / (-denom_x), -0.5, 0.5))

        fu = ((px + dx_sub) - cx) / float(w)
        fv = ((py + dy_sub) - cy) / float(h)
        freqs_uv.append((fu, fv))

        y0, y1 = max(0, py - sup_r), min(h, py + sup_r + 1)
        x0, x1 = max(0, px - sup_r), min(w, px + sup_r + 1)
        peak_Suppress[y0:y1, x0:x1] = 0.0

    if not freqs_uv:
        return False, img_f

    # 4. Fit continuous-frequency Fourier crystal series strictly on KNOWN unmasked ring pixels!
    ring_k = max(7, min(31, (min(h, w) // 4) | 1))
    dil_ring = (
        cv2.dilate(mask_bool.astype(np.uint8), np.ones((ring_k, ring_k), np.uint8)) > 0
    ) & known
    fit_mask = dil_ring if np.count_nonzero(dil_ring) >= max(48, 4 * len(freqs_uv)) else known
    ky, kx = np.nonzero(fit_mask)
    my, mx = np.nonzero(mask_bool)

    def build_design(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = [np.ones(xs.size, dtype=np.float64)]
        for fu, fv in freqs_uv:
            phase = 2.0 * np.pi * (fu * xs + fv * ys)
            cols.append(np.cos(phase))
            cols.append(np.sin(phase))
        return np.column_stack(cols)

    Phi_known = build_design(kx.astype(np.float64), ky.astype(np.float64))
    Phi_mask = build_design(mx.astype(np.float64), my.astype(np.float64))

    # Ridge-stabilized normal equations: (Phi^T Phi + reg*I) beta = Phi^T Y
    gram = Phi_known.T @ Phi_known + 1e-3 * Phi_known.shape[0] * np.eye(Phi_known.shape[1])
    rhs = Phi_known.T @ residual_rgb[ky, kx, :]
    beta = np.linalg.solve(gram, rhs)

    # Verify explained variance R^2 on known halftone ring—only accept crystal model if physically faithful!
    pred_known = Phi_known @ beta
    ss_res = float(np.sum((residual_rgb[ky, kx, :] - pred_known) ** 2))
    ss_tot = max(float(np.sum((residual_rgb[ky, kx, :] - np.mean(residual_rgb[ky, kx, :], axis=0)) ** 2)), 1e-9)
    r2 = 1.0 - ss_res / ss_tot
    if r2 < 0.35:
        return False, img_f

    pred_mask = Phi_mask @ beta
    # Match empirical amplitude quantization (binary ink vs paper crispness of printed screentone!)
    synthesized = img_f.copy()
    raw_recon = low_pass[my, mx, :] + pred_mask

    # Check if the surrounding unmasked screentone is bimodal black/white dots (sharp lithographic halftone)
    ring_vals = img_f[ky, kx, :]
    p5 = np.percentile(ring_vals, 5, axis=0)
    p95 = np.percentile(ring_vals, 95, axis=0)
    mid_ring = np.mean((ring_vals > p5 + 0.25 * (p95 - p5)) & (ring_vals < p5 + 0.75 * (p95 - p5)))
    if mid_ring < 0.22 and np.mean(p95 - p5) > 80.0:
        # Sharpen periodic wave via steepness matched to empirical edge contrast while conserving local mean
        mid_level = low_pass[my, mx, :]
        # Threshold wave so local duty cycle equals (mid_level - p5) / (p95 - p5)
        span = np.maximum(p95 - p5, 1.0)
        # Rank-order quantile matching per pixel neighborhood preserves exact dot crispness
        gain = np.std(ring_vals, axis=0) / np.maximum(np.std(pred_mask, axis=0), 1.0)
        raw_recon = mid_level + pred_mask * np.clip(gain, 1.0, 2.2)

    synthesized[my, mx, :] = np.clip(raw_recon, np.min(ring_vals, axis=0), np.max(ring_vals, axis=0))
    return True, synthesized


def align_cross_edition_clean_plate(
    target_rgb: np.ndarray, ref_rgb: np.ndarray
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Sub-pixel Cross-Edition Registration (`CLAHE + SIFT + USAC_MAGSAC + Dense DIS Optical Flow`)
    for multi-source clean-plate extraction ("自动借底").
    Returns (success, aligned_ref_rgb, valid_coverage_mask).
    """
    th, tw = target_rgb.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_tgt = clahe.apply(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY))
    g_ref = clahe.apply(cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY))

    sift = cv2.SIFT_create(nfeatures=4000)
    kp1, des1 = sift.detectAndCompute(g_ref, None)
    kp2, des2 = sift.detectAndCompute(g_tgt, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return False, target_rgb.copy(), np.zeros((th, tw), dtype=bool)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn12 = bf.knnMatch(des1, des2, k=2)
    knn21 = bf.knnMatch(des2, des1, k=2)

    good12 = {}
    for m_pair in knn12:
        if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
            good12[m_pair[0].queryIdx] = m_pair[0].trainIdx
    mutual = []
    for m_pair in knn21:
        if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
            q2, t1 = m_pair[0].queryIdx, m_pair[0].trainIdx
            if good12.get(t1) == q2:
                mutual.append((kp1[t1].pt, kp2[q2].pt))

    if len(mutual) < 10:
        return False, target_rgb.copy(), np.zeros((th, tw), dtype=bool)

    pts_ref = np.array([p[0] for p in mutual], dtype=np.float32)
    pts_tgt = np.array([p[1] for p in mutual], dtype=np.float32)
    H_mat, inliers = cv2.findHomography(pts_ref, pts_tgt, cv2.USAC_MAGSAC, 3.0)
    if H_mat is None or inliers is None or int(np.sum(inliers)) < 10:
        return False, target_rgb.copy(), np.zeros((th, tw), dtype=bool)

    warped_ref = cv2.warpPerspective(ref_rgb, H_mat, (tw, th), flags=cv2.INTER_CUBIC)
    valid_mask = cv2.warpPerspective(
        np.full(ref_rgb.shape[:2], 255, dtype=np.uint8), H_mat, (tw, th), flags=cv2.INTER_NEAREST
    ) > 127

    # Dense DIS Optical Flow sub-pixel refinement to eliminate non-rigid scan curvature
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    g_warped = cv2.cvtColor(warped_ref, cv2.COLOR_RGB2GRAY)
    flow = dis.calc(g_tgt, g_warped, None)  # displacement from target -> warped_ref
    yy, xx = np.meshgrid(np.arange(th, dtype=np.float32), np.arange(tw, dtype=np.float32), indexing="ij")
    refined_ref = cv2.remap(
        warped_ref, xx + flow[:, :, 0], yy + flow[:, :, 1], interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # Robust affine radiometric tone alignment (median & IQR matching per channel on consistent pixels)
    diff_gray = np.abs(
        cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        - cv2.cvtColor(refined_ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    )
    sigma_n = estimate_noise_sigma(g_tgt)
    consistent = valid_mask & (diff_gray <= max(18.0, 4.0 * sigma_n))
    if np.count_nonzero(consistent) >= 128:
        aligned_f = refined_ref.astype(np.float32)
        tgt_f = target_rgb.astype(np.float32)
        for c in range(3):
            r_med = np.median(aligned_f[consistent, c])
            t_med = np.median(tgt_f[consistent, c])
            r_iqr = max(float(np.subtract(*np.percentile(aligned_f[consistent, c], [75, 25]))), 1.0)
            t_iqr = max(float(np.subtract(*np.percentile(tgt_f[consistent, c], [75, 25]))), 1.0)
            gain = np.clip(t_iqr / r_iqr, 0.7, 1.4)
            aligned_f[:, :, c] = (aligned_f[:, :, c] - r_med) * gain + t_med
        refined_ref = np.clip(np.rint(aligned_f), 0, 255).astype(np.uint8)

    return True, refined_ref, valid_mask


def _measure_ring_coherence(
    rgb: np.ndarray, active: np.ndarray, unknown: np.ndarray, known: np.ndarray
) -> tuple[float, float, float, float]:
    """Measure observed Scharr structure tensor coherence on the known ring and stroke radius."""
    radii = distance_transform_edt(active)[active]
    radius = float(np.quantile(radii, 0.75)) if radii.size else 1.0
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0) / 16.0
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1) / 16.0
    guard = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gx[guard] = 0.0
    gy[guard] = 0.0
    sig = max(1.0, radius)
    j_xx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    j_xy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    j_yy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    tr = j_xx + j_yy
    det_disc = np.hypot(j_xx - j_yy, 2.0 * j_xy)
    lam1 = 0.5 * (tr + det_disc)
    lam2 = 0.5 * (tr - det_disc)
    eps = np.finfo(float).eps
    coherence = ((lam1 - lam2) / np.maximum(lam1 + lam2, eps)) ** 2
    obs_known = known & ~guard
    ring = validation_collar(active, obs_known) if np.any(obs_known) else obs_known
    if not np.any(ring):
        return 0.0, 0.0, 1.0, radius
    noise_rgb = estimate_substrate_noise(rgb, obs_known if np.any(obs_known) else known)
    scharr_x, scharr_y = cv2.getDerivKernels(1, 0, cv2.FILTER_SCHARR, normalize=False)
    noise_energy = float(
        np.dot(np.square([0.299, 0.587, 0.114]), noise_rgb ** 2)
        * np.sum(np.square(scharr_y @ scharr_x.T))
        / 256.0
    )
    return float(np.mean(coherence[ring])), float(np.mean(lam1[ring])), noise_energy, radius


def _boundary_discontinuity_residual(
    rgb: np.ndarray, predicted: np.ndarray, unknown: np.ndarray, known: np.ndarray
) -> float:
    """Measure observable color jump between unknown inner perimeter and nearest known pixels."""
    if not np.any(unknown) or not np.any(known):
        return 0.0
    eroded = cv2.erode(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    inner_edge = unknown & ~eroded
    if not np.any(inner_edge):
        inner_edge = unknown
    _, nn_idx = distance_transform_edt(~known, return_indices=True)
    ey, ex = np.nonzero(inner_edge)
    ky = nn_idx[0, ey, ex]
    kx = nn_idx[1, ey, ex]
    diff = predicted[ey, ex].astype(np.float64) - rgb[ky, kx].astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)) / 255.0)


def _fast_physics_gated_restoration(
    rgb: np.ndarray,
    active: np.ndarray,
    known: np.ndarray,
    unknown: np.ndarray,
    allow_learned: bool,
    precomputed_lama_roi: np.ndarray | None = None,
) -> tuple[np.ndarray, RestorationPath, float, np.ndarray]:
    """Physics-gated restoration without running all 4 brute-force PDE solvers."""
    h, w = active.shape
    holdout = validation_collar(active, known)
    mean_coh, lam1_ring, noise_energy, radius = _measure_ring_coherence(rgb, active, unknown, known)

    if allow_learned:
        unk_frac = float(np.count_nonzero(unknown)) / float(max(1, unknown.size))
        narrow_slit = (radius <= 0.08 * float(min(h, w))) and (unk_frac <= 0.15)
        strong_directional = (mean_coh >= 0.78) and (lam1_ring > 4.0 * noise_energy) and narrow_slit
        if not strong_directional:
            if precomputed_lama_roi is not None and precomputed_lama_roi.shape == rgb.shape:
                candidate = precomputed_lama_roi
            else:
                from .learned import learned_clean_plate
                candidate = learned_clean_plate(rgb, unknown)
            out = rgb.copy()
            out[active] = candidate[active]
            bg = np.clip(np.rint(candidate), 0, 255).astype(np.uint8)
            res = _boundary_discontinuity_residual(rgb, bg, unknown, known)
            return out, "lama_masked", res, bg

        # Strong unidirectional coherence (speedlines/hatching) with narrow slit:
        # compare only structure_tensor_line vs lama_masked on holdout ring.
        validation_mask = unknown | holdout
        gradient_domain = holdout & (cv2.erode(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        dx_k, dy_k = cv2.getDerivKernels(1, 0, cv2.FILTER_SCHARR, normalize=False)
        gradient_gain = float(np.sum((dy_k @ dx_k.T) ** 2))
        truth = rgb.astype(float)
        truth_grads = [cv2.Scharr(truth, cv2.CV_64F, dx, dy) for dx, dy in ((1, 0), (0, 1))]

        def _val_loss(pred: np.ndarray) -> float:
            p = pred.astype(float)
            errs = np.mean((p - truth) ** 2, axis=2)
            for grad, (dx, dy) in zip(truth_grads, ((1, 0), (0, 1))):
                delta = cv2.Scharr(p, cv2.CV_64F, dx, dy) - grad
                errs[gradient_domain] += np.mean(delta[gradient_domain] ** 2, axis=1) / gradient_gain
            errs[gradient_domain] /= 3.0
            return float(np.mean(errs[holdout])) if np.any(holdout) else 0.0

        from .learned import learned_clean_plate
        cand_tensor = solve_anisotropic_tensor_inpaint(rgb, validation_mask)[0]
        cand_lama = learned_clean_plate(rgb, validation_mask)
        loss_tensor = _val_loss(cand_tensor)
        loss_lama = _val_loss(cand_lama)
        if loss_tensor <= loss_lama:
            solved = solve_anisotropic_tensor_inpaint(rgb, unknown)[0]
            bg = np.clip(np.rint(solved), 0, 255).astype(np.uint8)
            out = rgb.copy()
            out[active] = bg[active]
            return out, "structure_tensor_line", float(np.sqrt(loss_tensor) / 255.0), bg
        if precomputed_lama_roi is not None and precomputed_lama_roi.shape == rgb.shape:
            bg = precomputed_lama_roi
        else:
            bg = learned_clean_plate(rgb, unknown)
        out = rgb.copy()
        out[active] = bg[active]
        return out, "lama_masked", float(np.sqrt(loss_lama) / 255.0), bg

    # Enclosed non-flat bubble (allow_learned=False): compare solid_fill vs structure_tensor_line;
    # include narrow_slit_pde only when stroke geometry is a genuine narrow slit relative to ROI.
    validation_mask = unknown | holdout
    gradient_domain = holdout & (cv2.erode(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    dx_k, dy_k = cv2.getDerivKernels(1, 0, cv2.FILTER_SCHARR, normalize=False)
    gradient_gain = float(np.sum((dy_k @ dx_k.T) ** 2))
    truth = rgb.astype(float)
    truth_grads = [cv2.Scharr(truth, cv2.CV_64F, dx, dy) for dx, dy in ((1, 0), (0, 1))]

    def _val_loss_enc(pred: np.ndarray) -> float:
        p = pred.astype(float)
        errs = np.mean((p - truth) ** 2, axis=2)
        for grad, (dx, dy) in zip(truth_grads, ((1, 0), (0, 1))):
            delta = cv2.Scharr(p, cv2.CV_64F, dx, dy) - grad
            errs[gradient_domain] += np.mean(delta[gradient_domain] ** 2, axis=1) / gradient_gain
        errs[gradient_domain] /= 3.0
        return float(np.mean(errs[holdout])) if np.any(holdout) else 0.0

    solvers: dict[RestorationPath, object] = {
        "solid_fill": lambda image, mask: solve_laplace_dirichlet(image, mask),
        "structure_tensor_line": lambda image, mask: solve_anisotropic_tensor_inpaint(image, mask)[0],
    }
    roi_short = float(max(1, min(h, w)))
    unk_frac = float(np.count_nonzero(validation_mask)) / float(max(1, validation_mask.size))
    if radius <= 0.10 * roi_short and unk_frac <= 0.25:
        solvers["narrow_slit_pde"] = lambda image, mask: solve_biharmonic_slit(image.astype(np.float64), mask)

    best_name: RestorationPath = "solid_fill"
    best_loss = np.inf
    for name, fn in solvers.items():
        loss = _val_loss_enc(fn(rgb, validation_mask))
        if loss < best_loss:
            best_loss = loss
            best_name = name
    solved = solvers[best_name](rgb, unknown)
    bg = np.clip(np.rint(solved), 0, 255).astype(np.uint8)
    out = rgb.copy()
    out[active] = bg[active]
    return out, best_name, float(np.sqrt(best_loss) / 255.0), bg


def restore_clean_plate_roi(
    roi_rgb: np.ndarray,
    stroke_alpha: np.ndarray,
    z2_occluder_mask: np.ndarray | None = None,
    aligned_ref_roi: np.ndarray | None = None,
    substrate_mask: np.ndarray | None = None,
    context_exclusion: np.ndarray | None = None,
    precomputed_lama_roi: np.ndarray | None = None,
) -> tuple[np.ndarray, RestorationPath, float, np.ndarray]:
    """
    Physics-Gated Clean-Plate Restoration Engine with strict bit-exact invariant:
      clean_roi[mask == 0] == roi_rgb[mask == 0] (L_inf == 0 guaranteed).
    When `substrate_mask` is provided (interior of an enclosed speech bubble excluding its border wall),
    boundary statistics and Dirichlet boundary conditions are strictly confined to `substrate_mask`
    so dark/light bubble borders NEVER bleed smudges into clean bubble interiors.
    Returns:
      (clean_roi_uint8, chosen_path, boundary_residual, inferred_background)
    """
    # Active erasure mask: where stroke alpha > 0, STRICTLY EXCLUDING Z2 foreground occluders!
    # Alpha already includes detected outlines and anti-alias support. Never
    # enlarge it here: that would change pixels explicitly declared protected.
    active = stroke_alpha > 0
    if z2_occluder_mask is not None:
        active = active & (~(z2_occluder_mask > 0))
    sub_bool = (substrate_mask > 0) if (substrate_mask is not None and np.any(substrate_mask > 0)) else None
    # The recorded stroke alpha defines erasure support. An inferred substrate
    # boundary supplies context but cannot silently veto observed text pixels.

    if not np.any(active):
        return roi_rgb.copy(), "solid_fill", 0.0, roi_rgb.copy()

    unknown = active if context_exclusion is None else active | context_exclusion
    known = (~unknown) & sub_bool if sub_bool is not None else (~unknown)
    if not np.any(known):
        known = ~unknown
    if not np.any(known):
        raise ValueError("clean-plate reconstruction has no observable background pixels")

    ring = validation_collar(active, known)

    # Path 0: Cross-edition physical clean-plate transfer when an unoccluded reference edition exists
    if aligned_ref_roi is not None and aligned_ref_roi.shape == roi_rgb.shape:
        # Check if reference has clean background (much lower stroke contrast inside active mask)
        if np.any(ring):
            ring_err = float(np.median(np.abs(roi_rgb[ring].astype(np.float32) - aligned_ref_roi[ring].astype(np.float32))))
            ref_var_in = float(np.std(aligned_ref_roi[active].astype(np.float32)))
            tgt_var_in = float(np.std(roi_rgb[active].astype(np.float32)))
            if ring_err < 16.0 and ref_var_in < tgt_var_in * 0.85:
                out = roi_rgb.copy()
                # Seamless Poisson/feather blend only inside active mask, bit-exact outside
                out[active] = aligned_ref_roi[active]
                return out, "cross_edition_fusion", ring_err / 255.0, aligned_ref_roi.copy()

    ring_pixels = roi_rgb[ring].astype(np.float64) if np.any(ring) else roi_rgb[known].astype(np.float64)

    # Isolate working image outside enclosed bubble substrate so bubble outline walls never pollute PDEs!
    work_rgb = roi_rgb
    if sub_bool is not None:
        med_sub = np.median(ring_pixels, axis=0).astype(np.uint8)
        work_rgb = roi_rgb.copy()
        work_rgb[~sub_bool] = med_sub

    # Path 1: the visible substrate must fit its measured noise distribution.
    # Test the upper decile too: sparse halftone dots must not disappear behind
    # a median-only decision. Quantiles are distributional, not pixel cutoffs.
    center = np.median(ring_pixels, axis=0)
    sigma = estimate_substrate_noise(roi_rgb, known)
    null_distances = substrate_squared_distance(ring_pixels, center, sigma)
    if all(np.quantile(null_distances, q) <= chi2.ppf(q, 3) for q in (0.75, 0.95)):
        solved = solve_laplace_dirichlet(work_rgb.astype(np.float64), unknown)
        out = roi_rgb.copy()
        out[active] = np.clip(np.rint(solved[active]), 0, 255).astype(np.uint8)
        holdout = validation_collar(active, known)
        outer_known = known & ~holdout
        outer_flat = True
        if np.any(outer_known):
            outer_dev = substrate_squared_distance(work_rgb[outer_known], center, sigma)
            outer_flat = bool(np.quantile(outer_dev, 0.95) <= chi2.ppf(0.95, 3))
        if outer_flat:
            diff_h = center[None, :] - work_rgb[holdout].astype(np.float64)
            res_h = float(np.sqrt(np.mean(diff_h * diff_h)) / 255.0) if np.any(holdout) else 0.0
        else:
            predicted = solve_laplace_dirichlet(work_rgb.astype(np.float64), unknown | holdout)
            res_h = normalized_prediction_error(predicted, work_rgb, holdout)
        return out, "solid_fill", res_h, np.clip(np.rint(solved), 0, 255).astype(np.uint8)

    # Path 1b (before 2D FFT on enclosed bubbles): smooth-shaded enclosed speech bubble
    # verified via harmonic Laplace holdout so aged-paper bubbles skip 2D FFT!
    if sub_bool is not None:
        holdout = validation_collar(active, known)
        if np.any(holdout) and np.any(~(unknown | holdout)):
            predicted = solve_laplace_dirichlet(work_rgb.astype(np.float64), unknown | holdout)
            harmonic_distances = substrate_squared_distance(work_rgb[holdout], predicted[holdout], sigma)
            if all(np.quantile(harmonic_distances, q) <= chi2.ppf(q, 3) for q in (0.75, 0.95)):
                solved = solve_laplace_dirichlet(work_rgb.astype(np.float64), unknown)
                out = roi_rgb.copy()
                out[active] = np.clip(np.rint(solved[active]), 0, 255).astype(np.uint8)
                return out, "solid_fill", normalized_prediction_error(predicted, work_rgb, holdout), np.clip(np.rint(solved), 0, 255).astype(np.uint8)

    # Path 2: Fisher's exact 2D FFT periodic halftone crystal synthesis
    is_halftone, fft_recon = detect_and_synthesize_halftone_fft(work_rgb, unknown)
    if is_halftone:
        holdout = validation_collar(active, known)
        validated, predicted = detect_and_synthesize_halftone_fft(work_rgb, unknown | holdout)
        if validated:
            out = roi_rgb.copy()
            out[active] = np.clip(np.rint(fft_recon[active]), 0, 255).astype(np.uint8)
            return out, "fft_halftone_lattice", normalized_prediction_error(predicted, work_rgb, holdout), np.clip(np.rint(fft_recon), 0, 255).astype(np.uint8)

    # Path 3: Physics-gated complex background restoration
    solved_out, path, residual, background = _fast_physics_gated_restoration(
        work_rgb,
        active,
        known,
        unknown,
        allow_learned=(sub_bool is None),
        precomputed_lama_roi=precomputed_lama_roi,
    )
    out = roi_rgb.copy()
    out[active] = solved_out[active]
    return out, path, residual, background
