from __future__ import annotations

import os
from pathlib import Path
import re
import threading
import warnings
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import UnivariateSpline
from .preprocess import estimate_noise_sigma, estimate_substrate_noise, substrate_squared_distance, otsu_threshold_and_eta
from .restoration import solve_laplace_dirichlet
from scipy.stats import chi2
from .segmentation import separate_kanji_ruby_components
from .schemas import RubyPair

_ocr_lock = threading.Lock()
_rapid_inference_lock = threading.Lock()
_ocr_instances: dict[tuple[str, ...], object] = {}
_mocr_instances: dict[tuple[str, ...], object] = {}


def _rapid_detect(ocr, bgr):
    # RapidOCR stores its per-image preprocessing operator on the shared
    # detector instance. Concurrent page calls must not exchange that state.
    with _rapid_inference_lock:
        return ocr.text_det(bgr)


def _rapid_recognize(ocr, bgr):
    with _rapid_inference_lock:
        return ocr(bgr)


def fuse_text_line_geometry(primary: list[np.ndarray], independent: list[np.ndarray],
                            probability: np.ndarray | None = None) -> list[np.ndarray]:
    """Refine merged paragraphs reciprocally before joining matched extents.

    Either detector can merge neighboring tracks. A candidate containing two
    parallel centerlines outside one another's cross-axis footprint is coarse
    evidence, not a line that may absorb those finer tracks. Unmatched native
    contours survive. Observed neural ink inside the native contours is assigned
    to its closest retained track so a narrow refinement cannot clip glyphs.
    """
    if not independent:
        return primary
    primary = [cv2.convexHull(np.asarray(p,np.float32)).reshape(-1,2) for p in primary]
    independent = [cv2.convexHull(np.asarray(p,np.float32)).reshape(-1,2) for p in independent]
    def match(first,second):
        matches=[]
        for p in first:
            hits=[]
            for j,q in enumerate(second):
                overlap,_=cv2.intersectConvexConvex(p,q)
                if overlap>min(cv2.contourArea(p),cv2.contourArea(q))/2:
                    hits.append(j)
            matches.append(hits)
        return matches

    def merges_tracks(p,others,hits):
        if len(hits)<2:
            return False
        # Principal axes and interval midpoints come directly from geometry;
        # no page-size, font-size or angle tolerance decides whether to split.
        corners=cv2.boxPoints(cv2.minAreaRect(p))
        edges=np.roll(corners,-1,axis=0)-corners
        length=np.linalg.norm(edges,axis=1)
        axis=edges[np.argmax(length)]/np.max(length)
        normal=np.array([-axis[1],axis[0]])
        tracks=[]
        for index in hits:
            q=others[index]
            along,cross=q@axis,q@normal
            if np.ptp(along)<=np.ptp(cross):
                continue
            lo,hi=cross.min(),cross.max()
            tracks.append(((lo+hi)/2,(hi-lo)/2))
        return any(abs(a-b)>max(ra,rb) for i,(a,ra) in enumerate(tracks) for b,rb in tracks[i+1:])

    primary_hits=match(primary,independent)
    independent_hits=match(independent,primary)
    fine_primary=[p for p,hits in zip(primary,primary_hits) if not merges_tracks(p,independent,hits)]
    fine_independent=[q for q,hits in zip(independent,independent_hits) if not merges_tracks(q,primary,hits)]
    matches = match(fine_primary,fine_independent)
    fused = []
    for j,q in enumerate(fine_independent):
        parts = [q]+[p for p,hits in zip(fine_primary,matches) if hits == [j]]
        fused.append(cv2.boxPoints(cv2.minAreaRect(np.vstack(parts))))
    fused.extend(p for p,hits in zip(fine_primary,matches) if not hits)
    return complete_text_track_ink_coverage(fused, primary+independent, probability) if probability is not None else fused


def complete_text_track_ink_coverage(tracks: list[np.ndarray], native: list[np.ndarray],
                                     probability: np.ndarray) -> list[np.ndarray]:
    """Keep detected ink when replacing a coarse paragraph by finer tracks.

    Blank contour margins may shrink; ink within either native detector's
    support may not. A pixelwise distance transform assigns that observed ink
    to the closest retained track before extending its enclosing quadrilateral. This
    is geometry conservation, not a test that a neural detection is true text.
    """
    if not tracks:
        return native
    ownership=np.zeros(probability.shape,np.int32)
    for index,p in enumerate(tracks,1):
        cv2.fillPoly(ownership,[np.rint(p).astype(np.int32)],index)
    support=np.zeros(probability.shape,np.uint8)
    for p in native:
        cv2.fillPoly(support,[np.rint(p).astype(np.int32)],1)
    cut,_=otsu_threshold_and_eta(probability)
    missing=(support>0)&(probability>cut)&(ownership==0)
    if not missing.any():
        return tracks
    from scipy.ndimage import distance_transform_edt
    nearest=distance_transform_edt(ownership==0,return_distances=False,return_indices=True)
    yy,xx=np.nonzero(missing)
    assigned=ownership[tuple(nearest[:,yy,xx])]
    result=list(tracks)
    for index in np.unique(assigned):
        pixels=np.column_stack((xx[assigned==index],yy[assigned==index])).astype(np.float32)
        result[index-1]=cv2.boxPoints(cv2.minAreaRect(np.vstack((tracks[index-1],pixels))))
    return result


def split_disconnected_text_tracks(polygons: list[np.ndarray], probability: np.ndarray) -> list[np.ndarray]:
    """Partition a native track across observed inter-paragraph whitespace.

    Cuts require a gap larger than both the observed cross-axis extent and
    the gap-distribution Tukey fence. The entire original quadrilateral is
    partitioned, including its blank ends: no weak glyph is discarded.
    """
    cut,_=otsu_threshold_and_eta(probability)
    support=probability>cut
    result=[]
    for polygon in polygons:
        if cv2.contourArea(np.asarray(polygon,np.float32))==0:
            result.append(polygon)
            continue
        points=cv2.boxPoints(cv2.minAreaRect(np.asarray(polygon,np.float32)))
        edges=np.roll(points,-1,axis=0)-points
        lengths=np.linalg.norm(edges,axis=1)
        longest=int(np.argmax(lengths))
        along=edges[longest]/lengths[longest]
        cross=np.array([-along[1],along[0]])
        lo,hi=(points@along).min(),(points@along).max()
        left,right=(points@cross).min(),(points@cross).max()
        origin=along*lo+cross*left
        transform=np.column_stack((along,cross,origin)).astype(np.float32)
        occupancy=cv2.warpAffine(support.astype(np.uint8),cv2.invertAffineTransform(transform),
            (max(1,int(np.ceil(hi-lo))),max(1,int(np.ceil(right-left)))),
            flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).any(axis=0)
        transitions=np.diff(np.r_[False,occupancy,False].astype(np.int8))
        starts,ends=np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)
        gaps=starts[1:]-ends[:-1]
        if gaps.size:
            q1,q3=np.quantile(gaps,[.25,.75])
            fence=q3+1.5*(q3-q1) if gaps.size>1 else 0.
            separated=gaps>max(right-left,fence)
            cuts=(ends[:-1][separated]+starts[1:][separated])/2
        else:
            cuts=[]
        if not len(cuts):
            result.append(polygon)
            continue
        bounds=[0.,*cuts,float(hi-lo)]
        for start,end in zip(bounds[:-1],bounds[1:]):
            result.append(np.array([origin+along*start,origin+along*end,
                origin+along*end+cross*(right-left),origin+along*start+cross*(right-left)],np.float32))
    return result


def observed_line_is_vertical(polygons: list[np.ndarray], default_vertical: bool) -> bool:
    """Writing-axis second moment, weighted by actual line extent.

    Tiny punctuation fragments cannot outvote a long intact row/column.
    Square glyphs contribute no direction; their page prior breaks a true tie.
    """
    moment = 0.
    for polygon in polygons:
        points = np.asarray(polygon,float)
        centered = points-points.mean(axis=0)
        moment += float(np.mean(centered[:,1]**2-centered[:,0]**2))
    return moment > 0 if moment != 0 else default_vertical


def medial_stroke_radii(ink: np.ndarray) -> np.ndarray:
    """Lift medial-axis radii to stroke pixels, instead of measuring the edge.

    An EDT pixel next to the boundary of a broad patch has radius one too.
    Normal-direction ridges measure the local stroke's width; nearest-ridge
    transport assigns that width to its boundary and stops narrow connections
    from claiming the interior and edge of a wider artwork region.
    """
    from scipy.ndimage import distance_transform_edt
    padded = np.pad(ink, 1)
    distance = distance_transform_edt(padded)
    center = distance[1:-1,1:-1]
    ridge = np.zeros(ink.shape,bool)
    for dy, dx in ((0,1),(1,0),(1,1),(1,-1)):
        forward = distance[1+dy:1+dy+ink.shape[0],1+dx:1+dx+ink.shape[1]]
        backward = distance[1-dy:1-dy+ink.shape[0],1-dx:1-dx+ink.shape[1]]
        ridge |= (center>=forward)&(center>=backward)&((center>forward)|(center>backward))
    ridge &= ink
    if not ridge.any():
        return np.zeros(ink.shape,float)
    _, nearest = distance_transform_edt(~ridge,return_indices=True)
    return np.where(ink,center[nearest[0],nearest[1]],0)


def complete_text_track_endpoints(rgb: np.ndarray, tracks: list[np.ndarray],
                                  probability: np.ndarray) -> list[np.ndarray]:
    """Recover terminal stroke components before fixing recognition geometry.

    Search one measured character width beyond each end. Independently fit
    source pigments enriched over the flanking context; only components that
    cross an existing end and retain the observed stroke scale can extend it.
    This supplies evidence of geometry, not a claim that every mark is text.
    """
    from scipy.ndimage import distance_transform_edt
    from .learned import stroke_palette, stroke_pigment_distance
    from .preprocess import estimate_noise_sigma
    height, width = probability.shape
    cut, _ = otsu_threshold_and_eta(probability)
    page_lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    page_sigma = np.array(
        [max(estimate_noise_sigma(page_lab[:, :, k]), 100 / (255 * np.sqrt(12))) for k in range(3)],
        dtype=np.float64,
    )
    page_norm = page_lab.astype(np.float64) / page_sigma
    completed = []
    for track in tracks:
        corners = cv2.boxPoints(cv2.minAreaRect(np.asarray(track, np.float32)))
        edges = np.roll(corners, -1, axis=0)-corners
        lengths = np.linalg.norm(edges, axis=1)
        axis = edges[np.argmax(lengths)]/np.max(lengths)
        normal = np.array([-axis[1], axis[0]])
        along, cross = corners@axis, corners@normal
        lo, hi, left, right = along.min(), along.max(), cross.min(), cross.max()
        em = right-left
        envelope = np.array([axis*a+normal*b for a, b in
                             ((lo-em,left-em/2), (hi+em,left-em/2),
                              (hi+em,right+em/2), (lo-em,right+em/2))])
        x0, y0 = np.maximum(0, np.floor(envelope.min(axis=0))).astype(int)
        x1, y1 = np.minimum([width,height], np.ceil(envelope.max(axis=0))+1).astype(int)
        raw = rgb[y0:y1, x0:x1]
        inside = np.zeros(raw.shape[:2], np.uint8)
        cv2.fillPoly(inside, [np.rint(corners-[x0,y0]).astype(np.int32)], 1)
        seeds = (inside>0)&(probability[y0:y1,x0:x1]>cut)
        if not seeds.any() or not np.any(inside==0):
            completed.append(track)
            continue
        lab_cache = (page_lab[y0:y1, x0:x1], page_sigma, page_norm[y0:y1, x0:x1])
        _, palette_labels = stroke_palette(raw, seeds, lab_cache=lab_cache)
        yy, xx = np.indices(inside.shape)
        xy = np.stack((xx+x0, yy+y0), axis=-1)
        t, c = xy@axis, xy@normal
        # Inspect the complete component in a flanking envelope. Clipping it
        # to the line width first turns a crossing panel edge into a seemingly
        # short, narrow terminal stroke.
        window = ((c>=left-em/2)&(c<=right+em/2)
                  &(t>=lo-em)&(t<=hi+em))
        along_pixel = np.abs(axis).sum()/2
        cross_pixel = np.abs(normal).sum()/2
        interior = ((t>lo+along_pixel)&(t<hi-along_pixel)
                    &(c>left+cross_pixel)&(c<right-cross_pixel))
        extensions = []
        for label in np.unique(palette_labels[seeds]):
            source = palette_labels==label
            _, distance = stroke_pigment_distance(raw, source, inside==0, lab_cache=lab_cache)
            ink = window & (distance<=chi2.isf(.05/max(1,int(window.sum())),3))
            measured = ink & (inside>0)
            if not measured.any():
                continue
            local_radius = medial_stroke_radii(ink)
            # Source palette samples are neural observations inside the
            # detected line; common-color artwork elsewhere cannot set width.
            radii = local_radius[ink & source]
            if not radii.size:
                continue
            q1, q3 = np.quantile(radii, [.25,.75])
            upper_radius = q3+1.5*(q3-q1)
            # A wide patch has narrow medial branches at its corners too.
            # Remove the opening supported by genuinely wider disks, including
            # their boundary, rather than retaining those corners as strokes.
            wide_core = distance_transform_edt(np.pad(ink,1))[1:-1,1:-1] > upper_radius
            wide_parent = np.zeros_like(ink)
            if wide_core.any():
                _, parent_labels = cv2.connectedComponents(ink.astype(np.uint8),connectivity=8)
                wide_parent = np.isin(parent_labels,np.unique(parent_labels[wide_core]))
                ink &= distance_transform_edt(~wide_core) > upper_radius
            measured &= ink
            anchor = measured & source & interior
            anchor_distance = distance_transform_edt(~anchor) if anchor.any() else None
            count, components = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
            for component in range(1,count):
                hit = components==component
                if not np.any(hit & ((t<lo)|(t>hi))):
                    continue
                component_t, component_c = t[hit], c[hit]
                crosses_end = np.any(hit & measured & interior)
                # A clipped bracket can be wholly detached from the last dot
                # or letter. Require the whole component beyond one endpoint,
                # within a measured radius of that plane and one observed
                # character width of source pigment, inside the owned column.
                # A boundary-touching
                # artwork corner does not satisfy this detached-mark contract.
                detached_end = (anchor_distance is not None and
                    not np.any(hit & wide_parent) and
                    ((component_t.max()<lo+along_pixel and lo-component_t.max()<=upper_radius)
                     or (component_t.min()>hi-along_pixel and component_t.min()-hi<=upper_radius))
                    and component_c.min()>=left-cross_pixel
                    and component_c.max()<=right+cross_pixel
                    and np.min(anchor_distance[hit])<=em)
                if not crosses_end and not detached_end:
                    continue
                # Reject broad artwork and structures continuing beyond the
                # terminal search window, rather than truncating and claiming it.
                if (component_t.min()<=lo-em+along_pixel or component_t.max()>=hi+em-along_pixel
                    or component_c.min()<=left-em/2+cross_pixel or component_c.max()>=right+em/2-cross_pixel
                    or np.quantile(distance_transform_edt(hit)[hit],.75)>upper_radius):
                    continue
                extensions.append(xy[hit])
        if extensions:
            points = np.vstack([np.asarray(track)]+extensions)
            # Preserve the original axes and cross extent; adding an endpoint
            # must not rotate the entire line or absorb a neighboring column.
            lo, hi = min(lo,(points@axis).min()), max(hi,(points@axis).max())
            completed.append(np.array([axis*a+normal*b for a,b in
                              ((lo,left),(hi,left),(hi,right),(lo,right))],np.float32))
        else:
            completed.append(track)
    return completed


def observed_background_contrast(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Interpolate a background hypothesis from genuinely observed exterior.

    A local Gaussian has zero support deep inside a large text block. Leaving
    that interior equal to its source falsely measures zero glyph contrast.
    Harmonic extension supplies a value throughout the unknown domain.
    This is segmentation evidence only; it never writes the clean plate.
    """
    background = solve_laplace_dirichlet(rgb,~known)
    sigma = estimate_substrate_noise(rgb,known)
    return np.sqrt(substrate_squared_distance(rgb,background,sigma))
from .models import create_ort_session, inference_threads, resolve_ort_providers

_ocr_diagnostics: dict[str, object] = {
    "rapidocr_loaded": False,
    "rapidocr_error": None,
    "manga_ocr_loaded": False,
    "manga_ocr_error": None,
    "providers": [],
}


def _find_manga_ocr_dir() -> Path | None:
    env_dir = os.environ.get("PRISM_T_MANGA_OCR_DIR", "").strip()
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        Path(env_dir) if env_dir else None,
        Path.cwd() / "data" / "models" / "manga_ocr_onnx",
        repo_root / "data" / "models" / "manga_ocr_onnx",
    ]
    for cand in candidates:
        if cand is not None and (cand / "encoder_model.onnx").is_file() and (cand / "decoder_model.onnx").is_file():
            return cand
    return None


def get_ocr_runtime_status() -> dict[str, object]:
    providers = resolve_ort_providers()
    mocr_dir = _find_manga_ocr_dir()
    return {
        "requested_device": os.environ.get("PRISM_DEVICE", "auto").strip() or "auto",
        "resolved_providers": providers,
        "manga_ocr_model_present": mocr_dir is not None,
        "rapidocr_loaded": _ocr_diagnostics["rapidocr_loaded"],
        "rapidocr_error": _ocr_diagnostics["rapidocr_error"],
        "manga_ocr_loaded": _ocr_diagnostics["manga_ocr_loaded"],
        "manga_ocr_error": _ocr_diagnostics["manga_ocr_error"],
    }


def _get_rapidocr():
    providers = tuple(resolve_ort_providers())
    cached = _ocr_instances.get(providers)
    if cached is not None:
        return cached if cached is not False else None
    with _ocr_lock:
        cached = _ocr_instances.get(providers)
        if cached is not None:
            return cached if cached is not False else None
        _ocr_diagnostics["providers"] = list(providers)
        try:
            from rapidocr_onnxruntime import RapidOCR

            ocr = RapidOCR(
                det_use_cuda=False,
                cls_use_cuda=False,
                rec_use_cuda=False,
                det_use_dml=False,
                cls_use_dml=False,
                rec_use_dml=False,
                intra_op_num_threads=inference_threads(),
                inter_op_num_threads=1,
            )
            if providers != ("CPUExecutionProvider",):
                for wrapper in (
                    getattr(getattr(ocr, "text_det", None), "infer", None),
                    getattr(getattr(ocr, "text_rec", None), "session", None),
                ):
                    sess = getattr(wrapper, "session", None)
                    model_file = getattr(sess, "_model_path", None) if sess is not None else None
                    if model_file:
                        wrapper.session = create_ort_session(model_file, providers)
            _ocr_instances[providers] = ocr
            _ocr_diagnostics["rapidocr_loaded"] = True
            _ocr_diagnostics["rapidocr_error"] = None
            return ocr
        except Exception as exc:
            _ocr_instances[providers] = False
            _ocr_diagnostics["rapidocr_loaded"] = False
            _ocr_diagnostics["rapidocr_error"] = str(exc)
            warnings.warn(f"RapidOCR failed to initialize: {exc}", RuntimeWarning, stacklevel=2)
            return None


class _MangaOcrONNX:
    """
    Pure ONNX Runtime Vision-Encoder-Decoder recognizer for vertical & horizontal Japanese manga text
    (kanji, hiragana, katakana, dakuten, prolonged sound marks, and stylized manga typography).
    """

    def __init__(self, model_dir: Path, providers: tuple[str, ...] | list[str] | None = None) -> None:
        self.vocab = (model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        resolved = list(providers) if providers is not None else resolve_ort_providers()
        self.providers = resolved
        self.enc = create_ort_session(model_dir / "encoder_model.onnx", resolved)
        self.dec = create_ort_session(model_dir / "decoder_model.onnx", resolved)
        self._lock = threading.Lock()

    def recognize(self, rgb_crop: np.ndarray) -> tuple[str, float, float]:
        """Returns (text, mean_token_prob, mean_shannon_entropy)."""
        if rgb_crop.size == 0 or rgb_crop.shape[0] < 3 or rgb_crop.shape[1] < 3:
            return "", 0.0, 1.0
        pil = Image.fromarray(rgb_crop).convert("L").convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = (np.asarray(pil, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pv = np.transpose(arr, (2, 0, 1))[None, ...]

        ids = [2]  # [CLS]
        probs: list[float] = []
        with self._lock:
            hidden = self.enc.run(None, {"pixel_values": pv})[0]
            for _ in range(96):
                logits = self.dec.run(
                    None,
                    {
                        "input_ids": np.array([ids], dtype=np.int64),
                        "encoder_hidden_states": hidden,
                    },
                )[0][0, -1]
                shifted = logits - float(np.max(logits))
                exps = np.exp(shifted)
                nid = int(np.argmax(logits))
                p = float(exps[nid] / np.sum(exps))
                if nid == 3:  # [SEP]
                    break
                ids.append(nid)
                probs.append(p)
                # Prevent degenerate token loops on non-text background noise
                if len(ids) >= 8 and len(set(ids[-6:])) == 1:
                    break

        if not probs:
            return "", 0.0, 1.0

        chars: list[str] = []
        for tid in ids[1:]:
            if tid in (0, 1, 2, 3, 4) or tid >= len(self.vocab):
                continue
            tok = self.vocab[tid]
            if tok.startswith("##"):
                tok = tok[2:]
            chars.append(tok)

        raw_txt = "".join(chars).strip()
        # Normalize ASCII ellipsis/punctuation runs common in manga-ocr vocab
        raw_txt = re.sub(r"\.{2,}", "…", raw_txt)
        raw_txt = raw_txt.replace("?", "？").replace("!", "！")
        mean_p = float(np.mean(probs))
        mean_ent = float(np.mean([_shannon_binary_entropy(p) for p in probs]))
        return raw_txt, mean_p, mean_ent


def _get_manga_ocr() -> _MangaOcrONNX | None:
    providers = tuple(resolve_ort_providers())
    cached = _mocr_instances.get(providers)
    if cached is not None:
        return cached if cached is not False else None
    with _ocr_lock:
        cached = _mocr_instances.get(providers)
        if cached is not None:
            return cached if cached is not False else None
        cand = _find_manga_ocr_dir()
        if cand is not None:
            try:
                inst = _MangaOcrONNX(cand, providers=providers)
                _mocr_instances[providers] = inst
                _ocr_diagnostics["manga_ocr_loaded"] = True
                _ocr_diagnostics["manga_ocr_error"] = None
                return inst
            except Exception as exc:
                _mocr_instances[providers] = False
                _ocr_diagnostics["manga_ocr_loaded"] = False
                _ocr_diagnostics["manga_ocr_error"] = str(exc)
                warnings.warn(f"MangaOCR ONNX failed to initialize from {cand}: {exc}", RuntimeWarning, stacklevel=2)
                return None
        _mocr_instances[providers] = False
        _ocr_diagnostics["manga_ocr_loaded"] = False
        _ocr_diagnostics["manga_ocr_error"] = "manga_ocr_onnx model files not found"
        return None


def solve_tps_weights(
    src_pts: np.ndarray, dst_pts: np.ndarray, reg: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Bookstein (TPAMI 1989) Thin-Plate Spline (TPS) biharmonic solver:
        [ K + reg*I   P ] [ W ]   [ dst_pts ]
        [    P^T      0 ] [ A ] = [    0    ]
    where U(r) = r^2 * ln(r^2).
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = src.shape[0]
    if n < 3:
        raise ValueError("Thin-Plate Spline requires at least 3 non-collinear control points")

    diff = src[:, None, :] - src[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    K = np.where(r2 > 1e-18, r2 * np.log(np.maximum(r2, 1e-18)), 0.0)
    K += reg * np.eye(n, dtype=np.float64)

    P = np.column_stack([np.ones(n, dtype=np.float64), src])
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    L[n:, n:] = 1e-8 * np.eye(3, dtype=np.float64)

    rhs = np.zeros((n + 3, 2), dtype=np.float64)
    rhs[:n, :] = dst
    sol = np.linalg.solve(L, rhs)
    return sol[:n, :], sol[n:, :]


def evaluate_tps_grid(
    grid_xy: np.ndarray, ctrl_pts: np.ndarray, W: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """Evaluate exact Bookstein TPS mapping over an (H, W_grid, 2) coordinate grid."""
    H_g, W_g = grid_xy.shape[:2]
    pts = grid_xy.reshape(-1, 2).astype(np.float64)
    ctrl = np.asarray(ctrl_pts, dtype=np.float64)

    diff = pts[:, None, :] - ctrl[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    U = np.where(r2 > 1e-18, r2 * np.log(np.maximum(r2, 1e-18)), 0.0)

    mapped = A[0, :][None, :] + pts @ A[1:, :] + U @ W
    return mapped.reshape(H_g, W_g, 2).astype(np.float32)


def fit_centerline_spine(ink_mask: np.ndarray, vertical: bool) -> np.ndarray:
    """
    Fit a smooth cubic spline centerline spine (x(s), y(s)) along curved or angled text strokes.
    Returns (K, 2) float64 points in ROI-local (x, y) coordinates.
    """
    ys, xs = np.nonzero(ink_mask > 0)
    h, w = ink_mask.shape[:2]
    if ys.size < 12:
        if vertical:
            return np.array([[w * 0.5, 0.0], [w * 0.5, float(h - 1)]], dtype=np.float64)
        return np.array([[0.0, h * 0.5], [float(w - 1), h * 0.5]], dtype=np.float64)

    primary = ys.astype(np.float64) if vertical else xs.astype(np.float64)
    cross = xs.astype(np.float64) if vertical else ys.astype(np.float64)

    p_min, p_max = float(np.min(primary)), float(np.max(primary))
    span = p_max - p_min
    if span < 6.0:
        if vertical:
            return np.array([[w * 0.5, p_min], [w * 0.5, p_max]], dtype=np.float64)
        return np.array([[p_min, h * 0.5], [p_max, h * 0.5]], dtype=np.float64)

    n_bins = max(4, min(16, int(span // max(4.0, (w if vertical else h) * 0.25))))
    edges = np.linspace(p_min, p_max, n_bins + 1)
    bin_p: list[float] = []
    bin_c: list[float] = []
    bin_w: list[float] = []
    for b in range(n_bins):
        sel = (primary >= edges[b]) & (primary <= edges[b + 1] if b == n_bins - 1 else primary < edges[b + 1])
        if np.count_nonzero(sel) >= 3:
            bin_p.append(float(np.median(primary[sel])))
            bin_c.append(float(np.median(cross[sel])))
            bin_w.append(float(np.sqrt(np.count_nonzero(sel))))

    if len(bin_p) < 2:
        if vertical:
            return np.array([[w * 0.5, p_min], [w * 0.5, p_max]], dtype=np.float64)
        return np.array([[p_min, h * 0.5], [p_max, h * 0.5]], dtype=np.float64)

    bp = np.array(bin_p, dtype=np.float64)
    bc = np.array(bin_c, dtype=np.float64)
    bw = np.array(bin_w, dtype=np.float64)

    order = np.argsort(bp)
    bp, bc, bw = bp[order], bc[order], bw[order]
    uniq_mask = np.concatenate([[True], np.diff(bp) > 0.5])
    bp, bc, bw = bp[uniq_mask], bc[uniq_mask], bw[uniq_mask]

    if bp.size < 4:
        p_eval = np.linspace(p_min, p_max, 8)
        c_eval = np.interp(p_eval, bp, bc)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = UnivariateSpline(bp, bc, w=bw, k=min(3, bp.size - 1), s=float(bp.size) * 2.0)
        p_eval = np.linspace(p_min, p_max, 12)
        c_eval = spline(p_eval)

    if vertical:
        return np.column_stack([c_eval, p_eval])
    return np.column_stack([p_eval, c_eval])


def rectify_curved_strip_tps(
    roi_rgb: np.ndarray, spine_xy: np.ndarray, strip_thickness: float, vertical: bool
) -> np.ndarray:
    """
    Unwrap a curved or tilted text strip along spine_xy into a canonical axis-aligned image
    using exact Bookstein Thin-Plate Splines (TPS).
    """
    if spine_xy.shape[0] < 2:
        return roi_rgb.copy()

    diffs = np.diff(spine_xy, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(arc[-1])
    if total_len < 4.0:
        return roi_rgb.copy()

    chord = spine_xy[-1] - spine_xy[0]
    chord_len = float(np.hypot(chord[0], chord[1]))
    sin_tilt = (
        abs(float(chord[0])) / max(chord_len, 1e-6)
        if vertical
        else abs(float(chord[1])) / max(chord_len, 1e-6)
    )
    if chord_len > 1e-3:
        perp_max = float(
            np.max(
                np.abs(
                    (spine_xy[:, 0] - spine_xy[0, 0]) * (-float(chord[1]))
                    + (spine_xy[:, 1] - spine_xy[0, 1]) * float(chord[0])
                )
            )
            / chord_len
        )
    else:
        perp_max = 0.0
    if sin_tilt < 0.06 and perp_max < 0.22 * max(strip_thickness, 1.0):
        return roi_rgb.copy()

    tangents = np.gradient(spine_xy, axis=0)
    tnorm = np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-6)
    tx, ty = tangents[:, 0] / tnorm, tangents[:, 1] / tnorm
    if vertical and np.max(np.abs(tx)) < 0.03:
        return roi_rgb.copy()
    if (not vertical) and np.max(np.abs(ty)) < 0.03:
        return roi_rgb.copy()

    half_t = max(4.0, strip_thickness * 0.55)
    if vertical:
        nx, ny = ty, -tx
        out_w = max(8, int(np.ceil(2.0 * half_t)))
        out_h = max(8, int(np.ceil(total_len)))
    else:
        nx, ny = -ty, tx
        out_w = max(8, int(np.ceil(total_len)))
        out_h = max(8, int(np.ceil(2.0 * half_t)))

    k_pts = spine_xy.shape[0]
    src_ctrl: list[list[float]] = []
    dst_ctrl: list[list[float]] = []
    for i in range(k_pts):
        s_pos = float(arc[i])
        sx, sy = float(spine_xy[i, 0]), float(spine_xy[i, 1])
        for sign_off, u_cross in [(-1.0, 0.0), (1.0, float(out_w - 1 if vertical else out_h - 1))]:
            qx = sx + sign_off * half_t * float(nx[i])
            qy = sy + sign_off * half_t * float(ny[i])
            if vertical:
                src_ctrl.append([u_cross, s_pos])
            else:
                src_ctrl.append([s_pos, u_cross])
            dst_ctrl.append([qx, qy])

    W_tps, A_tps = solve_tps_weights(np.array(src_ctrl), np.array(dst_ctrl), reg=1e-3)
    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    grid_uv = np.stack([uu, vv], axis=-1)
    map_xy = evaluate_tps_grid(grid_uv, np.array(src_ctrl), W_tps, A_tps)

    rectified = cv2.remap(
        roi_rgb,
        map_xy[:, :, 0],
        map_xy[:, :, 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified


def _shannon_binary_entropy(conf: float) -> float:
    p = float(np.clip(conf, 1e-6, 1.0 - 1e-6))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _order_text_boxes_1d(
    box_items: list[tuple[float, float, float, float, str, float]], vertical: bool
) -> list[tuple[str, float]]:
    """
    Order OCR box fragments using exact 1D interval-overlap clustering (never Bankers' round()),
    eliminating two-row bucket collisions such as `"What a" / "relief."` -> `"relief What a"`.
    Each item is `(x0, y0, x1, y1, text, score)`.
    """
    cleaned = [it for it in box_items if it[4] and it[4].strip()]
    if not cleaned:
        return []

    if vertical:
        # Sort by right edge / center X descending first so rightmost columns seed clusters
        cleaned.sort(key=lambda it: -0.5 * (it[0] + it[2]))
        cols: list[dict] = []
        for it in cleaned:
            x0, y0, x1, y1, _, _ = it
            cx = 0.5 * (x0 + x1)
            iw = max(1.0, x1 - x0)
            placed = False
            for col in cols:
                c_w = max(1.0, col["x1"] - col["x0"])
                ov = min(col["x1"], x1) - max(col["x0"], x0)
                if ov >= 0.38 * min(c_w, iw) and abs(cx - col["cx"]) <= 0.55 * max(c_w, iw):
                    col["items"].append(it)
                    col["cx"] = float(np.mean([0.5 * (m[0] + m[2]) for m in col["items"]]))
                    col["x0"] = float(np.mean([m[0] for m in col["items"]]))
                    col["x1"] = float(np.mean([m[2] for m in col["items"]]))
                    placed = True
                    break
            if not placed:
                cols.append({"x0": x0, "x1": x1, "cx": cx, "items": [it]})
        cols.sort(key=lambda c: -c["cx"])
        out: list[tuple[str, float]] = []
        for col in cols:
            col["items"].sort(key=lambda m: 0.5 * (m[1] + m[3]))
            for m in col["items"]:
                out.append((m[4], m[5]))
        return out

    # Horizontal rows: cluster by vertical interval overlap, sort top-to-bottom then left-to-right
    cleaned.sort(key=lambda it: 0.5 * (it[1] + it[3]))
    rows: list[dict] = []
    for it in cleaned:
        x0, y0, x1, y1, _, _ = it
        cy = 0.5 * (y0 + y1)
        ih = max(1.0, y1 - y0)
        placed = False
        for row in rows:
            r_h = max(1.0, row["y1"] - row["y0"])
            ov = min(row["y1"], y1) - max(row["y0"], y0)
            if ov >= 0.38 * min(r_h, ih) and abs(cy - row["cy"]) <= 0.55 * max(r_h, ih):
                row["items"].append(it)
                row["cy"] = float(np.mean([0.5 * (m[1] + m[3]) for m in row["items"]]))
                row["y0"] = float(np.mean([m[1] for m in row["items"]]))
                row["y1"] = float(np.mean([m[3] for m in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"y0": y0, "y1": y1, "cy": cy, "items": [it]})
    rows.sort(key=lambda r: r["cy"])
    out = []
    for row in rows:
        row["items"].sort(key=lambda m: 0.5 * (m[0] + m[2]))
        for m in row["items"]:
            out.append((m[4], m[5]))
    return out


def _refine_small_line_ocr(
    img_rgb: np.ndarray, poly: np.ndarray, init_text: str, init_score: float, ocr
) -> tuple[str, float]:
    """2x cubic supersampled recognition for small borderless/stylized text lines."""
    if ocr is None:
        return init_text, init_score
    H, W = img_rgb.shape[:2]
    x0 = max(0, int(np.floor(float(np.min(poly[:, 0])))) - 4)
    y0 = max(0, int(np.floor(float(np.min(poly[:, 1])))) - 4)
    x1 = min(W, int(np.ceil(float(np.max(poly[:, 0])))) + 4)
    y1 = min(H, int(np.ceil(float(np.max(poly[:, 1])))) + 4)
    bw, bh = x1 - x0, y1 - y0
    if bw < 6 or bh < 6:
        return init_text, init_score
    if min(bw, bh) >= 40 and init_score >= 0.93:
        return init_text, init_score

    sub = img_rgb[y0:y1, x0:x1]
    up = cv2.resize(sub, (bw * 2, bh * 2), interpolation=cv2.INTER_CUBIC)
    med_bg = [int(c) for c in np.median(up.reshape(-1, 3), axis=0)]
    up_pad = cv2.copyMakeBorder(up, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=med_bg)
    up_bgr = cv2.cvtColor(up_pad, cv2.COLOR_RGB2BGR)

    cand_txt, cand_sc = "", -1.0
    try:
        if hasattr(ocr, "text_rec") and callable(ocr.text_rec):
            rec_res, _ = ocr.text_rec([up_bgr])
            if rec_res and len(rec_res) > 0:
                t_r = str(rec_res[0][0]).strip()
                s_r = float(rec_res[0][1])
                if t_r:
                    cand_txt, cand_sc = t_r, s_r
        else:
            res_u, _ = _rapid_recognize(ocr,up_bgr)
            if res_u:
                t_r = " ".join(str(it[1]).strip() for it in res_u if str(it[1]).strip())
                s_r = float(np.mean([float(it[2]) for it in res_u]))
                if t_r:
                    cand_txt, cand_sc = t_r, s_r
    except Exception:
        return init_text, init_score

    if cand_txt and cand_sc > init_score:
        return cand_txt, cand_sc
    return init_text, init_score


def _recognize_strip_with_mocr(
    mocr: _MangaOcrONNX, strip_rgb: np.ndarray, vertical: bool
) -> tuple[str, float, float]:
    """
    Run MangaOcrONNX on an elongated or compact text strip.
    For long thin lines (aspect ratio > 3.8), split at inter-character contrast valleys
    proportional to line thickness so 224x224 ViT patches never crush glyph resolution.
    """
    h, w = strip_rgb.shape[:2]
    long_side = float(h if vertical else w)
    short_side = float(max(12, w if vertical else h))
    med_bg = [int(c) for c in np.median(strip_rgb.reshape(-1, 3), axis=0)]

    def _run_padded(sub_rgb: np.ndarray) -> tuple[str, float, float]:
        pad = max(2, int(round(0.22 * min(sub_rgb.shape[:2]))))
        sub_bg = [int(c) for c in np.median(sub_rgb.reshape(-1, 3), axis=0)]
        padded = cv2.copyMakeBorder(sub_rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=sub_bg)
        return mocr.recognize(padded)

    if long_side <= 3.8 * short_side:
        return _run_padded(strip_rgb)

    # Compute 1D ink activity profile along the reading axis to cut only at whitespace valleys
    sub_f = strip_rgb.astype(np.float32)
    bg_f = np.array(med_bg, dtype=np.float32)
    diff_norm = np.linalg.norm(sub_f - bg_f[None, None, :], axis=2)
    axis_sum = 1 if vertical else 0
    profile = np.mean(diff_norm, axis=axis_sum)
    k_sm = max(3, int(round(short_side * 0.15)) | 1)
    profile_s = np.convolve(profile, np.ones(k_sm, dtype=np.float32) / float(k_sm), mode="same")

    L = int(long_side)
    target_len = max(2, int(round(3.4 * short_side)))
    chunks: list[tuple[int, int]] = []
    pos = 0
    while pos < L:
        if L - pos <= int(round(4.3 * short_side)):
            chunks.append((pos, L))
            break
        win_lo = min(L - 1, pos + int(round(2.4 * short_side)))
        win_hi = min(L, pos + target_len)
        if win_hi > win_lo + 1:
            cut = win_lo + int(np.argmin(profile_s[win_lo:win_hi]))
        else:
            cut = min(L, pos + target_len)
        cut = max(pos + 8, cut)
        chunks.append((pos, cut))
        pos = cut

    texts: list[str] = []
    probs: list[float] = []
    ents: list[float] = []
    for cs, ce in chunks:
        sub = strip_rgb[cs:ce, :] if vertical else strip_rgb[:, cs:ce]
        if min(sub.shape[:2]) < 4:
            continue
        t_c, p_c, e_c = _run_padded(sub)
        if t_c:
            texts.append(t_c)
            probs.append(p_c)
            ents.append(e_c)
    if not texts:
        return "", 0.0, 1.0
    sep = "" if vertical else " "
    return sep.join(texts), float(np.mean(probs)), float(np.mean(ents))


def recognize_region_crop(
    strip_rgb: np.ndarray, vertical: bool, prefer_japanese: bool = False
) -> tuple[str, float]:
    """
    Recognize an orthogonalized or multi-column text crop:
    - Uses dedicated Japanese Vision-Encoder-Decoder ONNX (`MangaOcrONNX`) when `prefer_japanese=True` or `vertical=True`
      (supporting Hiragana, Katakana, Kanji, vertical columns, and stylized manga fonts).
    - Uses `RapidOCR` CRNN for horizontal non-Japanese (Latin/Webtoon/Chinese) text, or when Latin signs appear.
    """
    mocr = _get_manga_ocr() if (prefer_japanese or vertical) else None
    ocr = _get_rapidocr()

    med_bg = [int(c) for c in np.median(strip_rgb.reshape(-1, 3), axis=0)]
    pad = max(4, min(strip_rgb.shape[:2]) // 6)
    padded = cv2.copyMakeBorder(strip_rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=med_bg)

    if mocr is not None:
        ja_txt, ja_prob, ja_ent = _recognize_strip_with_mocr(mocr, strip_rgb, vertical=vertical)
        core_chars = [c for c in ja_txt if c not in "…・。、！？!?ー〜～.,- \t\n"]
        is_degen_repeat = len(core_chars) >= 4 and len(set(core_chars)) == 1 and ja_prob < 0.94
        r_sc_max = 0.0
        res_r = None
        if ocr is not None:
            bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
            res_r, _ = _rapid_recognize(ocr,bgr)
            if (not res_r) and vertical and padded.shape[0] > padded.shape[1] * 1.2:
                res_r, _ = _rapid_recognize(ocr,cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))
            if res_r:
                r_str = " ".join(str(it[1]).strip() for it in res_r if str(it[1]).strip())
                r_sc_max = float(max(float(it[2]) for it in res_r))
                r_sc_mean = float(np.mean([float(it[2]) for it in res_r]))
                ascii_alpha = sum(1 for c in r_str if ("A" <= c <= "Z" or "a" <= c <= "z"))
                if (not vertical) and ascii_alpha >= 4 and r_sc_mean > max(0.88, ja_prob):
                    return r_str, _shannon_binary_entropy(r_sc_mean)
        # CTC evidence reduces artwork false positives; these scores do not
        # establish semantic correctness or guarantee hallucination rejection.
        has_ctc_glyphs = (ocr is None) or (r_sc_max >= 0.35)
        if len(core_chars) <= 2:
            short_ok = (ocr is None) or ((ja_prob >= 0.84 and r_sc_max >= 0.35) or r_sc_max >= 0.58)
        else:
            short_ok = ja_prob >= 0.65
        if core_chars and (not is_degen_repeat) and has_ctc_glyphs and short_ok:
            return ja_txt, ja_ent
        # If RapidOCR strongly confirms real text exists while MangaOCR tripped (e.g. heavy style/glow),
        # fall through to RapidOCR rather than dropping the entire dialogue block.
        if prefer_japanese and not (res_r and r_sc_max >= 0.72):
            return "", 1.0

    if ocr is None:
        return "", 1.0

    bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
    res, _ = _rapid_recognize(ocr,bgr)
    if not res:
        if vertical and padded.shape[0] > padded.shape[1] * 1.4:
            rot = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            res_rot, _ = _rapid_recognize(ocr,rot)
            if res_rot:
                res = res_rot
        if not res:
            return "", 0.85

    box_items = []
    for box, text, score in res:
        arr = np.array(box, dtype=np.float32)
        bx0, by0 = float(np.min(arr[:, 0])), float(np.min(arr[:, 1]))
        bx1, by1 = float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
        box_items.append((bx0, by0, bx1, by1, str(text).strip(), float(score)))

    ordered_pairs = _order_text_boxes_1d(box_items, vertical=vertical)
    texts = [p[0] for p in ordered_pairs]
    if not texts:
        return "", 0.85
    scores = [p[1] for p in ordered_pairs]
    mean_ent = float(np.mean([_shannon_binary_entropy(s) for s in scores]))
    return "".join(texts) if vertical else " ".join(texts), mean_ent


def detect_and_recognize_page_regions(
    img_rgb: np.ndarray, default_vertical: bool, text_probability: np.ndarray | None = None, recognize_text: bool = True,
    detected_lines: list[np.ndarray] | None = None, detected_blocks: list[list[float]] | None = None,
) -> list[dict]:
    """
    Detect text regions on a comic/manga/webtoon page, merge intra-bubble lines into cohesive dialogue blocks,
    peel off companion Ruby (ルビ) tracks prior to recognition, rectify curved spines via Bookstein TPS,
    and return structured recognition records.
    """
    H, W = img_rgb.shape[:2]
    ocr = _get_rapidocr()
    mocr = _get_manga_ocr() if recognize_text else None
    use_ja_detector_mode = bool((not recognize_text or (default_vertical and mocr is not None)) and ocr is not None and hasattr(ocr, "text_det"))

    raw_boxes: list[tuple[np.ndarray, str, float]] = []

    if detected_lines is not None:
        if text_probability is not None:
            detected_lines=split_disconnected_text_tracks(detected_lines,text_probability)
        if ocr is not None:
            independent,_ = _rapid_detect(ocr,cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR))
            independent=list(independent) if independent is not None else []
            if text_probability is not None:
                independent=split_disconnected_text_tracks(independent,text_probability)
            detected_lines = fuse_text_line_geometry(detected_lines,independent,text_probability)
        raw_boxes = [(p, "", 0.0) for p in detected_lines]
        if recognize_text and ocr is not None:
            # Keep independent page OCR evidence: a shrunken DB contour can
            # cut a terminal glyph that the OCR detector saw at full extent.
            recognized, _ = _rapid_recognize(ocr,cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR))
            for polygon, text, score in recognized or []:
                poly = np.array(polygon,np.float32)
                matches = []
                for k,(other,_,_) in enumerate(raw_boxes):
                    intersection,_ = cv2.intersectConvexConvex(cv2.convexHull(poly),cv2.convexHull(other))
                    area = min(cv2.contourArea(poly),cv2.contourArea(other))
                    if area > 0 and intersection > area/2:
                        matches.append((intersection/area,k))
                if matches:
                    _, k = max(matches)
                    merged = cv2.boxPoints(cv2.minAreaRect(np.vstack((poly,raw_boxes[k][0]))))
                    raw_boxes[k] = (merged,str(text).strip(),float(score))
                else:
                    raw_boxes.append((poly,str(text).strip(),float(score)))
    elif ocr is not None:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if H > int(round(2.0 * W)):
            # Aspect-preserving sliding-window tiling for tall Webtoon canvases so DBNet's
            # internal side-length normalizer never crushes horizontal glyph resolution!
            tile_h = max(W, int(round(1.5 * W)))
            stride = max(W // 2, int(round(1.05 * W)))
            edge_guard = max(4, int(round(0.04 * W)))
            y_starts = list(range(0, max(1, H - tile_h + 1), stride))
            if not y_starts or y_starts[-1] + tile_h < H:
                y_starts.append(max(0, H - tile_h))

            cand_boxes: list[tuple[np.ndarray, str, float]] = []
            for t_idx, ys in enumerate(y_starts):
                ye = min(H, ys + tile_h)
                if use_ja_detector_mode:
                    dt_boxes, _ = _rapid_detect(ocr,bgr[ys:ye, :, :])
                    if dt_boxes is not None:
                        for box in dt_boxes:
                            arr = np.array(box, dtype=np.float32)
                            arr[:, 1] += float(ys)
                            by0, by1 = float(np.min(arr[:, 1])), float(np.max(arr[:, 1]))
                            if t_idx > 0 and (by0 - ys) < edge_guard:
                                continue
                            if t_idx < len(y_starts) - 1 and (ye - by1) < edge_guard:
                                continue
                            cand_boxes.append((arr, "", 0.90))
                else:
                    res_t, _ = _rapid_recognize(ocr,bgr[ys:ye, :, :])
                    if not res_t:
                        continue
                    for box, text, score in res_t:
                        txt = str(text).strip()
                        if not txt:
                            continue
                        arr = np.array(box, dtype=np.float32)
                        arr[:, 1] += float(ys)
                        by0, by1 = float(np.min(arr[:, 1])), float(np.max(arr[:, 1]))
                        if t_idx > 0 and (by0 - ys) < edge_guard:
                            continue
                        if t_idx < len(y_starts) - 1 and (ye - by1) < edge_guard:
                            continue
                        cand_boxes.append((arr, txt, float(score)))

            cand_boxes.sort(key=lambda it: it[2], reverse=True)
            for arr, txt, sc in cand_boxes:
                ax0, ay0 = float(np.min(arr[:, 0])), float(np.min(arr[:, 1]))
                ax1, ay1 = float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
                a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
                dup = False
                for k_arr, _, _ in raw_boxes:
                    bx0, by0 = float(np.min(k_arr[:, 0])), float(np.min(k_arr[:, 1]))
                    bx1, by1 = float(np.max(k_arr[:, 0])), float(np.max(k_arr[:, 1]))
                    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
                    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
                    inter = inter_w * inter_h
                    b_area = max(1.0, (bx1 - bx0) * (by1 - by0))
                    if inter / min(a_area, b_area) > 0.60:
                        dup = True
                        break
                if not dup:
                    raw_boxes.append((arr, txt, sc))
        elif use_ja_detector_mode:
            # Pure stroke-geometry DBNet detection so pure-Hiragana/Katakana columns are NEVER dropped by Chinese CRNN!
            dt_boxes, _ = _rapid_detect(ocr,bgr)
            if dt_boxes is not None:
                for box in dt_boxes:
                    arr = np.array(box, dtype=np.float32)
                    raw_boxes.append((arr, "", 0.90))
        else:
            res, _ = _rapid_recognize(ocr,bgr)
            if res:
                for box, text, score in res:
                    t_init = str(text).strip()
                    if t_init:
                        arr = np.array(box, dtype=np.float32)
                        t_ref, sc_ref = _refine_small_line_ocr(img_rgb, arr, t_init, float(score), ocr)
                        raw_boxes.append((arr, t_ref, sc_ref))

    if text_probability is not None:
        from .learned import missing_text_polygons
        accounted = [b[0] for b in raw_boxes]
        missing = []
        page_cut, _ = otsu_threshold_and_eta(text_probability)
        known_lines = np.zeros((H,W),np.uint8)
        for poly in accounted:
            cv2.fillPoly(known_lines,[np.rint(poly).astype(np.int32)],1)
        known_ink = text_probability[(known_lines > 0) & (text_probability > page_cut)]
        evidence_floor = float(np.quantile(known_ink,.25)) if known_ink.size else None
        if detected_blocks:
            outside = text_probability.copy()
            for bx, by, bw, bh in detected_blocks:
                x0, y0 = max(0, int(np.floor(bx))), max(0, int(np.floor(by)))
                x1, y1 = min(W, int(np.ceil(bx+bw))), min(H, int(np.ceil(by+bh)))
                if x1 <= x0 or y1 <= y0:
                    continue
                offset = np.array([x0, y0], np.float32)
                for p in missing_text_polygons(text_probability[y0:y1, x0:x1], [p-offset for p in accounted], threshold=page_cut, join_components=False, evidence_floor=evidence_floor):
                    missing.append(p+offset)
                outside[y0:y1, x0:x1] = 0
            missing.extend(missing_text_polygons(outside, accounted+missing, threshold=page_cut, join_components=False, evidence_floor=evidence_floor))
        else:
            missing = missing_text_polygons(text_probability, accounted,threshold=page_cut,evidence_floor=evidence_floor)
        gray_lsd = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
        line_segments, line_widths, _, _ = cv2.createLineSegmentDetector().detect(gray_lsd)
        if line_segments is not None and len(line_segments) > 0:
            lsd_segs = line_segments.reshape(-1, 4).astype(np.float64)
            lsd_widths = line_widths.reshape(-1).astype(np.float64)
            lsd_start = lsd_segs[:, :2]
            lsd_end = lsd_segs[:, 2:]
            lsd_vec = lsd_end - lsd_start
            lsd_sq = np.sum(lsd_vec * lsd_vec, axis=1)
            lsd_thr = lsd_widths * 0.5 + 1.0 / np.sqrt(12.0)
            lsd_min_x = np.minimum(lsd_start[:, 0], lsd_end[:, 0]) - lsd_thr
            lsd_max_x = np.maximum(lsd_start[:, 0], lsd_end[:, 0]) + lsd_thr
            lsd_min_y = np.minimum(lsd_start[:, 1], lsd_end[:, 1]) - lsd_thr
            lsd_max_y = np.maximum(lsd_start[:, 1], lsd_end[:, 1]) + lsd_thr
        else:
            lsd_segs = None
        for polygon in missing:
            # A segmentation fragment on a panel edge is not a new letter.
            # Test the actual neural ink against measured line ribbons whose
            # endpoints extend beyond this candidate's entire diagonal.
            if lsd_segs is not None:
                px,py,pw,ph = cv2.boundingRect(polygon)
                x0,y0,x1,y1 = max(0,px),max(0,py),min(W,px+pw),min(H,py+ph)
                cand_idx = np.flatnonzero(
                    (lsd_sq > (pw * pw + ph * ph))
                    & (lsd_max_x >= x0) & (lsd_min_x <= x1)
                    & (lsd_max_y >= y0) & (lsd_min_y <= y1)
                )
                if cand_idx.size > 0:
                    yy,xx = np.nonzero(text_probability[y0:y1,x0:x1] > page_cut)
                    if yy.size > 0:
                        points = np.column_stack((xx+x0,yy+y0)).astype(np.float64)
                        on_frame = np.zeros(len(points), bool)
                        for s_i in range(0, cand_idx.size, 32):
                            batch_idx = cand_idx[s_i:s_i+32]
                            cs = lsd_start[batch_idx]
                            cv_b = lsd_vec[batch_idx]
                            csq = lsd_sq[batch_idx]
                            cthr = lsd_thr[batch_idx]
                            diff = points[:, None, :] - cs[None, :, :]
                            proj = np.clip(np.sum(diff * cv_b[None, :, :], axis=2) / csq[None, :], 0.0, 1.0)
                            dist = np.linalg.norm(diff - proj[:, :, None] * cv_b[None, :, :], axis=2)
                            on_frame |= np.any(dist <= cthr[None, :], axis=1)
                        if np.count_nonzero(on_frame) > on_frame.size / 2:
                            continue
            if not recognize_text:
                raw_boxes.append((polygon, "", 0.0))
                continue
            x, y, w, h = cv2.boundingRect(polygon)
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
            text, entropy = recognize_region_crop(img_rgb[y0:y1, x0:x1], vertical=h>w, prefer_japanese=default_vertical)
            if text:
                raw_boxes.append((polygon, text, 1.0 - entropy))

    # Source evidence lives on the pixel lattice. Collinear model contours have
    # zero sampled area and cannot define a crop or a two-dimensional text track.
    raw_boxes = [item for item in raw_boxes if cv2.contourArea(np.rint(item[0]).astype(np.int32)) > 0]
    if not raw_boxes:
        return []

    native_polygons = [item[0].copy() for item in raw_boxes]

    n = len(raw_boxes)
    rects = []
    scales = []
    for poly, _, _ in raw_boxes:
        x0, y0 = float(np.min(poly[:, 0])), float(np.min(poly[:, 1]))
        x1, y1 = float(np.max(poly[:, 0])), float(np.max(poly[:, 1]))
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        rects.append((x0, y0, x1, y1))
        scales.append(min(bw, bh))

    gray_full = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma_page = estimate_noise_sigma(gray_full)
    gx_f = cv2.Scharr(gray_full, cv2.CV_32F, 1, 0) / 16.0
    gy_f = cv2.Scharr(gray_full, cv2.CV_32F, 0, 1) / 16.0
    grad_full = np.hypot(gx_f, gy_f)
    g_otsu, _ = otsu_threshold_and_eta(grad_full)

    def _box_substrate_stats(box_rect: tuple[float, float, float, float]) -> tuple[float, float]:
        rx0_s = max(0, int(np.floor(box_rect[0])))
        ry0_s = max(0, int(np.floor(box_rect[1])))
        rx1_s = min(W, int(np.ceil(box_rect[2])) + 1)
        ry1_s = min(H, int(np.ceil(box_rect[3])) + 1)
        sub_g = gray_full[ry0_s:ry1_s, rx0_s:rx1_s]
        if sub_g.size < 4:
            return float(np.median(gray_full)), sigma_page
        t_s, _ = otsu_threshold_and_eta(sub_g)
        bg_s = sub_g[sub_g >= t_s] if float(np.mean(sub_g >= t_s)) >= 0.45 else sub_g[sub_g < t_s]
        if bg_s.size < 2:
            bg_s = sub_g.ravel()
        med_s = float(np.median(bg_s))
        mad_s = 1.4826 * float(np.median(np.abs(bg_s - med_s)))
        return med_s, mad_s

    box_sub_stats = [_box_substrate_stats(r) for r in rects]
    block_owners = []
    for poly, _, _ in raw_boxes:
        center = np.mean(poly, axis=0)
        containing = [(bw*bh, k) for k, (bx, by, bw, bh) in enumerate(detected_blocks or [])
                      if bx <= center[0] <= bx+bw and by <= center[1] <= by+bh]
        block_owners.append(min(containing)[1] if containing else None)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ax0, ay0, ax1, ay1 = rects[i]
        aw, ah = ax1 - ax0, ay1 - ay0
        for j in range(i + 1, n):
            if block_owners[i] is not None and block_owners[j] is not None and block_owners[i] != block_owners[j]:
                continue
            bx0, by0, bx1, by1 = rects[j]
            bw, bh = bx1 - bx0, by1 - by0
            if (ah > aw) != (bh > bw):
                continue
            char_s = max(scales[i], scales[j])
            gap_x = max(0.0, max(ax0, bx0) - min(ax1, bx1))
            gap_y = max(0.0, max(ay0, by0) - min(ay1, by1))
            overlap_y = max(0.0, min(ay1, by1) - max(ay0, by0)) / max(1.0, min(ah, bh))
            overlap_x = max(0.0, min(ax1, bx1) - max(ax0, bx0)) / max(1.0, min(aw, bw))

            close_columns = (gap_x <= 1.45 * char_s) and (overlap_y >= 0.25)
            close_rows = (gap_y <= 1.20 * char_s) and (overlap_x >= 0.25)
            if not (close_columns or close_rows):
                continue
            if block_owners[i] is not None and block_owners[i] == block_owners[j]:
                parent[find(j)] = find(i)
                continue

            mx0 = int(np.clip(min(ax1, bx1), 0, W - 1))
            mx1 = int(np.clip(max(ax0, bx0), 0, W - 1))
            my0 = int(np.clip(min(ay1, by1), 0, H - 1))
            my1 = int(np.clip(max(ay0, by0), 0, H - 1))
            crossed_border = False
            if mx1 > mx0 + 1 and close_columns:
                mid_strip = grad_full[int(max(ay0, by0)) : int(min(ay1, by1)) + 1, mx0:mx1]
                if mid_strip.size > 0 and float(np.max(np.mean(mid_strip, axis=0))) > g_otsu * 1.15:
                    crossed_border = True
            elif my1 > my0 + 1 and close_rows:
                mid_strip = grad_full[my0:my1, int(max(ax0, bx0)) : int(min(ax1, bx1)) + 1]
                if mid_strip.size > 0 and float(np.max(np.mean(mid_strip, axis=1))) > g_otsu * 1.15:
                    crossed_border = True

            if not crossed_border:
                med_a, mad_a = box_sub_stats[i]
                med_b, mad_b = box_sub_stats[j]
                if max(mad_a, mad_b) <= 4.5 * sigma_page:
                    if abs(med_a - med_b) > max(5.5 * sigma_page, 4.5 * max(mad_a, mad_b)):
                        crossed_border = True
                    elif my1 > my0 + 1 and close_rows:
                        ix0 = int(np.clip(max(ax0, bx0), 0, W - 1))
                        ix1 = int(np.clip(min(ax1, bx1), 0, W - 1))
                        if ix1 > ix0 + 3:
                            row_meds = np.median(gray_full[my0:my1, ix0 : ix1 + 1], axis=1)
                            sub_ref = 0.5 * (med_a + med_b)
                            if float(np.max(np.abs(row_meds - sub_ref))) > max(5.0 * sigma_page, 4.5 * max(mad_a, mad_b)):
                                crossed_border = True
                    elif mx1 > mx0 + 1 and close_columns:
                        iy0 = int(np.clip(max(ay0, by0), 0, H - 1))
                        iy1 = int(np.clip(min(ay1, by1), 0, H - 1))
                        if iy1 > iy0 + 3:
                            col_meds = np.median(gray_full[iy0 : iy1 + 1, mx0:mx1], axis=0)
                            sub_ref = 0.5 * (med_a + med_b)
                            if float(np.max(np.abs(col_meds - sub_ref))) > max(5.0 * sigma_page, 4.5 * max(mad_a, mad_b)):
                                crossed_border = True

            if not crossed_border:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    # Terminal evidence repairs a line's crop, not its paragraph identity.
    # Decide grouping from the detected tracks before extending their ends.
    if text_probability is not None:
        completed = complete_text_track_endpoints(img_rgb, native_polygons, text_probability)
        raw_boxes = [(polygon,text,score) for polygon,(_,text,score) in zip(completed,raw_boxes)]
        rects = [(float(poly[:,0].min()),float(poly[:,1].min()),
                  float(poly[:,0].max()),float(poly[:,1].max())) for poly in completed]

    records: list[dict] = []
    for member_indices in groups.values():
        polys = [raw_boxes[k][0] for k in member_indices]
        all_pts = np.vstack(polys)
        med_s = float(np.median([scales[k] for k in member_indices]))
        pad_b = max(4, int(round(med_s * 0.20)))
        x0 = max(0, int(np.floor(np.min(all_pts[:, 0]))) - pad_b)
        y0 = max(0, int(np.floor(np.min(all_pts[:, 1]))) - pad_b)
        x1 = min(W, int(np.ceil(np.max(all_pts[:, 0]))) + pad_b)
        y1 = min(H, int(np.ceil(np.max(all_pts[:, 1]))) + pad_b)
        bw, bh = x1 - x0, y1 - y0
        if bw < 6 or bh < 6:
            continue

        is_vert = observed_line_is_vertical(polys,default_vertical)

        roi = img_rgb[y0:y1, x0:x1]
        roi_f = roi.astype(np.float32)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ink_bin = np.zeros((bh, bw), dtype=np.uint8)
        poly_guard = np.zeros((bh, bw), dtype=np.uint8)
        pad_pg = max(3, int(round(med_s * 0.16)))
        pg_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad_pg + 1, 2 * pad_pg + 1))

        for p_arr in polys:
            loc_p = np.round(p_arr - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32)
            p_m = np.zeros((bh, bw), dtype=np.uint8)
            cv2.fillPoly(p_m, [loc_p], 1)
            poly_guard |= cv2.dilate(p_m, pg_kern)

            px0 = max(0, int(loc_p[:, 0].min()) - pad_pg)
            py0 = max(0, int(loc_p[:, 1].min()) - pad_pg)
            px1 = min(bw, int(loc_p[:, 0].max()) + pad_pg + 1)
            py1 = min(bh, int(loc_p[:, 1].max()) + pad_pg + 1)
            sub_g = roi_gray[py0:py1, px0:px1]
            if sub_g.size < 4:
                continue
            # Sample the exterior flanking collar from an expanded page crop so
            # axis-aligned polygons always retain genuine left/right/top/bottom
            # substrate samples instead of collapsing to four corner triangles.
            ext_pad = max(pad_pg + 4, 2 * pad_pg)
            gx0 = max(0, x0 + int(loc_p[:, 0].min()) - ext_pad)
            gy0 = max(0, y0 + int(loc_p[:, 1].min()) - ext_pad)
            gx1 = min(W, x0 + int(loc_p[:, 0].max()) + ext_pad + 1)
            gy1 = min(H, y0 + int(loc_p[:, 1].max()) + ext_pad + 1)
            sub_full_u8 = img_rgb[gy0:gy1, gx0:gx1]
            sub_full_f = sub_full_u8.astype(np.float32)
            p_ext = np.zeros((gy1 - gy0, gx1 - gx0), dtype=np.uint8)
            ext_poly = np.round(p_arr - np.array([[gx0, gy0]], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(p_ext, [ext_poly], 1)
            ext_ring_full = cv2.dilate(p_ext, pg_kern) == 0
            ox0, oy0 = (x0 + px0) - gx0, (y0 + py0) - gy0
            ox1, oy1 = ox0 + (px1 - px0), oy0 + (py1 - py0)

            t_loc, _ = otsu_threshold_and_eta(sub_g)
            lo_m = sub_g < t_loc
            hi_m = ~lo_m
            gap_loc = (
                abs(float(np.mean(sub_g[hi_m])) - float(np.mean(sub_g[lo_m])))
                if (np.any(lo_m) and np.any(hi_m))
                else 1.0
            )
            sub_f = roi_f[py0:py1, px0:px1]
            ext_mad = float("inf")
            ext_q75 = float("inf")
            em_rgb = None
            if np.count_nonzero(ext_ring_full) >= 8:
                ext_px = sub_full_f[ext_ring_full]
                em_rgb = np.median(ext_px, axis=0)
                ext_devs = np.linalg.norm(ext_px - em_rgb[None, :], axis=1)
                ext_mad = 1.4826 * float(np.median(ext_devs))
                ext_q75 = float(np.quantile(ext_devs, 0.75))

            unif_tol = max(4.5 * sigma_page, 0.20 * max(gap_loc, 1.0))
            if np.isfinite(ext_mad) and ext_mad <= unif_tol and ext_q75 <= 1.5 * unif_tol and em_rgb is not None:
                # Anchor luminance polarity directly to the observed exterior
                # substrate instead of pixel-area majority (which flips on tall
                # bright-on-dark columns).
                em_gray = float(0.299 * em_rgb[0] + 0.587 * em_rgb[1] + 0.114 * em_rgb[2])
                dark_first = em_gray >= t_loc
                sub_ink = (lo_m if dark_first else hi_m).astype(np.uint8)
                # Also recover chromatic fills/outlines whose luminance matches
                # the exterior substrate (e.g. pink text on a purple wall).
                ext_sigma = estimate_substrate_noise(sub_full_u8, ext_ring_full)
                c_sq = substrate_squared_distance(sub_f, em_rgb, ext_sigma)
                c_crit = chi2.isf(0.05 / max(1, sub_g.size), 3)
                sub_ink = sub_ink | (c_sq > c_crit).astype(np.uint8)
            elif np.count_nonzero(ext_ring_full) >= 8:
                env_diff_full = observed_background_contrast(sub_full_u8, ext_ring_full)
                env_diff = env_diff_full[oy0:oy1, ox0:ox1]
                te_loc, _ = otsu_threshold_and_eta(env_diff)
                thresh_env = max(te_loc, np.sqrt(chi2.isf(0.05 / max(1, env_diff.size), 3)))
                sub_ink = (env_diff > thresh_env).astype(np.uint8)
            else:
                if text_probability is not None:
                    p_sub = text_probability[y0 + py0 : y0 + py1, x0 + px0 : x0 + px1]
                    in_p = p_m[py0:py1, px0:px1] > 0
                    if np.any(in_p & lo_m) and np.any(in_p & hi_m):
                        dark_first = float(np.mean(p_sub[in_p & lo_m])) >= float(np.mean(p_sub[in_p & hi_m]))
                    else:
                        dark_first = float(np.mean(lo_m)) <= 0.55
                else:
                    dark_first = float(np.mean(lo_m)) <= 0.55
                sub_ink = (lo_m if dark_first else hi_m).astype(np.uint8)
            ink_bin[py0:py1, px0:px1] |= sub_ink

        # Prune continuous panel/box frame lines touching 2+ edges and spanning >= 80% of ROI,
        # without deleting full-column neural-supported outlined glyphs.
        roi_tp = text_probability[y0:y1, x0:x1] if text_probability is not None else None
        roi_tp_cut = float(otsu_threshold_and_eta(roi_tp)[0]) if (roi_tp is not None and roi_tp.size > 0) else 0.0
        num_cc_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats(ink_bin, connectivity=8)
        for cid_b in range(1, num_cc_b):
            cx_b, cy_b, cw_b, ch_b, _ = st_b[cid_b]
            touches = (
                int(cx_b == 0)
                + int(cy_b == 0)
                + int(cx_b + cw_b >= bw)
                + int(cy_b + ch_b >= bh)
            )
            if touches >= 2 and (cw_b >= 0.80 * bw or ch_b >= 0.80 * bh):
                if roi_tp is not None and roi_tp_cut > 0.0:
                    if float(np.mean(roi_tp[lbl_b == cid_b])) >= 0.25 * roi_tp_cut:
                        continue
                ink_bin[lbl_b == cid_b] = 0

        observed_strokes = ink_bin & poly_guard
        # Confine ink_bin strictly to union of detected polygon interiors (+proportional stroke tip margin)
        ink_bin = ink_bin & poly_guard

        base_mask, ruby_mask = separate_kanji_ruby_components(ink_bin, vertical=is_vert)
        if len(member_indices) == 1:
            spine_local = fit_centerline_spine(base_mask, vertical=is_vert)
            spine_global = [[float(p[0] + x0), float(p[1] + y0)] for p in spine_local]
        else:
            spine_local = np.zeros((0, 2), dtype=np.float64)
            spine_global = []

        def _recognize_ja_columns() -> tuple[str, float]:
            if len(member_indices) == 1:
                raw_text, raw_ent = recognize_region_crop(roi, vertical=is_vert, prefer_japanese=True)
                if raw_text:
                    rect_strip = rectify_curved_strip_tps(
                        roi, spine_local, float(bw if is_vert else bh), vertical=is_vert
                    )
                    tps_text, tps_ent = recognize_region_crop(rect_strip, vertical=is_vert, prefer_japanese=True)
                    if tps_text and tps_ent < raw_ent:
                        return tps_text, tps_ent
                    return raw_text, raw_ent
                return "", 1.0
            col_items = []
            for idx_c in member_indices:
                cx0, cy0, cx1, cy1 = rects[idx_c]
                col_items.append((cx0, cy0, cx1, cy1, str(idx_c), 0.90))
            ordered_cols = _order_text_boxes_1d(col_items, vertical=is_vert)
            col_texts: list[str] = []
            col_ents: list[float] = []
            for idx_str, _ in ordered_cols:
                rx0, ry0, rx1, ry1 = rects[int(idx_str)]
                pad_c = max(3, int(round(scales[int(idx_str)] * 0.14)))
                cx0 = max(0, int(np.floor(rx0)) - pad_c)
                cy0 = max(0, int(np.floor(ry0)) - pad_c)
                cx1 = min(W, int(np.ceil(rx1)) + pad_c)
                cy1 = min(H, int(np.ceil(ry1)) + pad_c)
                col_roi = img_rgb[cy0:cy1, cx0:cx1]
                if min(col_roi.shape[:2]) < 6:
                    continue
                ct, ce = recognize_region_crop(col_roi, vertical=is_vert, prefer_japanese=True)
                if ct:
                    col_texts.append(ct)
                    col_ents.append(ce)
            if not col_texts:
                return "", 1.0
            return ("" if is_vert else " ").join(col_texts), float(np.mean(col_ents))

        ruby_pairs: list[RubyPair] = []
        if not recognize_text:
            source_text, ocr_entropy = "", 0.0
        elif detected_lines is not None:
            # The checkpoint supplies geometry, not pre-recognized strings.
            # Read native line crops explicitly; do not compare an actual OCR
            # result against entropy computed from an empty placeholder string.
            line_items = [(rects[k][0],rects[k][1],rects[k][2],rects[k][3],str(k),0.) for k in member_indices]
            pieces, entropies = [], []
            for key, _ in _order_text_boxes_1d(line_items, vertical=is_vert):
                poly = raw_boxes[int(key)][0]
                points = cv2.boxPoints(cv2.minAreaRect(poly.astype(np.float32)))
                edges = np.roll(points,-1,axis=0)-points
                lengths = np.linalg.norm(edges,axis=1)
                axes = edges/np.maximum(lengths[:,None],np.finfo(float).eps)
                reference = np.array([0.,1.]) if is_vert else np.array([1.,0.])
                u = axes[np.argmax(np.abs(axes @ reference))]
                if u @ reference < 0:
                    u = -u
                v = np.array([u[1],-u[0]]) if is_vert else np.array([-u[1],u[0]])
                along,cross = points @ u,points @ v
                basis = np.column_stack((v,u)) if is_vert else np.column_stack((u,v))
                origin = u*along.min()+v*cross.min()
                width,height = (np.ptp(cross),np.ptp(along)) if is_vert else (np.ptp(along),np.ptp(cross))
                width,height = max(1,int(np.ceil(width))),max(1,int(np.ceil(height)))
                transform = np.column_stack((basis,origin)).astype(np.float32)
                crop = cv2.warpAffine(img_rgb,cv2.invertAffineTransform(transform),(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                text, entropy = recognize_region_crop(crop,vertical=is_vert,prefer_japanese=default_vertical)
                original, confidence = raw_boxes[int(key)][1:]
                if original and (not text or _shannon_binary_entropy(confidence) < entropy):
                    text, entropy = original, _shannon_binary_entropy(confidence)
                if text:
                    pieces.append(text);entropies.append(entropy)
            source_text = ("" if is_vert else " ").join(pieces)
            ocr_entropy = float(np.mean(entropies)) if entropies else 1.
        elif np.count_nonzero(ruby_mask) >= 6:
            sub_color = (
                np.median(roi[ink_bin == 0], axis=0).astype(np.uint8)
                if np.any(ink_bin == 0)
                else np.array([255, 255, 255], dtype=np.uint8)
            )
            if use_ja_detector_mode:
                # MangaOcrONNX natively reads intact columns without risking base radical clipping.
                base_text, base_ent = _recognize_ja_columns()
            else:
                clean_base_roi = roi.copy()
                dil_ruby = cv2.dilate(ruby_mask, np.ones((3, 3), np.uint8)) > 0
                clean_base_roi[dil_ruby] = sub_color
                base_text, base_ent = recognize_region_crop(
                    clean_base_roi, vertical=is_vert, prefer_japanese=default_vertical
                )

            only_ruby_roi = np.broadcast_to(sub_color, roi.shape).copy()
            dil_only_ruby = cv2.dilate(ruby_mask, np.ones((2, 2), np.uint8)) > 0
            only_ruby_roi[dil_only_ruby] = roi[dil_only_ruby]
            ruby_text, _ = recognize_region_crop(
                only_ruby_roi, vertical=is_vert, prefer_japanese=default_vertical
            )
            valid_ruby = bool(ruby_text) and (
                (not default_vertical)
                or all(("\u3040" <= ch <= "\u30ff") or ch in "ー・…" for ch in ruby_text)
            )
            if base_text and valid_ruby and ruby_text != base_text:
                ry_idx, rx_idx = np.nonzero(ruby_mask > 0)
                by_idx, bx_idx = np.nonzero(base_mask > 0)
                if by_idx.size > 0:
                    r_proj = ry_idx if is_vert else rx_idx
                    b_proj = by_idx if is_vert else bx_idx
                    b_min, b_span = float(b_proj.min()), max(1.0, float(b_proj.max() - b_proj.min()))
                    frac_s = float(np.clip((r_proj.min() - b_min) / b_span, 0.0, 1.0))
                    frac_e = float(np.clip((r_proj.max() - b_min) / b_span, 0.0, 1.0))
                    c_start = int(np.floor(frac_s * len(base_text)))
                    c_end = max(c_start + 1, int(np.ceil(frac_e * len(base_text))))
                    ruby_pairs.append(
                        RubyPair(
                            base=base_text[c_start:c_end],
                            ruby=ruby_text,
                            char_start=c_start,
                            char_end=min(len(base_text), c_end),
                        )
                    )
            source_text = base_text
            ocr_entropy = base_ent
        elif use_ja_detector_mode:
            source_text, ocr_entropy = _recognize_ja_columns()
        else:
            if len(member_indices) == 1:
                rect_strip = rectify_curved_strip_tps(
                    roi, spine_local, float(bw if is_vert else bh), vertical=is_vert
                )
                tps_text, tps_ent = recognize_region_crop(
                    rect_strip, vertical=is_vert, prefer_japanese=default_vertical
                )
                orig_text = raw_boxes[member_indices[0]][1]
                orig_ent = _shannon_binary_entropy(raw_boxes[member_indices[0]][2])
                if tps_text and (not orig_text or tps_ent < orig_ent):
                    source_text, ocr_entropy = tps_text, tps_ent
                else:
                    source_text, ocr_entropy = orig_text, orig_ent
            else:
                box_items = []
                for k in member_indices:
                    rx0, ry0, rx1, ry1 = rects[k]
                    box_items.append((rx0, ry0, rx1, ry1, raw_boxes[k][1], raw_boxes[k][2]))
                ordered_pairs = _order_text_boxes_1d(box_items, vertical=is_vert)
                if is_vert:
                    source_text = "".join(p[0] for p in ordered_pairs)
                else:
                    source_text = " ".join(p[0] for p in ordered_pairs)
                ocr_entropy = (
                    float(np.mean([_shannon_binary_entropy(p[1]) for p in ordered_pairs]))
                    if ordered_pairs
                    else 0.85
                )
                if not source_text:
                    source_text, ocr_entropy = recognize_region_crop(roi, vertical=is_vert, prefer_japanese=default_vertical)

        if recognize_text and (not source_text or not source_text.strip()):
            continue

        # Suppress single-character speedline noise ghosts spanning oversized elongated artwork patches
        core_chars_src = [c for c in source_text.strip() if c not in "…・。、！？!?ー〜～.,- \t\n"]
        if (
            recognize_text and len(member_indices) == 1
            and len(core_chars_src) <= 1
            and max(bw, bh) > 4.2 * max(med_s, 8.0)
            and min(bw, bh) > 2.2 * max(med_s, 8.0)
            and ocr_entropy > 0.35
        ):
            continue

        col_items_ord = [(rects[k][0], rects[k][1], rects[k][2], rects[k][3], str(k), 1.0) for k in member_indices]
        ord_cols_meta = _order_text_boxes_1d(col_items_ord, vertical=is_vert)
        col_boxes_ordered = [
            [
                int(np.floor(rects[int(idx_s)][0])),
                int(np.floor(rects[int(idx_s)][1])),
                int(np.ceil(rects[int(idx_s)][2] - rects[int(idx_s)][0])),
                int(np.ceil(rects[int(idx_s)][3] - rects[int(idx_s)][1])),
            ]
            for idx_s, _ in ord_cols_meta
        ]

        hull = cv2.convexHull(all_pts.astype(np.float32))
        poly_list = [[float(p[0][0]), float(p[0][1])] for p in hull]
        records.append(
            {
                "bbox": [x0, y0, bw, bh],
                "polygon": poly_list,
                "orientation": "vertical" if is_vert else "horizontal",
                "source_text": source_text,
                "ruby_pairs": ruby_pairs,
                "spine_points": spine_global,
                "ocr_entropy": ocr_entropy,
                # Independent observations flow to the sole final alpha pass.
                "ink_mask_roi": observed_strokes,
                "detector_polygons": [native_polygons[int(k)].astype(float).tolist() for k, _ in ord_cols_meta],
                "column_boxes": col_boxes_ordered,
                "line_polygons": [raw_boxes[int(k)][0].astype(float).tolist() for k, _ in ord_cols_meta],
                "col_scale_px": float(med_s),
            }
        )

    return records
