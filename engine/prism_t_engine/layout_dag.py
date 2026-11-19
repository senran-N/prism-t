from __future__ import annotations

import base64
import heapq
import cv2
import numpy as np
from .preprocess import otsu_threshold_and_eta
from .schemas import ResolvedFormatProfile


def extract_panel_boxes(
    img_rgb: np.ndarray, is_spread: bool = False, gutter_x: int | None = None
) -> list[tuple[int, int, int, int]]:
    """
    Scale-invariant panel frame extraction supporting framed manga/comics and frameless webtoons.
    Returns list of panel bounding boxes (x, y, w, h).
    """
    H, W = img_rgb.shape[:2]
    if is_spread and gutter_x is not None and 0 < gutter_x < W:
        right_panels = extract_panel_boxes(img_rgb[:, gutter_x:], is_spread=False)
        left_panels = extract_panel_boxes(img_rgb[:, :gutter_x], is_spread=False)
        shifted_right = [(x + gutter_x, y, w, h) for (x, y, w, h) in right_panels]
        return shifted_right + left_panels

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    thresh, eta = otsu_threshold_and_eta(gray)
    if eta < 0.08:
        return [(0, 0, W, H)]

    # Gutter substrate is typically the dominant border intensity (white or black gutter)
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    gutter_is_bright = float(np.median(border_pixels)) >= thresh
    non_gutter = (gray < thresh) if gutter_is_bright else (gray > thresh)

    # Close thin internal gaps smaller than 0.5% of short edge to keep panel frames unified
    k_sz = max(3, int(round(min(H, W) * 0.006)) | 1)
    closed = cv2.morphologyEx(
        non_gutter.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k_sz, k_sz), np.uint8)
    )
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(H * W)
    panels: list[tuple[int, int, int, int]] = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if float(w * h) >= 0.025 * page_area:
            panels.append((int(x), int(y), int(w), int(h)))

    if not panels:
        return [(0, 0, W, H)]

    # Suppress nested boxes that are >85% contained inside a larger panel
    panels.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for bx, by, bw, bh in panels:
        inside = False
        for kx, ky, kw, kh in kept:
            ix0, iy0 = max(bx, kx), max(by, ky)
            ix1, iy1 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
            if ix1 > ix0 and iy1 > iy0:
                inter = float((ix1 - ix0) * (iy1 - iy0))
                if inter / float(bw * bh) >= 0.85:
                    inside = True
                    break
        if not inside:
            kept.append((bx, by, bw, bh))

    return kept if kept else [(0, 0, W, H)]


def _topological_order_boxes(
    boxes: list[tuple[float, float, float, float]], rtl: bool
) -> list[int]:
    """
    Order axis-aligned boxes (x0, y0, x1, y1) via a strict 2D spatial precedence DAG:
      - Vertical rule: i precedes j if i lies strictly above j (up to minor overlap fraction)
      - Horizontal rule: when i and j overlap vertically, i precedes j right-to-left (if rtl) or left-to-right
    Cycles from diagonal/burst layouts are broken by dropping minimum-margin precedence edges,
    followed by deterministic Kahn topological sort.
    """
    n = len(boxes)
    if n <= 1:
        return list(range(n))

    edges: list[tuple[float, int, int]] = []  # (strength, u, v) meaning u -> v
    for i in range(n):
        ax0, ay0, ax1, ay1 = boxes[i]
        aw, ah = max(1.0, ax1 - ax0), max(1.0, ay1 - ay0)
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = boxes[j]
            bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)

            ov_y = min(ay1, by1) - max(ay0, by0)
            min_h = min(ah, bh)

            if ov_y <= 0.25 * min_h:
                # Clearly separated vertically: upper precedes lower
                if ay0 + 0.5 * ah <= by0 + 0.5 * bh:
                    strength = (by0 - ay1) / min_h + 2.0
                    edges.append((strength, i, j))
                else:
                    strength = (ay0 - by1) / min_h + 2.0
                    edges.append((strength, j, i))
            else:
                # Overlapping vertical band: apply reading-direction horizontal rule
                acx = 0.5 * (ax0 + ax1)
                bcx = 0.5 * (bx0 + bx1)
                min_w = min(aw, bw)
                if rtl:
                    if acx >= bcx:
                        edges.append(((acx - bcx) / min_w + 0.5, i, j))
                    else:
                        edges.append(((bcx - acx) / min_w + 0.5, j, i))
                else:
                    if acx <= bcx:
                        edges.append(((bcx - acx) / min_w + 0.5, i, j))
                    else:
                        edges.append(((acx - bcx) / min_w + 0.5, j, i))

    # Build acyclic DAG greedily from strongest geometric precedence to weakest
    edges.sort(key=lambda e: e[0], reverse=True)
    adj: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n

    def reachable(src: int, dst: int) -> bool:
        stack = [src]
        seen = {src}
        while stack:
            curr = stack.pop()
            if curr == dst:
                return True
            for nxt in adj[curr]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    for _, u, v in edges:
        if not reachable(v, u):
            adj[u].append(v)
            indeg[v] += 1

    # Tie-breaker key matching canonical reading sweep
    def tie_key(idx: int) -> tuple[float, float]:
        x0, y0, x1, y1 = boxes[idx]
        cy = 0.5 * (y0 + y1)
        cx = -0.5 * (x0 + x1) if rtl else 0.5 * (x0 + x1)
        return (cy, cx)

    heap: list[tuple[tuple[float, float], int]] = [
        (tie_key(i), i) for i in range(n) if indeg[i] == 0
    ]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (tie_key(v), v))

    if len(order) < n:
        rem = [i for i in range(n) if i not in set(order)]
        rem.sort(key=tie_key)
        order.extend(rem)
    return order


def ground_speaker_via_tail_ray(
    img_rgb: np.ndarray,
    tail_tip: list[float] | None,
    tail_dir: list[float] | None,
    panel_box: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, str]:
    """
    Raycast along the bubble tail vector (tail_tip + t * tail_dir) within panel_box to locate
    the speaking character region, extracting a 64-dim L2-normalized CIE-Lab + gradient-orientation
    appearance descriptor for cross-node speaker clustering.
    """
    if tail_tip is None or tail_dir is None:
        return None, ""
    H, W = img_rgb.shape[:2]
    px, py, pw, ph = panel_box
    panel_diag = float(np.hypot(pw, ph))
    max_reach = max(16.0, 0.45 * panel_diag)

    tx, ty = tail_tip
    vx, vy = tail_dir
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 16.0
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 16.0
    grad = np.hypot(gx, gy)
    g_thresh, _ = otsu_threshold_and_eta(grad)

    # March outward starting just past the tail tip border until striking rich structural edges (character art)
    hit_x, hit_y = -1, -1
    patch_r = max(12, int(round(min(H, W) * 0.035)))
    for step in np.linspace(4.0, max_reach, 32):
        cx = int(round(tx + vx * step))
        cy = int(round(ty + vy * step))
        if not (px <= cx < px + pw and py <= cy < py + ph):
            break
        if not (0 <= cx < W and 0 <= cy < H):
            break
        x0, y0 = max(0, cx - patch_r // 2), max(0, cy - patch_r // 2)
        x1, y1 = min(W, cx + patch_r // 2 + 1), min(H, cy + patch_r // 2 + 1)
        if float(np.mean(grad[y0:y1, x0:x1] >= g_thresh * 0.75)) >= 0.12:
            hit_x, hit_y = cx, cy
            break

    if hit_x < 0:
        return None, ""

    x0, y0 = max(0, hit_x - patch_r), max(0, hit_y - patch_r)
    x1, y1 = min(W, hit_x + patch_r), min(H, hit_y + patch_r)
    patch = img_rgb[y0:y1, x0:x1]
    if patch.size < 48:
        return None, ""

    # 64D compact appearance descriptor: 3x16 CIE Lab histograms + 16-bin weighted gradient orientation histogram
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    h_l, _ = np.histogram(lab[:, :, 0], bins=16, range=(0.0, 256.0), density=False)
    h_a, _ = np.histogram(lab[:, :, 1], bins=16, range=(0.0, 256.0), density=False)
    h_b, _ = np.histogram(lab[:, :, 2], bins=16, range=(0.0, 256.0), density=False)

    p_gx = gx[y0:y1, x0:x1]
    p_gy = gy[y0:y1, x0:x1]
    p_mag = grad[y0:y1, x0:x1]
    angles = (np.arctan2(p_gy, p_gx) + np.pi) % (2.0 * np.pi)
    h_ori, _ = np.histogram(angles, bins=16, range=(0.0, 2.0 * np.pi), weights=p_mag)

    vec = np.concatenate([h_l, h_a, h_b, h_ori]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None, ""
    vec /= norm
    b64 = base64.b64encode(vec.tobytes()).decode("ascii")
    return vec, b64


def build_narrative_dag_and_ground_speakers(
    img_rgb: np.ndarray,
    nodes_meta: list[dict],
    profile: ResolvedFormatProfile,
    is_spread: bool = False,
    gutter_x: int | None = None,
) -> list[dict]:
    """
    Assign panel hierarchy, topological reading_order, primary vs aside track, and clustered speaker IDs.
    """
    if not nodes_meta:
        return nodes_meta

    rtl = profile == "manga_rtl"
    panels = extract_panel_boxes(img_rgb, is_spread=is_spread, gutter_x=gutter_x)
    panel_rects = [(float(x), float(y), float(x + w), float(y + h)) for (x, y, w, h) in panels]
    ordered_panel_ids = _topological_order_boxes(panel_rects, rtl=rtl)
    panel_rank = {pid: rank for rank, pid in enumerate(ordered_panel_ids)}

    # Classify primary vs aside track: borderless floating text significantly smaller than median dialogue stroke/area
    enclosed_areas = [
        float(m["bbox"][2] * m["bbox"][3])
        for m in nodes_meta
        if m["bubble_kind"] not in ("floating_sfx", "diegetic_surface")
    ]
    med_area = float(np.median(enclosed_areas)) if enclosed_areas else 0.0

    panel_buckets: dict[int, list[int]] = {pid: [] for pid in range(len(panels))}
    for idx, m in enumerate(nodes_meta):
        bx, by, bw, bh = m["bbox"]
        cx, cy = bx + 0.5 * bw, by + 0.5 * bh
        best_p = 0
        best_dist = 1e18
        for pid, (px, py, pw, ph) in enumerate(panels):
            if px <= cx <= px + pw and py <= cy <= py + ph:
                best_p = pid
                best_dist = 0.0
                break
            dx = max(px - cx, 0.0, cx - (px + pw))
            dy = max(py - cy, 0.0, cy - (py + ph))
            d2 = dx * dx + dy * dy
            if d2 < best_dist:
                best_dist = d2
                best_p = pid
        panel_buckets[best_p].append(idx)
        m["_panel_id"] = best_p

        if m["bubble_kind"] == "floating_sfx" and med_area > 0 and float(bw * bh) < 0.30 * med_area:
            m["track"] = "aside"
        else:
            m["track"] = "primary"

    # Order nodes panel-by-panel, primary track before aside track within the same panel
    global_order: list[int] = []
    for pid in sorted(panel_buckets.keys(), key=lambda p: panel_rank[p]):
        members = panel_buckets[pid]
        if not members:
            continue
        prim = [i for i in members if nodes_meta[i]["track"] == "primary"]
        aside = [i for i in members if nodes_meta[i]["track"] == "aside"]
        for group in (prim, aside):
            if len(group) <= 1:
                global_order.extend(group)
            else:
                g_boxes = [
                    (
                        float(nodes_meta[i]["bbox"][0]),
                        float(nodes_meta[i]["bbox"][1]),
                        float(nodes_meta[i]["bbox"][0] + nodes_meta[i]["bbox"][2]),
                        float(nodes_meta[i]["bbox"][1] + nodes_meta[i]["bbox"][3]),
                    )
                    for i in group
                ]
                local_ord = _topological_order_boxes(g_boxes, rtl=rtl)
                global_order.extend(group[k] for k in local_ord)

    for rank, node_idx in enumerate(global_order):
        nodes_meta[node_idx]["reading_order"] = rank

    # Ground speakers via tail raycasting and cluster appearance vectors on the unit hypersphere
    centroids: list[np.ndarray] = []
    for m in nodes_meta:
        pbox = panels[m["_panel_id"]]
        vec, b64 = ground_speaker_via_tail_ray(img_rgb, m.get("tail_tip"), m.get("tail_dir"), pbox)
        m["speaker_appearance_b64"] = b64
        if vec is not None:
            assigned = -1
            best_sim = -1.0
            for cid, cvec in enumerate(centroids):
                sim = float(np.dot(vec, cvec))
                if sim > best_sim:
                    best_sim = sim
                    assigned = cid
            # Cosine threshold cos(pi/7) ~ 0.901 for tight local color/texture speaker consistency
            if assigned >= 0 and best_sim >= np.cos(np.pi / 7.0):
                new_c = centroids[assigned] + vec
                centroids[assigned] = new_c / max(float(np.linalg.norm(new_c)), 1e-8)
                m["speaker_cluster_id"] = f"spk_{assigned + 1}"
            else:
                centroids.append(vec.copy())
                m["speaker_cluster_id"] = f"spk_{len(centroids)}"
        else:
            m["speaker_cluster_id"] = ""
        m.pop("_panel_id", None)

    nodes_meta.sort(key=lambda item: item["reading_order"])
    return nodes_meta
