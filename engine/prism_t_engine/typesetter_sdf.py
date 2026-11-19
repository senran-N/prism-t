from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq
from .schemas import GlyphBox, RubyPair, StyleSpec, TypesetLayout

# JIS X 4051 / GB/T 15834 Kinsoku Shori character sets
_LINE_START_FORBIDDEN = set("，。、！？：；）》」』】〉〕—…～·,.!?:;)]}ゝゞーァィゥェォッャュョヮヵヶ")
_LINE_END_FORBIDDEN = set("（《「『【〈〔([{")

# Unicode Vertical Presentation Forms for authentic vertical manga typesetting
_VERTICAL_PUNCT_MAP = str.maketrans(
    {
        "（": "︵",
        "）": "︶",
        "「": "﹁",
        "」": "﹂",
        "『": "﹃",
        "』": "﹄",
        "【": "︻",
        "】": "︼",
        "《": "︽",
        "》": "︾",
        "〈": "︿",
        "〉": "﹀",
        "〔": "︹",
        "〕": "︺",
        "—": "︱",
        "ー": "︱",
        "～": "︱",
        "〜": "︱",
        "~": "︱",
        "…": "︙",
    }
)

_VERTICAL_GLYPH_SUB = {
    "，": "︐",
    "、": "︑",
    "。": "︒",
    "：": "︓",
    "；": "︔",
    "！": "︕",
    "？": "︖",
}
_VERTICAL_CORNER_PUNCT = set("，、。︐︑︒")
_VERTICAL_SYMMETRIC_BRACKETS = set("︵︶︽︾︻︼︿﹀︹︺")
_VERTICAL_CENTERED_PUNCT = _VERTICAL_SYMMETRIC_BRACKETS | {"︙", "︱"}
_VERTICAL_CORNER_BRACKETS = set("﹁﹂﹃﹄")


@lru_cache(maxsize=32768)
def _font_bbox(font: ImageFont.FreeTypeFont, text: str, anchor: str | None = None) -> tuple[int, int, int, int]:
    bb = font.getbbox(text, anchor=anchor) if anchor is not None else font.getbbox(text)
    return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])


@lru_cache(maxsize=32768)
def _font_tight_bbox(font: ImageFont.FreeTypeFont, ch: str) -> tuple[int, int, int, int]:
    """Return tight pixel-ink bounding box in font draw coordinates (trims empty glyph box padding)."""
    bb = _font_bbox(font, ch)
    mbb = font.getmask(ch).getbbox()
    if mbb is None:
        return bb
    return (bb[0] + int(mbb[0]), bb[1] + int(mbb[1]), bb[0] + int(mbb[2]), bb[1] + int(mbb[3]))


@lru_cache(maxsize=512)
def _font_metrics(font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    a, d = font.getmetrics()
    return int(a), int(d)


def _vertical_char_metrics(
    ch: str,
    next_ch: str | None,
    font: ImageFont.FreeTypeFont,
    font_size: float,
    char_spacing: float,
    has_next: bool,
) -> tuple[str, float, float, bool]:
    """Return (draw_char, advance_h, after_gap, is_upper_right_punct) for one vertical char."""
    draw_ch = _VERTICAL_GLYPH_SUB.get(ch, ch)
    if draw_ch != ch:
        path = getattr(font, "path", None)
        if not path or ord(draw_ch) not in _font_coverage(str(path)):
            draw_ch = ch

    if ch in _VERTICAL_CORNER_PUNCT:
        ch_h = font_size * 0.62
        is_corner = True
    elif ch == "︙":
        ch_h = font_size * (0.72 if next_ch == ch else 0.88)
        is_corner = False
    elif ch == "︱":
        ch_h = font_size * (0.72 if next_ch == ch else 0.86)
        is_corner = False
    elif ch in _VERTICAL_SYMMETRIC_BRACKETS or ch in _VERTICAL_CORNER_BRACKETS:
        tbb = _font_tight_bbox(font, draw_ch)
        ch_h = max(font_size * 0.56, float(tbb[3] - tbb[1]) + font_size * 0.16)
        is_corner = False
    elif _is_cjk_char(ch):
        ch_h = font_size
        is_corner = False
    else:
        bbox = _font_bbox(font, draw_ch)
        ch_h = max(font_size * 0.55, float(bbox[3] - bbox[1]))
        is_corner = False

    if not has_next:
        gap = 0.0
    elif (ch == "︙" and next_ch == "︙") or (ch == "︱" and next_ch == "︱"):
        gap = 0.0
    else:
        gap = font_size * char_spacing
    return draw_ch, ch_h, gap, is_corner


_FONT_CANDIDATES: dict[str, list[str]] = {
    "gothic": [
        "NotoSansSC-VF.ttf",
        "SourceHanSansSC-Regular.otf",
        "msyh.ttc",
        "Dengb.ttf",
        "YuGothM.ttc",
        "NotoSansCJK-Regular.ttc",
        "simhei.ttf",
        "Arial Unicode.ttf",
        "arial.ttf",
    ],
    "gothic_bold": [
        "msyhbd.ttc",
        "NotoSansCJK-Bold.ttc",
        "NotoSansSC-Bold.otf",
        "simhei.ttf",
        "Dengb.ttf",
        "YuGothB.ttc",
        "arialbd.ttf",
    ],
    "mincho": [
        "NotoSerifSC-VF.ttf",
        "STSONG.TTF",
        "simsun.ttc",
        "msmincho.ttc",
        "NotoSerifCJK-Regular.ttc",
        "times.ttf",
    ],
    "rounded": [
        "SIMYOU.TTF",
        "Deng.ttf",
        "msyh.ttc",
        "YuGothM.ttc",
        "NotoSansCJK-Regular.ttc",
        "arial.ttf",
    ],
    "brush": [
        "simkai.ttf",
        "STKAITI.TTF",
        "msyhbd.ttc",
        "HGRGY.TTC",
    ],
    "comic_sans": [
        "comicbd.ttf",
        "comic.ttf",
        "msyhbd.ttc",
        "arialbd.ttf",
    ],
}

_ASCII_ONLY_FONTS = {"comicbd.ttf", "comic.ttf", "arialbd.ttf", "arial.ttf", "times.ttf"}
_font_path_cache: dict[tuple, str | None] = {}
_font_dir_index: dict[Path, dict[str, str]] = {}


def _get_dir_font_index(d: Path) -> dict[str, str]:
    idx = _font_dir_index.get(d)
    if idx is None:
        idx = {}
        try:
            for p in d.rglob("*"):
                if p.is_file():
                    lname = p.name.lower()
                    if lname not in idx:
                        idx[lname] = str(p)
        except OSError:
            pass
        _font_dir_index[d] = idx
    return idx


@lru_cache(maxsize=128)
def _font_coverage(path: str) -> frozenset[int]:
    with TTFont(path, fontNumber=0, lazy=True) as font:
        return frozenset(font.getBestCmap() or ())


@lru_cache(maxsize=128)
def _font_variations(path: str, bold: bool) -> tuple[float, ...]:
    with TTFont(path,fontNumber=0,lazy=True) as font:
        if "fvar" not in font:
            return ()
        # OpenType/CSS nominal weights: a variable font's own default may be
        # Thin (100), rather than Regular (400). Select the requested weight.
        return tuple(float(np.clip(700 if bold else 400,axis.minValue,axis.maxValue))
                     if axis.axisTag == "wght" else axis.defaultValue for axis in font["fvar"].axes)


def resolve_font_path(category: str, needs_cjk: bool = False, bold: bool = False, required_text: str = "") -> str | None:
    required = frozenset(ord(ch) for ch in required_text if not ch.isspace())
    cache_key = (category, needs_cjk, bold, required)
    if cache_key in _font_path_cache:
        return _font_path_cache[cache_key]

    eff_cat = "gothic_bold" if (category == "gothic" and bold) else category
    search_dirs = [
        Path.cwd() / "fonts",
        Path(__file__).resolve().parent.parent / "fonts",
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / ".local/share/fonts",
    ]
    raw_names = _FONT_CANDIDATES.get(eff_cat, _FONT_CANDIDATES["gothic"]) + _FONT_CANDIDATES["gothic"]
    if needs_cjk:
        names = [n for n in raw_names if n.lower() not in _ASCII_ONLY_FONTS]
    else:
        names = raw_names
    for d in search_dirs:
        if not d.exists():
            continue
        for name in names:
            direct = d / name
            if direct.is_file() and required <= _font_coverage(str(direct)):
                _font_path_cache[cache_key] = str(direct)
                return str(direct)
        dir_idx = _get_dir_font_index(d)
        for name in names:
            hit = dir_idx.get(name.lower())
            if hit is not None and required <= _font_coverage(hit):
                _font_path_cache[cache_key] = hit
                return hit

    # Symbol-only effects can require a math or symbol face outside the chosen
    # family. Select an installed face by actual cmap coverage, preserving the
    # requested Unicode characters rather than substituting missing-glyph boxes.
    for d in search_dirs:
        for path in _get_dir_font_index(d).values():
            if Path(path).suffix.lower() in (".ttf", ".ttc", ".otf") and required <= _font_coverage(path):
                _font_path_cache[cache_key] = path
                return path
    _font_path_cache[cache_key] = None
    return None


def load_font(
    category: str, size_px: float, needs_cjk: bool = False, bold: bool = False, required_text: str = ""
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    sz = max(1, int(round(size_px)))
    fpath = resolve_font_path(category, needs_cjk=needs_cjk, bold=bold, required_text=required_text)
    if fpath is None:
        raise ValueError(f"No installed font covers requested text in category {category}")
    return _load_font_face(fpath,sz,bold)


@lru_cache(maxsize=512)
def _load_font_face(fpath: str, size: int, bold: bool) -> ImageFont.FreeTypeFont:
    # A face depends on path, raster size and axes, not the set/order of
    # characters that led cmap selection to that face.
    font = ImageFont.truetype(fpath, size)
    variations = _font_variations(fpath,bold)
    if variations:
        font.set_variation_by_axes(list(variations))
    return font


@lru_cache(maxsize=32768)
def _text_advance(font: ImageFont.FreeTypeFont, text: str) -> float:
    return float(font.getlength(text))


def build_interior_sdf(
    roi_shape: tuple[int, int],
    roi_offset: tuple[int, int],
    bubble_polygon: list[list[float]],
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Compute exact Euclidean Signed Distance Field inside the bubble/region polygon within ROI,
    plus its Pole of Inaccessibility (argmax SDF) as the natural optical center!
    """
    h, w = roi_shape
    ox, oy = roi_offset
    mask = np.zeros((h, w), dtype=np.uint8)
    if bubble_polygon and len(bubble_polygon) >= 3:
        pts = np.array([[p[0] - ox, p[1] - oy] for p in bubble_polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        if np.count_nonzero(mask) < 16:
            mask[:, :] = 1
    else:
        mask[:, :] = 1

    # Pad 0 border so image edges act as region boundaries
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    sdf = distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)
    max_idx = int(np.argmax(sdf))
    py, px = divmod(max_idx, w)
    return sdf, (float(px), float(py))


def estimate_char_budget_from_sdf(
    sdf: np.ndarray, source_text: str, source_font_size_px: float = 0.0,
    style: StyleSpec | None = None,
) -> tuple[int, int]:
    """Count font cells that actually fit at the measured source scale.

    The CJK em cell is a geometric unit, not a vocabulary entry. Integrating
    whole row/column bands respects curved bubbles instead of converting total
    area with an assumed packing ratio. No source-length language ratio or
    absolute minimum font size constrains the translation.
    """
    style = style or StyleSpec(source_font_size_px=source_font_size_px)
    scale = float(source_font_size_px or style.source_font_size_px)
    if scale <= 0:
        return 1, 0  # No measured type scale: no invented capacity limit.
    font = load_font(style.font_category, scale, required_text="田", bold=style.weight_scale >= 1.10)
    ascent, descent = _font_metrics(font)
    step = float(ascent + descent)
    advance = _text_advance(font, "田")
    vertical = style.orientation == "vertical"
    support = sdf > float(style.outline_width_px)
    # Each along-axis position must stay within the bubble across the whole
    # font-height band. Evaluate all distinct raster phases, then use the best.
    mask = support.T if vertical else support
    band = max(1, int(np.ceil(scale if vertical else step)))
    cell = float(scale if vertical else advance)
    if mask.shape[0] < band or cell <= 0:
        return 1, 0
    row_any = np.any(mask, axis=1)
    if not np.any(row_any):
        return 1, 0
    r_idx = np.flatnonzero(row_any)
    r0, r1 = int(r_idx[0]), int(r_idx[-1]) + 1
    if r1 - r0 < band:
        return 1, 0
    col_any = np.any(mask[r0:r1], axis=0)
    c_idx = np.flatnonzero(col_any)
    c0, c1 = int(c_idx[0]), int(c_idx[-1]) + 1
    sub_mask = mask[r0:r1, c0:c1]

    # Exact vectorized sliding-band AND via vertical cumulative sum of invalid pixels
    bad = (~sub_mask).astype(np.int32)
    cs = np.zeros((sub_mask.shape[0] + 1, sub_mask.shape[1]), dtype=np.int32)
    np.cumsum(bad, axis=0, out=cs[1:])
    band_ok = (cs[band:] - cs[:-band]) == 0

    padded = np.pad(band_ok, ((0, 0), (1, 1)), mode="constant", constant_values=False)
    diff = padded[:, 1:].view(np.int8) - padded[:, :-1].view(np.int8)
    r_s, c_s = np.nonzero(diff == 1)
    if r_s.size == 0:
        return 1, 0
    _, c_e = np.nonzero(diff == -1)
    run_counts = np.floor((c_e - c_s).astype(np.float64) / cell).astype(np.int64)
    row_totals = np.bincount(r_s, weights=run_counts, minlength=band_ok.shape[0]).astype(np.int64)
    best_count = max(int(row_totals[phase::band].sum()) for phase in range(band))
    return 1, best_count


class _Token:
    __slots__ = ("text", "ruby", "is_cjk", "leading")

    def __init__(self, text: str, ruby: str = "", is_cjk: bool = True):
        self.text = text
        self.ruby = ruby
        self.is_cjk = is_cjk
        self.leading = ""


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0xFF00 <= cp <= 0xFFEF
        or 0xAC00 <= cp <= 0xD7AF
    )


def _tokenize_with_ruby(text: str, ruby_pairs: list[RubyPair]) -> list[_Token]:
    """
    Tokenize target text into atomic line-breakable units while keeping:
      1) Annotated RubyPair base spans (`宿敌` -> `挚友`) strictly indivisible!
      2) Space-separated Latin words indivisible (for comic_ltr / mixed English).
    """
    if not text:
        return []
    span_map: dict[int, RubyPair] = {}
    for rp in ruby_pairs:
        if 0 <= rp.char_start < rp.char_end <= len(text) and rp.ruby:
            span_map[rp.char_start] = rp

    tokens: list[_Token] = []
    i = 0
    n = len(text)
    leading = ""
    while i < n:
        if text[i].isspace() and text[i] != "\n":
            leading += text[i]
            i += 1
            continue
        before = len(tokens)
        if i in span_map:
            rp = span_map[i]
            sub = text[rp.char_start : rp.char_end]
            tokens.append(_Token(sub, ruby=rp.ruby, is_cjk=any(_is_cjk_char(c) for c in sub)))
            tokens[-1].leading, leading = leading, ""
            i = rp.char_end
            continue

        ch = text[i]
        if ch == "\n":
            tokens.append(_Token("\n", is_cjk=True))
            i += 1
        elif _is_cjk_char(ch):
            tokens.append(_Token(ch, is_cjk=True))
            i += 1
        else:
            # Collect contiguous Latin/digit word
            j = i + 1
            while j < n and j not in span_map and (not _is_cjk_char(text[j])) and (not text[j].isspace()):
                j += 1
            tokens.append(_Token(text[i:j], is_cjk=False))
            i = j
        if len(tokens) > before:
            tokens[-1].leading, leading = leading, ""

    return tokens


def _semantic_breaks(text: str, segments: list[str] | None, tokens: list[_Token]) -> set[int] | None:
    """Map exact model-supplied linguistic boundaries onto rendering tokens.

    Ruby and Latin atoms remain intact; an incompatible proposed boundary
    cannot split either. A manually edited target invalidates its old segments.
    """
    if not segments or "".join(segments) != text:
        return None
    compact = lambda value: "".join(c for c in value if not c.isspace())
    ends, offset = {0}, 0
    for segment in segments:
        offset += len(compact(segment))
        ends.add(offset)
    allowed, offset = {0}, 0
    for index, token in enumerate(tokens):
        offset += len(compact(token.text))
        if offset in ends:
            allowed.add(index+1)
    return allowed


def _semantic_break_penalties(count: int, boundaries: set[int] | None) -> np.ndarray:
    """Short linguistic units resist splitting more than long clauses.

    A model's segmentation is evidence, not permission to shrink an entire
    paragraph into unreadable type. The reciprocal shortest fragment has no
    pixel scale or vocabulary dependency; complete atoms remain free breaks.
    """
    penalties=np.zeros(count+1,float)
    if boundaries is not None:
        ordered=sorted(boundaries|{0,count})
        for start,end in zip(ordered,ordered[1:]):
            for index in range(start+1,end):
                penalties[index]=1/min(index-start,end-index)
    return penalties


def _join_tokens(tokens: list[_Token], vertical: bool) -> str:
    parts: list[str] = []
    for k, tok in enumerate(tokens):
        if k > 0 and not vertical:
            parts.append(tok.leading)
        parts.append(tok.text)
    s = "".join(parts)
    return s.translate(_VERTICAL_PUNCT_MAP) if vertical else s


def _is_tcy_token(tok: _Token, vertical: bool) -> bool:
    """JIS X 4051 Tate-chu-yoko (縦中横): short horizontal ASCII numbers/punct (e.g. '100', '!?') inside vertical columns."""
    return (
        vertical
        and (not tok.is_cjk)
        and (1 <= len(tok.text) <= 3)
        and all(ord(c) < 128 for c in tok.text)
    )


def _precompute_token_prefix(
    tokens: list[_Token],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_size: float,
    char_spacing: float,
    vertical: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute O(1) prefix-sum advances for token runs without internal newlines."""
    n_toks = len(tokens)
    pref_seg = np.zeros(n_toks + 1, dtype=np.float64)
    pref_gap = np.zeros(n_toks + 1, dtype=np.float64)
    for idx, tok in enumerate(tokens):
        if tok.text == "\n":
            seg = 0.0
            gap = 0.0
        else:
            seg = _measure_token_run([tok], font, font_size, char_spacing, vertical=vertical)
            gap = 0.0
            if idx + 1 < n_toks and tokens[idx + 1].text != "\n":
                gap = _token_gap(tok, tokens[idx+1], font, font_size, char_spacing, vertical)
        pref_seg[idx + 1] = pref_seg[idx] + seg
        pref_gap[idx + 1] = pref_gap[idx] + gap
    return pref_seg, pref_gap


def _token_gap(left, right, font, font_size, char_spacing, vertical):
    if not vertical and right.leading:
        return _text_advance(font, right.leading)
    return font_size * char_spacing if vertical or left.is_cjk or right.is_cjk else 0.


def _measure_token_run(
    tokens: list[_Token],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_size: float,
    char_spacing: float,
    vertical: bool,
) -> float:
    if not tokens:
        return 0.0
    if vertical:
        total = 0.0
        n_toks = len(tokens)
        for ti, tok in enumerate(tokens):
            if _is_tcy_token(tok, True):
                total += font_size
                if ti + 1 < n_toks:
                    total += font_size * char_spacing
            else:
                seg = tok.text.translate(_VERTICAL_PUNCT_MAP)
                for idx, ch in enumerate(seg):
                    next_ch = seg[idx + 1] if idx + 1 < len(seg) else None
                    has_next = (idx + 1 < len(seg)) or (ti + 1 < n_toks)
                    _, ch_h, gap, _ = _vertical_char_metrics(ch, next_ch, font, font_size, char_spacing, has_next)
                    total += ch_h + gap
            if tok.ruby:
                ruby_len = len(tok.ruby) * (font_size * 0.48)
                base_len = len(tok.text) * font_size
                if ruby_len > base_len:
                    total += ruby_len - base_len
        return total
    else:
        if not tokens:
            return 0.0
        length = 0.0
        n_toks = len(tokens)
        for ti, tok in enumerate(tokens):
            if tok.is_cjk:
                seg_w = _text_advance(font,tok.text) + max(0, len(tok.text) - 1) * (font_size * char_spacing)
            else:
                seg_w = _text_advance(font,tok.text)
            if tok.ruby:
                ruby_w = len(tok.ruby) * (font_size * 0.50)
                if ruby_w > seg_w:
                    seg_w = ruby_w
            length += seg_w
            if ti + 1 < n_toks:
                nxt = tokens[ti + 1]
                length += _token_gap(tok, nxt, font, font_size, char_spacing, False)
        return length


def _ribbon_chord_bounds_at(
    sdf: np.ndarray,
    margin: float,
    cross_coord: float,
    half_thick: float,
    pole_along: float,
    vertical: bool,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Minkowski Ribbon SDF Chord Query:
    Returns the longest/pole-anchored contiguous interval [start, end] where ALL pixels across
    the glyph ribbon thickness [cross_coord - half_thick, cross_coord + half_thick] maintain SDF > margin!
    Eliminates curved wall clipping and pointer-tail corner protrusions.
    """
    h, w = sdf.shape[:2]
    c0 = int(np.floor(cross_coord - half_thick))
    c1 = int(np.ceil(cross_coord + half_thick)) + 1
    mask_src = valid_mask if valid_mask is not None else (sdf > margin)
    if vertical:
        if c0 < 0 or c1 > w or c1 <= c0:
            return 0.0, 0.0, 0.0
        valid_1d = np.all(mask_src[:, c0:c1], axis=1)
    else:
        if c0 < 0 or c1 > h or c1 <= c0:
            return 0.0, 0.0, 0.0
        valid_1d = np.all(mask_src[c0:c1, :], axis=0)

    if not np.any(valid_1d):
        return 0.0, 0.0, 0.0

    padded = np.concatenate([[False], valid_1d, [False]])
    starts = np.nonzero(padded[1:] & ~padded[:-1])[0]
    ends = np.nonzero(~padded[1:] & padded[:-1])[0]
    if starts.size == 0:
        return 0.0, 0.0, 0.0

    min_valid_len = max(4.0, 1.7 * half_thick)
    for s_i, e_i in zip(starts, ends):
        if s_i <= pole_along <= e_i and float(e_i - s_i) >= min_valid_len:
            return float(s_i), float(e_i), float(e_i - s_i)

    best_idx = int(np.argmax(ends - starts))
    s_b, e_b = float(starts[best_idx]), float(ends[best_idx])
    return s_b, e_b, max(0.0, e_b - s_b)


def _source_ribbon_plan(tokens, lines, style, vertical, char_spacing, required_text, allowed_breaks=None):
    """Partition translated text onto observed line tracks, preserving anchors.

    Source paragraph breaks are preferences: translation length can change, but
    a floating caption must retain its place in the drawing. Dynamic programming
    fits complete tokens with kinsoku constraints at the largest source scale.
    """
    # Detector boxes can split a column into several pieces. Merge overlapping
    # cross-axis intervals before fitting the translated sequence.
    cross = 0 if vertical else 1
    extent = 2 if vertical else 3
    ordered = sorted(lines, key=lambda b: b[cross])
    tracks = []
    for box in ordered:
        x, y, w, h = map(float, box)
        if w <= 0 or h <= 0:
            continue
        overlap = min(box[cross]+box[extent], tracks[-1][cross]+tracks[-1][extent]) - max(box[cross], tracks[-1][cross]) if tracks else 0
        if tracks and overlap > min(box[extent], tracks[-1][extent])/2:
            a = tracks[-1]
            x0, y0 = min(a[0], x), min(a[1], y)
            x1, y1 = max(a[0] + a[2], x + w), max(a[1] + a[3], y + h)
            tracks[-1] = [x0, y0, x1 - x0, y1 - y0]
        else:
            tracks.append([x, y, w, h])
    if vertical:
        tracks.reverse()
    breaks, flat = set(), []
    flat_allowed = {0} if allowed_breaks is not None else None
    for index, token in enumerate(tokens):
        if token.text == "\n":
            breaks.add(len(flat))
        else:
            flat.append(token)
        if flat_allowed is not None and index+1 in allowed_breaks:
            flat_allowed.add(len(flat))
    allowed_breaks = flat_allowed
    if not tracks or not flat:
        return None
    n, count = len(flat), len(tracks)
    semantic_penalties=_semantic_break_penalties(n,allowed_breaks)
    widths = np.array([b[2] if vertical else b[3] for b in tracks])
    maximum = min(float(np.median(widths)), float(style.source_font_size_px) or float(np.median(widths)))
    source_scale = float(style.source_font_size_px) or float(np.median(widths))
    # Integer font sizes are FreeType's actual raster scales. Search them rather
    # than repeatedly rasterizing the same rounded fractional candidate.
    best=None
    for fs in range(max(1, int(np.floor(maximum))), 0, -1):
        scale_cost=float(np.log(fs/source_scale)**2)
        if best is not None and scale_cost>best[0]:
            break
        halo = float(style.outline_width_px)*fs/source_scale
        font = load_font(style.font_category, fs, bold=float(style.weight_scale) >= 1.10, required_text=required_text)
        seg, gap = _precompute_token_prefix(flat, font, fs, char_spacing, vertical)
        chords = [(b[1], b[1]+b[3], b[3]) if vertical else (b[0], b[0]+b[2], b[2]) for b in tracks]
        capacities = np.array([b[2] - 2 * halo for b in chords])
        total_advance = seg[-1] + gap[-2]
        fill = min(1., total_advance / max(float(capacities.sum()), np.finfo(float).eps))
        dp = np.full((count + 1, n + 1), np.inf)
        parents = np.full((count + 1, n + 1), -1, int)
        dp[0, 0] = 0
        for k, capacity in enumerate(capacities):
            usable = widths[k] >= fs + 2*halo and capacity > 0
            for i in range(n+1):
                if not np.isfinite(dp[k, i]):
                    continue
                # Translation can use fewer tracks than the original. A tiny
                # punctuation/ruby fragment cannot force the whole paragraph
                # into a one-pixel font; skipping a track never skips tokens.
                skip_cost = fill**2 if usable else 0
                if dp[k,i]+skip_cost < dp[k+1,i]:
                    dp[k+1,i],parents[k+1,i] = dp[k,i]+skip_cost,i
                if not usable:
                    continue
                for j in range(i + 1, n + 1):
                    advance = seg[j] - seg[i] + gap[j-1] - gap[i]
                    if advance > capacity:
                        break
                    if k and flat[i].text[0] in _LINE_START_FORBIDDEN:
                        continue
                    if j < n and flat[j-1].text[-1] in _LINE_END_FORBIDDEN:
                        continue
                    cost = (advance / capacity - fill) ** 2 + count*semantic_penalties[j]
                    # A natural paragraph boundary has zero break penalty.
                    if breaks and j < n and j not in breaks:
                        cost += 1 / n
                    value = dp[k, i] + cost
                    if value < dp[k+1, j]:
                        dp[k+1, j], parents[k+1, j] = value, i
        if np.isfinite(dp[count, n]):
            splits, used, j = [], [], n
            for k in range(count, 0, -1):
                i = parents[k, j]
                if i < j:
                    splits.append(flat[i:j]);used.append(k-1)
                j = i
            used.reverse()
            coords = [tracks[i][0]+tracks[i][2]/2 if vertical else tracks[i][1]+tracks[i][3]/2 for i in used]
            objective=float(dp[count,n])/count+scale_cost
            if best is None or objective<best[0]:
                best=(objective,float(fs),splits[::-1],coords,[chords[i] for i in used])
    if best is not None:
        return best
    raise ValueError("Translated text cannot fit source line geometry without clipping")


def _quadrilateral_layout(tokens, quads, shape, style, required_text, allowed_breaks=None):
    """Fit in each observed line's own coordinates, then map ink to the page."""
    vertical = style.orientation == "vertical"
    reference = np.array([0., 1.]) if vertical else np.array([1., 0.])
    # A detector can split one original row/column into punctuation and word
    # fragments. Weight orientation estimation by fragment span along the
    # reading direction and merge overlapping tracks across all groups.
    def _quad_axis_and_span(points: np.ndarray) -> tuple[np.ndarray, float, float]:
        edges = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(edges, axis=1)
        units = edges / np.maximum(lengths[:, None], np.finfo(float).eps)
        best_i = int(np.argmax(np.abs(units @ reference)))
        u = units[best_i].astype(np.float64)
        if u @ reference < 0:
            u = -u
        v = np.array([u[1], -u[0]]) if vertical else np.array([-u[1], u[0]])
        span_u = float(np.ptp(points @ u))
        span_v = float(np.ptp(points @ v))
        return u, span_u, span_v

    dir_vecs = []
    dir_weights = []
    for quad in quads:
        pts = np.asarray(quad, np.float32)
        u, span_u, _ = _quad_axis_and_span(pts)
        dir_vecs.append(u)
        dir_weights.append(max(span_u, 1.0))
    along_axis = np.sum(np.asarray(dir_vecs) * np.asarray(dir_weights)[:, None], axis=0)
    norm_a = float(np.linalg.norm(along_axis))
    along_axis = (along_axis / norm_a) if norm_a > np.finfo(float).eps else reference.copy()
    if along_axis @ reference < 0:
        along_axis = -along_axis
    # Snap near-horizontal/vertical blocks to canonical axis cleanly
    all_raw_pts = np.vstack([np.asarray(q, np.float32) for q in quads])
    cross_axis_tmp = np.array([along_axis[1], -along_axis[0]]) if vertical else np.array([-along_axis[1], along_axis[0]])
    a_perp = abs(float(along_axis @ np.array([reference[1], -reference[0]])))
    if a_perp <= 0.035 and float(np.ptp(all_raw_pts @ along_axis)) * a_perp <= max(2.5, 0.30 * float(np.ptp(all_raw_pts @ cross_axis_tmp))):
        along_axis = reference.copy()
    cross_axis = np.array([along_axis[1], -along_axis[0]]) if vertical else np.array([-along_axis[1], along_axis[0]])

    grouped: list[list[np.ndarray]] = []
    for quad in sorted(quads, key=lambda q: float(np.min(np.asarray(q, np.float32) @ cross_axis))):
        points = np.asarray(quad, np.float32)
        lo, hi = float((points @ cross_axis).min()), float((points @ cross_axis).max())
        best_grp_idx = -1
        best_overlap = 0.0
        for g_idx, grp in enumerate(grouped):
            prior = max(grp, key=lambda p: float(np.ptp(p @ along_axis)))
            a, b = float((prior @ cross_axis).min()), float((prior @ cross_axis).max())
            overlap = min(hi, b) - max(lo, a)
            min_w = max(min(hi - lo, b - a), 1e-3)
            if overlap > 0.45 * min_w and overlap > best_overlap:
                best_overlap = overlap
                best_grp_idx = g_idx
        if best_grp_idx >= 0:
            grouped[best_grp_idx].append(points)
        else:
            grouped.append([points])

    frames = []
    for group in grouped:
        member_stats = [_quad_axis_and_span(pts) for pts in group]
        max_span = max(sp_u for _, sp_u, _ in member_stats)
        dom_vecs = [u * max(sp_u, 1.0) for u, sp_u, _ in member_stats if sp_u >= 0.5 * max_span]
        u_grp = np.sum(dom_vecs, axis=0)
        u_norm = float(np.linalg.norm(u_grp))
        u_grp = (u_grp / u_norm) if u_norm > np.finfo(float).eps else along_axis.copy()
        if u_grp @ reference < 0:
            u_grp = -u_grp
        all_pts = np.vstack(group)
        # Share the block's consensus principal axis across parallel lines so rotated/diagonal
        # multi-line blocks never drift in angle or misorder across fragments.
        if float(u_grp @ along_axis) >= 0.96:
            u_grp = along_axis.copy()
            v_grp = cross_axis.copy()
        else:
            v_raw = np.array([u_grp[1], -u_grp[0]]) if vertical else np.array([-u_grp[1], u_grp[0]])
            grp_along = float(np.ptp(all_pts @ u_grp))
            grp_cross = float(np.ptp(all_pts @ v_raw))
            u_perp = abs(float(u_grp @ np.array([reference[1], -reference[0]])))
            if u_perp <= 0.035 and grp_along * u_perp <= max(2.5, 0.30 * grp_cross):
                u_grp = reference.copy()
            v_grp = np.array([u_grp[1], -u_grp[0]]) if vertical else np.array([-u_grp[1], u_grp[0]])
        pu = all_pts @ u_grp
        pv = all_pts @ v_grp
        u0, u1 = float(pu.min()), float(pu.max())
        v0, v1 = float(pv.min()), float(pv.max())
        origin = u_grp * u0 + v_grp * v0
        center = u_grp * (0.5 * (u0 + u1)) + v_grp * (0.5 * (v0 + v1))
        frames.append((origin, u_grp, v_grp, max(1.0, u1 - u0), max(1.0, v1 - v0), center))

    # Order frames strictly along the shared cross-axis (top-to-bottom for horizontal, RTL for vertical)
    frames.sort(key=lambda f: (-(f[5] @ cross_axis), f[5] @ along_axis) if vertical else (f[5] @ cross_axis, f[5] @ along_axis))

    # Infer horizontal alignment from substantial line anchors (ignoring short trailing fragments)
    # so a short final line or punctuation box never pulls a left-aligned block off-anchor!
    alignment = 0.5
    max_along_f = max(f[3] for f in frames)
    anchor_frames = [f for f in frames if f[3] >= 0.50 * max_along_f]
    if len(anchor_frames) < 2:
        anchor_frames = frames
    if not vertical and len(frames) > 1:
        anchors = np.array([[f[0] @ along_axis + t * f[3] for t in (0.0, 0.5, 1.0)] for f in anchor_frames])
        dispersion = np.mean((anchors - anchors.mean(axis=0)) ** 2, axis=0)
        alignment = (0.0, 0.5, 1.0)[int(np.argmin(dispersion))]
    if not vertical and len(frames) > 1:
        # Latin cap height is smaller than an em. Use observed baseline spacing
        # as available row height when fitting CJK, rather than treating a thin
        # strip of capital-letter ink as the complete typographic line box.
        locations = np.array([f[5] @ cross_axis for f in frames])
        pitches = np.abs(np.diff(locations))
        for i, frame in enumerate(frames):
            origin, u, v, along, cross, center = frame
            neighbors = [pitches[j] for j in (i - 1, i) if 0 <= j < len(pitches) and pitches[j] > 0]
            if neighbors and along >= cross:
                available = max(cross, min(neighbors))
                frames[i] = (origin - v * (available - cross) / 2, u, v, along, available, center)
        style = style.model_copy(update={"source_font_size_px": float(np.median([f[4] for f in anchor_frames]))})

    block_u_min = float(np.min(all_raw_pts @ along_axis))
    block_u_max = float(np.max(all_raw_pts @ along_axis))
    target_anchor = float(np.median([f[0] @ along_axis + alignment * f[3] for f in anchor_frames]))

    def _side_inset(along_len: float, cross_len: float) -> float:
        return (0.03 * along_len) if (not vertical and along_len >= 2.0 * cross_len) else 0.0

    stride = max(f[4] for f in frames) + 1
    # The ribbon planner reads vertical tracks from right to left. Match its
    # synthetic coordinates to frames already ordered in reading direction.
    tracks = []
    for i, f in enumerate(frames):
        inset = _side_inset(f[3], f[4])
        if vertical:
            tracks.append([(len(frames) - 1 - i) * stride, 0, f[4], f[3]])
        else:
            tracks.append([inset, i * stride, max(1.0, f[3] - 2.0 * inset), f[4]])
    plan = _source_ribbon_plan(tokens, tracks, style, vertical, .04, required_text, allowed_breaks)
    _, fs, splits, positions, _ = plan
    alpha = np.zeros(shape, np.float32)
    glyphs, rendered = [], []
    centers = np.array([b[0] + b[2] / 2 if vertical else b[1] + b[3] / 2 for b in tracks])
    for position, line_tokens in zip(positions, splits):
        frame = frames[int(np.argmin(np.abs(centers - position)))]
        origin, u, v, along, cross, _ = frame
        line_text = _join_tokens(line_tokens, vertical=vertical)
        ruby_pairs, cursor = [], 0
        for token in line_tokens:
            position = line_text.find(token.text, cursor)
            if token.ruby:
                ruby_pairs.append(RubyPair(base=token.text, ruby=token.ruby, char_start=position, char_end=position + len(token.text)))
            cursor = position + len(token.text)
        height, width = (int(np.ceil(along)), int(np.ceil(cross))) if vertical else (int(np.ceil(cross)), int(np.ceil(along)))
        source_scale = float(style.source_font_size_px) or fs
        local_style = style.model_copy(update={
            "source_font_size_px": fs,
            "outline_width_px": float(style.outline_width_px) * fs / source_scale,
        })
        local_layout, ink = solve_sdf_knuth_plass_layout(
            line_text, ruby_pairs, np.ones((height, width), np.float32), (width / 2, height / 2), local_style,
            enclosed_bubble=False, source_lines=[[0, 0, width, height]],
        )
        if not vertical and np.any(ink > 0):
            _, columns = np.nonzero(ink > 0)
            ink_w = float(columns.max() - columns.min() + 1)
            inset = _side_inset(along, cross)
            frame_u0 = float(origin @ along_axis)
            if len(frames) > 1:
                desired_u_start = target_anchor - alignment * ink_w
                env_lo = block_u_min + inset
                env_hi = max(env_lo, block_u_max - inset - ink_w)
                desired_u_start = float(np.clip(desired_u_start, env_lo, env_hi))
                shift = (desired_u_start - frame_u0) - float(columns.min())
            else:
                shift = alignment * (along - ink_w) - float(columns.min())
            origin = origin + u * shift
        axes = np.column_stack((v, u)) if vertical else np.column_stack((u, v))
        transform = np.column_stack((axes, origin)).astype(np.float32)
        warped = cv2.warpAffine(ink, transform, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
        alpha = np.maximum(alpha, warped)
        rendered.extend(local_layout.lines)
        for glyph in local_layout.glyphs:
            corners = np.array([[glyph.x,glyph.y],[glyph.x+glyph.w,glyph.y],
                                [glyph.x+glyph.w,glyph.y+glyph.h],[glyph.x,glyph.y+glyph.h]]) @ axes.T + origin
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            glyphs.append(glyph.model_copy(update={"x":float(lo[0]),"y":float(lo[1]),"w":float(hi[0]-lo[0]),"h":float(hi[1]-lo[1])}))
    return TypesetLayout(font_size=fs, lines=rendered, glyphs=glyphs,
                         silhouette_score=float(1/(1+plan[0])), ink_density_rendered=float(alpha.mean())), alpha


def solve_sdf_knuth_plass_layout(
    target_text: str,
    ruby_pairs: list[RubyPair],
    sdf: np.ndarray,
    pole_xy: tuple[float, float],
    style: StyleSpec,
    enclosed_bubble: bool = True,
    source_lines: list[list[int]] | None = None,
    source_quadrilaterals: list[list[list[float]]] | None = None,
    target_segments: list[str] | None = None,
) -> tuple[TypesetLayout, np.ndarray]:
    """
    Exact Generalized Knuth-Plass Dynamic Programming Diamond/Ellipsoid Typesetting Solver
    constrained by the bubble's Euclidean Distance Field `sdf`, supporting:
      - Vertical RTL CJK columns (with Unicode Vertical Presentation Forms, Tate-chu-yoko & right-side Ruby)
      - Horizontal LTR/Webtoon rows (with top-side Ruby & word-boundary preservation)
      - Minkowski ribbon SDF clearance & source-scale anchoring
      - Strict JIS X 4051 / GB/T 15834 Kinsoku Shori line-start/line-end rules
    Returns:
      (TypesetLayout, rendered_grayscale_alpha_float32 [0..1] of shape sdf.shape)
    """
    h, w = sdf.shape[:2]
    clean_txt = target_text.strip()
    if not clean_txt:
        return TypesetLayout(lines=[]), np.zeros((h, w), dtype=np.float32)

    vertical = style.orientation == "vertical"
    tokens = _tokenize_with_ruby(clean_txt, ruby_pairs)
    allowed_breaks = _semantic_breaks(clean_txt, target_segments, tokens)
    semantic_penalties=_semantic_break_penalties(len(tokens),allowed_breaks)
    if not tokens:
        return TypesetLayout(lines=[]), np.zeros((h, w), dtype=np.float32)

    if not enclosed_bubble and source_quadrilaterals:
        required = clean_txt + "".join(r.ruby for r in ruby_pairs)
        if vertical:
            required = required.translate(_VERTICAL_PUNCT_MAP)
        return _quadrilateral_layout(tokens, source_quadrilaterals, (h, w), style, required, allowed_breaks)

    max_sdf = float(np.max(sdf))
    s0 = float(getattr(style, "source_font_size_px", 0.0) or 0.0)
    outline_pad = (float(style.outline_width_px) + 0.5) if float(style.outline_width_px) > 0.25 else 0.0
    if enclosed_bubble:
        base_margin = max(0.18 * max_sdf, 0.22 * s0)
        margin = min(base_margin + outline_pad, 0.34 * max_sdf)
    else:
        margin = min(max(0.08 * max_sdf, outline_pad), 0.18 * max_sdf)
    safe_mask = sdf > margin
    if np.count_nonzero(safe_mask) < 16:
        margin = 0.0
        safe_mask = sdf > 0.0

    geom_max_fs = max(11.0, min(float(min(h, w)) * 0.65, max(8.0, (max_sdf - margin) * 1.35)))
    if s0 > 4.0:
        scale_up = (0.96 if vertical else 0.92) if enclosed_bubble else 0.96
        max_up = 1.02
        raw_target = min(geom_max_fs, s0 * scale_up)
        max_fs = max(9.5, min(geom_max_fs, s0 * max_up))
        min_fs = max(8.5, min(raw_target * 0.52, max_fs * 0.55))
        hi_int = max(8, int(round(max_fs)))
        lo_int = max(8, min(hi_int, int(np.floor(min_fs))))
        target_fs = float(np.clip(round(raw_target), lo_int, hi_int))
        fs_candidates = np.arange(hi_int, lo_int - 1, -1, dtype=np.float64)
    else:
        max_fs = geom_max_fs
        target_fs = max_fs * 0.72
        min_fs = max(8.0, min(11.0, max_fs * 0.32))
        fs_candidates = np.linspace(max_fs, min_fs, 15)

    line_spacing = 1.22 if not ruby_pairs else 1.52
    char_spacing = 0.04
    pole_x, pole_y = pole_xy
    n_tok = len(tokens)
    is_bold = float(style.weight_scale) >= 1.10

    best_plan = None  # (total_score, font_size, line_token_slices, line_coords, chords)

    has_cjk = any(_is_cjk_char(ch) for ch in clean_txt) or any(
        any(_is_cjk_char(ch) for ch in rp.ruby) for rp in (ruby_pairs or [])
    )
    required_text = (clean_txt + "".join(rp.ruby for rp in ruby_pairs)).translate(_VERTICAL_PUNCT_MAP) if vertical else clean_txt + "".join(rp.ruby for rp in ruby_pairs)
    if not enclosed_bubble and source_lines:
        best_plan = _source_ribbon_plan(tokens, source_lines, style, vertical, char_spacing, required_text, allowed_breaks)
    for fs in ([] if best_plan is not None else fs_candidates):
        fs_f = float(fs)
        if s0 > 4.0 and fs_f < target_fs and best_plan is not None:
            err_lb = fs_f - target_fs
            lb_obj = 3.5 * ((err_lb / max(target_fs, 1.0)) ** 2) - 0.12 * (fs_f / max(target_fs, 1.0))
            if lb_obj >= best_plan[0]:
                break
        font = load_font(style.font_category, fs_f, needs_cjk=has_cjk, bold=is_bold, required_text=required_text)
        pref_seg, pref_gap = _precompute_token_prefix(tokens, font, fs_f, char_spacing, vertical=vertical)
        step_cross = fs_f * line_spacing

        cross_span_start, cross_span_end, cross_cap = _ribbon_chord_bounds_at(
            sdf,
            margin,
            pole_y if vertical else pole_x,
            half_thick=0.45 * fs_f,
            pole_along=pole_x if vertical else pole_y,
            vertical=(not vertical),
            valid_mask=safe_mask,
        )
        if cross_cap <= fs_f * 0.75:
            cross_cap = float(w if vertical else h)
            cross_span_start = 0.0
            cross_span_end = cross_cap

        max_lines = max(1, min(n_tok, int(np.floor((cross_cap + 0.20 * fs_f) / step_cross))))

        for K in range(1, max_lines + 1):
            ruby_headroom = (0.48 * fs_f) if (not vertical and ruby_pairs) else 0.0
            center_c = pole_x if vertical else pole_y
            c_min_allowed = cross_span_start + 0.5 * fs_f + ruby_headroom
            c_max_allowed = cross_span_end - 0.5 * fs_f
            half_span = 0.5 * (K - 1) * step_cross
            if c_max_allowed - half_span >= c_min_allowed + half_span:
                block_center = float(
                    np.clip(center_c, c_min_allowed + half_span, c_max_allowed - half_span)
                )
            else:
                block_center = 0.5 * (cross_span_start + ruby_headroom + cross_span_end)

            line_coords: list[float] = []
            chords: list[tuple[float, float, float]] = []
            feasible_chords = True
            for k in range(K):
                offset = (k - 0.5 * (K - 1)) * step_cross
                c_pos = (block_center - offset) if vertical else (block_center + offset)
                s_c, e_c, len_c = _ribbon_chord_bounds_at(
                    sdf,
                    margin,
                    c_pos - (0.5 * ruby_headroom if not vertical else 0.0),
                    half_thick=0.46 * fs_f + (0.5 * ruby_headroom if not vertical else 0.0),
                    pole_along=pole_y if vertical else pole_x,
                    vertical=vertical,
                    valid_mask=safe_mask,
                )
                if len_c < fs_f * 0.90:
                    feasible_chords = False
                    break
                line_coords.append(c_pos)
                chords.append((s_c, e_c, len_c))
            if not feasible_chords:
                continue

            max_chord_k = max((c[2] for c in chords), default=1.0)
            nat_fill = min(0.84, max(0.16, (n_tok * fs_f) / max(float(K) * max_chord_k, 1.0)))
            taus = []
            for k in range(K):
                if K <= 2:
                    t_base = 0.84
                else:
                    u = (k - 0.5 * (K - 1)) / (0.5 * K + 0.25)
                    t_base = float(0.88 * np.sqrt(max(0.25, 1.0 - u * u)))
                taus.append((0.5 * t_base + 0.5 * nat_fill) if (s0 > 4.0 and enclosed_bubble) else t_base)

            INF = 1e18
            dp = np.full((K + 1, n_tok + 1), INF, dtype=np.float64)
            parent = np.full((K + 1, n_tok + 1), -1, dtype=np.int32)
            dp[0, 0] = 0.0

            for k in range(K):
                chord_len = chords[k][2]
                tau_k = taus[k]
                for i in range(k, n_tok):
                    if dp[k, i] >= INF * 0.5:
                        continue
                    start_i = i
                    while start_i < n_tok and tokens[start_i].text == "\n":
                        start_i += 1
                    for j in range(start_i + 1, n_tok + 1):
                        if (n_tok - j) < (K - 1 - k):
                            break
                        if tokens[j - 1].text == "\n" and j > start_i + 1:
                            break
                        if tokens[j - 1].text == "\n":
                            continue

                        adv = float(
                            (pref_seg[j] - pref_seg[start_i])
                            + (pref_gap[j - 1] - pref_gap[start_i])
                        )
                        if adv > chord_len * 1.00:
                            break
                        sub_toks = tokens[start_i:j]

                        fill_ratio = adv / max(chord_len, 1.0)
                        shape_cost = 4.0 * ((fill_ratio - tau_k) ** 2)

                        kinsoku_pen = 0.0
                        first_ch = sub_toks[0].text[0]
                        last_ch = sub_toks[-1].text[-1]
                        if k > 0 and first_ch in _LINE_START_FORBIDDEN:
                            kinsoku_pen += 25.0
                        if j < n_tok and last_ch in _LINE_END_FORBIDDEN:
                            kinsoku_pen += 25.0

                        orphan_pen = 0.0
                        if k == K - 1 and K >= 2 and len(sub_toks) == 1 and len(sub_toks[0].text) == 1 and n_tok >= 5:
                            orphan_pen += 2.2

                        # Visual badness is averaged over rows below; splitting
                        # a word must not become cheaper simply by adding rows.
                        cand_cost = dp[k, i] + shape_cost + kinsoku_pen + orphan_pen + K*semantic_penalties[j]
                        if cand_cost < dp[k + 1, j]:
                            dp[k + 1, j] = cand_cost
                            parent[k + 1, j] = i

            if dp[K, n_tok] < INF * 0.5:
                mean_badness = dp[K, n_tok] / float(K)
                if s0 > 4.0:
                    err_fs = fs_f - target_fs
                    w_scale = 16.0 if err_fs > 0.0 else 3.5
                    scale_pen = w_scale * ((err_fs / max(target_fs, 1.0)) ** 2)
                    size_bonus = 0.12 * (min(fs_f, target_fs) / max(target_fs, 1.0))
                    total_obj = mean_badness + scale_pen - size_bonus
                else:
                    size_reward = 1.6 * (fs_f / max(max_fs, 1.0))
                    total_obj = mean_badness - size_reward
                if best_plan is None or total_obj < best_plan[0]:
                    splits: list[list[_Token]] = []
                    curr = n_tok
                    for k_rev in range(K, 0, -1):
                        prev = int(parent[k_rev, curr])
                        splits.append([t for t in tokens[prev:curr] if t.text != "\n"])
                        curr = prev
                    splits.reverse()
                    best_plan = (total_obj, fs_f, splits, line_coords, chords)

        if s0 <= 4.0 and best_plan is not None and best_plan[1] >= fs_f and best_plan[0] < -0.6:
            break

    # Fallback multi-line balanced shrink if text is slightly tighter than min_fs (never overflow 1 single line!)
    if best_plan is None:
        clean_toks = [t for t in tokens if t.text != "\n"]
        fallback_units = [[token] for token in clean_toks]
        cross_s, cross_e, cross_c = _ribbon_chord_bounds_at(
            sdf,
            0.0,
            pole_y if vertical else pole_x,
            half_thick=0.35 * min_fs,
            pole_along=pole_x if vertical else pole_y,
            vertical=(not vertical),
        )
        if cross_c <= min_fs * 0.75:
            cross_c = float(w if vertical else h)
            cross_s, cross_e = 0.0, cross_c
        k_fb = max(1, min(len(fallback_units), int(np.floor((cross_c + 0.25 * min_fs) / (min_fs * 1.12)))))
        splits_fb: list[list[_Token]] = []
        for k_i in range(k_fb):
            i0 = (k_i * len(fallback_units)) // k_fb
            i1 = ((k_i + 1) * len(fallback_units)) // k_fb
            if i1 > i0:
                splits_fb.append([token for unit in fallback_units[i0:i1] for token in unit])
        k_fb = len(splits_fb)
        center_c = 0.5 * (cross_s + cross_e)
        step_fb = min(min_fs * 1.12, cross_c / max(float(k_fb), 1.0))
        coords_fb: list[float] = []
        chords_fb: list[tuple[float, float, float]] = []
        min_chord_fb = float(h if vertical else w)
        for k_i in range(k_fb):
            off = (k_i - 0.5 * (k_fb - 1)) * step_fb
            cp = (center_c - off) if vertical else (center_c + off)
            sc, ec, lc = _ribbon_chord_bounds_at(
                sdf,
                0.0,
                cp,
                half_thick=0.35 * min_fs,
                pole_along=pole_y if vertical else pole_x,
                vertical=vertical,
            )
            if lc <= 1.0:
                sc, ec, lc = 0.0, float(h if vertical else w), float(h if vertical else w)
            coords_fb.append(cp)
            chords_fb.append((sc, ec, lc))
            min_chord_fb = min(min_chord_fb, lc)
        # Fit the actual glyph advances, including Latin kerning, spaces and
        # ruby. A fixed minimum font size could silently clip an unbreakable
        # proper name or model-supplied linguistic unit.
        for fs_fb in range(max(1, int(np.floor(min_fs))), 0, -1):
            font_fb = load_font(style.font_category, fs_fb, bold=is_bold, required_text=required_text)
            if all(_measure_token_run(line, font_fb, fs_fb, char_spacing, vertical)
                   <= chord[2] for line, chord in zip(splits_fb, chords_fb)):
                break
        else:
            raise ValueError("Complete translation cannot fit without clipping at the available raster resolution")
        best_plan = (0.0, fs_fb, splits_fb, coords_fb, chords_fb)

    _, opt_fs, opt_splits, opt_coords, opt_chords = best_plan
    main_font = load_font(style.font_category, opt_fs, needs_cjk=has_cjk, bold=is_bold, required_text=required_text)
    ruby_fs = max(7.0, opt_fs * 0.46)
    ruby_font = load_font(style.font_category, ruby_fs, needs_cjk=has_cjk, bold=is_bold, required_text=required_text)

    # Render sharp 2x supersampled anti-aliased grayscale alpha canvas for sub-pixel typography fidelity!
    ss = 2
    canvas_img = Image.new("L", (w * ss, h * ss), 0)
    draw = ImageDraw.Draw(canvas_img)
    main_font_ss = load_font(style.font_category, opt_fs * ss, needs_cjk=has_cjk, bold=is_bold, required_text=required_text)
    ruby_font_ss = load_font(style.font_category, ruby_fs * ss, needs_cjk=has_cjk, bold=is_bold, required_text=required_text)

    glyph_boxes: list[GlyphBox] = []
    rendered_lines: list[str] = []

    for k, line_toks in enumerate(opt_splits):
        line_str = _join_tokens(line_toks, vertical=vertical)
        rendered_lines.append(line_str)
        c_pos = opt_coords[k]
        s_c, e_c, _ = opt_chords[k]
        adv = _measure_token_run(line_toks, main_font, opt_fs, char_spacing, vertical=vertical)
        cursor = s_c + style.outline_width_px if not enclosed_bubble else 0.5 * (s_c + e_c) - 0.5 * adv
        if not vertical:
            # Keep punctuation and CJK on one font baseline. Per-character
            # top anchoring moved commas, periods and ellipses to cap height.
            line_bbox = _font_bbox(main_font_ss, line_str, anchor="ls")
            baseline = c_pos-(line_bbox[1]+line_bbox[3])/(2*ss)

        for token_index,tok in enumerate(line_toks):
            t_str = tok.text.translate(_VERTICAL_PUNCT_MAP) if vertical else tok.text
            seg_start = cursor
            if vertical:
                if _is_tcy_token(tok, True):
                    # Render short ASCII digits/punct horizontally (Tate-chu-yoko) centered across column c_pos
                    tcy_scale = 1.0 if len(tok.text) == 1 else (0.78 if len(tok.text) == 2 else 0.60)
                    tcy_font_ss = load_font(style.font_category, opt_fs * tcy_scale * ss, needs_cjk=has_cjk, bold=is_bold, required_text=required_text)
                    bb = _font_bbox(tcy_font_ss, tok.text)
                    gw = float(bb[2] - bb[0]) / ss
                    gh = float(bb[3] - bb[1]) / ss
                    gx = c_pos - 0.5 * gw - float(bb[0]) / ss
                    gy = cursor + 0.5 * (opt_fs - gh) - float(bb[1]) / ss
                    draw.text((gx * ss, gy * ss), tok.text, fill=255, font=tcy_font_ss)
                    glyph_boxes.append(GlyphBox(char=tok.text, x=float(c_pos - 0.5 * gw), y=float(cursor), w=gw, h=opt_fs, font_size=opt_fs, is_ruby=False))
                    cursor += opt_fs * (1.0 + char_spacing)
                else:
                    for ci, ch in enumerate(t_str):
                        next_ch = t_str[ci + 1] if ci + 1 < len(t_str) else None
                        draw_ch, ch_h, gap, is_corner = _vertical_char_metrics(
                            ch, next_ch, main_font, opt_fs, char_spacing, has_next=True
                        )
                        bb = _font_bbox(main_font_ss, draw_ch)
                        gw = float(bb[2] - bb[0]) / ss
                        gh = float(bb[3] - bb[1]) / ss
                        if is_corner:
                            target_cx = c_pos + opt_fs * (0.24 if draw_ch in ("︐", "︑", "︒") else 0.20)
                            gx = target_cx - 0.5 * gw - float(bb[0]) / ss
                            gy = cursor + 0.46 * ch_h - 0.5 * gh - float(bb[1]) / ss
                            box_x = float(target_cx - 0.5 * gw)
                        elif draw_ch in _VERTICAL_CENTERED_PUNCT:
                            tbb = _font_tight_bbox(main_font_ss, draw_ch)
                            tw_g = float(tbb[2] - tbb[0]) / ss
                            th_g = float(tbb[3] - tbb[1]) / ss
                            gx = c_pos - 0.5 * (tbb[0] + tbb[2]) / ss
                            gy = cursor + 0.5 * ch_h - 0.5 * (tbb[1] + tbb[3]) / ss
                            box_x = float(c_pos - 0.5 * tw_g)
                            gw = tw_g
                        elif draw_ch in _VERTICAL_CORNER_BRACKETS:
                            tbb = _font_tight_bbox(main_font_ss, draw_ch)
                            gx = c_pos - 0.5 * gw - float(bb[0]) / ss
                            gy = cursor + 0.5 * ch_h - 0.5 * (tbb[1] + tbb[3]) / ss
                            box_x = float(c_pos - 0.5 * gw)
                        else:
                            gx = c_pos - 0.5 * gw - float(bb[0]) / ss
                            gy = cursor + 0.5 * (ch_h - gh) - float(bb[1]) / ss
                            box_x = float(c_pos - 0.5 * gw)
                        draw.text((gx * ss, gy * ss), draw_ch, fill=255, font=main_font_ss)
                        glyph_boxes.append(GlyphBox(char=ch, x=box_x, y=float(cursor), w=gw, h=ch_h, font_size=opt_fs, is_ruby=False))
                        cursor += ch_h + gap

                if tok.ruby:
                    span_mid_y = 0.5 * (seg_start + cursor - opt_fs * char_spacing)
                    r_str = tok.ruby.translate(_VERTICAL_PUNCT_MAP)
                    r_total_h = len(r_str) * ruby_fs
                    ry = span_mid_y - 0.5 * r_total_h
                    rx_center = c_pos + 0.52 * opt_fs + 0.50 * ruby_fs
                    for rch in r_str:
                        rbb = _font_bbox(ruby_font_ss, rch)
                        rw = float(rbb[2] - rbb[0]) / ss
                        draw.text(((rx_center - 0.5 * rw - float(rbb[0]) / ss) * ss, (ry - float(rbb[1]) / ss) * ss), rch, fill=255, font=ruby_font_ss)
                        glyph_boxes.append(GlyphBox(char=rch, x=float(rx_center - 0.5 * rw), y=float(ry), w=rw, h=ruby_fs, font_size=ruby_fs, is_ruby=True))
                        ry += ruby_fs
            else:
                gy = c_pos - 0.5 * opt_fs
                if tok.is_cjk:
                    for ch in t_str:
                        adv_ch = _text_advance(main_font,ch)
                        draw.text((cursor * ss, baseline * ss), ch, fill=255, font=main_font_ss, anchor="ls")
                        glyph_boxes.append(GlyphBox(char=ch, x=float(cursor), y=float(gy), w=adv_ch, h=opt_fs, font_size=opt_fs, is_ruby=False))
                        cursor += adv_ch + opt_fs * char_spacing
                    cursor -= opt_fs * char_spacing
                else:
                    # Render non-CJK / Latin word runs atomically so native kerning & ligatures remain intact
                    run_adv = _text_advance(main_font,t_str)
                    draw.text((cursor * ss, baseline * ss), t_str, fill=255, font=main_font_ss, anchor="ls")
                    prev_prefix = 0.0
                    for idx_c, ch in enumerate(t_str):
                        next_prefix = _text_advance(main_font,t_str[: idx_c + 1])
                        ch_w = max(1.0, next_prefix - prev_prefix)
                        glyph_boxes.append(
                            GlyphBox(char=ch, x=float(cursor + prev_prefix), y=float(gy), w=ch_w, h=opt_fs, font_size=opt_fs, is_ruby=False)
                        )
                        prev_prefix = next_prefix
                    cursor += run_adv

                if tok.ruby:
                    span_mid_x = 0.5 * (seg_start + cursor)
                    r_total_w = _text_advance(ruby_font,tok.ruby)
                    rx = max(0.0, span_mid_x - 0.5 * r_total_w)
                    ry = max(0.0, c_pos - 0.55 * opt_fs - ruby_fs)
                    draw.text((rx * ss, ry * ss), tok.ruby, fill=255, font=ruby_font_ss)
                    glyph_boxes.append(GlyphBox(char=tok.ruby, x=float(rx), y=float(ry), w=r_total_w, h=ruby_fs, font_size=ruby_fs, is_ruby=True))
                if token_index+1 < len(line_toks):
                    cursor += _token_gap(tok, line_toks[token_index+1], main_font, opt_fs, char_spacing, False)

    downsampled = canvas_img.resize((w, h), resample=Image.Resampling.BOX)
    glyph_alpha = np.asarray(downsampled, dtype=np.float32) / 255.0

    sil_score = float(np.clip(1.0 / (1.0 + max(0.0, best_plan[0] + 1.6)), 0.0, 1.0))
    ink_rendered = float(np.sum(glyph_alpha)) / max(float(h * w), 1.0)
    layout = TypesetLayout(
        font_size=opt_fs,
        line_spacing=line_spacing,
        char_spacing=char_spacing,
        lines=rendered_lines,
        silhouette_score=sil_score,
        ink_density_rendered=ink_rendered,
        glyphs=glyph_boxes,
    )
    return layout, glyph_alpha


def conserve_optical_ink_mass(
    glyph_alpha: np.ndarray,
    style: StyleSpec,
    over_artwork: bool,
) -> tuple[np.ndarray, StyleSpec]:
    r"""
    Exact Optical Ink-Mass Conservation Law via Brent's Root-Finding (`scipy.optimize.brentq`)
    on the Euclidean Signed Distance Field of the rendered strokes!
    When sparse katakana SFX (`ドーン`) is localized into dense multi-stroke Hanzi (`轰隆隆`) over artwork,
    prevents black-brick visual crushing by solving for the exact hollow shell thickness t* (or halftone core)
    such that:
        (1 / |A|) * \int_A alpha_compensated(x; t*) dx == rho_orig
    """
    h, w = glyph_alpha.shape[:2]
    area = max(float(h * w), 1.0)
    rho_solid = float(np.sum(glyph_alpha)) / area
    rho_target = float(style.ink_density_orig)

    updated = style.model_copy(deep=True)
    explicit_hollow = updated.hollow_mode in ("double_outline_hollow", "halftone_core")
    needs_compensation = explicit_hollow or (
        over_artwork
        and float(updated.stroke_width_px) >= 4.2
        and rho_target > 0.015
        and rho_solid > 1.35 * rho_target
    )

    if not needs_compensation or rho_solid <= 1e-4:
        return glyph_alpha, updated
    if (not explicit_hollow) and abs(rho_solid - rho_target) <= max(1e-3, 0.05 * rho_target):
        return glyph_alpha, updated

    base_bin = glyph_alpha >= 0.45
    if not np.any(base_bin):
        return glyph_alpha, updated

    # Derive crop bounds from all nonzero ink support so faint disconnected components are preserved
    nz_y, nz_x = np.nonzero(glyph_alpha > 1e-4)
    if nz_y.size == 0:
        return glyph_alpha, updated
    y0 = max(0, int(nz_y.min()) - 2)
    y1 = min(h, int(nz_y.max()) + 3)
    x0 = max(0, int(nz_x.min()) - 2)
    x1 = min(w, int(nz_x.max()) + 3)
    ch, cw = y1 - y0, x1 - x0
    bin_c = base_bin[y0:y1, x0:x1]
    alpha_c = glyph_alpha[y0:y1, x0:x1]
    pad_bin = np.pad(bin_c, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    d_in = distance_transform_edt(pad_bin)[1:-1, 1:-1].astype(np.float32)
    max_depth = float(np.max(d_in))

    def _embed(crop_res: np.ndarray) -> np.ndarray:
        out = np.zeros_like(glyph_alpha)
        out[y0:y1, x0:x1] = crop_res
        return out

    if updated.hollow_mode == "halftone_core" or (
        (explicit_hollow or max_depth >= 4.0) and max_depth >= 2.2 and updated.hollow_mode != "stroke_thinned"
    ):
        if updated.hollow_mode == "halftone_core":
            # Periodic 45-degree halftone dot core inside thick strokes + crisp outer rim!
            yy, xx = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            u_rot = (xx + yy) / np.sqrt(2.0)
            v_rot = (xx - yy) / np.sqrt(2.0)
            dot_period = max(3.5, 0.85 * max_depth)
            carrier = 0.5 * (np.cos(2.0 * np.pi * u_rot / dot_period) * np.cos(2.0 * np.pi * v_rot / dot_period) + 1.0)
            rim = np.clip(2.0 - d_in, 0.0, 1.0) * alpha_c

            def mass_err_ht(thresh: float) -> float:
                dots = (carrier >= thresh).astype(np.float32) * alpha_c
                comb = np.maximum(rim, dots)
                return float(np.sum(comb)) / area - rho_target

            f_low, f_high = mass_err_ht(0.99), mass_err_ht(0.01)
            if f_low <= 0.0 <= f_high:
                t_opt = float(brentq(mass_err_ht, 0.01, 0.99, xtol=1e-3))
            else:
                t_opt = 0.55
            dots_opt = (carrier >= t_opt).astype(np.float32) * alpha_c
            return _embed(np.maximum(rim, dots_opt)), updated

        # Hollow double-outline shell: keep outer shell 0 < d_in <= t*, making core transparent (alpha=0)
        # so underlying character eyes/lines remain visible through the glyph!
        def shell_alpha(thickness: float) -> np.ndarray:
            # Smooth 1px anti-aliased inner wall at d_in == thickness
            inner_keep = np.clip(thickness + 0.5 - d_in, 0.0, 1.0)
            return alpha_c * inner_keep

        def mass_err(thickness: float) -> float:
            return float(np.sum(shell_alpha(thickness))) / area - rho_target

        t_min = 0.85
        err_min = mass_err(t_min)
        err_max = mass_err(max_depth + 1.0)
        if err_min <= 0.0 <= err_max:
            t_star = float(brentq(mass_err, t_min, max_depth + 1.0, xtol=1e-3))
        else:
            t_star = max(t_min, 1.35)

        updated.hollow_mode = "double_outline_hollow"
        return _embed(shell_alpha(t_star)), updated

    # Thin strokes (max_depth < 2.2): continuous sub-pixel SDF erosion to match exact ink density
    d_out = distance_transform_edt(~pad_bin)[1:-1, 1:-1].astype(np.float32)
    sdf = d_in - d_out

    def erode_alpha(shift: float) -> np.ndarray:
        return np.clip(sdf - shift + 0.5, 0.0, 1.0)

    def erode_err(shift: float) -> float:
        return float(np.sum(erode_alpha(shift))) / area - rho_target

    e_0 = erode_err(0.0)
    e_hi = erode_err(max_depth)
    if e_hi <= 0.0 <= e_0:
        s_star = float(brentq(erode_err, 0.0, max_depth, xtol=1e-3))
        updated.hollow_mode = "stroke_thinned"
        return _embed(erode_alpha(s_star)), updated

    return glyph_alpha, updated
