from __future__ import annotations

from typing import Annotated, Literal, TypeVar
from pydantic import BaseModel, BeforeValidator, Field

T = TypeVar("T")
NonNullList = Annotated[list[T], BeforeValidator(lambda v: [] if v is None else v)]

FormatProfile = Literal["auto", "manga_rtl", "webtoon_vertical", "comic_ltr"]
ResolvedFormatProfile = Literal["manga_rtl", "webtoon_vertical", "comic_ltr"]

BubbleKind = Literal[
    "speech_oval",
    "shout_spiky",
    "thought_cloud",
    "narration_box",
    "floating_sfx",
    "diegetic_surface",
]

FusionMode = Literal[
    "clean_interior",
    "halftone_backed",
    "speedline_backed",
    "alpha_blended",
    "occluded_z2",
    "diegetic_planar",
    "exemplar_stroke",
]

RestorationPath = Literal[
    "solid_fill",
    "fft_halftone_lattice",
    "structure_tensor_line",
    "narrow_slit_pde",
    "inverse_alpha_unmix",
    "cross_edition_fusion",
    "lama_masked",
]

HollowMode = Literal[
    "solid",
    "double_outline_hollow",
    "halftone_core",
    "stroke_thinned",
]


class RubyPair(BaseModel):
    base: str
    ruby: str
    char_start: int = 0
    char_end: int = 0


class StyleSpec(BaseModel):
    orientation: Literal["vertical", "horizontal"] = "vertical"
    font_category: Literal["gothic", "mincho", "rounded", "brush", "comic_sans"] = "gothic"
    stroke_width_px: float = 2.2
    stroke_modulation: float = 0.15
    tremor_amplitude: float = 0.0
    slant_deg: float = 0.0
    fg_color_rgba: list[int] = Field(default_factory=lambda: [12, 12, 14, 255])
    outline_color_rgba: list[int] = Field(default_factory=lambda: [255, 255, 255, 255])
    outline_width_px: float = 0.0
    blend_alpha: float = 1.0
    blend_mode: Literal["normal", "multiply", "screen"] = "normal"
    ink_density_orig: float = 0.25
    hollow_mode: HollowMode = "solid"
    weight_scale: float = 1.0
    source_font_size_px: float = 0.0


class GlyphBox(BaseModel):
    char: str
    x: float
    y: float
    w: float
    h: float
    font_size: float
    is_ruby: bool = False


class TypesetLayout(BaseModel):
    font_size: float = 20.0
    line_spacing: float = 1.18
    char_spacing: float = 0.04
    lines: NonNullList[str] = Field(default_factory=list)
    silhouette_score: float = 1.0
    ink_density_rendered: float = 0.25
    glyphs: NonNullList[GlyphBox] = Field(default_factory=list)


class PatchLayers(BaseModel):
    roi: list[int] = Field(default_factory=lambda: [0, 0, 0, 0])  # [x, y, w, h]
    clean_z0_png_b64: str = ""
    stroke_alpha_png_b64: str = ""
    source_observed_ink_png_b64: str = ""
    source_domain_png_b64: str = ""
    refined_stroke_png_b64: str = ""
    inpaint_unknown_png_b64: str = ""
    inferred_background_png_b64: str = ""
    boundary_opacity_png_b64: str = ""
    z2_occluder_mask_png_b64: str = ""
    shading_map_png_b64: str = ""
    exemplar_patch_png_b64: str = ""
    rendered_z1_png_b64: str = ""
    render_fingerprint: str = ""


class UncertaintySpec(BaseModel):
    score: float = 0.0
    ocr_entropy: float = 0.0
    occlusion_ambiguity: float = 0.0
    restoration_residual: float = 0.0
    source_ink_uncovered_ratio: float | None = None
    source_ink_residual_ratio: float | None = None
    automatic_repair_pixels: int = 0
    reasons: NonNullList[str] = Field(default_factory=list)
    needs_review: bool = False


class SeamSlice(BaseModel):
    index: int
    y_offset: int
    height: int
    filename: str = ""


class SourceReading(BaseModel):
    line_index: int
    recognizer: str
    text: str
    confidence: float | None = None
    truncated: bool = False


class TextNode(BaseModel):
    id: str
    reading_order: int = 0
    track: Literal["primary", "aside"] = "primary"
    bbox: list[int] = Field(default_factory=lambda: [0, 0, 0, 0])  # [x, y, w, h]
    polygon: NonNullList[list[float]] = Field(default_factory=list)
    source_lines: NonNullList[list[int]] = Field(default_factory=list)
    source_quadrilaterals: NonNullList[list[list[float]]] = Field(default_factory=list)
    detector_quadrilaterals: NonNullList[list[list[float]]] = Field(default_factory=list)
    source_readings: NonNullList[SourceReading] = Field(default_factory=list)
    bubble_polygon: NonNullList[list[float]] = Field(default_factory=list)
    tail_tip: list[float] | None = None
    tail_dir: list[float] | None = None
    spine_points: NonNullList[list[float]] = Field(default_factory=list)
    homography_3x3: list[list[float]] | None = None

    bubble_kind: BubbleKind = "speech_oval"
    fusion_mode: FusionMode = "clean_interior"
    restoration_path: RestorationPath = "solid_fill"

    speaker_cluster_id: str = ""
    speaker_appearance_b64: str = ""

    source_text: str = ""
    ocr_text: str = ""
    recognition_pending: bool = False
    ruby_pairs: NonNullList[RubyPair] = Field(default_factory=list)
    target_text: str = ""
    target_segments: NonNullList[str] = Field(default_factory=list)
    target_ruby_pairs: NonNullList[RubyPair] = Field(default_factory=list)
    char_budget_min: int = 1
    char_budget_max: int = 32

    style: StyleSpec = Field(default_factory=StyleSpec)
    layout: TypesetLayout = Field(default_factory=TypesetLayout)
    patches: PatchLayers = Field(default_factory=PatchLayers)
    uncertainty: UncertaintySpec = Field(default_factory=UncertaintySpec)


class PageMetrics(BaseModel):
    bit_exact_unmasked_ratio: float = 1.0
    unmasked_linf: int = 0
    protected_pixels: int = 0
    total_nodes: int = 0
    halftone_restored_nodes: int = 0
    deep_fusion_nodes: int = 0
    needs_review_nodes: int = 0
    analyze_ms: float = 0.0
    render_ms: float = 0.0


class RejectedRegion(BaseModel):
    id: str
    bbox: list[int]
    reason: str = ""
    source_lines: NonNullList[list[int]] = Field(default_factory=list)
    source_quadrilaterals: NonNullList[list[list[float]]] = Field(default_factory=list)
    source_readings: NonNullList[SourceReading] = Field(default_factory=list)


class PageDOM(BaseModel):
    recognition_progress: dict | None = None
    translation_progress: dict | None = None
    rejected_regions: NonNullList[RejectedRegion] = Field(default_factory=list)
    schema_version: int = 1
    width: int
    height: int
    format_profile: ResolvedFormatProfile = "manga_rtl"
    is_color: bool = False
    paper_tint_rgb: list[int] = Field(default_factory=lambda: [255, 255, 255])
    seam_slices: NonNullList[SeamSlice] = Field(default_factory=list)
    is_spread: bool = False
    gutter_x: int | None = None
    nodes: NonNullList[TextNode] = Field(default_factory=list)
    metrics: PageMetrics = Field(default_factory=PageMetrics)


class AnalyzeRequest(BaseModel):
    image_png_b64: str
    reference_edition_png_b64: str | None = None
    format_profile: FormatProfile = "auto"
    recognize_text: bool = True
    seam_slices: NonNullList[SeamSlice] = Field(default_factory=list)
    is_spread: bool = False
    gutter_x: int | None = None


class AnalyzeResponse(BaseModel):
    dom: PageDOM
    clean_preview_png_b64: str


class DetectLayoutResponse(BaseModel):
    dom: PageDOM
    layout_token: str


class RestoreRequest(BaseModel):
    layout_token: str = ""
    image_png_b64: str = ""
    reference_edition_png_b64: str | None = ""
    format_profile: FormatProfile = "auto"
    recognize_text: bool = False
    seam_slices: NonNullList[SeamSlice] = Field(default_factory=list)
    is_spread: bool = False
    gutter_x: int | None = None
    exclude_node_ids: NonNullList[str] = Field(default_factory=list)


class RenderRequest(BaseModel):
    raw_image_png_b64: str
    dom: PageDOM
    target_lang: str = "zh-CN"


class RenderResponse(BaseModel):
    dom: PageDOM
    composite_png_b64: str
    slice_pngs_b64: list[str] = Field(default_factory=list)
