package dom

import "encoding/json"

type TranslationProgress struct {
	ContextHash string                     `json:"context_hash"`
	Results     map[string]json.RawMessage `json:"results"`
}

type SourceTranscription struct {
	ID        string   `json:"id"`
	IsText    *bool    `json:"is_text"`
	LineTexts []string `json:"line_texts"`
	LineOrder []int    `json:"line_order"`
	Reason    string   `json:"reason"`
}

type RecognitionProgress struct {
	ContextHash string                                 `json:"context_hash"`
	Results     map[string]json.RawMessage             `json:"results"`
	LineParts   map[string]map[int]SourceTranscription `json:"line_parts,omitempty"`
}

type RubyPair struct {
	Base      string `json:"base"`
	Ruby      string `json:"ruby"`
	CharStart int    `json:"char_start"`
	CharEnd   int    `json:"char_end"`
}

type StyleSpec struct {
	Orientation      string  `json:"orientation"`
	FontCategory     string  `json:"font_category"`
	StrokeWidthPx    float64 `json:"stroke_width_px"`
	StrokeModulation float64 `json:"stroke_modulation"`
	TremorAmplitude  float64 `json:"tremor_amplitude"`
	SlantDeg         float64 `json:"slant_deg"`
	FgColorRGBA      []int   `json:"fg_color_rgba"`
	OutlineColorRGBA []int   `json:"outline_color_rgba"`
	OutlineWidthPx   float64 `json:"outline_width_px"`
	BlendAlpha       float64 `json:"blend_alpha"`
	BlendMode        string  `json:"blend_mode"`
	InkDensityOrig   float64 `json:"ink_density_orig"`
	HollowMode       string  `json:"hollow_mode"`
	WeightScale      float64 `json:"weight_scale"`
	SourceFontSizePx float64 `json:"source_font_size_px"`
}

type GlyphBox struct {
	Char     string  `json:"char"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	W        float64 `json:"w"`
	H        float64 `json:"h"`
	FontSize float64 `json:"font_size"`
	IsRuby   bool    `json:"is_ruby"`
}

type TypesetLayout struct {
	FontSize           float64    `json:"font_size"`
	LineSpacing        float64    `json:"line_spacing"`
	CharSpacing        float64    `json:"char_spacing"`
	Lines              []string   `json:"lines,omitempty"`
	SilhouetteScore    float64    `json:"silhouette_score"`
	InkDensityRendered float64    `json:"ink_density_rendered"`
	Glyphs             []GlyphBox `json:"glyphs,omitempty"`
}

type PatchLayers struct {
	ROI                      []int  `json:"roi"`
	CleanZ0PNGB64            string `json:"clean_z0_png_b64"`
	StrokeAlphaPNGB64        string `json:"stroke_alpha_png_b64"`
	SourceObservedInkPNGB64  string `json:"source_observed_ink_png_b64"`
	SourceDomainPNGB64       string `json:"source_domain_png_b64"`
	RefinedStrokePNGB64      string `json:"refined_stroke_png_b64"`
	InpaintUnknownPNGB64     string `json:"inpaint_unknown_png_b64"`
	InferredBackgroundPNGB64 string `json:"inferred_background_png_b64"`
	BoundaryOpacityPNGB64    string `json:"boundary_opacity_png_b64"`
	Z2OccluderMaskB64        string `json:"z2_occluder_mask_png_b64"`
	ShadingMapPNGB64         string `json:"shading_map_png_b64"`
	ExemplarPatchPNGB64      string `json:"exemplar_patch_png_b64"`
	RenderedZ1PNGB64         string `json:"rendered_z1_png_b64"`
	RenderFingerprint        string `json:"render_fingerprint"`
}

type UncertaintySpec struct {
	Score                   float64  `json:"score"`
	OCREntropy              float64  `json:"ocr_entropy"`
	OcclusionAmbiguity      float64  `json:"occlusion_ambiguity"`
	RestorationResidual     float64  `json:"restoration_residual"`
	SourceInkUncoveredRatio *float64 `json:"source_ink_uncovered_ratio"`
	SourceInkResidualRatio  *float64 `json:"source_ink_residual_ratio"`
	AutomaticRepairPixels   int      `json:"automatic_repair_pixels"`
	Reasons                 []string `json:"reasons,omitempty"`
	NeedsReview             bool     `json:"needs_review"`
}

type SeamSlice struct {
	Index    int    `json:"index"`
	YOffset  int    `json:"y_offset"`
	Height   int    `json:"height"`
	Filename string `json:"filename"`
}

type SourceReading struct {
	LineIndex  int      `json:"line_index"`
	Recognizer string   `json:"recognizer"`
	Text       string   `json:"text"`
	Confidence *float64 `json:"confidence"`
	Truncated  bool     `json:"truncated"`
}

type TextNode struct {
	ID                     string          `json:"id"`
	ReadingOrder           int             `json:"reading_order"`
	Track                  string          `json:"track"`
	BBox                   []int           `json:"bbox"`
	Polygon                [][]float64     `json:"polygon,omitempty"`
	SourceLines            [][]int         `json:"source_lines,omitempty"`
	SourceQuadrilaterals   [][][]float64   `json:"source_quadrilaterals,omitempty"`
	DetectorQuadrilaterals [][][]float64   `json:"detector_quadrilaterals,omitempty"`
	SourceReadings         []SourceReading `json:"source_readings,omitempty"`
	BubblePolygon          [][]float64     `json:"bubble_polygon,omitempty"`
	TailTip                []float64       `json:"tail_tip,omitempty"`
	TailDir                []float64       `json:"tail_dir,omitempty"`
	SpinePoints            [][]float64     `json:"spine_points,omitempty"`
	Homography3x3          [][]float64     `json:"homography_3x3,omitempty"`
	BubbleKind             string          `json:"bubble_kind"`
	FusionMode             string          `json:"fusion_mode"`
	RestorationPath        string          `json:"restoration_path"`
	SpeakerClusterID       string          `json:"speaker_cluster_id"`
	SpeakerAppearanceB64   string          `json:"speaker_appearance_b64"`
	SourceText             string          `json:"source_text"`
	OCRText                string          `json:"ocr_text,omitempty"`
	RecognitionPending     bool            `json:"recognition_pending"`
	RubyPairs              []RubyPair      `json:"ruby_pairs,omitempty"`
	TargetText             string          `json:"target_text"`
	TargetSegments         []string        `json:"target_segments"`
	TargetRubyPairs        []RubyPair      `json:"target_ruby_pairs,omitempty"`
	CharBudgetMin          int             `json:"char_budget_min"`
	CharBudgetMax          int             `json:"char_budget_max"`
	Style                  StyleSpec       `json:"style"`
	Layout                 TypesetLayout   `json:"layout"`
	Patches                PatchLayers     `json:"patches"`
	Uncertainty            UncertaintySpec `json:"uncertainty"`
}

type PageMetrics struct {
	UnmaskedLInf          int     `json:"unmasked_linf"`
	ProtectedPixels       int     `json:"protected_pixels"`
	BitExactUnmaskedRatio float64 `json:"bit_exact_unmasked_ratio"`
	TotalNodes            int     `json:"total_nodes"`
	HalftoneRestoredNodes int     `json:"halftone_restored_nodes"`
	DeepFusionNodes       int     `json:"deep_fusion_nodes"`
	NeedsReviewNodes      int     `json:"needs_review_nodes"`
	AnalyzeMs             float64 `json:"analyze_ms"`
	RenderMs              float64 `json:"render_ms"`
}

type RejectedRegion struct {
	ID                   string          `json:"id"`
	BBox                 []int           `json:"bbox"`
	Reason               string          `json:"reason"`
	SourceLines          [][]int         `json:"source_lines,omitempty"`
	SourceQuadrilaterals [][][]float64   `json:"source_quadrilaterals,omitempty"`
	SourceReadings       []SourceReading `json:"source_readings,omitempty"`
}

type PageDOM struct {
	RecognitionProgress *RecognitionProgress `json:"recognition_progress,omitempty"`
	TranslationProgress *TranslationProgress `json:"translation_progress,omitempty"`
	RejectedRegions     []RejectedRegion     `json:"rejected_regions,omitempty"`
	SchemaVersion       int                  `json:"schema_version"`
	Width               int                  `json:"width"`
	Height              int                  `json:"height"`
	FormatProfile       string               `json:"format_profile"`
	IsColor             bool                 `json:"is_color"`
	PaperTintRGB        []int                `json:"paper_tint_rgb"`
	SeamSlices          []SeamSlice          `json:"seam_slices,omitempty"`
	IsSpread            bool                 `json:"is_spread"`
	GutterX             *int                 `json:"gutter_x"`
	Nodes               []TextNode           `json:"nodes,omitempty"`
	Metrics             PageMetrics          `json:"metrics"`
}
