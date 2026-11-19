export interface RubyPair {
  base: string;
  ruby: string;
  char_start: number;
  char_end: number;
}

export interface StyleSpec {
  orientation: "vertical" | "horizontal";
  font_category: "gothic" | "mincho" | "rounded" | "brush" | "comic_sans";
  stroke_width_px: number;
  stroke_modulation: number;
  tremor_amplitude: number;
  slant_deg: number;
  fg_color_rgba: number[];
  outline_color_rgba: number[];
  outline_width_px: number;
  blend_alpha: number;
  blend_mode: "normal" | "multiply" | "screen";
  ink_density_orig: number;
  hollow_mode: "solid" | "double_outline_hollow" | "halftone_core" | "stroke_thinned";
  weight_scale: number;
  source_font_size_px?: number;
}

export interface TypesetLayout {
  font_size: number;
  line_spacing: number;
  char_spacing: number;
  lines: string[];
  silhouette_score: number;
  ink_density_rendered: number;
}

export interface PatchLayers {
  roi: number[];
  clean_z0_png_b64: string;
  stroke_alpha_png_b64: string;
  source_observed_ink_png_b64: string;
  source_domain_png_b64: string;
  refined_stroke_png_b64: string;
  inpaint_unknown_png_b64: string;
  inferred_background_png_b64: string;
  boundary_opacity_png_b64: string;
  z2_occluder_mask_png_b64: string;
  shading_map_png_b64: string;
  exemplar_patch_png_b64: string;
  rendered_z1_png_b64: string;
  render_fingerprint: string;
}

export interface UncertaintySpec {
  score: number;
  ocr_entropy: number;
  occlusion_ambiguity: number;
  restoration_residual: number;
  source_ink_uncovered_ratio: number | null;
  source_ink_residual_ratio: number | null;
  automatic_repair_pixels: number;
  reasons: string[];
  needs_review: boolean;
}

export interface SourceReading {
  line_index: number;
  recognizer: string;
  text: string;
  confidence: number | null;
  truncated: boolean;
}

export interface TextNode {
  id: string;
  reading_order: number;
  track: "primary" | "aside";
  bbox: number[];
  polygon: number[][];
  source_lines: number[][];
  source_quadrilaterals: number[][][];
  detector_quadrilaterals: number[][][];
  source_readings: SourceReading[];
  bubble_polygon: number[][];
  tail_tip?: number[] | null;
  tail_dir?: number[] | null;
  spine_points: number[][];
  bubble_kind: string;
  fusion_mode: string;
  restoration_path: string;
  speaker_cluster_id: string;
  speaker_appearance_b64: string;
  source_text: string;
  ocr_text: string;
  recognition_pending: boolean;
  ruby_pairs: RubyPair[];
  target_text: string;
  target_segments: string[];
  target_ruby_pairs: RubyPair[];
  char_budget_min: number;
  char_budget_max: number;
  style: StyleSpec;
  layout: TypesetLayout;
  patches: PatchLayers;
  uncertainty: UncertaintySpec;
}

export interface PageMetrics {
  bit_exact_unmasked_ratio: number;
  unmasked_linf: number;
  protected_pixels: number;
  total_nodes: number;
  halftone_restored_nodes: number;
  deep_fusion_nodes: number;
  needs_review_nodes: number;
  analyze_ms: number;
  render_ms: number;
}

export interface PageDOM {
  recognition_progress?: { context_hash: string; results: Record<string, { id: string; is_text: boolean; line_texts: string[]; line_order: number[]; reason: string }>; line_parts?: Record<string, Record<string, { id: string; is_text: boolean; line_texts: string[]; line_order: number[]; reason: string }>> } | null;
  translation_progress?: { context_hash: string; results: Record<string, unknown> } | null;
  schema_version: number;
  width: number;
  height: number;
  format_profile: string;
  is_color: boolean;
  paper_tint_rgb: number[];
  is_spread: boolean;
  gutter_x?: number | null;
  nodes: TextNode[];
  rejected_regions: { id: string; bbox: number[]; reason: string; source_lines: number[][]; source_quadrilaterals: number[][][]; source_readings: SourceReading[] }[];
  metrics: PageMetrics;
}

export interface WorkItem {
  id: number;
  title: string;
  source_lang: string;
  target_lang: string;
  format_profile: string;
  page_count: number;
  review_count: number;
  reviewed_count?: number;
  chapter_count?: number;
  cover_page_id?: number;
  updated_at?: string;
  created_at: string;
}

export interface PageSummary {
  id: number;
  work_id: number;
  chapter_title: string;
  page_index: number;
  filename: string;
  width: number;
  height: number;
  format_profile: string;
  status: string;
  needs_review: boolean;
  total_nodes: number;
  bit_exact_unmasked_ratio: number;
  updated_at: string;
}

export interface GlossaryTerm {
  id?: number;
  work_id: number;
  source_term: string;
  target_term: string;
  ruby_override: string;
  notes: string;
}

export interface CharacterPersona {
  id?: number;
  work_id: number;
  cluster_key: string;
  name_raw: string;
  name_target: string;
  first_person: string;
  speech_style: string;
}

export interface CharacterRelation {
  work_id: number;
  from_char_id: number;
  to_char_id: number;
  honorific_target: string;
  register_note: string;
}

export interface HardwareInfo {
  configured_device: string;
  active_device: string;
  cuda_available: boolean;
  gpu_name: string;
  vram_total_mb: number;
  cpu_threads: number;
  inference_threads: number;
}

export interface HealthResponse {
  status: string;
  engine_connected: boolean;
  engine?: {
    status: string;
    version: string;
    device: string;
    hardware?: HardwareInfo;
  };
  device: string;
  max_concurrency: number;
  llm_configured: boolean;
  llm_model: string;
  llm_vision: boolean;
  llm_reasoning_effort: string;
  default_target_lang: string;
}

export interface ModelPreset {
  provider: string;
  model_id: string;
  display_name: string;
  base_url: string;
  supports_vision: boolean;
  has_api_key: boolean;
}

export interface RuntimeSettings {
  device: string;
  max_concurrency: number;
  target_lang: string;
  llm_base_url: string;
  llm_model: string;
  llm_reasoning_effort: string;
  llm_timeout_seconds: number;
  llm_vision: boolean;
  llm_source_prepass: boolean;
  llm_vision_audit: boolean;
  llm_batch_size: number;
}

export interface RuntimeSettingsResponse {
  settings: RuntimeSettings;
  llm_api_key_set: boolean;
  llm_api_key_masked: string;
  engine_managed: boolean;
  settings_file: string;
  local_presets: ModelPreset[];
  warning?: string;
  engine_connected?: boolean;
  engine?: HealthResponse["engine"];
  llm_configured?: boolean;
}

export interface RuntimeSettingsUpdate extends RuntimeSettings {
  llm_api_key?: string;
  preset_provider?: string;
  preset_model?: string;
  clear_api_key?: boolean;
}

export interface TestLLMResponse {
  ok: boolean;
  latency_ms: number;
  model?: string;
  reasoning_effort?: string;
  error?: string;
}

export interface TranslationJob {
  id: string;
  status: "running" | "completed" | "failed" | "canceled";
  source_kind?: string;
  work_title: string;
  chapter_title: string;
  work_id: number;
  total_units: number;
  downloaded_units?: number;
  completed_units: number;
  current_stage: string;
  current_file: string;
  message: string;
  page_ids: number[];
  error: string;
  started_at: string;
  updated_at: string;
  elapsed_ms: number;
}

export interface ComicSourceItem {
  id: number;
  name: string;
  kind: string;
  base_url: string;
  headers?: Record<string, string>;
  default_profile: string;
  created_at: string;
}

export interface ComicEntry {
  id: string;
  title: string;
  description?: string;
  cover_url?: string;
  source_kind: string;
  url: string;
  format_profile?: string;
  languages?: string[];
  tags?: string[];
}

export interface PageAsset {
  index: number;
  filename: string;
  url: string;
  headers?: Record<string, string>;
}

export interface ChapterEntry {
  id: string;
  title: string;
  chapter_num?: string;
  volume?: string;
  language?: string;
  page_count?: number;
  source_kind: string;
  url: string;
  archive_url?: string;
  pages?: PageAsset[];
  publish_at?: string;
}

export interface BrowseSourceResult {
  kind: "comics" | "chapters" | "pages" | "archive";
  title?: string;
  source_kind: string;
  comics?: ComicEntry[];
  chapters?: ChapterEntry[];
  pages?: PageAsset[];
  archive_url?: string;
  next_url?: string;
}
