# Prism-T

[中文](README.md) | English

Prism-T is a local-first localization and typesetting pipeline for Japanese manga, vertical webtoons, and Western comics. Every page is represented as a reversible `Comic DOM` that separates the restored clean plate (`Z0`), localized typography (`Z1`), and foreground occluders (`Z2`), strictly preserving unmasked original pixels bit-for-bit (`L_inf == 0`), with an interactive Web Studio and native multi-layer `.psd` / `.cbz` export.

---

## Overview

- **Scale-Invariant Preflight & Seamless Webtoon Stitching**
  - Estimates sensor noise $\hat{\sigma}_n$ via 2D Haar wavelet lower-quartile MAD and detects layout profiles (`manga_rtl`, `webtoon_vertical`, `comic_ltr`) from image statistics;
  - Tests vertical slice boundaries for cross-seam bubbles/text, stitches active strips into virtual continuous canvases, and losslessly slices them back to original offsets after rendering.

- **Closed-Form Stroke Matting & Multi-Layer Outline Erasure**
  - Combines Comic Text Detector neural segmentation, full-perimeter exterior color contrast, and topological outer-rim shell recovery to erase double-outlined lettering (fill + inner rim + outer rim) across multi-zone backgrounds without bleeding into margins;
  - Separates Kanji and Furigana (`ルビ` / `義訓ルビ`) via 2-GMM BIC selection and rectifies curved text lines with Bookstein Thin-Plate Splines (TPS).

- **Multi-Path Clean-Plate Solver (`Z0`) & 4D Optical Compositing**
  - Routes each region across cross-edition optical-flow fusion (`CLAHE + SIFT + USAC_MAGSAC + DIS`), Laplace harmonic fill, 2D FFT crystal halftone reconstruction (Fisher's $g$-test), Weickert anisotropic diffusion, and native FFT masked-only LaMa;
  - Never feeds whole pages into VAE/diffusion re-encoders; only recorded mask pixels are modified;
  - Performs inverse semi-transparent unmixing, `Z2` foreground occluder extraction, and intrinsic illumination shading (`Albedo × Shading`).

- **SDF Knuth-Plass Typesetting & Optical Ink Conservation**
  - Computes exact Euclidean Signed Distance Fields (SDF) of bubble interiors and solves contour-hugging line breaks with generalized Knuth-Plass dynamic programming;
  - Implements JIS X 4051 / GB/T 15834 Kinsoku line-breaking rules, vertical punctuation rotation, paired `Gikun Ruby`, and Brent root-finding (`brentq`) stroke scaling to match source ink mass.

- **Context-Aware Narrative Compiler & Streaming Resumable Batching**
  - Supports vision-grounded holistic compilation (annotated scene overview + native-resolution crop sheet) and OCR-backed text mode with strict JSON Schema validation and verbatim digit/masking-symbol conservation;
  - Persists cross-chapter character persona clusters (64D cosine re-ID), directed honorific relation graphs, and work glossaries in SQLite WAL;
  - Supports constant-memory streaming directory translation (`--stream --resume --continue-on-error`), concurrent workers (`PRISM_T_CONCURRENCY`), optional reasoning effort (`PRISM_T_LLM_REASONING_EFFORT`), and whole-page visual semantic audit (`PRISM_T_LLM_VISION_AUDIT`).

- **Web Studio & Native Multi-Layer `.psd` Export**
  - Includes a local React Web Studio for bookshelf browsing, live read-while-translating, interactive node editing, fast re-rendering without re-inpainting, and local folder / remote source import;
  - Encodes native binary Adobe Photoshop `.psd` v1 files in pure Go (PackBits RLE + UTF-16BE `8BIMluni` Unicode layer names) with deterministic layer ordering: `00_Original_Raw` → `01_Restored_Clean_Z0` → `02..N_Node_Z1` → `99_Foreground_Occluders_Z2`.

### Scope & Non-Goals

- **Honest Translation Backend Boundary**: Prism-T contains no hardcoded translation lookup tables or fake offline translations. End-to-end semantic translation requires an OpenAI-compatible Chat Completions endpoint supporting strict `response_format.type=json_schema`. Without one, run `--analyze-only` for pure visual de-rendering, clean-plate restoration, and layered `.psd` export.
- **Single-Node Local Architecture**: Built around local SQLite WAL (`data/prism_t.db`) and a loopback Python operator service (`127.0.0.1:18775`) supervised by `prismtd`.
- **Automated Quality Limits**: Ultra-thin handwritten SFX missed by upstream detectors, extreme multi-layer occlusions, and nuanced literary localization may still benefit from Web Studio adjustments or layered `.psd` finishing.

---

## Architecture

```text
[CLI / Web Studio] ---> prismtd (Go, CGO_ENABLED=0, :3090) ---> data/prism_t.db (SQLite WAL)
                              |---> OpenAI-Compatible LLM Endpoint (Narrative Compiler & Audit)
                              |---> Native Multi-Layer PSD v1 & Seamless CBZ Exporter
                              |
                              v  HTTP loopback (127.0.0.1:18775, auto-supervised)
                        prism-t-engine (Python, FastAPI / SciPy / OpenCV / ONNX / PyTorch)
                              |---> Scale-Invariant Preprocessor & Webtoon Seam Stitcher
                              |---> CTD + Closed-Form Matting + Outer-Rim Shell Recovery + TPS OCR
                              |---> Multi-Path Z0 Clean-Plate Solver (Flow / Laplace / FFT / PDE / LaMa)
                              +---> 4D Optical Unmixing (Alpha / Z2 / Shading) + SDF Knuth-Plass Typesetter
```

---

## Quick Start

Requirements: **Go 1.24+**, **Python 3.11+** with [`uv`](https://docs.astral.sh/uv/), and **Node.js 20+**. Supports NVIDIA CUDA acceleration or CPU execution.

```bash
# 1. Install Python vision & math operator dependencies
uv --directory engine sync --extra dev

# 2. Download detection, inpainting, and source OCR weights (SHA-256 verified)
uv --directory engine run python -m prism_t_engine.models

# 3. Build Web Studio assets and static CGO-free Go binary
npm --prefix web ci
npm --prefix web run build
CGO_ENABLED=0 go build -ldflags="-s -w" -o prismtd ./cmd/prismtd

# 4. Launch Web Studio (auto-spawns and supervises local prism-t-engine)
./prismtd serve --listen 127.0.0.1:3090
```

> On Windows PowerShell, use `$env:CGO_ENABLED="0"; go build -ldflags="-s -w" -o prismtd.exe ./cmd/prismtd`. If `PRISM_T_ADMIN_TOKEN` is unset on first launch, a random admin token is generated at `data/admin.key`.

### Common CLI Commands

```bash
# Pure visual de-rendering, clean-plate restoration, and layered .psd export (no LLM required)
./prismtd translate --in ./page_01.png --out ./page_01_layered.psd --analyze-only

# End-to-end translation with an OpenAI-compatible vision endpoint
export PRISM_T_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export PRISM_T_LLM_API_KEY="your-api-key"
export PRISM_T_LLM_MODEL="your-vision-model"
export PRISM_T_LLM_VISION="true"
./prismtd translate --in ./raw_chapter.cbz --out ./en.cbz --work "Sample Manga" --chapter "Ch. 01" --lang en

# High-throughput resumable streaming translation for large image directories
export PRISM_T_CONCURRENCY="4"
export PRISM_T_LLM_REASONING_EFFORT="low"
export PRISM_T_LLM_VISION_AUDIT="true"
./prismtd translate --in ./raw_pages --out ./out_localized --work "Sample Manga" --stream --resume --continue-on-error

# Export layered .psd / .cbz or manage work glossary
./prismtd export --page-id 1 --out ./page_01.psd
./prismtd export --work-id 1 --out ./work_01.cbz
./prismtd glossary --work "Sample Manga" --import ./glossary_terms.json
./prismtd stats
```

---

## Documentation

Detailed developer documentation (in Chinese) is located under [`docs/`](docs/):

- **[Getting Started & Build Guide](docs/getting-started.md)**
- **[Architecture & Technical Design](docs/architecture.md)**
- **[CLI & Configuration Reference](docs/cli-and-configuration.md)**
- **[HTTP API Reference](docs/api-reference.md)**

---

## Testing

```bash
# Go unit and integration tests
CGO_ENABLED=0 go test -count=1 ./...

# Python operator math-invariant and API tests
uv --directory engine run pytest -v

# Web frontend typecheck and production build
npm --prefix web run build
```

---

## Model Attribution

Pretrained weights are downloaded on demand into `data/models/` and are not bundled in this repository:

- [Comic Text Detector](https://github.com/dmMaze/comic-text-detector)
- [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting) (fine-tuned [LaMa](https://github.com/advimman/lama))
- [manga-ocr-nar-small](https://huggingface.co/muscgab/manga-ocr-nar-small) (weights licensed under CC BY-NC-SA 4.0; see downloaded `NOTICE`)

---

## Contributing & Security

- See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, engineering invariants, and pull request guidelines.
- See [SECURITY.md](SECURITY.md) for security boundaries and vulnerability reporting.

## License

Licensed under the [GNU General Public License v3.0](LICENSE).
