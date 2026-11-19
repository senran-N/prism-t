from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from . import __version__
from .models import warmup_models
from .pipeline import (
    LayoutTokenExpiredError,
    analyze_and_restore_page,
    detect_page_layout,
    restore_page_from_layout,
    typeset_and_render_page,
)
from .preprocess import detect_seam_split_need
from .ocr_tps import get_ocr_runtime_status
from .codec import decode_image_b64
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectLayoutResponse,
    RenderRequest,
    RenderResponse,
    RestoreRequest,
)
from pydantic import BaseModel


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    warmup_models(blocking=False)
    yield


app = FastAPI(title="Prism-T Engine", version=__version__, lifespan=_lifespan)


class SeamCheckRequest(BaseModel):
    top_png_b64: str
    bottom_png_b64: str


class SeamCheckResponse(BaseModel):
    needs_stitch: bool


@app.get("/health")
def health() -> dict:
    from .models import hardware_status, status
    from .source_ocr import source_ocr_directory
    ocr_status = get_ocr_runtime_status()
    return {
        "status": "ok",
        "version": __version__,
        "device": ocr_status["requested_device"],
        "hardware": hardware_status(),
        "ocr": ocr_status,
        "source_ocr": {"path": str(source_ocr_directory()), "available": (source_ocr_directory() / "onnx/mangaocr_nar_small_fp32.onnx").is_file()},
        "models": status(),
    }


@app.post("/v1/detect_layout", response_model=DetectLayoutResponse)
def detect_layout_endpoint(req: AnalyzeRequest) -> DetectLayoutResponse:
    return detect_page_layout(req)


@app.post("/v1/restore_page", response_model=AnalyzeResponse)
def restore_page_endpoint(req: RestoreRequest) -> AnalyzeResponse:
    try:
        return restore_page_from_layout(req)
    except LayoutTokenExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest) -> AnalyzeResponse:
    return analyze_and_restore_page(req)


@app.post("/v1/render", response_model=RenderResponse)
def render_endpoint(req: RenderRequest) -> RenderResponse:
    return typeset_and_render_page(req)


@app.post("/v1/check_seam", response_model=SeamCheckResponse)
def check_seam_endpoint(req: SeamCheckRequest) -> SeamCheckResponse:
    top_rgb = decode_image_b64(req.top_png_b64, mode="RGB")
    bot_rgb = decode_image_b64(req.bottom_png_b64, mode="RGB")
    return SeamCheckResponse(needs_stitch=detect_seam_split_need(top_rgb, bot_rgb))
