"""Pretrained segmentation and native FFT restoration assets with provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Callable, Sequence, TypeVar
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / "data" / "models"
ASSETS = {
    "comic_text": ("comictextdetector.onnx", "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx"),
    "lama": ("anime-manga-big-lama.pt", "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"),
    "source_ocr": ("manga_ocr_nar_small/onnx/mangaocr_nar_small_fp32.onnx", "https://huggingface.co/muscgab/manga-ocr-nar-small/resolve/81ff4f3313263d1da349946a9dd20d14e0fb0e66/onnx/mangaocr_nar_small_fp32.onnx"),
    "source_vocabulary": ("manga_ocr_nar_small/nar_vocabulary_6817.json", "https://huggingface.co/muscgab/manga-ocr-nar-small/resolve/81ff4f3313263d1da349946a9dd20d14e0fb0e66/nar_vocabulary_6817.json"),
    "source_license": ("manga_ocr_nar_small/LICENSE_MODEL", "https://huggingface.co/muscgab/manga-ocr-nar-small/resolve/81ff4f3313263d1da349946a9dd20d14e0fb0e66/LICENSE_MODEL"),
    "source_notice": ("manga_ocr_nar_small/NOTICE.md", "https://huggingface.co/muscgab/manga-ocr-nar-small/resolve/81ff4f3313263d1da349946a9dd20d14e0fb0e66/NOTICE.md"),
}
ASSET_DIGESTS = {
    "lama": "479d3afdcb7ed2fd944ed4ebcc39ca45b33491f0f2e43eb1000bd623cfb41823",
    "source_ocr": "5ee628dfd40c59c522022d87c31f8f903104198877b9ba4cf795603dc540dcf5",
    "source_vocabulary": "9f1059427b54216c869dbf1244157f86bef74b1a0061c46a6001402c6f0c1dbf",
    "source_license": "e66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd",
    "source_notice": "2ac6cbd920832dd5c7038866a3e36c4bb9174d074c2a492ba0e5165948cd173f",
}


def inference_threads() -> int:
    # Share cores across page jobs and detector/recognizer inference sessions.
    conc = max(1, int(os.environ.get("PRISM_T_CONCURRENCY", "1")))
    return max(2, (os.cpu_count() or 2) // max(1, 2*conc))


def model_path(name: str) -> Path:
    return Path(os.environ.get("PRISM_T_" + name.upper() + "_MODEL", str(ROOT / ASSETS[name][0])))


_DLL_DIR_LOCK = threading.Lock()
_DLL_DIR_HANDLES: list[object] = []
_CUDA_DLL_INITIALIZED = False


def ensure_cuda_dll_search_path() -> None:
    """Register PyTorch's bundled CUDA/cuDNN DLLs for onnxruntime-gpu on Windows."""
    global _CUDA_DLL_INITIALIZED
    if _CUDA_DLL_INITIALIZED:
        return
    with _DLL_DIR_LOCK:
        if _CUDA_DLL_INITIALIZED:
            return
        try:
            if os.name == "nt":
                import torch

                torch_lib = Path(torch.__file__).resolve().parent / "lib"
                if torch_lib.is_dir():
                    lib_str = str(torch_lib)
                    if hasattr(os, "add_dll_directory"):
                        _DLL_DIR_HANDLES.append(os.add_dll_directory(lib_str))
                    current_path = os.environ.get("PATH", "")
                    if lib_str.lower() not in current_path.lower():
                        os.environ["PATH"] = lib_str + (os.pathsep + current_path if current_path else "")
                    try:
                        import onnxruntime as ort

                        if hasattr(ort, "preload_dlls"):
                            ort.preload_dlls(directory=lib_str)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            _CUDA_DLL_INITIALIZED = True


def resolve_torch_device(device: str | None = None) -> str:
    import torch

    requested = (device if device is not None else os.environ.get("PRISM_DEVICE", "auto")).strip().lower()
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def resolve_ort_providers(device: str | None = None) -> list[str]:
    req = (device if device is not None else os.environ.get("PRISM_DEVICE", "auto")).strip().lower()
    if req != "cpu":
        ensure_cuda_dll_search_path()
    try:
        import onnxruntime as ort

        avail = list(ort.get_available_providers())
    except Exception:
        avail = ["CPUExecutionProvider"]

    if req == "cpu":
        order = ["CPUExecutionProvider"]
    elif req == "cuda":
        order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif req in ("dml", "directml"):
        order = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

    resolved = [p for p in order if p in avail]
    return resolved if resolved else ["CPUExecutionProvider"]


def ort_provider_specs(providers: Sequence[str]) -> list:
    specs: list = []
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            specs.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        else:
            specs.append(provider)
    return specs


_ORT_PINNED_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prism-ort")
_T = TypeVar("_T")


def run_ort_on_pinned_thread(fn: Callable[[], _T]) -> _T:
    """Execute ONNX Runtime inference on a single pinned OS thread so CUDAExecutionProvider
    reuses one PerThreadContext/cublasHandle_t instead of leaking per-thread cuBLAS handles
    across FastAPI's worker threadpool."""
    return _ORT_PINNED_EXECUTOR.submit(fn).result()


def create_ort_session(model_file: str | Path, providers: Sequence[str] | None = None):
    import onnxruntime as ort

    resolved = tuple(providers) if providers is not None else tuple(resolve_ort_providers())
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = inference_threads()
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    try:
        return ort.InferenceSession(str(model_file), sess_options=options, providers=ort_provider_specs(resolved))
    except Exception as exc:
        if resolved != ("CPUExecutionProvider",):
            logging.getLogger("uvicorn.error").warning(
                "ORT session creation on %s failed for %s (%s); falling back to CPUExecutionProvider",
                list(resolved),
                model_file,
                exc,
            )
            return ort.InferenceSession(str(model_file), sess_options=options, providers=["CPUExecutionProvider"])
        raise


_LAMA_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _lama_model_for_device(device: str):
    import torch

    torch.set_num_threads(inference_threads())
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch._C._jit_set_profiling_mode(False)
        except Exception:
            pass
    lama_file = model_path("lama")
    logging.getLogger("uvicorn.error").info("loading native FFT LaMa (%s) on %s", lama_file.name, device)
    graph = torch.jit.load(str(lama_file), map_location=device).eval()
    return graph, device


def lama_model():
    device = resolve_torch_device()
    with _LAMA_LOAD_LOCK:
        return _lama_model_for_device(device)


def _ort_device_label(providers: Sequence[str]) -> str:
    if not providers:
        return "cpu"
    first = providers[0]
    if first == "CUDAExecutionProvider":
        return "cuda"
    if first == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def status() -> dict:
    torch_dev = resolve_torch_device()
    ort_dev = _ort_device_label(resolve_ort_providers())
    device_by_name = {
        "lama": torch_dev,
        "comic_text": ort_dev,
        "source_ocr": ort_dev,
    }
    return {
        name: {
            "available": model_path(name).is_file(),
            "path": str(model_path(name)),
            "device": device_by_name.get(name, "cpu"),
        }
        for name in ASSETS
    }


def hardware_status() -> dict:
    """Return active compute device, GPU capabilities, and thread budget telemetry."""
    configured = os.environ.get("PRISM_DEVICE", "auto").strip().lower() or "auto"
    ort_dev = _ort_device_label(resolve_ort_providers())
    cuda_available = False
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    torch_dev = "cpu"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_dev = resolve_torch_device()
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
            vram_total_mb = int(round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)))
    except Exception:
        pass
    active_device = "cuda" if ("cuda" in (ort_dev, torch_dev)) else ort_dev
    return {
        "configured_device": configured,
        "active_device": active_device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb,
        "cpu_threads": int(os.cpu_count() or 1),
        "inference_threads": int(inference_threads()),
    }


_WARMUP_LOCK = threading.Lock()
_WARMUP_STARTED = False


def _do_warmup() -> None:
    import numpy as np

    logger = logging.getLogger("uvicorn.error")
    try:
        if model_path("comic_text").is_file():
            from .learned import comic_detector

            session, lock = comic_detector()
            dummy = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
            with lock:
                session.run(["seg", "det", "blk"], {"images": dummy})
            logger.info("warmup: comic_text detector ready")
    except Exception as exc:
        logger.debug("warmup: comic_text skipped (%s)", exc)

    try:
        lama_file = model_path("lama")
        if lama_file.is_file():
            import contextlib
            import torch
            from .learned import _lama_inference_lock

            net, device = lama_model()
            dummy_img = torch.zeros((1, 3, 512, 512), dtype=torch.float32, device=device)
            dummy_mask = torch.zeros((1, 1, 512, 512), dtype=torch.float32, device=device)
            with (_lama_inference_lock if device == "cuda" else contextlib.nullcontext()):
                with torch.inference_mode():
                    _ = net(dummy_img, dummy_mask)
            logger.info("warmup: native FFT LaMa (%s) ready on %s", lama_file.name, device)
    except Exception as exc:
        logger.debug("warmup: lama skipped (%s)", exc)

    try:
        from .ocr_tps import _get_rapidocr

        _get_rapidocr()
    except Exception as exc:
        logger.debug("warmup: rapidocr skipped (%s)", exc)

    try:
        from .source_ocr import get_source_ocr, source_ocr_directory

        if (source_ocr_directory() / "onnx/mangaocr_nar_small_fp32.onnx").is_file():
            get_source_ocr()
    except Exception as exc:
        logger.debug("warmup: source_ocr skipped (%s)", exc)


def warmup_models(blocking: bool = False) -> None:
    """Preload and warmup ONNX/PyTorch kernels so first page request pays zero JIT/cuDNN overhead."""
    global _WARMUP_STARTED
    skip = os.environ.get("PRISM_T_SKIP_WARMUP", "").strip().lower()
    if skip in ("1", "true", "yes"):
        return
    with _WARMUP_LOCK:
        if _WARMUP_STARTED:
            return
        _WARMUP_STARTED = True
    if blocking:
        _do_warmup()
    else:
        t = threading.Thread(target=_do_warmup, name="prism-t-warmup", daemon=True)
        t.start()


def download(name: str) -> dict:
    target = model_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = ASSETS[name][1]
    if not target.is_file():
        partial = target.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
        partial.replace(target)
    with target.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if name in ASSET_DIGESTS and digest != ASSET_DIGESTS[name]:
        raise ValueError(f"{name}: installed model differs from the pinned publisher digest")
    provenance = {"name": name, "source": url, "sha256": digest, "bytes": target.stat().st_size}
    target.with_name(target.name + ".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install pretrained comic segmentation, source recognition and native FFT restoration assets")
    parser.add_argument("models", nargs="*", choices=list(ASSETS), default=list(ASSETS))
    args = parser.parse_args()
    with ThreadPoolExecutor() as pool:
        for result in pool.map(download, args.models):
            print(json.dumps(result), flush=True)
