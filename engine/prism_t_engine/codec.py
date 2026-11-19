from __future__ import annotations

import base64
import io
import cv2
import numpy as np
from PIL import Image


def decode_image_b64(b64_str: str, mode: str = "RGB") -> np.ndarray:
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as im:
        converted = im.convert(mode)
        return np.array(converted, dtype=np.uint8)


def encode_png_b64(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        encoded_in = np.ascontiguousarray(arr)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        encoded_in = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        encoded_in = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"Unsupported channel shape: {arr.shape}")
    ok, buf = cv2.imencode(".png", encoded_in, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
