from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from prism_t_engine.source_ocr import get_source_ocr, prepare_source_patches


def test_model_preprocessing_matches_the_pinned_author_implementation():
    reference = Path(__file__).with_name("fixtures") / "source_ocr_author_preprocess.py"
    spec = importlib.util.spec_from_file_location("author_source_preprocessing", reference)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    for height, width in ((545, 36), (36, 545), (224, 224), (17, 18), (1, 100), (100, 1)):
        yy, xx = np.indices((height, width))
        image = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=2).astype(np.uint8)
        expected = author.empty_inputs()
        author.preprocess(Image.fromarray(image), expected)
        actual = prepare_source_patches(image, author.BUDGET, author.PATCH)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{height}x{width}, {key}")


def test_source_preprocessing_is_scale_covariant_and_has_no_shared_buffers():
    image = np.full((30, 180, 3), [13, 57, 109], np.uint8)
    scaled = np.repeat(np.repeat(image, 3, axis=0), 3, axis=1)
    a, b = prepare_source_patches(image, 324, 14), prepare_source_patches(scaled, 324, 14)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    inputs = [np.full((30 + i, 180 - i, 3), i * 20, np.uint8) for i in range(8)]
    expected = [prepare_source_patches(raw, 324, 14) for raw in inputs]
    with ThreadPoolExecutor(4) as executor:
        concurrent = list(executor.map(lambda raw: prepare_source_patches(raw, 324, 14), inputs))
    for first, second in zip(expected, concurrent):
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_real_source_model_decodes_its_actual_contract_and_reports_output_limit():
    model = get_source_ocr()
    inputs = [np.full((40, 160, 3), fill, np.uint8) for fill in (0, 127, 255)]
    sequential = [model.recognize(raw) for raw in inputs]
    with ThreadPoolExecutor(3) as executor:
        concurrent = list(executor.map(model.recognize, inputs))
    assert concurrent == sequential
    assert all(isinstance(text, str) and isinstance(truncated, bool) for text, truncated in sequential)
