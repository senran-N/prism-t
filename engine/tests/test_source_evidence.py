import cv2
import numpy as np

from prism_t_engine import source_evidence
from prism_t_engine.schemas import TextNode


def rectangle(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_rectification_preserves_native_rows_columns_and_rotation():
    yy, xx = np.indices((120, 130))
    image = np.stack((xx, yy, xx + yy), axis=2).astype(np.uint8)
    for vertical, (x, y, w, h) in [(False, (15, 30, 80, 12)), (True, (20, 15, 10, 80))]:
        crop = source_evidence.rectify_source_line(image, rectangle(x, y, w, h), vertical)
        np.testing.assert_array_equal(crop, image[y:y+h, x:x+w])
    for angle in (-27, 31):
        for vertical, size in [(False, (60, 10)), (True, (10, 60))]:
            polygon = cv2.boxPoints(((65, 60), size, angle))
            crop = source_evidence.rectify_source_line(image, polygon, vertical)
            assert abs(crop.shape[1] - size[0]) <= 1
            assert abs(crop.shape[0] - size[1]) <= 1
            assert crop[-1, 0, 1] > crop[0, 0, 1]
            assert crop[0, -1, 0] > crop[0, 0, 0]


def test_empty_and_conflicting_readings_survive_without_node_filtering(monkeypatch):
    class Manga:
        calls = 0
        def recognize(self, crop):
            self.calls += 1
            return ("違う読取", False) if self.calls == 1 else ("", True)

    class Rapid:
        def text_rec(self, crops):
            assert [crop.shape[:2] for crop in crops] == [(12, 60), (8, 55)]
            return [("別の読取", .91), ("123", .8)], 0.

    monkeypatch.setattr(source_evidence, "get_source_ocr", lambda: Manga())
    monkeypatch.setattr(source_evidence, "_get_rapidocr", lambda: Rapid())
    raw = np.full((80, 100, 3), 255, np.uint8)
    polygons = [rectangle(10, 10, 60, 12), rectangle(10, 40, 55, 8)]
    readings = source_evidence.collect_source_readings(raw, polygons, False, True)
    assert [(r["line_index"], r["recognizer"], r["text"]) for r in readings] == [
        (0, "manga_ocr_nar_small", "違う読取"), (1, "manga_ocr_nar_small", ""),
        (0, "rapidocr_ctc", "別の読取"), (1, "rapidocr_ctc", "123"),
    ]
    node = TextNode(id="small_note", source_readings=readings, recognition_pending=True)
    restored = TextNode.model_validate_json(node.model_dump_json())
    assert restored.source_readings == node.source_readings
    assert restored.recognition_pending and restored.source_text == ""
    assert restored.source_readings[1].confidence is None and restored.source_readings[1].truncated


def test_vertical_column_is_not_passed_as_sideways_glyphs_to_ctc(monkeypatch):
    class Manga:
        def recognize(self, crop):
            assert crop.shape[:2] == (60, 10)
            return "縦書き", False

    monkeypatch.setattr(source_evidence, "get_source_ocr", lambda: Manga())
    def invalid_ctc():
        raise AssertionError("horizontal CTC must not read rotated vertical glyphs")
    monkeypatch.setattr(source_evidence, "_get_rapidocr", invalid_ctc)
    readings = source_evidence.collect_source_readings(np.zeros((100, 100, 3), np.uint8), [rectangle(20, 10, 10, 60)], True, False)
    assert len(readings) == 1 and readings[0]["text"] == "縦書き"


def test_comic_ltr_vertical_fragment_does_not_invoke_manga_ocr(monkeypatch):
    def fail_mocr():
        raise AssertionError("manga_ocr_nar_small must not run on comic_ltr vertical fragments when japanese=False")

    monkeypatch.setattr(source_evidence, "get_source_ocr", fail_mocr)
    monkeypatch.setattr(source_evidence, "_get_rapidocr", lambda: None)
    readings = source_evidence.collect_source_readings(
        np.zeros((100, 100, 3), np.uint8),
        [rectangle(20, 10, 10, 60)],
        vertical=True,
        japanese=False,
        format_profile="comic_ltr",
    )
    assert readings == []
    readings_wt = source_evidence.collect_source_readings(
        np.zeros((100, 100, 3), np.uint8),
        [rectangle(20, 10, 10, 60)],
        vertical=True,
        japanese=False,
        format_profile="webtoon_vertical",
    )
    assert readings_wt == []


def test_collect_page_source_readings_batches_horizontal_ctc_calls(monkeypatch):
    calls = []

    class RapidBatch:
        def text_rec(self, crops):
            calls.append(len(crops))
            return [(f"line_{i}", 0.9) for i in range(len(crops))], 0.0

    monkeypatch.setattr(source_evidence, "_get_rapidocr", lambda: RapidBatch())
    raw = np.full((120, 200, 3), 255, np.uint8)
    node_specs = [
        ([rectangle(10, 10, 50, 12), rectangle(10, 30, 50, 12)], False, False),
        ([rectangle(80, 10, 60, 12)], False, False),
        ([rectangle(10, 60, 40, 12)], False, False),
    ]
    batched = source_evidence.collect_page_source_readings(raw, node_specs, format_profile="comic_ltr")
    assert calls == [4]
    assert [len(r) for r in batched] == [2, 1, 1]
    assert batched[0][0]["text"] == "line_0"
    assert batched[0][1]["text"] == "line_1"
    assert batched[1][0]["line_index"] == 0 and batched[1][0]["text"] == "line_2"
    assert batched[2][0]["line_index"] == 0 and batched[2][0]["text"] == "line_3"
