import numpy as np
import pytest

from prism_t_engine.deep_fusion import composite_node_layers
from prism_t_engine.restoration import restore_clean_plate_roi, fisher_g_survival
from prism_t_engine.schemas import StyleSpec
from prism_t_engine.learned import learned_clean_plate, learned_clean_plate_full_context, missing_text_polygons, text_line_polygons
from prism_t_engine.models import model_path


def test_real_restoration_cannot_expand_alpha_support():
    raw = np.full((31,31,3),240,np.uint8)
    raw[14:17,14:17] = 230
    raw[15,15] = 0
    alpha = np.zeros((31,31),np.float32)
    alpha[15,15] = 1
    clean, _, _, _ = restore_clean_plate_roi(raw,alpha)
    layer = np.zeros((31,31,4),np.uint8)
    out = composite_node_layers(raw,clean,alpha,layer,None,None,StyleSpec(),"clean_interior")
    assert np.array_equal(out[alpha == 0],raw[alpha == 0])
    assert out[15,15,0] > raw[15,15,0]


def test_fisher_tail_matches_exact_small_cases_and_multiple_testing():
    assert fisher_g_survival(.75,2) == pytest.approx(.5)
    assert fisher_g_survival(.5,3) == pytest.approx(.75)
    assert fisher_g_survival(.75,3) == pytest.approx(.1875)
    assert fisher_g_survival(.1,10) == 1
    assert fisher_g_survival(.2,1000) < fisher_g_survival(.2,100)


def test_single_connected_effect_is_a_text_candidate():
    probability = np.zeros((80,120),np.float32)
    probability[15:65,35:80] = .9
    polygons = missing_text_polygons(probability,[])
    assert len(polygons) == 1
    assert polygons[0][:,0].min() <= 35 and polygons[0][:,0].max() >= 79


def test_line_head_recovers_complete_glyphs_from_shrunken_db_contour():
    glyphs = np.zeros((80,160),np.float32)
    glyphs[25:48,20:45] = .95
    glyphs[25:48,55:80] = .95
    lines = np.zeros_like(glyphs)
    lines[30:44,22:78] = .95
    polygons = text_line_polygons(glyphs,lines,np.full_like(lines,.5))
    assert len(polygons) == 1
    assert polygons[0][:,1].min() <= 25 and polygons[0][:,1].max() >= 47


@pytest.mark.skipif(not model_path("lama").is_file(),reason="pretrained LaMa asset not installed")
def test_learned_inpainting_native_shape_and_strict_protection():
    yy,xx=np.mgrid[:96,:80]
    raw=np.stack([xx*2,yy*2,xx+yy],axis=-1).astype(np.uint8)
    mask=np.zeros((96,80),bool)
    mask[30:62,34:45]=True
    damaged=raw.copy();damaged[mask]=[255,0,255]
    out=learned_clean_plate(damaged,mask)
    changed_hidden=damaged.copy()
    changed_hidden[mask]=[0,255,0]
    alternate=learned_clean_plate(changed_hidden,mask)
    assert out.shape == raw.shape
    # Hidden source values must have no path into the prediction, including
    # reflection padding beyond this small native-resolution ROI.
    assert np.array_equal(out[mask],alternate[mask])
    assert np.array_equal(out[~mask],damaged[~mask])
    assert np.mean(np.abs(out[mask].astype(float)-raw[mask])) < np.mean(np.abs(damaged[mask].astype(float)-raw[mask]))


@pytest.mark.skipif(not model_path("lama").is_file(), reason="pretrained LaMa asset not installed")
def test_learned_clean_plate_full_context_uses_real_neighbors_and_preserves_unmasked():
    yy, xx = np.mgrid[:560, :560]
    raw = np.stack(
        [
            np.clip(40 + xx // 3, 0, 255),
            np.clip(70 + yy // 3, 0, 255),
            np.clip(110 + (xx + yy) // 6, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    mask = np.zeros((560, 560), dtype=bool)
    mask[240:320, 260:280] = True
    damaged = raw.copy()
    damaged[mask] = [255, 0, 255]
    altered = raw.copy()
    altered[mask] = [0, 255, 0]

    roi = (240, 220, 60, 120)  # (x, y, w, h)
    rx0, ry0, rw, rh = roi
    roi_mask = mask[ry0 : ry0 + rh, rx0 : rx0 + rw]

    out_a = learned_clean_plate_full_context(damaged, mask, roi=roi)
    out_b = learned_clean_plate_full_context(altered, mask, roi=roi)
    assert out_a.shape == (rh, rw, 3)
    assert out_a.dtype == np.uint8
    np.testing.assert_array_equal(out_a[roi_mask], out_b[roi_mask])
    np.testing.assert_array_equal(out_a[~roi_mask], damaged[ry0 : ry0 + rh, rx0 : rx0 + rw][~roi_mask])
    assert np.mean(np.abs(out_a[roi_mask].astype(float) - raw[ry0 : ry0 + rh, rx0 : rx0 + rw][roi_mask])) < 20.0


def test_lama_cuda_oom_falls_back_to_cpu_cleanly(monkeypatch):
    import weakref
    from prism_t_engine import learned, models

    scratch_refs = []

    class ScratchBuffer:
        pass

    class FailingCudaNet:
        def __call__(self, img_t, mask_t):
            buf = ScratchBuffer()
            scratch_refs.append(weakref.ref(buf))
            raise RuntimeError("CUDA out of memory")

    class FakeCpuNet:
        def __call__(self, img_t, mask_t):
            # Failed forward's exception frame and scratch objects must already be freed!
            assert all(ref() is None for ref in scratch_refs)
            return img_t * 0.0 + 0.5

    monkeypatch.setattr(learned, "lama_model", lambda: (FailingCudaNet(), "cuda"))
    monkeypatch.setattr(models, "_lama_model_for_device", lambda dev: (FakeCpuNet(), "cpu"))
    monkeypatch.setattr(learned, "_LAMA_FP16_SAFE", False)

    raw = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    out = learned_clean_plate(raw, mask)
    assert len(scratch_refs) == 1
    np.testing.assert_array_equal(out[~mask], raw[~mask])
    assert np.all(out[mask] == 128)
