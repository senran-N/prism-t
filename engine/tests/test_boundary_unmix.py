import numpy as np
from prism_t_engine.restoration import unmix_stroke_boundary


def test_boundary_inverse_retains_unpredicted_texture_and_exact_protected_pixels():
    for scale in (1,2):
        h,w=50*scale,70*scale
        y,x=np.indices((h,w))
        prediction=np.full((h,w,3),[65,100,130],np.uint8)
        texture=np.stack([np.where(x%2,2,-2),np.where(y%2,2,-2),np.where((x+y)%2,2,-2)],axis=2)
        truth=(prediction.astype(int)+texture).astype(np.uint8)
        core=np.zeros((h,w),bool);core[10*scale:40*scale,25*scale:31*scale]=True
        boundary=np.zeros_like(core);boundary[10*scale:40*scale,31*scale:32*scale]=True
        pigment=np.array([235,235,235])
        raw=truth.copy();raw[core]=pigment
        raw[boundary]=np.rint(.25*pigment+.75*truth[boundary]).astype(np.uint8)
        clean=raw.copy();clean[core]=prediction[core]
        unknown=core|boundary
        forbidden=np.zeros_like(core)
        out,support,opacity=unmix_stroke_boundary(raw,clean,core.astype(float),prediction,core,unknown,forbidden,0.)
        assert np.all((opacity[boundary]>0)&(opacity[boundary]<1))
        # Inversion carries raw RGB's orthogonal texture, unlike hard replacement.
        assert np.max(np.abs(out[boundary].astype(int)-truth[boundary].astype(int)))<=3
        assert np.var(out[boundary],axis=0).sum()>0
        np.testing.assert_array_equal(out[support==0],raw[support==0])
        np.testing.assert_array_equal(out[core],clean[core])


def test_boundary_inverse_does_not_acquire_foreground_neighbor_or_matching_art():
    raw=np.full((45,70,3),[40,90,130],np.uint8)
    base=raw.copy()
    core=np.zeros(raw.shape[:2],bool);core[10:35,20:26]=True
    raw[core]=240
    unknown=np.zeros_like(core);unknown[8:37,18:34]=True
    forbidden=np.zeros_like(core);forbidden[10:35,26]=True
    raw[forbidden]=np.rint(.3*240+.7*raw[forbidden]).astype(np.uint8)
    # A disconnected matching pigment and unrelated colored illustration.
    raw[12:32,30]=240
    raw[12:32,28]=[200,40,50]
    clean=raw.copy();clean[core]=base[core]
    out,support,opacity=unmix_stroke_boundary(raw,clean,core.astype(float),base,core,unknown,forbidden,0.)
    assert not opacity.any()
    np.testing.assert_array_equal(support,core)
    np.testing.assert_array_equal(out,clean)


def test_boundary_ycbcr_chroma_subsampling_unmix_recovers_sky_and_rejects_unrelated_art():
    # Simulates JPEG 4:2:0 chroma bleeding from a white outline onto deep blue sky:
    # luma stays near sky while Cb desaturates toward white (pulling B down from 90 to 56),
    # which falls off the coupled 1D RGB segment but obeys decoupled YCbCr physics.
    h, w = 48, 64
    base = np.full((h, w, 3), [13, 36, 90], dtype=np.uint8)
    core = np.zeros((h, w), bool)
    core[10:38, 22:28] = True
    bleed = np.zeros((h, w), bool)
    bleed[10:38, 28:29] = True
    unrelated = np.zeros((h, w), bool)
    unrelated[10:38, 21:22] = True  # directly touching left edge of core, opposite red hue!

    raw = base.copy()
    raw[core] = [244, 253, 254]
    raw[bleed] = [20, 44, 56]
    raw[unrelated] = [200, 40, 50]
    clean = raw.copy()
    clean[core] = base[core]
    unknown = core | bleed | unrelated
    forbidden = np.zeros_like(core)

    out, support, opacity = unmix_stroke_boundary(raw, clean, core.astype(float), base, core, unknown, forbidden, 0.0)
    assert np.all(opacity[bleed] > 0)
    assert np.max(np.abs(out[bleed].astype(int) - base[bleed].astype(int))) <= 4
    # Unrelated red artwork touching core must remain strictly untouched (L_inf == 0)
    assert not np.any(opacity[unrelated] > 0)
    assert not np.any(support[unrelated] > 0)
    np.testing.assert_array_equal(out[unrelated], raw[unrelated])
    np.testing.assert_array_equal(out[support == 0], raw[support == 0])
