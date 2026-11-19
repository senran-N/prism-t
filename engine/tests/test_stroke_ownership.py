import cv2
import numpy as np

from prism_t_engine.learned import refine_text_alpha


def test_touching_same_pigment_art_does_not_inherit_whole_component():
    for scale in (1, 2, 3):
        yy, xx = np.indices((120*scale, 200*scale))
        raw = np.stack((30+xx//(5*scale), 70+yy//(3*scale),
                        140+xx//(8*scale)), axis=2).astype(np.uint8)
        ink = np.zeros(yy.shape, np.uint8)
        for x in (25, 55, 85):
            ink[20*scale:80*scale, x*scale:(x+5)*scale] = 1
        observed = ink.copy()
        observed[55*scale:60*scale, 85*scale:145*scale] = 1
        observed[35*scale:100*scale, 125*scale:180*scale] = 1
        raw[observed > 0] = [210, 25, 180]
        domain = np.zeros(yy.shape, np.uint8)
        domain[10*scale:110*scale, 15*scale:190*scale] = 1
        for filled in (False, True):
            probability = domain.astype(np.float32) if filled else ink.astype(np.float32)
            for rotate in (False, True):
                image = raw.transpose(1, 0, 2) if rotate else raw
                p, d, o = [a.T if rotate else a for a in (probability, domain, observed)]
                alpha = refine_text_alpha(image, p, d, o)
                alpha = alpha.T if rotate else alpha
                assert np.all(alpha[ink > 0] > 0)
                assert not np.any(alpha[40*scale:95*scale, 140*scale:175*scale])


def test_a_wide_source_glyph_sets_its_own_width_instead_of_a_pixel_cap():
    for scale in (1, 3):
        yy, xx = np.indices((100*scale, 150*scale))
        raw = np.stack((40+xx//(5*scale), 80+yy//(4*scale),
                        130+xx//(7*scale)), axis=2).astype(np.uint8)
        ink = np.zeros(yy.shape, np.uint8)
        cv2.putText(ink, 'H', (30*scale, 80*scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 2*scale, 1, 13*scale)
        raw[ink > 0] = [210, 25, 180]
        domain = np.zeros(yy.shape, np.uint8)
        domain[10*scale:95*scale, 15*scale:110*scale] = 1
        alpha = refine_text_alpha(raw, ink.astype(np.float32), domain, ink)
        assert np.all(alpha[ink > 0] > 0)
        assert not np.any(alpha[:, 120*scale:])


def test_refine_text_alpha_captures_connected_aa_halo_without_confetti():
    rng = np.random.default_rng(11)
    bg = np.clip(np.stack([np.full((96, 120), 60), np.full((96, 120), 110), np.full((96, 120), 190)], axis=-1) + rng.normal(0, 1.0, (96, 120, 3)), 0, 255).astype(np.uint8)
    raw = bg.copy()
    core = np.zeros((96, 120), dtype=bool)
    core[25:70, 48:55] = True
    # Soft 1.5px anti-aliased glow ring around core (35% pigment mixture)
    halo = np.zeros_like(core)
    halo[24:71, 47] = True
    halo[24:71, 55] = True
    pigment = np.array([250, 245, 255], dtype=np.float64)
    raw[core] = pigment.astype(np.uint8)
    raw[halo] = np.rint(0.35 * pigment + 0.65 * bg[halo]).astype(np.uint8)

    domain = np.zeros((96, 120), dtype=np.uint8)
    domain[15:80, 35:68] = 1
    prob = np.where(core, 0.95, np.where(halo, 0.35, 0.0)).astype(np.float32)

    alpha = refine_text_alpha(raw, prob, domain, core.astype(np.uint8))
    assert np.all(alpha[core] > 0)
    # Continuous connected AA halo must be captured rather than leaving speckled gaps
    assert float(np.mean(alpha[halo] > 0)) >= 0.90
    # Distant background must stay completely untouched
    assert not np.any(alpha[:, :30])
    assert not np.any(alpha[:, 75:])
