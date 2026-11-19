import numpy as np
import pytest

from prism_t_engine.preprocess import estimate_substrate_noise
from prism_t_engine import restoration, learned


def test_noise_calibration_cannot_observe_hidden_pixels():
    rng=np.random.default_rng(53)
    raw=np.clip(180+rng.normal(0,1,(96,120,3)),0,255).astype(np.uint8)
    known=np.ones(raw.shape[:2],bool)
    known[15:80,25:95]=False
    changed=raw.copy()
    changed[~known]=rng.integers(0,256,(int((~known).sum()),3),dtype=np.uint8)
    assert np.array_equal(estimate_substrate_noise(raw,known),estimate_substrate_noise(changed,known))


def test_visible_edge_error_rejects_a_lower_rgb_error_blurry_candidate(monkeypatch):
    # Controlled operators isolate the observable selection criterion.
    def smooth(image,mask):
        out=image.copy();out[mask]=139
        return out,0.
    def oscillating(image,mask):
        out=image.astype(float).copy()
        y,x=np.indices(mask.shape)
        pattern=128+10*np.sin(y*np.pi/2)
        out[mask]=np.repeat(pattern[:,:,None],3,axis=2)[mask]
        return out
    monkeypatch.setattr(restoration,'solve_anisotropic_tensor_inpaint',smooth)
    monkeypatch.setattr(restoration,'solve_laplace_dirichlet',oscillating)
    monkeypatch.setattr(restoration,'solve_biharmonic_slit',oscillating)
    monkeypatch.setattr(learned,'learned_clean_plate',oscillating)
    raw=np.full((100,120,3),128,np.uint8)
    active=np.zeros(raw.shape[:2],bool);active[20:80,30:90]=True
    raw[active]=0
    holdout=restoration.validation_collar(active,~active)
    validation=active|holdout
    assert np.mean((oscillating(raw,validation)[holdout]-raw[holdout])**2)<11**2
    result,path,residual, _ = restoration.select_context_restoration(raw,active,~active)
    assert path=='structure_tensor_line'
    assert np.array_equal(result[~active],raw[~active])
    assert residual>0


def test_real_restoration_does_not_read_erased_rgb_or_touch_known_pixels():
    y,x=np.mgrid[:60,:90]
    raw=np.stack([110+x,150+y,90+x//2],axis=2).astype(np.uint8)
    active=np.zeros(raw.shape[:2],bool);active[20:40,25:65]=True
    first=raw.copy();first[active]=0
    second=raw.copy();second[active]=255
    a,_,_, _ = restoration.select_context_restoration(first,active,~active)
    b,_,_, _ = restoration.select_context_restoration(second,active,~active)
    assert np.array_equal(a[active],b[active])
    assert np.array_equal(a[~active],first[~active])


def test_neighbor_text_is_hidden_during_validation_and_actual_reconstruction(monkeypatch):
    y, x = np.indices((55, 85))
    raw = np.stack([70+x, 90+y, 130+x//2], axis=2).astype(np.uint8)
    active = np.zeros(raw.shape[:2], bool); active[15:40, 20:28] = True
    neighbor = np.zeros_like(active); neighbor[10:45, 30:38] = True
    calls = []

    def reconstruct(image, mask):
        assert np.all(mask[neighbor])
        calls.append(mask.copy())
        return restoration.solve_laplace_dirichlet(image, mask)

    monkeypatch.setattr(learned, 'learned_clean_plate', reconstruct)
    first = raw.copy(); first[neighbor] = [255, 0, 190]
    second = raw.copy(); second[neighbor] = 0
    a, _, _, _ = restoration.select_context_restoration(first, active, ~active, neighbor)
    b, _, _, _ = restoration.select_context_restoration(second, active, ~active, neighbor)
    assert len(calls) >= 2
    np.testing.assert_array_equal(a[active], b[active])
    np.testing.assert_array_equal(a[~active], first[~active])
    np.testing.assert_array_equal(b[~active], second[~active])


def test_neighbor_exclusion_never_grants_permission_to_write_neighbor_pixels():
    raw = np.full((45, 65, 3), [220, 230, 240], np.uint8)
    active = np.zeros(raw.shape[:2], bool); active[12:33, 20:25] = True
    neighbor = np.zeros_like(active); neighbor[10:35, 27:32] = True
    raw[active] = 0; raw[neighbor] = [190, 0, 160]
    result, _, _, _ = restoration.restore_clean_plate_roi(
        raw, active.astype(np.float32), context_exclusion=neighbor)
    np.testing.assert_array_equal(result[active], np.tile([220, 230, 240], (active.sum(), 1)))
    np.testing.assert_array_equal(result[~active], raw[~active])


def test_spectral_decision_and_reconstruction_cannot_read_hidden_lettering():
    y, x = np.indices((96, 112))
    wave = np.cos(2*np.pi*(x+y)/9)*np.cos(2*np.pi*(x-y)/9)
    raw = np.repeat(np.where(wave > .25, 35, 242)[:, :, None], 3, axis=2).astype(np.uint8)
    unknown = np.zeros(raw.shape[:2], bool)
    unknown[15:80, 28:34] = True
    unknown[20:75, 38:44] = True
    black, white = raw.copy(), raw.copy()
    black[unknown], white[unknown] = 0, 255
    first_ok, first = restoration.detect_and_synthesize_halftone_fft(black, unknown)
    second_ok, second = restoration.detect_and_synthesize_halftone_fft(white, unknown)
    assert first_ok and second_ok
    np.testing.assert_array_equal(first[unknown], second[unknown])


def test_directional_solver_preserves_constant_color_without_diagonal_bias():
    for scale in (1,2):
        raw=np.full((35*scale,45*scale,3),[83,147,219],np.uint8)
        mask=np.zeros(raw.shape[:2],bool);mask[10*scale:25*scale,15*scale:30*scale]=True
        raw[mask]=0
        result,_=restoration.solve_anisotropic_tensor_inpaint(raw,mask)
        assert np.allclose(result[mask],[83,147,219],atol=1e-8,rtol=0)
        assert np.array_equal(result[~mask],raw[~mask])


@pytest.mark.parametrize("solver", [
    restoration.solve_laplace_dirichlet,
    restoration.solve_biharmonic_slit,
    lambda raw,mask: restoration.solve_anisotropic_tensor_inpaint(raw,mask)[0],
])
def test_boundary_holes_cannot_observe_hidden_rgb(solver):
    y,x=np.mgrid[:40,:54]
    raw=np.stack([70+x+12*np.sin(y/3),100+y+8*np.cos(x/4),80+(x+y)/2],axis=-1).astype(np.uint8)
    masks=[]
    for rows,columns in [(slice(0,14),slice(0,17)),(slice(0,13),slice(20,38)),
                         (slice(24,40),slice(35,54)),(slice(11,29),slice(0,13)),
                         (slice(15,32),slice(42,54))]:
        mask=np.zeros(raw.shape[:2],bool); mask[rows,columns]=True; masks.append(mask)
    for mask in masks:
        black=raw.copy(); black[mask]=0
        white=raw.copy(); white[mask]=255
        first=solver(black,mask); second=solver(white,mask)
        np.testing.assert_array_equal(first[mask],second[mask])
        np.testing.assert_array_equal(first[~mask],raw[~mask])
        assert np.isfinite(first).all()


@pytest.mark.parametrize("shape",[(1,11),(13,1),(19,23)])
def test_harmonic_and_biharmonic_edge_reconstruct_constant_without_color_shrinkage(shape):
    mask=np.ones(shape,bool); mask.flat[-1]=False
    for solver in (restoration.solve_laplace_dirichlet,restoration.solve_biharmonic_slit):
        raw=np.zeros(shape+(3,),float); raw[~mask]=[83.25,147.5,219.75]
        result=solver(raw,mask)
        np.testing.assert_allclose(result[mask],np.tile([83.25,147.5,219.75],(mask.sum(),1)),atol=1e-7,rtol=0)
        np.testing.assert_array_equal(result[~mask],raw[~mask])


def test_directional_reconstruction_rotates_with_observed_image():
    y,x=np.mgrid[:40,:54]
    raw=np.stack([90+x+15*np.sin((x+y)/4),130+y,90+x/2],axis=-1).astype(np.uint8)
    mask=np.zeros(raw.shape[:2],bool); mask[:14,15:38]=True; raw[mask]=0
    first,_=restoration.solve_anisotropic_tensor_inpaint(raw,mask)
    rotated,_=restoration.solve_anisotropic_tensor_inpaint(np.rot90(raw),np.rot90(mask))
    np.testing.assert_allclose(first,np.rot90(rotated,-1),atol=1e-7,rtol=0)


def test_pde_without_observable_boundary_does_not_return_source_as_repaired():
    mask=np.ones((7,9),bool); raw=np.zeros((7,9,3),np.uint8)
    for solver in (restoration.solve_laplace_dirichlet,restoration.solve_biharmonic_slit,restoration.solve_anisotropic_tensor_inpaint):
        with pytest.raises(ValueError,match="observable Dirichlet"):
            solver(raw,mask)
    with pytest.raises(ValueError,match="no observable background"):
        restoration.restore_clean_plate_roi(raw,mask.astype(np.float32))


def test_quarantined_outline_cannot_recolor_core_or_grant_write_permission():
    for scale in (1, 2):
        raw = np.full((60*scale,80*scale,3),[35,65,125],np.uint8)
        active = np.zeros(raw.shape[:2],bool)
        active[15*scale:45*scale,35*scale:41*scale] = True
        unknown = restoration.reconstruction_unknown(active,active,np.zeros_like(active))
        edge = unknown & ~active
        raw[active] = [201,17,194]
        raw[edge] = [250,248,253]
        contaminated,_,_, _ = restoration.restore_clean_plate_roi(raw,active.astype(float))
        first,path_a,_, _ = restoration.restore_clean_plate_roi(raw,active.astype(float),context_exclusion=unknown)
        altered = raw.copy();altered[edge] = [0,0,0]
        second,path_b,_, _ = restoration.restore_clean_plate_roi(altered,active.astype(float),context_exclusion=unknown)
        assert np.mean(np.abs(contaminated[active].astype(float)-[35,65,125])) > 20
        np.testing.assert_array_equal(first[active],np.tile([35,65,125],(active.sum(),1)))
        np.testing.assert_array_equal(first[active],second[active])
        assert path_a == path_b
        np.testing.assert_array_equal(first[~active],raw[~active])
        np.testing.assert_array_equal(second[~active],altered[~active])


def test_paper_residue_audit_ignores_quarantined_pigment_as_substrate():
    raw = np.full((70,90,3),[220,230,240],np.uint8)
    observed = np.zeros(raw.shape[:2],bool);observed[20:50,30:36] = True
    unknown = restoration.reconstruction_unknown(observed,observed,np.zeros_like(observed))
    raw[observed] = 0
    raw[unknown & ~observed] = [255,0,190]
    clean = raw.copy()
    result,alpha,_,_,count = restoration.repair_observed_paper_residue(
        raw,clean,observed.astype(float),observed,np.ones(observed.shape,np.uint8),
        context_exclusion=unknown,
    )
    assert count == observed.sum()
    np.testing.assert_array_equal(result[observed],np.tile([220,230,240],(observed.sum(),1)))
    np.testing.assert_array_equal(result[alpha==0],raw[alpha==0])


def test_smooth_shaded_enclosed_bubble_skips_learned_and_falls_back_cleanly(monkeypatch):
    yy, xx = np.mgrid[0:64, 0:80]
    substrate = np.zeros((64, 80), dtype=np.uint8)
    substrate[6:58, 6:74] = 1
    base = np.clip(180.0 + 0.7 * yy + 0.5 * xx, 0, 255).astype(np.uint8)
    raw = np.repeat(base[:, :, None], 3, axis=2)
    raw[substrate == 0] = [20, 20, 20]
    active = np.zeros((64, 80), dtype=bool)
    active[26:36, 30:36] = True
    raw[active] = [15, 15, 15]

    calls = []
    from prism_t_engine import learned

    def fake_learned(rgb, mask):
        calls.append(int(np.count_nonzero(mask)))
        return rgb.copy()

    monkeypatch.setattr(learned, "learned_clean_plate", fake_learned)

    clean_sub, path_sub, _, _ = restoration.restore_clean_plate_roi(
        raw, active.astype(np.float32), substrate_mask=substrate
    )
    assert calls == []
    assert path_sub == "solid_fill"
    np.testing.assert_array_equal(clean_sub[~active], raw[~active])

    _, path_sel, _, _ = restoration.select_context_restoration(
        raw, active, substrate > 0, allow_learned=False
    )
    assert calls == []
    assert path_sel in {"solid_fill", "structure_tensor_line", "narrow_slit_pde"}

    restoration.select_context_restoration(
        raw, active, np.ones_like(active), allow_learned=True
    )
    assert len(calls) >= 1


def test_paper_residue_repair_clears_aged_paper_fringe():
    rng = np.random.default_rng(7)
    base = np.clip(225.0 + rng.normal(0.0, 2.0, (70, 90, 3)), 0, 255).astype(np.uint8)
    raw = base.copy()
    observed = np.zeros((70, 90), dtype=bool)
    observed[20:45, 30:35] = True
    # Faint 1px anti-aliased fringe at ~12 units below paper median (inside source_domain)
    fringe = np.zeros_like(observed)
    fringe[20:45, 29] = True
    fringe[20:45, 35] = True
    raw[observed] = [20, 20, 20]
    raw[fringe] = np.clip(base[fringe].astype(np.int16) - 12, 0, 255).astype(np.uint8)

    clean = raw.copy()
    clean[observed] = 225
    text_domain = np.zeros((70, 90), dtype=np.uint8)
    text_domain[18:47, 28:37] = 1
    substrate = np.ones((70, 90), dtype=np.uint8)

    result, alpha, _, _, repaired = restoration.repair_observed_paper_residue(
        raw, clean, observed.astype(np.float32), observed, substrate,
        source_domain=text_domain,
    )
    assert repaired >= int(fringe.sum())
    assert np.all(alpha[fringe] > 0)
    assert np.max(np.abs(result[fringe].astype(np.int16) - 225)) <= 2
    np.testing.assert_array_equal(result[alpha == 0], raw[alpha == 0])
