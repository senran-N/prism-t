import numpy as np
import cv2
from PIL import ImageFont
from concurrent.futures import ThreadPoolExecutor
import time

from prism_t_engine.schemas import StyleSpec
from prism_t_engine.ocr_tps import observed_background_contrast, observed_line_is_vertical, fuse_text_line_geometry, split_disconnected_text_tracks
from prism_t_engine.deep_fusion import extract_stroke_style_and_halo, unmix_optical_alpha_layer
from prism_t_engine.typesetter_sdf import (
    _font_coverage,
    build_interior_sdf,
    estimate_char_budget_from_sdf,
    resolve_font_path,
    load_font,
    solve_sdf_knuth_plass_layout,
)


def test_terminal_geometry_recovers_weak_curved_ink_at_both_ends_and_orientations():
    from prism_t_engine.ocr_tps import complete_text_track_endpoints
    for scale in (1,2):
        yy,xx=np.indices((120*scale,140*scale))
        background=np.stack((25+xx//(8*scale),70+yy//(6*scale),140+xx//(9*scale)),axis=2).astype(np.uint8)
        ink=np.zeros(yy.shape,np.uint8)
        cv2.line(ink,(62*scale,32*scale),(62*scale,83*scale),1,3*scale)
        punctuation=np.zeros_like(ink)
        cv2.ellipse(punctuation,(62*scale,26*scale),(8*scale,8*scale),0,190,350,1,2*scale)
        cv2.ellipse(punctuation,(62*scale,90*scale),(8*scale,8*scale),0,10,170,1,2*scale)
        raw=background.copy();raw[(ink|punctuation)>0]=[210,20,180]
        probability=ink.astype(np.float32);probability[punctuation>0]=.05
        track=np.array([[48,24],[76,24],[76,92],[48,92]],np.float32)*scale
        for rotate in (False,True):
            rgb,p,polygon=raw,probability,track
            if rotate:
                rgb,p=raw.transpose(1,0,2),probability.T
                polygon=track[:,::-1].copy()
            result=complete_text_track_endpoints(rgb,[polygon],p)[0]
            mask=np.zeros(p.shape,np.uint8);cv2.fillPoly(mask,[np.rint(result).astype(np.int32)],1)
            expected=(ink|punctuation).T if rotate else ink|punctuation
            assert np.all(mask[expected>0])
            assert cv2.contourArea(result)>=cv2.contourArea(polygon)


def test_terminal_geometry_does_not_follow_a_broad_connected_artwork_patch():
    from prism_t_engine.ocr_tps import complete_text_track_endpoints
    raw=np.full((100,120,3),[25,70,140],np.uint8)
    ink=np.zeros(raw.shape[:2],np.uint8);ink[20:65,52:57]=1
    raw[ink>0]=[210,20,180]
    raw[65:90,40:70]=[210,20,180]
    polygon=np.array([[40,15],[70,15],[70,65],[40,65]],np.float32)
    result=complete_text_track_endpoints(raw,[polygon],ink.astype(np.float32))[0]
    assert np.allclose(result,polygon)


def test_terminal_geometry_keeps_crossing_thin_artwork_out_of_a_line_crop():
    from prism_t_engine.ocr_tps import complete_text_track_endpoints
    for scale in (1,2):
        raw=np.full((130*scale,150*scale,3),[25,70,140],np.uint8)
        ink=np.zeros(raw.shape[:2],np.uint8)
        cv2.line(ink,(70*scale,35*scale),(70*scale,85*scale),1,3*scale)
        raw[ink>0]=[210,20,180]
        # A frame diagonal crosses the end but continues well beyond the
        # track's sides. Clipping to the track first conceals that evidence.
        cv2.line(raw,(10*scale,50*scale),(130*scale,130*scale),(210,20,180),2*scale)
        polygon=np.array([[55,25],[85,25],[85,95],[55,95]],np.float32)*scale
        result=complete_text_track_endpoints(raw,[polygon],ink.astype(np.float32))[0]
        assert np.allclose(result,polygon)


def test_endpoint_completion_does_not_reassign_paragraph_membership(monkeypatch):
    from prism_t_engine import ocr_tps
    raw=np.full((110,160,3),245,np.uint8)
    probability=np.zeros(raw.shape[:2],np.float32)
    polygons=[np.array(p,np.float32) for p in
              ([[20,20],[70,20],[70,40],[20,40]],[[75,60],[125,60],[125,80],[75,80]])]
    cv2.putText(raw,'AB',(24,35),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
    cv2.putText(raw,'CD',(80,76),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
    probability[np.min(raw,axis=2)<100]=1
    # Keep the independently detected tracks unchanged for this ownership
    # check; the production completion itself is exercised above.
    monkeypatch.setattr(ocr_tps,'_get_rapidocr',lambda:None)
    monkeypatch.setattr(ocr_tps,'split_disconnected_text_tracks',lambda tracks,p:tracks)
    import prism_t_engine.learned as learned
    monkeypatch.setattr(learned,'missing_text_polygons',lambda *args,**kwargs:[])
    completed=[p.copy() for p in polygons]
    completed[0][[1,2],0]=85
    completed[1][[0,3],0]=60
    monkeypatch.setattr(ocr_tps,'complete_text_track_endpoints',lambda *args:completed)
    result=ocr_tps.detect_and_recognize_page_regions(raw,False,probability,False,polygons,[])
    assert len(result)==2
    assert all(len(r['detector_polygons'])==len(r['line_polygons'])==1 for r in result)


def test_terminal_curve_survives_a_narrow_connection_to_broad_same_color_artwork():
    from prism_t_engine.ocr_tps import complete_text_track_endpoints
    for scale in (1,2):
        yy,xx=np.indices((130*scale,150*scale))
        background=np.stack((25+xx//(8*scale),70+yy//(6*scale),140+xx//(9*scale)),axis=2).astype(np.uint8)
        ink=np.zeros(yy.shape,np.uint8)
        cv2.line(ink,(62*scale,32*scale),(62*scale,83*scale),1,3*scale)
        curve=np.zeros_like(ink)
        cv2.ellipse(curve,(62*scale,90*scale),(8*scale,8*scale),0,10,170,1,2*scale)
        raw=background.copy();raw[(ink|curve)>0]=[210,20,180]
        cv2.rectangle(raw,(78*scale,84*scale),(125*scale,115*scale),(210,20,180),-1)
        cv2.line(raw,(69*scale,94*scale),(83*scale,94*scale),(210,20,180),2*scale)
        probability=ink.astype(np.float32);probability[curve>0]=.05
        polygon=np.array([[48,24],[76,24],[76,92],[48,92]],np.float32)*scale
        for rotate in (False,True):
            rgb,p,track=raw,probability,polygon
            if rotate:
                rgb,p,track=raw.transpose(1,0,2),probability.T,polygon[:,::-1].copy()
            result=complete_text_track_endpoints(rgb,[track],p)[0]
            mask=np.zeros(p.shape,np.uint8);cv2.fillPoly(mask,[np.rint(result).astype(np.int32)],1)
            expected=(ink|curve).T if rotate else ink|curve
            assert np.all(mask[expected>0])
            axis=0 if rotate else 1
            assert result[:,axis].max()<=np.nonzero(expected)[1-axis].max()+1


def test_detached_terminal_arc_does_not_need_a_pixel_inside_the_clipped_track():
    from prism_t_engine.ocr_tps import complete_text_track_endpoints
    for scale in (1,2):
        raw=np.full((115*scale,130*scale,3),[25,70,140],np.uint8)
        ink=np.zeros(raw.shape[:2],np.uint8)
        cv2.line(ink,(62*scale,30*scale),(62*scale,86*scale),1,3*scale)
        curve=np.zeros_like(ink)
        cv2.ellipse(curve,(62*scale,89*scale),(8*scale,8*scale),0,10,170,1,2*scale)
        raw[(ink|curve)>0]=[210,20,180]
        probability=ink.astype(np.float32)
        polygon=np.array([[48,24],[76,24],[76,89],[48,89]],np.float32)*scale
        for rotate in (False,True):
            rgb,p,track=raw,probability,polygon
            if rotate:
                rgb,p,track=raw.transpose(1,0,2),probability.T,polygon[:,::-1].copy()
            result=complete_text_track_endpoints(rgb,[track],p)[0]
            mask=np.zeros(p.shape,np.uint8);cv2.fillPoly(mask,[np.rint(result).astype(np.int32)],1)
            expected=(ink|curve).T if rotate else ink|curve
            assert np.all(mask[expected>0])


def test_page_has_one_final_refinement_and_persists_its_actual_evidence(monkeypatch):
    from prism_t_engine import learned,ocr_tps,pipeline
    from prism_t_engine.codec import encode_png_b64,decode_image_b64
    from prism_t_engine.schemas import AnalyzeRequest
    raw=np.full((90,180,3),245,np.uint8)
    ink=np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink,'INK',(25,60),cv2.FONT_HERSHEY_SIMPLEX,1.2,1,3)
    raw[ink>0]=20
    polygon=np.array([[20,20],[100,20],[100,65],[20,65]],np.float32)
    class Detector:
        def text_det(self,image): return [polygon],0.
    monkeypatch.setattr(ocr_tps,'_get_rapidocr',lambda:Detector())
    monkeypatch.setattr(pipeline,'comic_text_evidence',lambda rgb:(ink.astype(np.float32),[polygon],[]))
    monkeypatch.setattr(pipeline,'collect_source_readings',lambda *args:[])
    def rejected_intermediate(*args,**kwargs):
        raise AssertionError('OCR geometry cannot refine the final erasure support')
    monkeypatch.setattr(learned,'refine_text_alpha',rejected_intermediate)
    final=pipeline.refine_text_alpha
    calls=[]
    def capture(rgb,probability,seed,observed_ink=None):
        result=final(rgb,probability,seed,observed_ink)
        calls.append((seed.copy(),observed_ink.copy(),result.copy()))
        return result
    monkeypatch.setattr(pipeline,'refine_text_alpha',capture)
    response=pipeline.analyze_and_restore_page(AnalyzeRequest(image_png_b64=encode_png_b64(raw),recognize_text=False))
    assert len(calls)==len(response.dom.nodes)==1
    node=response.dom.nodes[0]
    domain,observed,refined=calls[0]
    for stored,expected in [(node.patches.source_domain_png_b64,domain),
                            (node.patches.source_observed_ink_png_b64,observed),
                            (node.patches.refined_stroke_png_b64,refined)]:
        actual=decode_image_b64(stored,mode='L')
        np.testing.assert_array_equal(actual>0,expected>0)
    assert len(node.detector_quadrilaterals)==len(node.source_quadrilaterals)


def test_shared_rapidocr_preprocessor_does_not_cross_page_boundaries():
    from prism_t_engine.ocr_tps import _rapid_detect, _rapid_recognize
    class StatefulDetector:
        def text_det(self,image):
            self.preprocessor=image.shape
            time.sleep(.01)
            return self.preprocessor
        def __call__(self,image):
            return self.text_det(image)
    detector=StatefulDetector()
    inputs=[np.zeros((20+i,30+2*i,3),np.uint8) for i in range(8)]
    def infer(item):
        i,image=item
        return (_rapid_detect if i%2 else _rapid_recognize)(detector,image)
    with ThreadPoolExecutor(4) as executor:
        results=list(executor.map(infer,enumerate(inputs)))
    assert results==[image.shape for image in inputs]


def test_small_punctuation_cannot_change_the_measured_row_direction():
    def box(x,y,w,h):return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],float)
    assert not observed_line_is_vertical([box(0,0,300,18)]+[box(i*10,0,3,8) for i in range(8)],True)
    assert observed_line_is_vertical([box(0,0,18,300)]+[box(0,i*10,8,3) for i in range(8)],False)
    assert observed_line_is_vertical([box(0,0,10,10)],True)
    assert not observed_line_is_vertical([box(0,0,10,10)],False)


def test_independent_rows_split_a_merged_paragraph_and_keep_unmatched_text():
    def box(x,y,w,h):return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
    for scale in (1,2):
        primary=[box(10,10,280,100)*scale,box(30,131,5,7)*scale,box(40,132,200,12)*scale,box(300,150,25,40)*scale]
        native=[box(12,12,275,35)*scale,box(12,65,275,40)*scale,box(29,127,230,20)*scale]
        fused=fuse_text_line_geometry(primary,native)
        assert len(fused)==4
        # The title remains two distinct large rows; its small caption does
        # not appear again as an independently owned punctuation fragment.
        assert max(np.ptp(p[:,1]) for p in fused[:2]) <= 40*scale
        assert np.min(fused[2][:,0]) <= 30*scale
        assert np.array_equal(fused[-1],cv2.convexHull(primary[-1]).reshape(-1,2))


def test_either_detector_can_merge_neighboring_tracks_without_destroying_fine_geometry():
    def box(x,y,w,h):return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
    for scale in (1,3):
        for angle in (0,23):
            affine=cv2.getRotationMatrix2D((0,0),angle,scale)
            transform=lambda p: cv2.transform(p[None],affine)[0]
            coarse=[transform(box(8,8,50,240))]
            fine=[transform(box(10,10,19,234)),transform(box(35,10,20,100))]
            first=fuse_text_line_geometry(coarse,fine)
            second=fuse_text_line_geometry(fine,coarse)
            assert len(first)==len(second)==2
            for polygon in fine:
                center=polygon.mean(axis=0)
                assert sum(cv2.pointPolygonTest(p,tuple(center.astype(float)),False)>=0 for p in first)==1
                assert sum(cv2.pointPolygonTest(p,tuple(center.astype(float)),False)>=0 for p in second)==1


def test_inline_fragments_are_not_mistaken_for_parallel_neighboring_tracks():
    def box(x,y,w,h):return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
    whole=[box(10,10,20,200)]
    fragments=[box(11,12,18,80),box(11,105,18,100)]
    fused=fuse_text_line_geometry(fragments,whole)
    assert len(fused)==1
    np.testing.assert_allclose(fused[0].min(axis=0),[10,10])
    np.testing.assert_allclose(fused[0].max(axis=0),[30,210])


def test_refining_parallel_tracks_keeps_weak_edges_and_detached_punctuation():
    def box(x,y,w,h):return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.float32)
    for scale in (1,3):
        for transpose in (False,True):
            probability=np.zeros((130*scale,90*scale),np.float32)
            probability[15*scale:110*scale,25*scale:30*scale]=1
            probability[15*scale:110*scale,55*scale:60*scale]=1
            # Curved glyph edges and disconnected punctuation extend beyond
            # the fine columns but stay inside the coarse detector's support.
            probability[20*scale:25*scale,18*scale:25*scale]=.7
            probability[112*scale:115*scale,61*scale:64*scale]=.7
            fine=[box(23,12,9,100)*scale,box(53,12,9,100)*scale]
            coarse=[box(12,8,60,115)*scale]
            if transpose:
                probability=probability.T.copy()
                fine=[p[:,::-1].copy() for p in fine]
                coarse=[p[:,::-1].copy() for p in coarse]
            for a,b in ((coarse,fine),(fine,coarse)):
                fused=fuse_text_line_geometry(a,b,probability)
                assert len(fused)==2
                assert all(p.shape==(4,2) for p in fused)
                coverage=np.zeros(probability.shape,np.uint8)
                for p in fused:cv2.fillPoly(coverage,[np.rint(p).astype(np.int32)],1)
                assert np.all(coverage[probability>0])
                # The blank coarse margin does not expand back into a paragraph.
                assert sum(cv2.contourArea(p) for p in fused)<cv2.contourArea(coarse[0])
                for p in fine:
                    center=tuple(p.mean(axis=0).astype(float))
                    assert sum(cv2.pointPolygonTest(q,center,False)>=0 for q in fused)==1


def test_track_whitespace_split_partitions_geometry_without_losing_any_pixel():
    for scale in (1,3):
        probability=np.zeros((120*scale,50*scale),np.float32)
        for start in (10,22,34,80,92):
            probability[start*scale:(start+8)*scale,15*scale:25*scale]=1
        polygon=np.array([[12,5],[28,5],[28,110],[12,110]],np.float32)*scale
        for transpose in (False,True):
            native=probability.T if transpose else probability
            p=polygon[:,::-1].copy() if transpose else polygon
            parts=split_disconnected_text_tracks([p],native)
            assert len(parts)==2
            original=np.zeros(native.shape,np.uint8); coverage=np.zeros_like(original)
            cv2.fillPoly(original,[np.rint(p).astype(np.int32)],1)
            for piece in parts:cv2.fillPoly(coverage,[np.rint(piece).astype(np.int32)],1)
            np.testing.assert_array_equal(coverage,original)
        contiguous=probability.copy()
        contiguous[42*scale:80*scale,15*scale:25*scale]=1
        assert len(split_disconnected_text_tracks([polygon],contiguous))==1


def test_background_contrast_reaches_the_interior_of_a_large_title():
    for scale in (1,2):
        h,w=100*scale,220*scale
        yy,xx=np.indices((h,w))
        background=np.stack((30+xx//8,60+yy//8,130+xx//12),axis=2).astype(np.uint8)
        raw=background.copy()
        strokes=np.zeros((h,w),np.uint8)
        cv2.putText(strokes,"HI",(65*scale,72*scale),cv2.FONT_HERSHEY_SIMPLEX,2*scale,1,5*scale)
        raw[strokes>0]=250
        known=np.zeros((h,w),bool)
        known[[0,-1],:]=True;known[:,[0,-1]]=True
        contrast=observed_background_contrast(raw,known)
        assert np.median(contrast[strokes>0])>np.max(contrast[known])
        assert np.quantile(contrast[strokes>0],.05)>np.quantile(contrast[strokes==0],.95)


def test_erasure_collar_is_not_evidence_of_transmitted_light():
    yy,xx=np.indices((100,280))
    raw=np.stack((50+xx//3,100+yy//2,180+xx//5),axis=2).astype(np.uint8)
    ink=np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink,"TIME",(35,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,1,3)
    raw[ink>0]=255
    erasure=cv2.dilate(ink,np.ones((15,15),np.uint8)).astype(float)
    is_semi,*_=unmix_optical_alpha_layer(raw,erasure,observed_ink=ink)
    assert not is_semi


def test_optical_inverse_rejects_a_pigment_that_requires_out_of_gamut_background():
    for scale in (1,2):
        yy,xx=np.indices((100*scale,260*scale))
        raw=np.stack((40+xx/(3*scale),40+yy/scale,80+xx/(2*scale)),axis=2).clip(0,255).astype(np.uint8)
        glyph=np.zeros(raw.shape[:2],np.uint8)
        cv2.putText(glyph,"OPAQUE",(10*scale,70*scale),cv2.FONT_HERSHEY_SIMPLEX,1.4*scale,1,3*scale)
        outline=cv2.dilate(glyph,np.ones((5*scale,5*scale),np.uint8))
        raw[outline>0]=255;raw[glyph>0]=[230,10,210]
        support=cv2.dilate(outline,np.ones((7*scale,7*scale),np.uint8))
        # Contrast evidence can include erasure-collar/background pixels;
        # this cannot make an impossible inverse a valid optical explanation.
        semi,*_=unmix_optical_alpha_layer(raw,support.astype(float),observed_ink=support)
        assert not semi


def test_pending_visual_recognition_has_a_geometric_budget():
    sdf, _ = build_interior_sdf((400, 120), (0, 0), [])
    _, maximum = estimate_char_budget_from_sdf(sdf, "", source_font_size_px=22)
    assert maximum > 20


def test_budget_uses_geometry_not_source_length_and_has_no_dpi_floor():
    counts=[]
    for scale in (1, 2, 3):
        sdf,_=build_interior_sdf((120*scale,240*scale),(0,0),[])
        style=StyleSpec(orientation="horizontal",source_font_size_px=20*scale)
        a=estimate_char_budget_from_sdf(sdf,"",20*scale,style)
        b=estimate_char_budget_from_sdf(sdf,"x"*1000,20*scale,style)
        assert a==b
        counts.append(a[1])
    assert max(counts)-min(counts) <= 12
    sdf,_=build_interior_sdf((100,200),(0,0),[])
    assert estimate_char_budget_from_sdf(sdf,"unknown type scale")== (1,0)


def test_linguistic_units_survive_bubble_line_breaking_without_dropping_glyphs():
    text="这是自动校准模组。"
    segments=["这是","自动校准","模组。"]
    for scale in (1,2):
        sdf,pole=build_interior_sdf((130*scale,105*scale),(0,0),[])
        layout,alpha=solve_sdf_knuth_plass_layout(text,[],sdf,pole,
            StyleSpec(orientation="horizontal",source_font_size_px=29*scale),target_segments=segments)
        assert "".join(layout.lines)==text
        boundaries=set(np.cumsum([len(s) for s in segments]))
        assert set(np.cumsum([len(s) for s in layout.lines])) <= boundaries
        assert np.any(alpha>0)


def test_mixed_script_spaces_survive_segmentation_and_actual_layout():
    from prism_t_engine.typesetter_sdf import _tokenize_with_ruby, _join_tokens
    text="7 - 我想，mod A 确实有效。"
    assert _join_tokens(_tokenize_with_ruby(text,[]),False)==text
    sdf,pole=build_interior_sdf((85,700),(0,0),[])
    layout,_=solve_sdf_knuth_plass_layout(text,[],sdf,pole,
        StyleSpec(orientation="horizontal",source_font_size_px=26),target_segments=[text])
    assert layout.lines==[text]


def test_clause_segmentation_cannot_force_tiny_ribbon_type():
    text="我们正在寻找一条穿越森林通往海边的道路。"
    quads=[[[10,y],[200,y],[200,y+30],[10,y+30]] for y in (10,55,100)]
    layout,alpha=solve_sdf_knuth_plass_layout(text,[],np.ones((145,215),np.float32),(107,72),
        StyleSpec(orientation="horizontal",source_font_size_px=30),enclosed_bubble=False,
        source_quadrilaterals=quads,target_segments=[text])
    assert layout.font_size>=20
    assert "".join(layout.lines)==text
    assert len(layout.lines)>1
    assert np.any(alpha>0)


def test_semantic_boundaries_cannot_split_ruby_or_break_source_tracks():
    from prism_t_engine.schemas import RubyPair
    text="我们是最佳搭档。"
    layout,alpha=solve_sdf_knuth_plass_layout(text,[RubyPair(base="搭档",ruby="伙伴",char_start=5,char_end=7)],
        np.ones((110,230),np.float32),(115,55),
        StyleSpec(orientation="horizontal",source_font_size_px=30),enclosed_bubble=False,
        source_quadrilaterals=[[[10,10],[220,10],[220,45],[10,45]],[[10,60],[220,60],[220,95],[10,95]]],
        target_segments=["我们","是","最佳","搭档。"])
    assert "".join(layout.lines)==text
    assert any(g.is_ruby for g in layout.glyphs)
    assert np.any(alpha>0)


def test_symbols_and_vertical_punctuation_are_present_in_selected_font():
    text = "译文×♪︵︶︙"
    path = resolve_font_path("brush", needs_cjk=True, required_text=text)
    assert path is not None
    assert {ord(ch) for ch in text} <= _font_coverage(path)


def test_borderless_translation_keeps_staggered_source_anchors_and_all_characters():
    lines = [[160, 12, 30, 210], [105, 48, 30, 300], [50, 22, 30, 270]]
    sdf, pole = build_interior_sdf((380, 220), (0, 0), [[40, 0], [200, 0], [200, 370], [40, 370]])
    text = "译文应保持原始列的位置与阅读顺序并完整保留每一个字符。"
    layout, alpha = solve_sdf_knuth_plass_layout(
        text, [], sdf, pole,
        StyleSpec(orientation="vertical", source_font_size_px=25),
        enclosed_bubble=False, source_lines=lines,
    )
    assert "".join(layout.lines) == text
    assert len(layout.lines) == len(lines)
    assert alpha.any()
    for box, rendered in zip(lines, layout.lines):
        first = next(g for g in layout.glyphs if g.char == rendered[0] and abs(g.x + g.w/2 - (box[0]+box[2]/2)) < 1)
        assert first.y == box[1]
    assert all(40 <= g.x and g.x + g.w <= 200 and 0 <= g.y and g.y + g.h <= 370 for g in layout.glyphs)


def test_outline_palette_is_not_replaced_by_background_pigment():
    for scale in (1, 2):
        raw = np.zeros((100*scale, 140*scale, 3), np.uint8)
        raw[50*scale:] = [40, 65, 130]
        ink = np.zeros(raw.shape[:2], np.uint8)
        for x in (40, 85):
            cv2.line(ink, (x*scale, 15*scale), (x*scale, 85*scale), 1, 5*scale)
        rim = cv2.dilate(ink, np.ones((7*scale, 7*scale), np.uint8)) > 0
        raw[rim] = [255, 255, 255]
        raw[ink > 0] = [190, 25, 180]
        support = cv2.dilate(rim.astype(np.uint8), np.ones((3*scale, 3*scale), np.uint8)).astype(float)
        style, _, _ = extract_stroke_style_and_halo(raw, support, 5*scale, .1, "vertical", "floating_sfx")
        assert np.linalg.norm(np.array(style.fg_color_rgba[:3]) - [190, 25, 180]) < 15
        assert np.linalg.norm(np.array(style.outline_color_rgba[:3]) - 255) < 20
        assert style.slant_deg == 0


def test_scanned_overlapping_row_boxes_do_not_collapse_into_one_line():
    # OCR rectangles include ascenders/descenders and commonly overlap even
    # though their baselines are distinct, as on older scanned Western comics.
    lines = [[10, 10 + 15*i, 120, 22] for i in range(4)]
    sdf, pole = build_interior_sdf((100, 150), (0, 0), [])
    text = "完整译文应按原来的四行排列，不能将重叠的检测框合成一条细线。"
    layout, _ = solve_sdf_knuth_plass_layout(
        text, [], sdf, pole, StyleSpec(orientation="horizontal", source_font_size_px=16),
        enclosed_bubble=False, source_lines=lines,
    )
    assert len(layout.lines) == 4
    assert layout.font_size >= 10
    assert "".join(layout.lines) == text


def test_rotated_caption_keeps_the_baseline_without_shrinking_to_aabb_height():
    angle = np.deg2rad(28)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-u[1], u[0]])
    origin = np.array([45., 25.])
    quad = np.array([[0,0],[210,0],[210,28],[0,28]]) @ np.column_stack((u,v)).T + origin
    sdf = np.ones((180, 300), np.float32)
    text = "斜排译文保留方向和完整文字"
    layout, alpha = solve_sdf_knuth_plass_layout(
        text, [], sdf, (150,90), StyleSpec(orientation="horizontal", source_font_size_px=24),
        enclosed_bubble=False, source_quadrilaterals=[quad.tolist()],
    )
    assert "".join(layout.lines) == text
    assert layout.font_size >= 14
    yy, xx = np.nonzero(alpha > 0)
    points = np.column_stack((xx,yy)) - origin
    assert np.ptp(points @ v) < 28
    assert np.corrcoef(xx,yy)[0,1] > .9


def test_translated_heading_retains_the_original_row_centers():
    quads=[[[30,10],[430,10],[430,70],[30,70]],[[20,100],[440,100],[440,160],[20,160]]]
    layout,alpha=solve_sdf_knuth_plass_layout(
        "甲乙丙丁",[],np.ones((180,470),np.float32),(235,90),
        StyleSpec(orientation="horizontal",source_font_size_px=60),
        enclosed_bubble=False,source_quadrilaterals=quads,
    )
    assert "".join(layout.lines)=="甲乙丙丁"
    assert len(layout.lines)==2
    for y0,y1 in [(10,85),(85,175)]:
        _,xs=np.nonzero(alpha[y0:y1]>0)
        assert abs((xs.min()+xs.max()+1)/2-230)<1


def test_native_vertical_quadrilaterals_read_right_to_left():
    quads = [[[x,10],[x+28,10],[x+28,170],[x,170]] for x in [20,70,120]]
    text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
    layout,_ = solve_sdf_knuth_plass_layout(
        text,[],np.ones((190,170),np.float32),(85,95),
        StyleSpec(orientation="vertical",source_font_size_px=24),
        enclosed_bubble=False,source_quadrilaterals=quads,
    )
    assert "".join(layout.lines) == text
    starts = [next(g for g in layout.glyphs if g.char == line[0]).x for line in layout.lines]
    assert len(starts) > 1
    assert all(a > b for a,b in zip(starts,starts[1:]))


def test_small_source_track_scales_the_outline_with_its_font():
    text = "唔"
    layout,alpha = solve_sdf_knuth_plass_layout(
        text,[],np.ones((30,30),np.float32),(15,15),
        StyleSpec(orientation="vertical",source_font_size_px=12,outline_width_px=2),
        enclosed_bubble=False,source_quadrilaterals=[[[11,10],[14,10],[14,16],[11,16]]],
    )
    assert "".join(layout.lines) == text
    assert layout.font_size <= 3
    assert np.any(alpha>0)


def test_independent_strokes_survive_partial_and_missing_neural_ink():
    from prism_t_engine.learned import refine_text_alpha
    for scale in (1,2):
        raw = np.full((90*scale,210*scale,3),245,np.uint8)
        ink = np.zeros(raw.shape[:2],np.uint8)
        cv2.putText(ink,"TEXT",(15*scale,65*scale),cv2.FONT_HERSHEY_SIMPLEX,1.5*scale,1,3*scale)
        raw[ink>0] = 15
        probability = ink.astype(np.float32)
        probability[:,:90*scale] = 0
        domain = np.zeros_like(ink)
        domain[15*scale:75*scale,10*scale:190*scale] = 1
        alpha = refine_text_alpha(raw,probability,domain,observed_ink=ink)
        assert np.all(alpha[ink>0] > 0)
        assert not alpha[0,0]


def test_glowing_ink_edges_are_recorded_as_erasure_support():
    from prism_t_engine.learned import refine_text_alpha
    for scale in (1,2):
        yy,xx=np.indices((100*scale,240*scale))
        background=np.stack((30+xx/(10*scale),100+yy/(20*scale),160+xx/(20*scale)),axis=2)
        ink=np.zeros(background.shape[:2],np.uint8)
        cv2.putText(ink,"GLOW",(25*scale,65*scale),cv2.FONT_HERSHEY_SIMPLEX,1.4*scale,1,3*scale)
        glow=cv2.GaussianBlur(ink.astype(float),(0,0),scale)
        opacity=np.maximum(ink,glow)
        raw=np.rint(background*(1-opacity[:,:,None])+255*opacity[:,:,None]).astype(np.uint8)
        domain=np.zeros_like(ink);domain[20*scale:76*scale,20*scale:210*scale]=1
        alpha=refine_text_alpha(raw,ink.astype(float),domain,observed_ink=ink)
        halo=(ink==0)&(glow>.1)&(domain>0)
        assert np.all(alpha[halo]>0)
        assert not np.any(alpha[:10*scale])


def test_substrate_context_does_not_veto_recorded_erasure_support():
    from prism_t_engine.restoration import restore_clean_plate_roi
    raw = np.full((60,90,3),245,np.uint8)
    alpha = np.zeros(raw.shape[:2],np.float32)
    alpha[15:42,25:33] = 1
    raw[alpha>0] = 15
    substrate = np.zeros(raw.shape[:2],np.uint8)
    substrate[:,30:] = 1
    clean,_,_, _ = restore_clean_plate_roi(raw,alpha,substrate_mask=substrate)
    assert np.all(clean[alpha>0] > 230)
    assert np.array_equal(clean[alpha==0],raw[alpha==0])


def test_neural_component_cannot_claim_another_text_region():
    from prism_t_engine.learned import refine_text_alpha
    raw = np.full((100,280,3),245,np.uint8)
    ink = np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink,"HI",(15,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,1,3)
    raw[ink>0] = 15
    raw[:,170:] = [40,90,150]
    probability = np.zeros(raw.shape[:2],np.float32)
    probability[20:80,10:260] = 1
    domain = np.zeros_like(ink)
    domain[15:80,10:90] = 1
    alpha = refine_text_alpha(raw,probability,domain,observed_ink=ink)
    assert np.all(alpha[ink>0] > 0)
    assert not np.any(alpha[:,150:])


def test_text_probability_tube_does_not_become_artwork_erasure():
    from prism_t_engine.learned import refine_text_alpha
    yy,xx = np.indices((110,230))
    raw = np.stack((30+xx//3,50+yy,120+xx//5),axis=2).astype(np.uint8)
    ink = np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink,"HI",(20,80),cv2.FONT_HERSHEY_SIMPLEX,2,1,4)
    raw[ink>0] = [190,20,180]
    probability = np.zeros(raw.shape[:2],np.float32)
    probability[10:100,10:220] = 1
    domain = probability.astype(np.uint8)
    alpha = refine_text_alpha(raw,probability,domain,observed_ink=ink)
    assert np.all(alpha[ink>0] > 0)
    assert not np.any(alpha[:,140:])


def test_measured_pigment_recovers_a_glyph_missed_by_local_contrast():
    from prism_t_engine.learned import refine_text_alpha
    yy,xx = np.indices((110,230))
    raw = np.stack((30+xx//3,50+yy,120+xx//5),axis=2).astype(np.uint8)
    ink = np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink,"HI",(20,80),cv2.FONT_HERSHEY_SIMPLEX,2,1,4)
    raw[ink>0] = [190,20,180]
    observed = ink.copy()
    observed[:,75:] = 0
    probability = np.zeros(raw.shape[:2],np.float32)
    probability[10:100,10:220] = 1
    alpha = refine_text_alpha(raw,probability,probability.astype(np.uint8),observed_ink=observed)
    assert np.all(alpha[ink>0] > 0)
    assert not np.any(alpha[:,140:])


def test_stroke_collar_does_not_erase_artwork_merely_because_it_is_near_ink():
    from scipy.ndimage import distance_transform_edt
    from prism_t_engine.learned import refine_text_alpha
    for scale in (1,2):
        yy,xx=np.indices((80*scale,100*scale))
        raw=np.stack((30+xx//(4*scale),80+yy//(2*scale),130+xx//(5*scale)),axis=2).astype(np.uint8)
        ink=np.zeros(raw.shape[:2],np.uint8)
        cv2.circle(ink,(50*scale,40*scale),15*scale,1,4*scale)
        raw[ink>0]=[200,15,180]
        probability=np.zeros(raw.shape[:2],np.float32)
        probability[15*scale:65*scale,20*scale:80*scale]=1
        alpha=refine_text_alpha(raw,probability,probability.astype(np.uint8),observed_ink=ink)
        assert np.all(alpha[ink>0]>0)
        # Both outer artwork and the visible counter touch the lettering.
        # Their colors do not match the ink or a mixture with the background.
        adjacent=(distance_transform_edt(ink==0)<=scale)&(ink==0)
        assert not np.any(alpha[adjacent])
        assert not alpha[40*scale,50*scale]


def test_observed_pigment_variation_is_not_split_into_ink_and_background():
    from prism_t_engine.learned import refine_text_alpha
    # A smooth within-pigment scan variation is visible on both the observed
    # stroke core and its missing edge. The edge must not be rejected merely
    # for lying in Otsu's upper residual partition.
    for scale in (1,2):
        h,w=100*scale,130*scale
        yy,xx=np.indices((h,w))
        raw=np.full((h,w,3),[30,90,130],np.uint8)
        pigment=np.stack((175+yy//(4*scale),20+yy//(3*scale),165+yy//(4*scale)),axis=2).astype(np.uint8)
        ink=np.zeros((h,w),np.uint8);ink[15*scale:85*scale,45*scale:65*scale]=1
        raw[ink>0]=pigment[ink>0]
        observed=ink.copy();observed[:,(64*scale):]=0
        probability=observed.astype(np.float32)
        domain=np.zeros((h,w),np.uint8);domain[10*scale:90*scale,40*scale:75*scale]=1
        alpha=refine_text_alpha(raw,probability,domain,observed)
        assert np.all(alpha[ink>0])
        assert not np.any(alpha[(ink==0)&(xx>70*scale)])


def test_pigment_covariance_follows_scan_variation_without_accepting_other_colors():
    from scipy.stats import chi2
    from prism_t_engine.learned import stroke_pigment_distance
    for scale in (1,2):
        yy,xx=np.indices((90*scale,140*scale))
        raw=np.full((*yy.shape,3),[20,80,160],np.uint8)
        # A family of scanned pigment shades with correlated RGB changes.
        t=yy/(90*scale)
        shades=np.stack((165+55*t,18+70*t,155+45*t),axis=2).astype(np.uint8)
        observed=np.zeros(yy.shape,bool);observed[:,35*scale:55*scale]=True
        continuation=np.zeros_like(observed);continuation[:,55*scale:57*scale]=True
        raw[observed|continuation]=shades[observed|continuation]
        _,distance=stroke_pigment_distance(raw,observed)
        critical=chi2.isf(.05/np.count_nonzero(continuation),3)
        assert np.all(distance[continuation]<=critical)
        assert np.all(distance[:,100*scale:]>critical)
        # No spatial stencil or pigment covariance is allowed to depend on
        # translation text; this test supplies pixels and support only.


def test_recovered_outline_color_cannot_claim_matching_blank_background():
    from scipy.ndimage import distance_transform_edt
    from prism_t_engine.learned import refine_text_alpha
    for scale in (1,2):
        raw=np.full((90*scale,180*scale,3),245,np.uint8)
        ink=np.zeros(raw.shape[:2],np.uint8)
        cv2.putText(ink,'HI',(15*scale,65*scale),cv2.FONT_HERSHEY_SIMPLEX,1.5*scale,1,3*scale)
        outline=distance_transform_edt(ink==0)<=scale
        observed=outline.astype(np.uint8)
        raw[ink>0]=[185,15,170]
        probability=np.zeros(raw.shape[:2],np.float32)
        probability[10*scale:80*scale,5*scale:165*scale]=1
        alpha=refine_text_alpha(raw,probability,probability.astype(np.uint8),observed)
        assert np.all(alpha[ink>0])
        assert not np.any(alpha[:,100*scale:])
        # White is a genuine observed outline color but also the paper color;
        # that shared color does not turn the filled neural tube into ink.


def test_unanchored_observed_punctuation_survives_on_multicolor_artwork():
    from prism_t_engine.learned import refine_text_alpha
    for scale in (1, 2):
        yy, xx = np.indices((100 * scale, 180 * scale))
        raw = np.stack((30 + xx // (4 * scale),
                        80 + yy // (3 * scale),
                        150 + xx // (9 * scale)), axis=2).astype(np.uint8)
        ink = np.zeros(yy.shape, np.uint8)
        ink[20*scale:70*scale, 35*scale:40*scale] = 1
        detached = np.zeros_like(ink)
        cv2.ellipse(detached, (65*scale, 75*scale), (7*scale, 12*scale),
                    0, 0, 130, 1, 2*scale)
        ink |= detached
        raw[ink > 0] = [210, 25, 180]
        observed = ink.copy()
        # A separate observed artwork edge shares no lettering pigment.
        observed[25*scale:80*scale, 100*scale:103*scale] = 1
        raw[25*scale:80*scale, 100*scale:103*scale] = [20, 45, 90]
        probability = ink.astype(np.float32)
        probability[detached > 0] = .05
        domain = np.zeros_like(ink)
        domain[10*scale:95*scale, 25*scale:115*scale] = 1
        alpha = refine_text_alpha(raw, probability, domain, observed)
        assert np.all(alpha[ink > 0])
        assert not np.any(alpha[25*scale:80*scale, 100*scale:103*scale])


def test_common_artwork_pigment_cannot_anchor_a_disconnected_component():
    from prism_t_engine.learned import stroke_pigment_distance
    raw = np.full((60, 100, 3), [30, 80, 140], np.uint8)
    support = np.zeros(raw.shape[:2], bool)
    support[10:45, 20:30] = True
    support[10:45, 35:45] = True
    raw[10:45, 20:30] = [210, 25, 180]
    reference = np.ones(support.shape, bool)
    reference[5:55, 15:50] = False
    pigments, distance = stroke_pigment_distance(raw, support, reference)
    assert len(pigments) == 1
    assert np.all(distance[10:45, 20:30] < 1)
    assert np.all(distance[10:45, 35:45] > 100)


def test_detector_boundary_cannot_be_used_as_known_background_for_missing_ink():
    from prism_t_engine.learned import refine_text_alpha
    for scale in (1, 2):
        yy, xx = np.indices((90*scale, 140*scale))
        raw = np.stack((30 + xx // (7*scale), 90 + yy // (5*scale),
                        150 + xx // (9*scale)), axis=2).astype(np.uint8)
        ink = np.zeros(yy.shape, np.uint8)
        ink[15*scale:70*scale, 45*scale:55*scale] = 1
        raw[ink > 0] = [210, 25, 180]
        domain = np.zeros_like(ink)
        domain[10*scale:68*scale, 35*scale:65*scale] = 1
        observed = ink & domain
        alpha = refine_text_alpha(raw, observed.astype(np.float32), domain, observed)
        assert np.all(alpha[ink > 0])
        assert not np.any(alpha[(ink == 0) & (yy > 75*scale)])


def test_zero_noise_bubble_paper_is_not_observed_as_foreground():
    from prism_t_engine.pipeline import analyze_and_restore_page
    from prism_t_engine.schemas import AnalyzeRequest
    from prism_t_engine.codec import encode_png_b64,decode_image_b64
    raw = np.full((220,220,3),245,np.uint8)
    cv2.ellipse(raw,(110,110),(85,90),0,0,360,(255,255,255),-1)
    cv2.ellipse(raw,(110,110),(85,90),0,0,360,(10,10,10),2)
    ink = np.zeros((130,90),np.uint8)
    ink[15:115,15:19] = 1
    ink[15:115,65:69] = 1
    raw[45:175,65:155][ink>0] = 12
    region = dict(bbox=[65,45,90,130],polygon=[[65,45],[155,45],[155,175],[65,175]],
                  orientation="vertical",source_text="文字",ruby_pairs=[],spine_points=[],
                  ocr_entropy=0.,ink_mask_roi=ink)
    result = analyze_and_restore_page(AnalyzeRequest(image_png_b64=encode_png_b64(raw),format_profile="manga_rtl"),[region])
    assert len(result.dom.nodes) == 1
    node = result.dom.nodes[0]
    x,y,w,h = node.patches.roi
    alpha = decode_image_b64(node.patches.stroke_alpha_png_b64,mode="L")
    assert alpha[110-y,110-x] == 0
    assert alpha[100-y,81-x] > 0


def test_flat_paper_residual_audit_repairs_with_explicit_alpha_support():
    from prism_t_engine.restoration import repair_observed_paper_residue
    for scale in (1,3):
        raw = np.full((40*scale,80*scale,3),245,np.uint8)
        ink = np.zeros(raw.shape[:2],np.uint8)
        ink[10*scale:30*scale,20*scale:30*scale] = 1
        ink[10*scale:30*scale,45*scale:55*scale] = 1
        raw[ink>0] = 15
        alpha = ink.astype(np.float32)
        alpha[:,40*scale:] = 0
        clean = raw.copy()
        clean[alpha>0] = 245
        final,recorded,uncovered,residual,count = repair_observed_paper_residue(
            raw,clean,alpha,ink,np.ones_like(ink),
        )
        assert uncovered == residual == .5
        assert count == 200*scale*scale
        assert np.all(final[ink>0] == 245)
        assert np.all(recorded[ink>0] > 0)
        assert np.array_equal(final[recorded==0],raw[recorded==0])


def test_nonuniform_artwork_is_not_certified_as_flat_paper():
    from prism_t_engine.restoration import repair_observed_paper_residue
    raw = np.full((60,100,3),[20,50,130],np.uint8)
    raw[30:] = [160,70,40]
    ink = np.zeros(raw.shape[:2],np.uint8)
    ink[10:50,35:55] = 1
    raw[ink>0] = 15
    result,alpha,uncovered,residual,count = repair_observed_paper_residue(raw,raw,ink.astype(float),ink,np.ones_like(ink))
    assert uncovered is None and residual is None and count == 0
    assert np.array_equal(result,raw)


def test_quantized_paper_with_unit_quartile_differences_keeps_its_ink_audit():
    from prism_t_engine.restoration import repair_observed_paper_residue
    from prism_t_engine.preprocess import estimate_substrate_noise, substrate_squared_distance
    rng = np.random.default_rng(157)
    level = np.clip(255-rng.poisson(.8,(96,160)),0,255).astype(np.uint8)
    raw = np.repeat(level[:,:,None],3,axis=2)
    ink = np.zeros(level.shape,np.uint8)
    ink[25:65,50:55] = 1
    raw[ink>0] = 15
    result,alpha,uncovered,residual,count = repair_observed_paper_residue(
        raw,raw,ink.astype(float),ink,np.ones_like(ink),
    )
    assert uncovered == 0 and residual == 1 and count == int(ink.sum())
    assert np.all(result[ink>0] > 250)
    assert np.array_equal(result[alpha==0],raw[alpha==0])
    known = ink==0
    assert np.all(estimate_substrate_noise(raw,known) < 2)
    # Adjacent uint8 values have touching quantization intervals, not a
    # perfectly resolved difference of one continuous intensity unit.
    assert substrate_squared_distance(np.array([[254,255,254]],np.uint8),np.array([255,255,255]),np.ones(3))[0] == 0


def test_paper_repair_cannot_clear_observed_strokes_outside_the_paper():
    from prism_t_engine.restoration import repair_observed_paper_residue
    raw = np.full((70,120,3),250,np.uint8)
    paper = np.zeros(raw.shape[:2],np.uint8)
    paper[5:65,5:65] = 1
    observed = np.zeros_like(paper)
    observed[25:45,30:35] = 1
    observed[10:60,90:95] = 1
    raw[observed>0] = 15
    alpha = np.zeros(raw.shape[:2],float)
    result,recorded,uncovered,residual,count = repair_observed_paper_residue(raw,raw,alpha,observed,paper)
    assert count == 20*5 and uncovered == residual == 1
    assert np.all(result[25:45,30:35] == 250)
    assert np.array_equal(result[paper==0],raw[paper==0])
    assert not recorded[paper==0].any()


def test_saturated_noisy_paper_and_generated_ink_are_automatically_repaired():
    from prism_t_engine.restoration import repair_observed_paper_residue
    from prism_t_engine.preprocess import estimate_noise_sigma
    rng = np.random.default_rng(27)
    level = np.clip(np.rint(255+rng.normal(1,2,(96,160))),0,255).astype(np.uint8)
    raw = np.repeat(level[:,:,None],3,axis=2)
    # The clipped majority still has a zero lower-quartile Haar noise MAD.
    raw[:24] = 255
    ink = np.zeros(raw.shape[:2],np.uint8)
    ink[35:65,55:60] = 1
    raw[ink>0] = 12
    alpha = np.zeros(raw.shape[:2],np.float32)
    alpha[32:68,52:65] = 1
    clean = raw.copy()
    clean[alpha>0] = 255
    # Hallucinated continuation falls on formerly blank paper, away from ink.
    clean[35:65,61:64] = 30
    result,recorded,uncovered,residual,count = repair_observed_paper_residue(
        raw,clean,alpha,ink,np.ones_like(ink),
    )
    assert estimate_noise_sigma(raw[:,:,0]) == 0
    assert uncovered == residual == 0
    assert count >= 30*3
    assert np.all(result[35:65,61:64] > 250)
    assert np.array_equal(result[recorded==0],raw[recorded==0])


def test_paper_audit_recovers_ink_missing_from_the_initial_stroke_detector():
    from prism_t_engine.restoration import repair_observed_paper_residue
    raw = np.full((80,140,3),250,np.uint8)
    raw[25:55,35:40] = 10
    raw[25:55,85:90] = 10
    observed = np.zeros(raw.shape[:2],np.uint8)
    observed[25:55,35:40] = 1
    source_domain = np.zeros_like(observed)
    source_domain[20:60,30:100] = 1
    alpha = observed.astype(float)
    clean = raw.copy()
    clean[observed>0] = 250
    result,recorded,uncovered,residual,count = repair_observed_paper_residue(
        raw,clean,alpha,observed,np.ones_like(observed),source_domain=source_domain,
    )
    assert uncovered == residual == .5
    assert count == 150
    assert np.all(result[25:55,85:90] == 250)
    assert np.all(recorded[25:55,85:90] == 1)
    assert np.array_equal(result[recorded==0],raw[recorded==0])


def test_opaque_colored_outline_is_not_transparent_background_transmission():
    from prism_t_engine.deep_fusion import unmix_optical_alpha_layer
    h,w = 120,180
    yy,xx = np.indices((h,w))
    raw = np.stack((30+xx//2,40+yy,120+xx//4),axis=2).astype(np.uint8)
    ink = np.zeros((h,w),np.uint8)
    cv2.putText(ink,"HI",(25,88),cv2.FONT_HERSHEY_SIMPLEX,2,1,7)
    rim = cv2.dilate(ink,np.ones((7,7),np.uint8))>0
    raw[rim] = 250
    raw[ink>0] = [190,20,180]
    semi,_,_,_,_ = unmix_optical_alpha_layer(raw,rim.astype(float))
    assert not semi


def test_scanned_paper_counters_are_not_selected_as_white_ink():
    raw = np.full((90,160,3),245,np.uint8)
    cv2.putText(raw,"TEST",(10,65),cv2.FONT_HERSHEY_SIMPLEX,1.5,(12,12,12),3,cv2.LINE_AA)
    support = np.zeros(raw.shape[:2],np.float32)
    support[25:70,8:150] = 1
    style,_,_ = extract_stroke_style_and_halo(raw,support,3,.1,"horizontal","speech_oval")
    assert max(style.fg_color_rgba[:3]) < 40
    assert style.outline_width_px == 0


def test_source_pigment_survives_aged_paper_and_reversed_ink_polarity():
    for scale in (1, 2):
        for reverse in (False, True):
            h, w = 100*scale, 180*scale
            yy, xx = np.indices((h, w))
            grain = ((xx//scale + 3*(yy//scale)) % 9 - 4)[:, :, None]
            paper = np.array([221, 204, 178] if not reverse else [30, 45, 65])
            pigment = np.array([81, 67, 47] if not reverse else [238, 225, 195])
            raw = np.clip(paper + grain, 0, 255).astype(np.uint8)
            ink = np.zeros((h, w), np.uint8)
            cv2.putText(ink, "B8", (15*scale, 76*scale), cv2.FONT_HERSHEY_SIMPLEX,
                        2*scale, 255, 4*scale, cv2.LINE_AA)
            coverage = ink[:, :, None]/255
            raw = np.rint(raw*(1-coverage) + pigment*coverage).astype(np.uint8)
            envelope = np.zeros((h, w), np.float32)
            envelope[20*scale:82*scale, 10*scale:120*scale] = 1
            # Independent contrast evidence can contain a small paper island.
            observed = ink > 0
            observed[30*scale:33*scale, 100*scale:103*scale] = True
            style, _, exemplar = extract_stroke_style_and_halo(
                raw, envelope, 4*scale, .1, "horizontal", "speech_oval",
                pigment_support=observed,
            )
            assert np.linalg.norm(np.array(style.fg_color_rgba[:3])-pigment) < 20
            assert style.outline_width_px == 0
            assert np.linalg.norm(np.median(exemplar, axis=(0, 1))-pigment) < 20


def test_thin_ink_and_bright_scan_speck_do_not_use_palette_order_for_fill():
    for scale in (1, 2):
        raw = np.full((70*scale, 140*scale, 3), [214, 195, 166], np.uint8)
        observed = np.zeros(raw.shape[:2], bool)
        for x in (40, 60, 80):
            raw[20*scale:50*scale, x*scale:(x+1)*scale] = [83, 69, 48]
            observed[20*scale:50*scale, x*scale:(x+1)*scale] = True
        raw[30*scale:31*scale, 95*scale:105*scale] = [235, 220, 195]
        observed[30*scale:31*scale, 95*scale:105*scale] = True
        envelope = np.zeros(raw.shape[:2], np.float32)
        envelope[15*scale:55*scale, 30*scale:110*scale] = 1
        style, _, _ = extract_stroke_style_and_halo(
            raw, envelope, scale, .1, "horizontal", "speech_oval", pigment_support=observed,
        )
        assert np.linalg.norm(np.array(style.fg_color_rgba[:3])-[83, 69, 48]) < 10


def test_antialias_mode_is_not_fill_and_shared_background_white_can_be_outline():
    for scale in (1, 2):
        raw = np.full((110*scale, 180*scale, 3), [35, 60, 125], np.uint8)
        # The same white also belongs to genuine visible artwork.
        raw[60*scale:, :] = [250, 247, 252]
        ink = np.zeros(raw.shape[:2], np.uint8)
        cv2.putText(ink, "HI", (22*scale, 85*scale), cv2.FONT_HERSHEY_SIMPLEX,
                    2*scale, 1, 5*scale)
        rim = cv2.dilate(ink, np.ones((7*scale, 7*scale), np.uint8)) > 0
        raw[rim] = [250, 247, 252]
        aa = (cv2.dilate(ink, np.ones((3*scale, 3*scale), np.uint8)) > 0) & (ink == 0)
        raw[aa] = [243, 169, 241]
        raw[ink > 0] = [201, 17, 194]
        envelope = cv2.dilate(rim.astype(np.uint8), np.ones((3*scale, 3*scale), np.uint8)).astype(float)
        style, _, _ = extract_stroke_style_and_halo(
            raw, envelope, 5*scale, .1, "horizontal", "floating_sfx", pigment_support=rim,
        )
        assert np.linalg.norm(np.array(style.fg_color_rgba[:3])-[201, 17, 194]) < 15
        assert style.outline_width_px > 0
        assert min(style.outline_color_rgba[:3]) > 230


def test_bubble_source_style_excludes_surrounding_artwork_from_substrate():
    raw = np.full((100,180,3), [83,69,48], np.uint8)
    raw[15:85,20:160] = [221,204,178]
    ink = np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink, "B8", (40,72), cv2.FONT_HERSHEY_SIMPLEX,1.5,255,3,cv2.LINE_AA)
    coverage = ink[:,:,None]/255
    raw = np.rint(raw*(1-coverage)+np.array([83,69,48])*coverage).astype(np.uint8)
    envelope = cv2.dilate((ink>0).astype(np.uint8),np.ones((3,3),np.uint8)).astype(float)
    substrate = np.zeros(raw.shape[:2],bool)
    substrate[17:83,22:158] = True
    style,_,_ = extract_stroke_style_and_halo(
        raw,envelope,3,.1,"horizontal","speech_oval",
        pigment_support=ink>0,substrate_support=substrate,
    )
    assert np.linalg.norm(np.array(style.fg_color_rgba[:3])-[83,69,48]) < 15
    assert style.outline_width_px == 0


def test_horizontal_punctuation_uses_the_text_baseline():
    text = "译文，完整。"
    layout,alpha = solve_sdf_knuth_plass_layout(
        text,[],np.ones((48,240),np.float32),(120,24),
        StyleSpec(orientation="horizontal",source_font_size_px=30),
        enclosed_bubble=False,source_lines=[[0,0,240,48]],
    )
    assert "".join(layout.lines) == text
    def center_y(character):
        glyph = next(g for g in layout.glyphs if g.char == character)
        x0,x1 = int(np.ceil(glyph.x)),int(np.floor(glyph.x+glyph.w))
        yy,_ = np.nonzero(alpha[:,x0:x1] > .25)
        return np.mean(yy)
    assert center_y("，") > center_y("文")
    assert center_y("。") > center_y("整")


def test_shorter_translation_can_leave_tiny_original_tracks_empty():
    lines = [[10,10,150,24],[10,40,2,2],[10,50,150,24]]
    text = "译文完整保留"
    layout,alpha = solve_sdf_knuth_plass_layout(
        text,[],np.ones((100,180),np.float32),(90,50),StyleSpec(orientation="horizontal",source_font_size_px=22),
        enclosed_bubble=False,source_lines=lines,
    )
    assert "".join(layout.lines) == text
    assert layout.font_size >= 18
    assert alpha.any()


def test_variable_cjk_font_renders_regular_weight_instead_of_thin_default():
    text = "文字︙"
    font = load_font("gothic",28,required_text=text)
    baseline = ImageFont.truetype(font.path,28)
    from fontTools.ttLib import TTFont
    with TTFont(font.path,fontNumber=0) as metadata:
        if "fvar" not in metadata:
            return
        weight = next((a for a in metadata['fvar'].axes if a.axisTag == 'wght'),None)
        if weight is not None and weight.defaultValue < 400:
            assert np.asarray(font.getmask(text)).sum() > np.asarray(baseline.getmask(text)).sum()


def test_adding_outline_preserves_fractional_glyph_coverage():
    from prism_t_engine.deep_fusion import apply_exemplar_stroke_shader
    alpha = np.zeros((20,30),np.float32)
    alpha[5:15,8:12] = 1
    alpha[5:15,12:15] = [.4,.2,.1]
    rendered = apply_exemplar_stroke_shader(alpha,None,StyleSpec(outline_width_px=1))
    assert np.all(rendered[:,:,3] >= np.rint(alpha*255))


def test_bright_outline_is_not_a_nonunit_mixture_of_dark_pigments():
    raw = np.full((120,180,3),[25,45,80],np.uint8)
    raw[60:] = [90,95,125]
    ink = np.zeros(raw.shape[:2],np.uint8)
    cv2.putText(ink,"HI",(25,88),cv2.FONT_HERSHEY_SIMPLEX,2,1,7)
    ring = cv2.dilate(ink,np.ones((7,7),np.uint8)) > 0
    raw[ring] = [250,248,252]
    raw[ink>0] = [197,20,191]
    alpha = cv2.dilate(ring.astype(np.uint8),np.ones((3,3),np.uint8)).astype(float)
    style,_,_ = extract_stroke_style_and_halo(raw,alpha,7,.1,"horizontal","floating_sfx")
    assert style.outline_width_px > 0
    assert min(style.outline_color_rgba[:3]) > 230


def test_render_cache_reuses_only_unchanged_nodes(monkeypatch):
    from prism_t_engine import pipeline
    from prism_t_engine.codec import encode_png_b64
    from prism_t_engine.schemas import PageDOM,TextNode,PatchLayers,RenderRequest
    raw = np.full((100,240,3),245,np.uint8)
    nodes = [TextNode(
        id=str(i),bbox=[10+i*120,10,100,80],
        bubble_polygon=[[10+i*120,10],[110+i*120,10],[110+i*120,90],[10+i*120,90]],
        target_text="完整译文",style=StyleSpec(orientation="horizontal",source_font_size_px=20),
        patches=PatchLayers(roi=[i*120,0,120,100],clean_z0_png_b64=encode_png_b64(raw[:,:120]),
                            stroke_alpha_png_b64=encode_png_b64(np.zeros((100,120),np.uint8))),
    ) for i in range(2)]
    calls = []
    solve = pipeline.solve_sdf_knuth_plass_layout
    def measured(*args,**kwargs):
        calls.append(kwargs['target_text'])
        return solve(*args,**kwargs)
    monkeypatch.setattr(pipeline,'solve_sdf_knuth_plass_layout',measured)
    raw_b64 = encode_png_b64(raw)
    transcription = {"context_hash":"source-image-implementation", "results":{"0":{"id":"0","is_text":True,"line_texts":["source evidence"],"line_order":[0],"reason":""}}}
    request = RenderRequest.model_validate_json(RenderRequest(raw_image_png_b64=raw_b64,dom=PageDOM(width=240,height=100,nodes=nodes,recognition_progress=transcription)).model_dump_json())
    first = pipeline.typeset_and_render_page(request)
    assert first.dom.recognition_progress == transcription
    assert len(calls) == 2
    calls.clear()
    again = pipeline.typeset_and_render_page(RenderRequest(raw_image_png_b64=raw_b64,dom=first.dom))
    assert again.dom.recognition_progress == transcription
    assert not calls
    assert again.composite_png_b64 == first.composite_png_b64
    again.dom.nodes[0].target_text = "修改后的译文"
    edited = pipeline.typeset_and_render_page(RenderRequest(raw_image_png_b64=raw_b64,dom=again.dom))
    assert calls == ["修改后的译文"]
    assert edited.dom.nodes[1].patches.rendered_z1_png_b64 == first.dom.nodes[1].patches.rendered_z1_png_b64
    calls.clear()
    edited.dom.nodes[1].patches.clean_z0_png_b64 = encode_png_b64(np.full((100,120,3),230,np.uint8))
    pipeline.typeset_and_render_page(RenderRequest(raw_image_png_b64=raw_b64,dom=edited.dom))
    assert calls == ["完整译文"]


def test_harmonize_page_typography_unifies_vision_mode_siblings_and_enclosed_ovals():
    from types import SimpleNamespace
    from prism_t_engine.pipeline import harmonize_dom_nodes_typography, harmonize_page_typography

    sib_a = StyleSpec(
        font_category="gothic", orientation="vertical", source_font_size_px=22.0,
        fg_color_rgba=[245, 245, 245, 255], stroke_width_px=2.4, tremor_amplitude=0.0,
        outline_width_px=2.2, outline_color_rgba=[20, 20, 22, 255],
    )
    sib_b = StyleSpec(
        font_category="mincho", orientation="vertical", source_font_size_px=18.0,
        fg_color_rgba=[238, 240, 242, 255], stroke_width_px=2.5, tremor_amplitude=0.0,
        outline_width_px=0.0,
    )
    sfx_big = StyleSpec(
        font_category="brush", orientation="vertical", source_font_size_px=44.0,
        fg_color_rgba=[210, 30, 45, 255], stroke_width_px=3.0, tremor_amplitude=0.0,
        outline_width_px=0.0,
    )
    oval_0 = StyleSpec(
        font_category="gothic", orientation="vertical", source_font_size_px=20.0,
        fg_color_rgba=[12, 12, 14, 255], outline_width_px=0.0,
    )
    oval_1 = StyleSpec(
        font_category="gothic", orientation="vertical", source_font_size_px=20.0,
        fg_color_rgba=[22, 24, 26, 255], outline_width_px=1.2, outline_color_rgba=[230, 230, 230, 255],
    )
    oval_2 = StyleSpec(
        font_category="gothic", orientation="vertical", source_font_size_px=20.0,
        fg_color_rgba=[16, 16, 18, 255], outline_width_px=0.0,
    )
    meta = [
        {"bubble_kind": "floating_sfx", "style": sib_a, "source_text": ""},
        {"bubble_kind": "floating_sfx", "style": sib_b, "source_text": ""},
        {"bubble_kind": "floating_sfx", "style": sfx_big, "source_text": ""},
        {"bubble_kind": "speech_oval", "style": oval_0, "source_text": ""},
        {"bubble_kind": "speech_oval", "style": oval_1, "source_text": ""},
        {"bubble_kind": "speech_oval", "style": oval_2, "source_text": ""},
    ]
    harmonize_page_typography(meta)
    assert sib_a.font_category == sib_b.font_category
    assert sib_a.fg_color_rgba == sib_b.fg_color_rgba
    assert sib_a.source_font_size_px == sib_b.source_font_size_px == 20.0
    assert sib_a.outline_width_px == sib_b.outline_width_px == 2.2
    assert sib_b.outline_color_rgba == [20, 20, 22, 255]
    assert sfx_big.font_category == "brush"
    assert sfx_big.fg_color_rgba[:3] == [210, 30, 45]
    assert sfx_big.source_font_size_px == 44.0
    assert oval_1.outline_width_px == 0.0
    assert oval_0.fg_color_rgba == oval_1.fg_color_rgba == oval_2.fg_color_rgba

    dom_a = StyleSpec(font_category="gothic", orientation="vertical", source_font_size_px=22.0, fg_color_rgba=[245, 245, 245, 255], stroke_width_px=2.0)
    dom_b = StyleSpec(font_category="mincho", orientation="vertical", source_font_size_px=18.0, fg_color_rgba=[238, 240, 242, 255], stroke_width_px=2.0)
    harmonize_dom_nodes_typography([
        SimpleNamespace(bubble_kind="floating_sfx", style=dom_a),
        SimpleNamespace(bubble_kind="floating_sfx", style=dom_b),
    ])
    assert dom_a.font_category == dom_b.font_category
    assert dom_a.fg_color_rgba == dom_b.fg_color_rgba
    assert dom_a.source_font_size_px == dom_b.source_font_size_px == 20.0


def test_enclosed_bubble_antialias_has_no_false_outline_and_solid_outlined_text_has_no_exemplar():
    raw_oval = np.full((120, 160, 3), [228, 211, 186], np.uint8)
    core_oval = np.zeros((120, 160), np.uint8)
    for x in (45, 95):
        cv2.line(core_oval, (x, 20), (x, 100), 1, 5)
    aa_ring = (cv2.dilate(core_oval, np.ones((3, 3), np.uint8)) > 0) & (core_oval == 0)
    raw_oval[aa_ring] = [123, 114, 103]
    raw_oval[core_oval > 0] = [18, 18, 20]
    support_oval = cv2.dilate(core_oval, np.ones((7, 7), np.uint8)).astype(np.float32)
    style_oval, _, _ = extract_stroke_style_and_halo(raw_oval, support_oval, 4.0, 0.02, "vertical", "speech_oval")
    assert style_oval.outline_width_px == 0.0

    raw_sfx = np.full((140, 180, 3), [40, 50, 75], np.uint8)
    ink_sfx = np.zeros((140, 180), np.uint8)
    for x in (50, 110):
        cv2.line(ink_sfx, (x, 20), (x, 120), 1, 7)
    rim_sfx = cv2.dilate(ink_sfx, np.ones((7, 7), np.uint8)) > 0
    raw_sfx[rim_sfx] = [252, 252, 252]
    raw_sfx[ink_sfx > 0] = [30, 205, 225]
    support_sfx = cv2.dilate(rim_sfx.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    style_sfx, fusion_sfx, _ = extract_stroke_style_and_halo(raw_sfx, support_sfx, 6.0, 0.05, "vertical", "floating_sfx")
    assert style_sfx.outline_width_px > 0.5
    assert fusion_sfx == "clean_interior"


def test_vertical_punctuation_quadrant_and_double_ellipsis_tight_join():
    text = "甲，乙。……"
    sdf, pole = build_interior_sdf((320, 140), (0, 0), [])
    layout, _ = solve_sdf_knuth_plass_layout(
        text, [], sdf, pole,
        StyleSpec(orientation="vertical", source_font_size_px=24),
    )
    assert len(layout.lines) == 1
    g_jia = next(g for g in layout.glyphs if g.char == "甲")
    g_comma = next(g for g in layout.glyphs if g.char == "，")
    g_period = next(g for g in layout.glyphs if g.char == "。")
    g_dots = [g for g in layout.glyphs if g.char == "︙"]
    col_cx = g_jia.x + 0.5 * g_jia.w
    assert g_comma.h < layout.font_size * 0.75
    assert g_period.h < layout.font_size * 0.75
    assert g_comma.x + 0.5 * g_comma.w > col_cx + 0.05 * layout.font_size
    assert g_period.x + 0.5 * g_period.w > col_cx + 0.05 * layout.font_size
    assert len(g_dots) == 2
    assert (g_dots[1].y - g_dots[0].y) <= layout.font_size * 0.76


def test_inner_cyan_core_over_sky_is_never_rejected_as_shallow_outline_blend():
    raw = np.full((160, 180, 3), [38, 65, 121], np.uint8)
    ink = np.zeros((160, 180), np.uint8)
    for x in (55, 115):
        cv2.line(ink, (x, 20), (x, 140), 1, 7)
    rim = cv2.dilate(ink, np.ones((7, 7), np.uint8)) > 0
    raw[rim] = [242, 252, 253]
    raw[ink > 0] = [126, 175, 203]
    support = cv2.dilate(rim.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    style, fusion, _ = extract_stroke_style_and_halo(raw, support, 7.0, 0.05, "vertical", "floating_sfx")
    assert np.linalg.norm(np.array(style.fg_color_rgba[:3]) - [126, 175, 203]) < 20
    assert style.outline_width_px > 0.5
    assert np.linalg.norm(np.array(style.outline_color_rgba[:3]) - [242, 252, 253]) < 20
    assert fusion == "clean_interior"
