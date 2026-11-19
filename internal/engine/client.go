package engine

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"prism-t/internal/dom"
)

// ErrEndpointNotFound indicates the operator server does not implement a newer
// endpoint (e.g., /v1/detect_layout on legacy mock servers).
var ErrEndpointNotFound = errors.New("engine endpoint not found")

type httpStatusError struct {
	Path       string
	StatusCode int
	Body       string
}

func (e *httpStatusError) Error() string {
	return fmt.Sprintf("engine %s returned %d: %s", e.Path, e.StatusCode, e.Body)
}

// IsEndpointNotFound reports whether err indicates an unimplemented endpoint
// (HTTP 404/405/501 or empty 200 stub response).
func IsEndpointNotFound(err error) bool {
	if errors.Is(err, ErrEndpointNotFound) {
		return true
	}
	var se *httpStatusError
	if errors.As(err, &se) {
		return se.StatusCode == http.StatusNotFound ||
			se.StatusCode == http.StatusMethodNotAllowed ||
			se.StatusCode == http.StatusNotImplemented
	}
	return false
}

type Client struct {
	baseURL string
	http    *http.Client
}

type HardwareStatus struct {
	ConfiguredDevice string  `json:"configured_device"`
	ActiveDevice     string  `json:"active_device"`
	CUDAAvailable    bool    `json:"cuda_available"`
	GPUName          *string `json:"gpu_name,omitempty"`
	VRAMTotalMB      *int    `json:"vram_total_mb,omitempty"`
	CPUThreads       int     `json:"cpu_threads"`
	InferenceThreads int     `json:"inference_threads"`
}

type HealthStatus struct {
	Status   string          `json:"status"`
	Version  string          `json:"version"`
	Device   string          `json:"device"`
	Hardware *HardwareStatus `json:"hardware,omitempty"`
	OCR      map[string]any  `json:"ocr,omitempty"`
	Models   map[string]any  `json:"models,omitempty"`
}

type AnalyzeRequest struct {
	ImagePNGB64            string          `json:"image_png_b64"`
	ReferenceEditionPNGB64 string          `json:"reference_edition_png_b64,omitempty"`
	FormatProfile          string          `json:"format_profile"`
	RecognizeText          bool            `json:"recognize_text"`
	SeamSlices             []dom.SeamSlice `json:"seam_slices,omitempty"`
	IsSpread               bool            `json:"is_spread"`
	GutterX                *int            `json:"gutter_x,omitempty"`
}

type AnalyzeResponse struct {
	DOM                dom.PageDOM `json:"dom"`
	CleanPreviewPNGB64 string      `json:"clean_preview_png_b64"`
}

type DetectLayoutResponse struct {
	DOM         dom.PageDOM `json:"dom"`
	LayoutToken string      `json:"layout_token"`
}

type RestoreRequest struct {
	LayoutToken            string          `json:"layout_token"`
	ImagePNGB64            string          `json:"image_png_b64"`
	ReferenceEditionPNGB64 string          `json:"reference_edition_png_b64,omitempty"`
	FormatProfile          string          `json:"format_profile"`
	RecognizeText          bool            `json:"recognize_text"`
	SeamSlices             []dom.SeamSlice `json:"seam_slices,omitempty"`
	IsSpread               bool            `json:"is_spread"`
	GutterX                *int            `json:"gutter_x,omitempty"`
	ExcludeNodeIDs         []string        `json:"exclude_node_ids,omitempty"`
}

type RenderRequest struct {
	RawImagePNGB64 string      `json:"raw_image_png_b64"`
	DOM            dom.PageDOM `json:"dom"`
	TargetLang     string      `json:"target_lang"`
}

type RenderResponse struct {
	DOM             dom.PageDOM `json:"dom"`
	CompositePNGB64 string      `json:"composite_png_b64"`
	SlicePNGsB64    []string    `json:"slice_pngs_b64"`
}

func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http: &http.Client{
			Timeout: 15 * time.Minute,
		},
	}
}

func (c *Client) Health(ctx context.Context) (HealthStatus, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return HealthStatus{}, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return HealthStatus{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return HealthStatus{}, fmt.Errorf("engine health status %d", resp.StatusCode)
	}
	var hs HealthStatus
	if err := json.NewDecoder(resp.Body).Decode(&hs); err != nil {
		return HealthStatus{}, err
	}
	return hs, nil
}

func buildAnalyzeRequest(rawPNG []byte, refPNG []byte, profile string, slices []dom.SeamSlice, recognizeText bool) AnalyzeRequest {
	if profile == "" {
		profile = "auto"
	}
	reqPayload := AnalyzeRequest{
		ImagePNGB64:   base64.StdEncoding.EncodeToString(rawPNG),
		FormatProfile: profile,
		RecognizeText: recognizeText,
		SeamSlices:    slices,
	}
	if len(refPNG) > 0 {
		reqPayload.ReferenceEditionPNGB64 = base64.StdEncoding.EncodeToString(refPNG)
	}
	return reqPayload
}

func (c *Client) Analyze(ctx context.Context, rawPNG []byte, refPNG []byte, profile string, slices []dom.SeamSlice, recognizeText bool) (AnalyzeResponse, error) {
	reqPayload := buildAnalyzeRequest(rawPNG, refPNG, profile, slices, recognizeText)
	var out AnalyzeResponse
	if err := c.postJSON(ctx, "/v1/analyze", reqPayload, &out); err != nil {
		return AnalyzeResponse{}, err
	}
	return out, nil
}

func (c *Client) DetectLayout(ctx context.Context, rawPNG []byte, refPNG []byte, profile string, slices []dom.SeamSlice, recognizeText bool) (DetectLayoutResponse, error) {
	reqPayload := buildAnalyzeRequest(rawPNG, refPNG, profile, slices, recognizeText)
	var out DetectLayoutResponse
	if err := c.postJSON(ctx, "/v1/detect_layout", reqPayload, &out); err != nil {
		if errors.Is(err, io.EOF) || IsEndpointNotFound(err) {
			return DetectLayoutResponse{}, fmt.Errorf("%w: %v", ErrEndpointNotFound, err)
		}
		return DetectLayoutResponse{}, err
	}
	return out, nil
}

func (c *Client) RestorePage(ctx context.Context, layoutToken string, rawPNG []byte, refPNG []byte, profile string, slices []dom.SeamSlice, recognizeText bool, excludeNodeIDs []string) (AnalyzeResponse, error) {
	if profile == "" {
		profile = "auto"
	}
	reqPayload := RestoreRequest{
		LayoutToken:    layoutToken,
		ImagePNGB64:    base64.StdEncoding.EncodeToString(rawPNG),
		FormatProfile:  profile,
		RecognizeText:  recognizeText,
		SeamSlices:     slices,
		ExcludeNodeIDs: excludeNodeIDs,
	}
	if len(refPNG) > 0 {
		reqPayload.ReferenceEditionPNGB64 = base64.StdEncoding.EncodeToString(refPNG)
	}
	var out AnalyzeResponse
	if err := c.postJSON(ctx, "/v1/restore_page", reqPayload, &out); err != nil {
		return AnalyzeResponse{}, err
	}
	return out, nil
}

type debugPatchLayers struct {
	SourceObservedInkPNGB64  string
	SourceDomainPNGB64       string
	RefinedStrokePNGB64      string
	InpaintUnknownPNGB64     string
	InferredBackgroundPNGB64 string
	BoundaryOpacityPNGB64    string
}

// stripDebugPatchLayers omits ledger-only diagnostic base64 blobs from the /v1/render
// HTTP payload without mutating the caller's PageDOM, returning them for re-attachment.
func stripDebugPatchLayers(pageDOM dom.PageDOM) (dom.PageDOM, map[string]debugPatchLayers) {
	if len(pageDOM.Nodes) == 0 {
		return pageDOM, nil
	}
	saved := make(map[string]debugPatchLayers, len(pageDOM.Nodes))
	strippedNodes := make([]dom.TextNode, len(pageDOM.Nodes))
	copy(strippedNodes, pageDOM.Nodes)
	for i := range strippedNodes {
		p := &strippedNodes[i].Patches
		if p.SourceObservedInkPNGB64 != "" || p.SourceDomainPNGB64 != "" ||
			p.RefinedStrokePNGB64 != "" || p.InpaintUnknownPNGB64 != "" ||
			p.InferredBackgroundPNGB64 != "" || p.BoundaryOpacityPNGB64 != "" {
			saved[strippedNodes[i].ID] = debugPatchLayers{
				SourceObservedInkPNGB64:  p.SourceObservedInkPNGB64,
				SourceDomainPNGB64:       p.SourceDomainPNGB64,
				RefinedStrokePNGB64:      p.RefinedStrokePNGB64,
				InpaintUnknownPNGB64:     p.InpaintUnknownPNGB64,
				InferredBackgroundPNGB64: p.InferredBackgroundPNGB64,
				BoundaryOpacityPNGB64:    p.BoundaryOpacityPNGB64,
			}
		}
		p.SourceObservedInkPNGB64 = ""
		p.SourceDomainPNGB64 = ""
		p.RefinedStrokePNGB64 = ""
		p.InpaintUnknownPNGB64 = ""
		p.InferredBackgroundPNGB64 = ""
		p.BoundaryOpacityPNGB64 = ""
	}
	pageDOM.Nodes = strippedNodes
	return pageDOM, saved
}

func restoreDebugPatchLayers(pageDOM *dom.PageDOM, saved map[string]debugPatchLayers) {
	if pageDOM == nil || len(saved) == 0 {
		return
	}
	for i := range pageDOM.Nodes {
		orig, ok := saved[pageDOM.Nodes[i].ID]
		if !ok {
			continue
		}
		p := &pageDOM.Nodes[i].Patches
		if p.SourceObservedInkPNGB64 == "" {
			p.SourceObservedInkPNGB64 = orig.SourceObservedInkPNGB64
		}
		if p.SourceDomainPNGB64 == "" {
			p.SourceDomainPNGB64 = orig.SourceDomainPNGB64
		}
		if p.RefinedStrokePNGB64 == "" {
			p.RefinedStrokePNGB64 = orig.RefinedStrokePNGB64
		}
		if p.InpaintUnknownPNGB64 == "" {
			p.InpaintUnknownPNGB64 = orig.InpaintUnknownPNGB64
		}
		if p.InferredBackgroundPNGB64 == "" {
			p.InferredBackgroundPNGB64 = orig.InferredBackgroundPNGB64
		}
		if p.BoundaryOpacityPNGB64 == "" {
			p.BoundaryOpacityPNGB64 = orig.BoundaryOpacityPNGB64
		}
	}
}

func (c *Client) Render(ctx context.Context, rawPNG []byte, pageDOM dom.PageDOM, targetLang string) (RenderResponse, error) {
	strippedDOM, savedDebug := stripDebugPatchLayers(pageDOM)
	reqPayload := RenderRequest{
		RawImagePNGB64: base64.StdEncoding.EncodeToString(rawPNG),
		DOM:            strippedDOM,
		TargetLang:     targetLang,
	}
	var out RenderResponse
	if err := c.postJSON(ctx, "/v1/render", reqPayload, &out); err != nil {
		return RenderResponse{}, err
	}
	restoreDebugPatchLayers(&out.DOM, savedDebug)
	return out, nil
}

func (c *Client) CheckSeam(ctx context.Context, topPNG, bottomPNG []byte) (bool, error) {
	payload := map[string]string{
		"top_png_b64":    base64.StdEncoding.EncodeToString(topPNG),
		"bottom_png_b64": base64.StdEncoding.EncodeToString(bottomPNG),
	}
	var out struct {
		NeedsStitch bool `json:"needs_stitch"`
	}
	if err := c.postJSON(ctx, "/v1/check_seam", payload, &out); err != nil {
		return false, err
	}
	return out.NeedsStitch, nil
}

func (c *Client) postJSON(ctx context.Context, path string, reqBody any, respOut any) error {
	b, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		errBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &httpStatusError{
			Path:       path,
			StatusCode: resp.StatusCode,
			Body:       string(errBytes),
		}
	}
	return json.NewDecoder(resp.Body).Decode(respOut)
}
