package narrative

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func writeChatStream(w http.ResponseWriter, content string) {
	w.Header().Set("Content-Type", "text/event-stream")
	chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{
		"index": 0, "delta": map[string]string{"content": content}, "finish_reason": "stop",
	}}})
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
}

func TestBackendRequestsExactStructuredContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Stream         bool            `json:"stream"`
			Messages       json.RawMessage `json:"messages"`
			ResponseFormat struct {
				Type       string `json:"type"`
				JSONSchema struct {
					Strict bool           `json:"strict"`
					Schema map[string]any `json:"schema"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		format := request.ResponseFormat
		if !request.Stream {
			t.Error("structured completions must stream without weakening the schema")
		}
		if !strings.Contains(string(request.Messages), "source_quadrilaterals") || !strings.Contains(string(request.Messages), "owned writing") {
			t.Error("visual translation must receive the node's owned line geometry and grounding instruction")
		}
		if format.Type != "json_schema" || !format.JSONSchema.Strict {
			t.Error("response must be constrained by the strict schema")
		}
		object := format.JSONSchema.Schema["properties"].(map[string]any)["translations"].(map[string]any)
		if object["type"] != "object" || object["additionalProperties"] != false || len(object["required"].([]any)) != 1 {
			t.Error("schema must require the complete requested batch")
		}
		ids := object["properties"].(map[string]any)["requested"].(map[string]any)["properties"].(map[string]any)["id"].(map[string]any)["enum"].([]any)
		fields := object["properties"].(map[string]any)["requested"].(map[string]any)["properties"].(map[string]any)
		if fields["target_segments"].(map[string]any)["type"] != "array" {
			t.Error("missing semantic line breaking contract")
		}
		if fields["target_text"] != nil {
			t.Error("redundant independently generated target_text returned to the wire contract")
		}
		if len(ids) != 1 || ids[0] != "requested" {
			t.Error("schema must constrain node IDs")
		}
		writeChatStream(w, `{"translations":{"requested":{"id":"requested","is_text":true,"source_text":"source","target_segments":["完整","\n译文"],"reason":"","rubies":[]}}}`)
	}))
	defer server.Close()
	backend := &OpenAICompatibleBackend{baseURL: server.URL, httpClient: server.Client(), diagnosticsPath: t.TempDir()}
	result, err := backend.TranslateNodes(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "requested", SourceQuadrilaterals: [][][]float64{{{1, 2}, {3, 2}, {3, 4}, {1, 4}}}}})
	if err != nil || len(result) != 1 || result[0].TargetText != "完整\n译文" || strings.Join(result[0].TargetSegments, "") != result[0].TargetText {
		t.Fatalf("actual response was not decoded: %+v, %v", result, err)
	}
}

func TestWrongResponseEnvelopeIsAnExplicitContractError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeChatStream(w, `{"nodes":[{"id":"requested","target_text":"译文"}]}`)
	}))
	defer server.Close()
	backend := &OpenAICompatibleBackend{baseURL: server.URL, httpClient: server.Client(), diagnosticsPath: t.TempDir()}
	_, err := backend.TranslateNodes(context.Background(), "zh-CN", nil, nil, []NodeTranslationRequest{{ID: "requested"}})
	if err == nil || !strings.Contains(err.Error(), "required translations object") {
		t.Fatalf("wrong envelope must not become an empty translation batch: %v", err)
	}
}

func TestKeyedTranslationRejectsDuplicateAndConflictingIdentity(t *testing.T) {
	for _, raw := range []string{
		`{"node":{"id":"node","target_segments":["a"]},"node":{"id":"node","target_segments":["b"]}}`,
		`{"node":{"id":"other","target_segments":["a"]}}`,
		`[{"id":"node","target_segments":["a"]}]`,
		`{"node":{"id":"node","target_segments":["a"],"target_text":"a different generated answer"}}`,
	} {
		var decoded keyedTranslations
		if err := json.Unmarshal([]byte(raw), &decoded); err == nil {
			t.Fatalf("ambiguous identity accepted: %s", raw)
		}
	}
}
