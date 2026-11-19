package narrative

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"testing/iotest"
)

func streamEvent(delta map[string]string, finish any) string {
	data, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{
		"index": 0, "delta": delta, "finish_reason": finish,
	}}})
	return fmt.Sprintf("data: %s\r\n\r\n", data)
}

func TestChatStreamReassemblesNativeUnicodeWithoutReasoningOrTruncation(t *testing.T) {
	want := `{"source":"×××君、ッ！","target":"完整译文"}`
	stream := ": keepalive\r\n\r\n" +
		streamEvent(map[string]string{"reasoning_content": "not a source or translation"}, nil) +
		streamEvent(map[string]string{"content": want[:10]}, nil) +
		streamEvent(map[string]string{"content": want[10:]}, nil) +
		streamEvent(nil, "stop") + "data: {\"choices\":[]}\r\n\r\ndata: [DONE]\r\n\r\n"
	first := 0
	got, err := readChatCompletionStream(iotest.OneByteReader(strings.NewReader(stream)), func() { first++ })
	if err != nil || got.Content != want || got.FinishReason != "stop" || first != 1 {
		t.Fatalf("stream changed source bytes or mixed reasoning into answer: %+v, %v, first=%d", got, err, first)
	}
	// Dense-page JSON may exceed bufio.Scanner's default token size.
	large := strings.Repeat("字", 40000)
	got, err = readChatCompletionStream(strings.NewReader(streamEvent(map[string]string{"content": large}, "stop")+"data: [DONE]\n\n"), nil)
	if err != nil || got.Content != large {
		t.Fatalf("dense answer truncated: bytes=%d, %v", len(got.Content), err)
	}
}

func TestChatStreamNeverAcceptsPartialOrRefusedAnswers(t *testing.T) {
	answer := streamEvent(map[string]string{"content": `{"valid_json":true}`}, nil)
	for name, stream := range map[string]string{
		"eof_after_parseable_json": answer,
		"eof_after_stop":           answer + streamEvent(nil, "stop"),
		"done_without_stop":        answer + "data: [DONE]\n\n",
		"length":                   answer + streamEvent(nil, "length") + "data: [DONE]\n\n",
		"filter":                   answer + streamEvent(nil, "content_filter") + "data: [DONE]\n\n",
		"refusal":                  streamEvent(map[string]string{"refusal": "refused"}, "stop") + "data: [DONE]\n\n",
		"upstream":                 "data: {\"error\":{\"message\":\"overloaded\"}}\n\n",
		"malformed":                "data: not-json\n\n",
		"empty":                    streamEvent(nil, "stop") + "data: [DONE]\n\n",
		"content_after_stop":       answer + streamEvent(nil, "stop") + streamEvent(map[string]string{"content": "extra"}, nil) + "data: [DONE]\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := readChatCompletionStream(strings.NewReader(stream), nil); err == nil {
				t.Fatal("an incomplete/refused stream became an accepted answer")
			}
		})
	}
}
