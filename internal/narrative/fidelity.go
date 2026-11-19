package narrative

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"prism-t/internal/dom"
)

// Digits are source evidence, not a translation dictionary. Requiring their
// spelling to survive also protects chapter labels and repeated quantities.
var writtenDigits = regexp.MustCompile(`\p{Nd}+`)

// Repeated masking marks carry anonymous identity. These are source symbols,
// not names or dialogue mappings; their spelling and multiplicity must survive.
var anonymousMarks = regexp.MustCompile(`[×〇○●]{2,}`)

func translationFidelityIssue(request NodeTranslationRequest, result NodeTranslationResult, vision bool) string {
	if request.SourceTextLocked {
		if (result.IsText == nil || *result.IsText) && result.SourceText != request.SourceText {
			return fmt.Sprintf("source_text is locked by the independent source-only transcription; return it byte-for-byte unchanged: %q", request.SourceText)
		}
		if result.IsText != nil && !*result.IsText {
			return ""
		}
	}
	if strings.TrimSpace(result.TargetText) != result.TargetText {
		return "target_text must not have leading or trailing whitespace"
	}
	if len(result.TargetSegments) == 0 || strings.Join(result.TargetSegments, "") != result.TargetText {
		return "target_segments must be a nonempty exact partition of target_text; preserve all spaces and punctuation"
	}
	for _, segment := range result.TargetSegments {
		if segment == "" {
			return "target_segments cannot contain empty units"
		}
	}
	source := request.SourceText
	if vision && !request.SourceTextLocked && result.SourceText != "" {
		source = result.SourceText
	}
	for _, evidence := range []struct {
		pattern *regexp.Regexp
		label   string
	}{{writtenDigits, "written digit runs"}, {anonymousMarks, "anonymous masking-symbol runs"}} {
		balance := make(map[string]int)
		for _, token := range evidence.pattern.FindAllString(source, -1) {
			balance[token]++
		}
		for _, token := range evidence.pattern.FindAllString(result.TargetText, -1) {
			balance[token]--
		}
		var missing, extra []string
		for token, count := range balance {
			if count > 0 {
				for i := 0; i < count; i++ {
					missing = append(missing, token)
				}
			} else if count < 0 {
				for i := 0; i < -count; i++ {
					extra = append(extra, token)
				}
			}
		}
		if len(missing) > 0 || len(extra) > 0 {
			sort.Strings(missing)
			sort.Strings(extra)
			return fmt.Sprintf("%s must match the source exactly, including repetitions (missing=%q, extra=%q; keep source digits and symbols verbatim rather than spelling numbers out in words); source=%q, draft=%q", evidence.label, missing, extra, source, result.TargetText)
		}
	}
	return ""
}

func isChineseTargetLang(lang string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(lang)), "zh")
}

// normalizeChineseEllipses collapses nonstandard '…'+ runs (1, 3+, or split across segments)
// to standard Chinese double-ellipsis "……" without disturbing concatenation invariants,
// digits, or anonymous masking marks.
func normalizeChineseEllipses(targetText string, segments []string) (string, []string) {
	if len(segments) == 0 {
		if !strings.ContainsRune(targetText, '…') {
			return targetText, segments
		}
		var b strings.Builder
		b.Grow(len(targetText))
		inEllipsis := false
		for _, r := range targetText {
			if r == '…' {
				if !inEllipsis {
					b.WriteString("……")
					inEllipsis = true
				}
				continue
			}
			inEllipsis = false
			b.WriteRune(r)
		}
		return b.String(), segments
	}

	hasEllipsis := strings.ContainsRune(targetText, '…')
	if !hasEllipsis {
		for _, seg := range segments {
			if strings.ContainsRune(seg, '…') {
				hasEllipsis = true
				break
			}
		}
	}
	if !hasEllipsis {
		return targetText, segments
	}

	cleanSegments := make([]string, 0, len(segments))
	inEllipsis := false
	for _, seg := range segments {
		var b strings.Builder
		b.Grow(len(seg))
		for _, r := range seg {
			if r == '…' {
				if !inEllipsis {
					b.WriteString("……")
					inEllipsis = true
				}
				continue
			}
			inEllipsis = false
			b.WriteRune(r)
		}
		if b.Len() > 0 {
			cleanSegments = append(cleanSegments, b.String())
		}
	}
	if len(cleanSegments) == 0 {
		return targetText, segments
	}
	return strings.Join(cleanSegments, ""), cleanSegments
}

// annotateDiscourseTopology derives purely punctuation-based structural hints (mixed
// spoken-clause vs parenthetical monologue, plus cross-node unclosed quotes/parentheses).
func annotateDiscourseTopology(requests []NodeTranslationRequest) {
	if len(requests) == 0 {
		return
	}
	order := make([]int, len(requests))
	for i := range order {
		order[i] = i
		requests[i].DiscourseHint = ""
	}
	sort.SliceStable(order, func(a, b int) bool {
		return requests[order[a]].ReadingOrder < requests[order[b]].ReadingOrder
	})

	type openMarker struct {
		reqIdx int
		ch     rune
	}
	matchOpener := map[rune]rune{
		'」': '「',
		'』': '『',
		'）': '（',
	}
	var openStack []openMarker
	mixedHints := make([]bool, len(requests))
	opensCross := make([]bool, len(requests))
	closesCross := make([]bool, len(requests))
	clauseContinues := make([]bool, len(requests))
	clauseFromPrev := make([]bool, len(requests))

	for pos, idx := range order {
		src := discourseSourceText(requests[idx])
		trimmedSrc := strings.TrimSpace(src)
		if trimmedSrc == "" {
			continue
		}
		// Detect explicit mid-sentence clause continuation across consecutive primary nodes
		// (e.g. a node ending in a trailing comma '、'/'，' whose predicate lands in the next node).
		if requests[idx].Track != "aside" && pos+1 < len(order) {
			nextIdx := order[pos+1]
			nextSrc := strings.TrimSpace(discourseSourceText(requests[nextIdx]))
			if requests[nextIdx].Track != "aside" && nextSrc != "" {
				var lastRune rune
				for _, r := range trimmedSrc {
					lastRune = r
				}
				if lastRune == '、' || lastRune == '，' || lastRune == ',' {
					clauseContinues[idx] = true
					clauseFromPrev[nextIdx] = true
				}
			}
		}
		parenDepth := 0
		hasInside := false
		hasOutside := false
		for _, r := range src {
			switch r {
			case '（', '(':
				parenDepth++
			case '）', ')':
				if parenDepth > 0 {
					parenDepth--
				}
			default:
				if !unicode.IsSpace(r) {
					if parenDepth > 0 {
						hasInside = true
					} else {
						hasOutside = true
					}
				}
			}
			switch r {
			case '「', '『', '（':
				openStack = append(openStack, openMarker{reqIdx: idx, ch: r})
			case '」', '』', '）':
				want := matchOpener[r]
				for k := len(openStack) - 1; k >= 0; k-- {
					if openStack[k].ch == want {
						if openStack[k].reqIdx != idx {
							opensCross[openStack[k].reqIdx] = true
							closesCross[idx] = true
						}
						openStack = openStack[:k]
						break
					}
				}
			}
		}
		if hasInside && hasOutside {
			mixedHints[idx] = true
		}
	}

	for i := range requests {
		var parts []string
		if mixedHints[i] {
			parts = append(parts, "mixed_outer_and_parenthetical_monologue")
		}
		if opensCross[i] {
			parts = append(parts, "unclosed_quote_or_paren_continues_to_next")
		}
		if closesCross[i] {
			parts = append(parts, "closes_quote_or_paren_from_prev")
		}
		if clauseContinues[i] {
			parts = append(parts, "clause_continues_to_next")
		}
		if clauseFromPrev[i] {
			parts = append(parts, "continues_clause_from_prev")
		}
		requests[i].DiscourseHint = strings.Join(parts, "; ")
	}
}

func discourseSourceText(req NodeTranslationRequest) string {
	if strings.TrimSpace(req.SourceText) != "" || len(req.SourceReadings) == 0 {
		return req.SourceText
	}
	readings := append([]dom.SourceReading(nil), req.SourceReadings...)
	sort.SliceStable(readings, func(i, j int) bool {
		return readings[i].LineIndex < readings[j].LineIndex
	})
	seenLine := make(map[int]bool, len(readings))
	var parts []string
	for _, r := range readings {
		if seenLine[r.LineIndex] {
			continue
		}
		txt := strings.TrimSpace(r.Text)
		if txt == "" {
			continue
		}
		seenLine[r.LineIndex] = true
		parts = append(parts, txt)
	}
	return strings.Join(parts, "")
}
