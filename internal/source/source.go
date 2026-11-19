package source

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"prism-t/internal/archive"
)

const (
	DefaultMangaDexBase = "https://api.mangadex.org"
	defaultUserAgent    = "Prism-T/1.0 (Lossless Comic Localization Workbench)"
	maxDownloadBytes    = 256 << 20 // 256 MiB safety cap for remote archives
	maxImageBytes       = 40 << 20  // 40 MiB per page
)

// ComicEntry represents a comic book returned by a catalog or search query.
type ComicEntry struct {
	ID            string   `json:"id"`
	Title         string   `json:"title"`
	Description   string   `json:"description,omitempty"`
	CoverURL      string   `json:"cover_url,omitempty"`
	SourceKind    string   `json:"source_kind"`
	URL           string   `json:"url"`
	FormatProfile string   `json:"format_profile,omitempty"`
	Languages     []string `json:"languages,omitempty"`
	Tags          []string `json:"tags,omitempty"`
}

// ChapterEntry represents a selectable chapter or downloadable unit inside a comic source.
type ChapterEntry struct {
	ID         string      `json:"id"`
	Title      string      `json:"title"`
	ChapterNum string      `json:"chapter_num,omitempty"`
	Volume     string      `json:"volume,omitempty"`
	Language   string      `json:"language,omitempty"`
	PageCount  int         `json:"page_count,omitempty"`
	SourceKind string      `json:"source_kind"`
	URL        string      `json:"url"`
	ArchiveURL string      `json:"archive_url,omitempty"`
	Pages      []PageAsset `json:"pages,omitempty"`
	PublishAt  string      `json:"publish_at,omitempty"`
}

// PageAsset describes one remote comic page ready to stream-download.
type PageAsset struct {
	Index    int               `json:"index"`
	Filename string            `json:"filename"`
	URL      string            `json:"url"`
	Headers  map[string]string `json:"headers,omitempty"`
}

// BrowseResult is the unified response of probing/browsing any comic source.
type BrowseResult struct {
	Kind       string         `json:"kind"` // "comics" | "chapters" | "pages" | "archive"
	Title      string         `json:"title,omitempty"`
	SourceKind string         `json:"source_kind"`
	Comics     []ComicEntry   `json:"comics,omitempty"`
	Chapters   []ChapterEntry `json:"chapters,omitempty"`
	Pages      []PageAsset    `json:"pages,omitempty"`
	ArchiveURL string         `json:"archive_url,omitempty"`
	NextURL    string         `json:"next_url,omitempty"`
}

// StreamedPage carries a downloaded page (or error) in reading order.
type StreamedPage struct {
	Index int
	Total int
	Page  archive.InputPage
	Err   error
}

// Client handles remote comic source discovery and streaming page downloads.
type Client struct {
	HTTP         *http.Client
	MangaDexBase string
}

// NewClient creates a comic source client honoring standard HTTP(S)_PROXY settings.
func NewClient() *Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = http.ProxyFromEnvironment
	return &Client{
		HTTP: &http.Client{
			Transport: tr,
			Timeout:   45 * time.Second,
		},
		MangaDexBase: DefaultMangaDexBase,
	}
}

func (c *Client) httpClient() *http.Client {
	if c != nil && c.HTTP != nil {
		return c.HTTP
	}
	return NewClient().HTTP
}

func (c *Client) mdBase() string {
	if c != nil && strings.TrimSpace(c.MangaDexBase) != "" {
		return strings.TrimRight(c.MangaDexBase, "/")
	}
	return DefaultMangaDexBase
}

// ValidateHTTPURL ensures only http/https URLs are fetched.
func ValidateHTTPURL(raw string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil, fmt.Errorf("invalid URL %q: %w", raw, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("unsupported URL scheme %q (only http/https allowed)", u.Scheme)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("URL %q is missing host", raw)
	}
	return u, nil
}

func (c *Client) doGet(ctx context.Context, targetURL string, headers map[string]string, limit int64) ([]byte, string, error) {
	if _, err := ValidateHTTPURL(targetURL); err != nil {
		return nil, "", err
	}
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, "", ctx.Err()
			case <-time.After(time.Duration(attempt) * 600 * time.Millisecond):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
		if err != nil {
			return nil, "", err
		}
		req.Header.Set("User-Agent", defaultUserAgent)
		for k, v := range headers {
			if strings.TrimSpace(k) != "" {
				req.Header.Set(k, v)
			}
		}
		resp, err := c.httpClient().Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		body, rErr := io.ReadAll(io.LimitReader(resp.Body, limit+1))
		_ = resp.Body.Close()
		if rErr != nil {
			lastErr = rErr
			continue
		}
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("HTTP %d from %s", resp.StatusCode, targetURL)
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, "", fmt.Errorf("HTTP %d from %s: %s", resp.StatusCode, targetURL, strings.TrimSpace(string(body[:min(len(body), 200)])))
		}
		if int64(len(body)) > limit {
			return nil, "", fmt.Errorf("response from %s exceeds size limit (%d bytes)", targetURL, limit)
		}
		return body, resp.Header.Get("Content-Type"), nil
	}
	return nil, "", fmt.Errorf("request %s failed after retries: %w", targetURL, lastErr)
}

// ==================== MangaDex Public API v5 ====================

var (
	reMangaDexTitle   = regexp.MustCompile(`(?i)mangadex\.org/title/([0-9a-f-]{36})`)
	reMangaDexChapter = regexp.MustCompile(`(?i)mangadex\.org/chapter/([0-9a-f-]{36})`)
	reUUID            = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)
)

func ExtractMangaDexTitleID(input string) string {
	s := strings.TrimSpace(input)
	if m := reMangaDexTitle.FindStringSubmatch(s); len(m) == 2 {
		return m[1]
	}
	return ""
}

func ExtractMangaDexChapterID(input string) string {
	s := strings.TrimSpace(input)
	if m := reMangaDexChapter.FindStringSubmatch(s); len(m) == 2 {
		return m[1]
	}
	return ""
}

func pickLocalized(m map[string]string) string {
	for _, k := range []string{"zh", "zh-hk", "ja", "en", "ja-ro", "ko"} {
		if v := strings.TrimSpace(m[k]); v != "" {
			return v
		}
	}
	for _, v := range m {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func (c *Client) SearchMangaDex(ctx context.Context, query string, limit int) (BrowseResult, error) {
	query = strings.TrimSpace(query)
	if limit <= 0 || limit > 24 {
		limit = 12
	}
	if cid := ExtractMangaDexChapterID(query); cid != "" {
		pages, err := c.ResolveMangaDexChapterPages(ctx, cid, false)
		if err != nil {
			return BrowseResult{}, err
		}
		return BrowseResult{
			Kind:       "pages",
			Title:      "MangaDex Chapter " + cid[:8],
			SourceKind: "mangadex",
			Pages:      pages,
		}, nil
	}
	if tid := ExtractMangaDexTitleID(query); tid != "" || reUUID.MatchString(query) {
		id := tid
		if id == "" {
			id = query
		}
		return c.ListMangaDexChapters(ctx, id, "")
	}

	u := fmt.Sprintf("%s/manga?limit=%d&includes[]=cover_art&order[relevance]=desc&contentRating[]=safe&contentRating[]=suggestive",
		c.mdBase(), limit)
	if query != "" {
		u += "&title=" + url.QueryEscape(query)
	}
	raw, _, err := c.doGet(ctx, u, nil, 8<<20)
	if err != nil {
		return BrowseResult{}, err
	}
	var payload struct {
		Data []struct {
			ID         string `json:"id"`
			Attributes struct {
				Title                        map[string]string   `json:"title"`
				Description                  map[string]string   `json:"description"`
				OriginalLanguage             string              `json:"originalLanguage"`
				AvailableTranslatedLanguages []string            `json:"availableTranslatedLanguages"`
				Tags                         []struct {
					Attributes struct {
						Name map[string]string `json:"name"`
					} `json:"attributes"`
				} `json:"tags"`
			} `json:"attributes"`
			Relationships []struct {
				Type       string `json:"type"`
				Attributes struct {
					FileName string `json:"fileName"`
				} `json:"attributes"`
			} `json:"relationships"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return BrowseResult{}, fmt.Errorf("parse MangaDex search: %w", err)
	}
	comics := make([]ComicEntry, 0, len(payload.Data))
	for _, item := range payload.Data {
		title := pickLocalized(item.Attributes.Title)
		if title == "" {
			title = item.ID
		}
		coverURL := ""
		for _, rel := range item.Relationships {
			if rel.Type == "cover_art" && rel.Attributes.FileName != "" {
				coverURL = fmt.Sprintf("https://uploads.mangadex.org/covers/%s/%s.256.jpg", item.ID, rel.Attributes.FileName)
				break
			}
		}
		profile := "manga_rtl"
		if item.Attributes.OriginalLanguage == "ko" {
			profile = "webtoon_vertical"
		} else if item.Attributes.OriginalLanguage == "en" {
			profile = "comic_ltr"
		}
		var tags []string
		for _, t := range item.Attributes.Tags {
			if name := pickLocalized(t.Attributes.Name); name != "" {
				tags = append(tags, name)
			}
			if len(tags) >= 4 {
				break
			}
		}
		desc := pickLocalized(item.Attributes.Description)
		if len([]rune(desc)) > 140 {
			desc = string([]rune(desc)[:140]) + "…"
		}
		comics = append(comics, ComicEntry{
			ID:            item.ID,
			Title:         title,
			Description:   desc,
			CoverURL:      coverURL,
			SourceKind:    "mangadex",
			URL:           "https://mangadex.org/title/" + item.ID,
			FormatProfile: profile,
			Languages:     item.Attributes.AvailableTranslatedLanguages,
			Tags:          tags,
		})
	}
	return BrowseResult{
		Kind:       "comics",
		Title:      "MangaDex",
		SourceKind: "mangadex",
		Comics:     comics,
	}, nil
}

func (c *Client) ListMangaDexChapters(ctx context.Context, mangaIDOrURL, lang string) (BrowseResult, error) {
	id := strings.TrimSpace(mangaIDOrURL)
	if extracted := ExtractMangaDexTitleID(id); extracted != "" {
		id = extracted
	}
	if !reUUID.MatchString(id) {
		return BrowseResult{}, fmt.Errorf("invalid MangaDex manga ID: %q", mangaIDOrURL)
	}
	title := "MangaDex " + id[:8]
	if metaRaw, _, err := c.doGet(ctx, fmt.Sprintf("%s/manga/%s", c.mdBase(), id), nil, 2<<20); err == nil {
		var meta struct {
			Data struct {
				Attributes struct {
					Title map[string]string `json:"title"`
				} `json:"attributes"`
			} `json:"data"`
		}
		if json.Unmarshal(metaRaw, &meta) == nil {
			if t := pickLocalized(meta.Data.Attributes.Title); t != "" {
				title = t
			}
		}
	}

	u := fmt.Sprintf("%s/manga/%s/feed?limit=96&order[volume]=asc&order[chapter]=asc&includeExternalUrl=0&contentRating[]=safe&contentRating[]=suggestive",
		c.mdBase(), id)
	if strings.TrimSpace(lang) != "" {
		u += "&translatedLanguage[]=" + url.QueryEscape(strings.TrimSpace(lang))
	}
	raw, _, err := c.doGet(ctx, u, nil, 8<<20)
	if err != nil {
		return BrowseResult{}, err
	}
	var payload struct {
		Data []struct {
			ID         string `json:"id"`
			Attributes struct {
				Title              string `json:"title"`
				Chapter            string `json:"chapter"`
				Volume             string `json:"volume"`
				TranslatedLanguage string `json:"translatedLanguage"`
				Pages              int    `json:"pages"`
				PublishAt          string `json:"publishAt"`
				ExternalURL        string `json:"externalUrl"`
			} `json:"attributes"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return BrowseResult{}, fmt.Errorf("parse MangaDex feed: %w", err)
	}
	chapters := make([]ChapterEntry, 0, len(payload.Data))
	for _, ch := range payload.Data {
		if ch.Attributes.Pages <= 0 || ch.Attributes.ExternalURL != "" {
			continue
		}
		label := strings.TrimSpace(ch.Attributes.Title)
		chNum := strings.TrimSpace(ch.Attributes.Chapter)
		if chNum != "" && label != "" {
			label = fmt.Sprintf("Ch.%s %s", chNum, label)
		} else if chNum != "" {
			label = fmt.Sprintf("Chapter %s", chNum)
		} else if label == "" {
			label = "Oneshot"
		}
		chapters = append(chapters, ChapterEntry{
			ID:         ch.ID,
			Title:      label,
			ChapterNum: chNum,
			Volume:     ch.Attributes.Volume,
			Language:   ch.Attributes.TranslatedLanguage,
			PageCount:  ch.Attributes.Pages,
			SourceKind: "mangadex",
			URL:        "https://mangadex.org/chapter/" + ch.ID,
			PublishAt:  ch.Attributes.PublishAt,
		})
	}
	return BrowseResult{
		Kind:       "chapters",
		Title:      title,
		SourceKind: "mangadex",
		Chapters:   chapters,
	}, nil
}

func (c *Client) ResolveMangaDexChapterPages(ctx context.Context, chapterIDOrURL string, dataSaver bool) ([]PageAsset, error) {
	id := strings.TrimSpace(chapterIDOrURL)
	if extracted := ExtractMangaDexChapterID(id); extracted != "" {
		id = extracted
	}
	if !reUUID.MatchString(id) {
		return nil, fmt.Errorf("invalid MangaDex chapter ID: %q", chapterIDOrURL)
	}
	u := fmt.Sprintf("%s/at-home/server/%s", c.mdBase(), id)
	raw, _, err := c.doGet(ctx, u, nil, 4<<20)
	if err != nil {
		return nil, err
	}
	var payload struct {
		BaseURL string `json:"baseUrl"`
		Chapter struct {
			Hash      string   `json:"hash"`
			Data      []string `json:"data"`
			DataSaver []string `json:"dataSaver"`
		} `json:"chapter"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("parse MangaDex at-home: %w", err)
	}
	files := payload.Chapter.Data
	mode := "data"
	if dataSaver && len(payload.Chapter.DataSaver) > 0 {
		files = payload.Chapter.DataSaver
		mode = "data-saver"
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("MangaDex chapter %s returned 0 pages", id)
	}
	base := strings.TrimRight(payload.BaseURL, "/")
	pages := make([]PageAsset, 0, len(files))
	for i, fn := range files {
		ext := filepath.Ext(fn)
		if ext == "" {
			ext = ".jpg"
		}
		pages = append(pages, PageAsset{
			Index:    i,
			Filename: fmt.Sprintf("%03d%s", i+1, ext),
			URL:      fmt.Sprintf("%s/%s/%s/%s", base, mode, payload.Chapter.Hash, fn),
		})
	}
	return pages, nil
}

// ==================== OPDS 1.2 Atom XML & JSON Manifest ====================

type atomFeed struct {
	Title   string      `xml:"title"`
	Links   []atomLink  `xml:"link"`
	Entries []atomEntry `xml:"entry"`
}

type atomEntry struct {
	ID      string     `xml:"id"`
	Title   string     `xml:"title"`
	Summary string     `xml:"summary"`
	Content string     `xml:"content"`
	Updated string     `xml:"updated"`
	Links   []atomLink `xml:"link"`
}

type atomLink struct {
	Rel   string `xml:"rel,attr"`
	Href  string `xml:"href,attr"`
	Type  string `xml:"type,attr"`
	Title string `xml:"title,attr"`
}

type jsonManifest struct {
	Name   string `json:"name"`
	Title  string `json:"title"`
	Pages  []string `json:"pages"`
	Comics []struct {
		ID            string `json:"id"`
		Title         string `json:"title"`
		Description   string `json:"description"`
		CoverURL      string `json:"cover_url"`
		URL           string `json:"url"`
		FormatProfile string `json:"format_profile"`
		Chapters      []struct {
			ID         string   `json:"id"`
			Title      string   `json:"title"`
			ChapterNum string   `json:"chapter_num"`
			Language   string   `json:"language"`
			ArchiveURL string   `json:"archive_url"`
			URL        string   `json:"url"`
			Pages      []string `json:"pages"`
		} `json:"chapters"`
	} `json:"comics"`
	Chapters []struct {
		ID         string   `json:"id"`
		Title      string   `json:"title"`
		ChapterNum string   `json:"chapter_num"`
		Language   string   `json:"language"`
		ArchiveURL string   `json:"archive_url"`
		URL        string   `json:"url"`
		Pages      []string `json:"pages"`
	} `json:"chapters"`
}

func resolveRef(baseURL, href string) string {
	href = strings.TrimSpace(href)
	if href == "" {
		return ""
	}
	base, err := url.Parse(baseURL)
	if err != nil {
		return href
	}
	ref, err := url.Parse(href)
	if err != nil {
		return href
	}
	return base.ResolveReference(ref).String()
}

func (c *Client) BrowseOPDSOrJSON(ctx context.Context, feedURL string, headers map[string]string) (BrowseResult, error) {
	body, ct, err := c.doGet(ctx, feedURL, headers, 16<<20)
	if err != nil {
		return BrowseResult{}, err
	}
	trimmed := bytes.TrimSpace(body)
	if strings.HasPrefix(string(trimmed), "{") || strings.Contains(strings.ToLower(ct), "json") {
		return parseJSONManifest(feedURL, trimmed, headers)
	}
	return parseOPDSAtom(feedURL, trimmed)
}

func parseJSONManifest(baseURL string, body []byte, headers map[string]string) (BrowseResult, error) {
	var m jsonManifest
	if err := json.Unmarshal(body, &m); err != nil {
		return BrowseResult{}, fmt.Errorf("parse JSON comic source: %w", err)
	}
	title := strings.TrimSpace(m.Title)
	if title == "" {
		title = strings.TrimSpace(m.Name)
	}
	if len(m.Pages) > 0 {
		pages := make([]PageAsset, 0, len(m.Pages))
		for i, u := range m.Pages {
			resolved := resolveRef(baseURL, u)
			pages = append(pages, PageAsset{
				Index:    i,
				Filename: filenameFromURL(resolved, i),
				URL:      resolved,
				Headers:  headers,
			})
		}
		return BrowseResult{Kind: "pages", Title: title, SourceKind: "json_manifest", Pages: pages}, nil
	}
	if len(m.Chapters) > 0 {
		chs := make([]ChapterEntry, 0, len(m.Chapters))
		for i, ch := range m.Chapters {
			pages := make([]PageAsset, 0, len(ch.Pages))
			for pi, pu := range ch.Pages {
				res := resolveRef(baseURL, pu)
				pages = append(pages, PageAsset{Index: pi, Filename: filenameFromURL(res, pi), URL: res, Headers: headers})
			}
			chs = append(chs, ChapterEntry{
				ID:         firstNonEmpty(ch.ID, fmt.Sprintf("ch_%d", i+1)),
				Title:      firstNonEmpty(ch.Title, fmt.Sprintf("Chapter %d", i+1)),
				ChapterNum: ch.ChapterNum,
				Language:   ch.Language,
				PageCount:  len(pages),
				SourceKind: "json_manifest",
				URL:        resolveRef(baseURL, ch.URL),
				ArchiveURL: resolveRef(baseURL, ch.ArchiveURL),
				Pages:      pages,
			})
		}
		return BrowseResult{Kind: "chapters", Title: title, SourceKind: "json_manifest", Chapters: chs}, nil
	}
	if len(m.Comics) > 0 {
		if len(m.Comics) == 1 && len(m.Comics[0].Chapters) > 0 && m.Comics[0].URL == "" {
			co := m.Comics[0]
			chs := make([]ChapterEntry, 0, len(co.Chapters))
			for i, ch := range co.Chapters {
				pages := make([]PageAsset, 0, len(ch.Pages))
				for pi, pu := range ch.Pages {
					res := resolveRef(baseURL, pu)
					pages = append(pages, PageAsset{Index: pi, Filename: filenameFromURL(res, pi), URL: res, Headers: headers})
				}
				chs = append(chs, ChapterEntry{
					ID:         firstNonEmpty(ch.ID, fmt.Sprintf("ch_%d", i+1)),
					Title:      firstNonEmpty(ch.Title, fmt.Sprintf("Chapter %d", i+1)),
					ChapterNum: ch.ChapterNum,
					Language:   ch.Language,
					PageCount:  len(pages),
					SourceKind: "json_manifest",
					URL:        resolveRef(baseURL, ch.URL),
					ArchiveURL: resolveRef(baseURL, ch.ArchiveURL),
					Pages:      pages,
				})
			}
			return BrowseResult{Kind: "chapters", Title: firstNonEmpty(co.Title, title), SourceKind: "json_manifest", Chapters: chs}, nil
		}
		comics := make([]ComicEntry, 0, len(m.Comics))
		for i, co := range m.Comics {
			comics = append(comics, ComicEntry{
				ID:            firstNonEmpty(co.ID, fmt.Sprintf("comic_%d", i+1)),
				Title:         firstNonEmpty(co.Title, fmt.Sprintf("Comic %d", i+1)),
				Description:   co.Description,
				CoverURL:      resolveRef(baseURL, co.CoverURL),
				SourceKind:    "json_manifest",
				URL:           resolveRef(baseURL, co.URL),
				FormatProfile: co.FormatProfile,
			})
		}
		return BrowseResult{Kind: "comics", Title: title, SourceKind: "json_manifest", Comics: comics}, nil
	}
	return BrowseResult{}, fmt.Errorf("JSON source at %s contains no comics, chapters, or pages", baseURL)
}

func parseOPDSAtom(baseURL string, body []byte) (BrowseResult, error) {
	var feed atomFeed
	if err := xml.Unmarshal(body, &feed); err != nil {
		return BrowseResult{}, fmt.Errorf("parse OPDS Atom XML: %w", err)
	}
	var nextURL string
	for _, l := range feed.Links {
		if l.Rel == "next" && l.Href != "" {
			nextURL = resolveRef(baseURL, l.Href)
		}
	}
	var comics []ComicEntry
	var chapters []ChapterEntry
	for i, e := range feed.Entries {
		var coverURL, navURL, acqURL string
		for _, l := range e.Links {
			href := resolveRef(baseURL, l.Href)
			rel := strings.ToLower(l.Rel)
			typ := strings.ToLower(l.Type)
			if strings.Contains(rel, "image") && coverURL == "" {
				coverURL = href
			}
			if strings.Contains(rel, "acquisition") || strings.Contains(typ, "comicbook") || strings.Contains(typ, "application/zip") {
				if acqURL == "" {
					acqURL = href
				}
			}
			if (strings.Contains(typ, "application/atom+xml") || rel == "subsection" || rel == "http://opds-spec.org/sort/new") && navURL == "" {
				navURL = href
			}
		}
		if acqURL != "" {
			chapters = append(chapters, ChapterEntry{
				ID:         firstNonEmpty(e.ID, fmt.Sprintf("opds_ch_%d", i+1)),
				Title:      firstNonEmpty(strings.TrimSpace(e.Title), fmt.Sprintf("Chapter %d", i+1)),
				SourceKind: "opds",
				URL:        acqURL,
				ArchiveURL: acqURL,
				PublishAt:  e.Updated,
			})
		} else if navURL != "" {
			comics = append(comics, ComicEntry{
				ID:          firstNonEmpty(e.ID, fmt.Sprintf("opds_entry_%d", i+1)),
				Title:       firstNonEmpty(strings.TrimSpace(e.Title), fmt.Sprintf("Entry %d", i+1)),
				Description: strings.TrimSpace(firstNonEmpty(e.Summary, e.Content)),
				CoverURL:    coverURL,
				SourceKind:  "opds",
				URL:         navURL,
			})
		}
	}
	if len(chapters) > 0 && len(comics) == 0 {
		return BrowseResult{Kind: "chapters", Title: strings.TrimSpace(feed.Title), SourceKind: "opds", Chapters: chapters, NextURL: nextURL}, nil
	}
	return BrowseResult{Kind: "comics", Title: strings.TrimSpace(feed.Title), SourceKind: "opds", Comics: comics, Chapters: chapters, NextURL: nextURL}, nil
}

// ==================== Direct URLs / Remote CBZ / Gallery Webpage ====================

var reImgSrc = regexp.MustCompile(`(?i)<img[^>]+(?:data-src|data-original|src)=["']([^"']+)["']`)

func (c *Client) ResolveDirectInput(ctx context.Context, rawInput string, headers map[string]string) (BrowseResult, error) {
	lines := strings.Split(strings.TrimSpace(rawInput), "\n")
	var urls []string
	for _, line := range lines {
		s := strings.TrimSpace(line)
		if s == "" || strings.HasPrefix(s, "#") {
			continue
		}
		urls = append(urls, s)
	}
	if len(urls) == 0 {
		return BrowseResult{}, fmt.Errorf("请输入至少一个有效的 http/https 链接")
	}

	// Single URL fast paths
	if len(urls) == 1 {
		single := urls[0]
		if cid := ExtractMangaDexChapterID(single); cid != "" {
			pages, err := c.ResolveMangaDexChapterPages(ctx, cid, false)
			if err != nil {
				return BrowseResult{}, err
			}
			return BrowseResult{Kind: "pages", Title: "MangaDex " + cid[:8], SourceKind: "mangadex", Pages: pages}, nil
		}
		if tid := ExtractMangaDexTitleID(single); tid != "" {
			return c.ListMangaDexChapters(ctx, tid, "")
		}
		parsed, err := ValidateHTTPURL(single)
		if err != nil {
			return BrowseResult{}, err
		}
		ext := strings.ToLower(path.Ext(parsed.Path))
		if ext == ".cbz" || ext == ".zip" {
			return BrowseResult{
				Kind:       "archive",
				Title:      strings.TrimSuffix(path.Base(parsed.Path), ext),
				SourceKind: "direct_archive",
				ArchiveURL: single,
			}, nil
		}
		if archive.IsSupportedImage(parsed.Path) {
			return BrowseResult{
				Kind:       "pages",
				Title:      strings.TrimSuffix(path.Base(parsed.Path), ext),
				SourceKind: "direct_urls",
				Pages:      []PageAsset{{Index: 0, Filename: filenameFromURL(single, 0), URL: single, Headers: headers}},
			}, nil
		}
		if ext == ".json" || ext == ".xml" || ext == ".opds" || strings.Contains(strings.ToLower(parsed.Path), "opds") {
			return c.BrowseOPDSOrJSON(ctx, single, headers)
		}
		// Fetch page and inspect content type: could be JSON, OPDS XML, zip/cbz, image, or HTML gallery.
		body, ct, err := c.doGet(ctx, single, headers, 12<<20)
		if err != nil {
			return BrowseResult{}, err
		}
		lct := strings.ToLower(ct)
		if strings.HasPrefix(lct, "image/") {
			return BrowseResult{
				Kind:       "pages",
				Title:      filenameFromURL(single, 0),
				SourceKind: "direct_urls",
				Pages:      []PageAsset{{Index: 0, Filename: filenameFromURL(single, 0), URL: single, Headers: headers}},
			}, nil
		}
		if strings.Contains(lct, "zip") || strings.Contains(lct, "comicbook") {
			return BrowseResult{Kind: "archive", Title: path.Base(parsed.Path), SourceKind: "direct_archive", ArchiveURL: single}, nil
		}
		trimmed := bytes.TrimSpace(body)
		if strings.HasPrefix(string(trimmed), "{") || strings.Contains(lct, "json") {
			return parseJSONManifest(single, trimmed, headers)
		}
		if strings.Contains(lct, "atom+xml") || strings.HasPrefix(string(trimmed), "<?xml") || bytes.Contains(trimmed[:min(len(trimmed), 256)], []byte("<feed")) {
			return parseOPDSAtom(single, trimmed)
		}
		// Standard HTML gallery image extraction
		matches := reImgSrc.FindAllStringSubmatch(string(body), -1)
		seen := make(map[string]bool)
		var pages []PageAsset
		pageHeaders := copyHeaders(headers)
		if pageHeaders["Referer"] == "" {
			pageHeaders["Referer"] = single
		}
		for _, m := range matches {
			if len(m) < 2 {
				continue
			}
			imgURL := resolveRef(single, m[1])
			u, uErr := ValidateHTTPURL(imgURL)
			if uErr != nil || seen[imgURL] {
				continue
			}
			low := strings.ToLower(u.Path)
			if strings.Contains(low, "avatar") || strings.Contains(low, "icon") || strings.Contains(low, "logo") || strings.HasSuffix(low, ".svg") {
				continue
			}
			if !archive.IsSupportedImage(u.Path) && !strings.Contains(low, "image") && !strings.Contains(low, "page") {
				continue
			}
			seen[imgURL] = true
			pages = append(pages, PageAsset{
				Index:    len(pages),
				Filename: filenameFromURL(imgURL, len(pages)),
				URL:      imgURL,
				Headers:  pageHeaders,
			})
		}
		if len(pages) == 0 {
			return BrowseResult{}, fmt.Errorf("网页 %s 中未提取到漫画图片直链（若是加密/动态站点，建议使用 OPDS / JSON 书源或直接粘贴图片列表）", single)
		}
		return BrowseResult{Kind: "pages", Title: parsed.Host, SourceKind: "direct_gallery", Pages: pages}, nil
	}

	// Multiple lines: ordered list of direct image URLs
	pages := make([]PageAsset, 0, len(urls))
	for i, raw := range urls {
		if _, err := ValidateHTTPURL(raw); err != nil {
			return BrowseResult{}, err
		}
		pages = append(pages, PageAsset{
			Index:    i,
			Filename: filenameFromURL(raw, i),
			URL:      raw,
			Headers:  headers,
		})
	}
	return BrowseResult{
		Kind:       "pages",
		Title:      fmt.Sprintf("在线图集 (%d 页)", len(pages)),
		SourceKind: "direct_urls",
		Pages:      pages,
	}, nil
}

// ==================== Streaming Page & Archive Downloader ====================

func (c *Client) DownloadPage(ctx context.Context, asset PageAsset, extraHeaders map[string]string) (archive.InputPage, error) {
	hdr := copyHeaders(extraHeaders)
	for k, v := range asset.Headers {
		hdr[k] = v
	}
	raw, _, err := c.doGet(ctx, asset.URL, hdr, maxImageBytes)
	if err != nil {
		return archive.InputPage{}, err
	}
	pBytes, hash, w, h, err := archive.NormalizeToPNG(raw)
	if err != nil {
		return archive.InputPage{}, fmt.Errorf("decode %s: %w", asset.Filename, err)
	}
	name := asset.Filename
	if name == "" {
		name = filenameFromURL(asset.URL, asset.Index)
	}
	return archive.InputPage{
		Filename:   name,
		PNGBytes:   pBytes,
		Blake3Hash: hash,
		Width:      w,
		Height:     h,
	}, nil
}

func (c *Client) DownloadArchivePages(ctx context.Context, archiveURL string, extraHeaders map[string]string, maxPages int) ([]archive.InputPage, error) {
	raw, _, err := c.doGet(ctx, archiveURL, extraHeaders, maxDownloadBytes)
	if err != nil {
		return nil, err
	}
	zr, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return nil, fmt.Errorf("open remote archive: %w", err)
	}
	var files []*zip.File
	for _, f := range zr.File {
		if !f.FileInfo().IsDir() && !strings.HasPrefix(filepath.Base(f.Name), ".") && archive.IsSupportedImage(f.Name) {
			files = append(files, f)
		}
	}
	sort.Slice(files, func(i, j int) bool { return archive.NaturalLess(files[i].Name, files[j].Name) })
	if maxPages > 0 && len(files) > maxPages {
		files = files[:maxPages]
	}
	pages := make([]archive.InputPage, 0, len(files))
	for _, f := range files {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		imgRaw, rErr := io.ReadAll(io.LimitReader(rc, maxImageBytes))
		_ = rc.Close()
		if rErr != nil {
			return nil, rErr
		}
		pBytes, hash, w, h, err := archive.NormalizeToPNG(imgRaw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f.Name, err)
		}
		pages = append(pages, archive.InputPage{
			Filename:   filepath.Base(f.Name),
			PNGBytes:   pBytes,
			Blake3Hash: hash,
			Width:      w,
			Height:     h,
		})
	}
	if len(pages) == 0 {
		return nil, fmt.Errorf("remote archive contains no supported comic images")
	}
	return pages, nil
}

// StreamPages concurrently prefetches pages (up to prefetch workers ahead) and emits them in strict
// reading order so page 1 immediately enters translation while pages 2..N download in the background.
func (c *Client) StreamPages(
	ctx context.Context,
	assets []PageAsset,
	extraHeaders map[string]string,
	prefetch int,
	onDownloaded func(downloaded int, total int, filename string),
) <-chan StreamedPage {
	out := make(chan StreamedPage, max(1, prefetch))
	if prefetch < 1 {
		prefetch = 3
	}
	if prefetch > len(assets) && len(assets) > 0 {
		prefetch = len(assets)
	}
	go func() {
		defer close(out)
		n := len(assets)
		if n == 0 {
			return
		}
		slots := make([]chan StreamedPage, n)
		for i := range slots {
			slots[i] = make(chan StreamedPage, 1)
		}
		sem := make(chan struct{}, prefetch)
		var dlCountMu sync.Mutex
		dlCount := 0
		var wg sync.WaitGroup

		// Launch bounded prefetch downloads
		go func() {
			for i, asset := range assets {
				select {
				case <-ctx.Done():
					slots[i] <- StreamedPage{Index: i, Total: n, Err: ctx.Err()}
					continue
				case sem <- struct{}{}:
				}
				wg.Add(1)
				go func(idx int, a PageAsset) {
					defer wg.Done()
					defer func() { <-sem }()
					p, err := c.DownloadPage(ctx, a, extraHeaders)
					if err == nil && onDownloaded != nil {
						dlCountMu.Lock()
						dlCount++
						cur := dlCount
						dlCountMu.Unlock()
						onDownloaded(cur, n, p.Filename)
					}
					slots[idx] <- StreamedPage{Index: idx, Total: n, Page: p, Err: err}
				}(i, asset)
			}
			wg.Wait()
		}()

		// Emit strictly in page order
		for i := 0; i < n; i++ {
			select {
			case <-ctx.Done():
				return
			case item := <-slots[i]:
				select {
				case <-ctx.Done():
					return
				case out <- item:
				}
				if item.Err != nil {
					return
				}
			}
		}
	}()
	return out
}

// ProxyImage fetches an external thumbnail image for safe UI display (e.g., MangaDex covers).
func (c *Client) ProxyImage(ctx context.Context, imageURL string, headers map[string]string) ([]byte, string, error) {
	return c.doGet(ctx, imageURL, headers, 8<<20)
}

func filenameFromURL(raw string, idx int) string {
	u, err := url.Parse(raw)
	if err == nil {
		base := path.Base(u.Path)
		if base != "" && base != "/" && base != "." && archive.IsSupportedImage(base) {
			return fmt.Sprintf("%03d_%s", idx+1, base)
		}
	}
	return fmt.Sprintf("%03d.png", idx+1)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func copyHeaders(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
