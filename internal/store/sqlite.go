package store

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"

	"prism-t/internal/archive"
	"prism-t/internal/dom"
)

const SchemaVersion = 1

type Store struct {
	db        *sql.DB
	speakerMu sync.Mutex
}

type Work struct {
	ID            int64  `json:"id"`
	Title         string `json:"title"`
	SourceLang    string `json:"source_lang"`
	TargetLang    string `json:"target_lang"`
	FormatProfile string `json:"format_profile"`
	PageCount     int    `json:"page_count"`
	ReviewCount   int    `json:"review_count"`
	ReviewedCount int    `json:"reviewed_count"`
	ChapterCount  int    `json:"chapter_count"`
	CoverPageID   int64  `json:"cover_page_id"`
	UpdatedAt     string `json:"updated_at"`
	CreatedAt     string `json:"created_at"`
}

type ComicSource struct {
	ID             int64             `json:"id"`
	Name           string            `json:"name"`
	Kind           string            `json:"kind"`
	BaseURL        string            `json:"base_url"`
	Headers        map[string]string `json:"headers,omitempty"`
	DefaultProfile string            `json:"default_profile"`
	CreatedAt      string            `json:"created_at"`
}

type Character struct {
	ID          int64     `json:"id"`
	WorkID      int64     `json:"work_id"`
	ClusterKey  string    `json:"cluster_key"`
	NameRaw     string    `json:"name_raw"`
	NameTarget  string    `json:"name_target"`
	FirstPerson string    `json:"first_person"`
	SpeechStyle string    `json:"speech_style"`
	Appearance  []float32 `json:"-"`
}

type CharacterRelation struct {
	WorkID          int64  `json:"work_id"`
	FromCharID      int64  `json:"from_char_id"`
	ToCharID        int64  `json:"to_char_id"`
	HonorificTarget string `json:"honorific_target"`
	RegisterNote    string `json:"register_note"`
}

type GlossaryTerm struct {
	ID           int64  `json:"id"`
	WorkID       int64  `json:"work_id"`
	SourceTerm   string `json:"source_term"`
	TargetTerm   string `json:"target_term"`
	RubyOverride string `json:"ruby_override"`
	Notes        string `json:"notes"`
}

type PageSummary struct {
	ID                    int64   `json:"id"`
	WorkID                int64   `json:"work_id"`
	ChapterTitle          string  `json:"chapter_title"`
	PageIndex             int     `json:"page_index"`
	Filename              string  `json:"filename"`
	Width                 int     `json:"width"`
	Height                int     `json:"height"`
	FormatProfile         string  `json:"format_profile"`
	Status                string  `json:"status"`
	NeedsReview           bool    `json:"needs_review"`
	TotalNodes            int     `json:"total_nodes"`
	BitExactUnmaskedRatio float64 `json:"bit_exact_unmasked_ratio"`
	UpdatedAt             string  `json:"updated_at"`
}

type PageRecord struct {
	Summary      PageSummary
	Blake3Hash   string
	DOM          dom.PageDOM
	RawPNG       []byte
	CompositePNG []byte
}

type Stats struct {
	Works        int `json:"works"`
	Pages        int `json:"pages"`
	NeedsReview  int `json:"needs_review"`
	Characters   int `json:"characters"`
	GlossarySize int `json:"glossary_size"`
}

func Open(dbPath string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return nil, fmt.Errorf("mkdir db dir: %w", err)
	}
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)", filepath.ToSlash(dbPath))
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1)
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) migrate() error {
	ddl := `
	CREATE TABLE IF NOT EXISTS prism_t_meta (
		key TEXT PRIMARY KEY,
		value TEXT NOT NULL
	);
	CREATE TABLE IF NOT EXISTS works (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		title TEXT NOT NULL UNIQUE,
		source_lang TEXT NOT NULL DEFAULT 'ja',
		target_lang TEXT NOT NULL DEFAULT 'zh-CN',
		format_profile TEXT NOT NULL DEFAULT 'auto',
		created_at TEXT NOT NULL
	);
	CREATE TABLE IF NOT EXISTS pages (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
		chapter_title TEXT NOT NULL DEFAULT 'Chapter 1',
		page_index INTEGER NOT NULL,
		filename TEXT NOT NULL,
		blake3_hash TEXT NOT NULL,
		width INTEGER NOT NULL,
		height INTEGER NOT NULL,
		format_profile TEXT NOT NULL,
		status TEXT NOT NULL DEFAULT 'translated',
		needs_review INTEGER NOT NULL DEFAULT 0,
		total_nodes INTEGER NOT NULL DEFAULT 0,
		bit_exact_ratio REAL NOT NULL DEFAULT 1.0,
		dom_json TEXT NOT NULL,
		raw_png BLOB NOT NULL,
		composite_png BLOB NOT NULL,
		updated_at TEXT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_pages_work_chap ON pages(work_id, chapter_title, page_index);

	CREATE TABLE IF NOT EXISTS characters (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
		cluster_key TEXT NOT NULL,
		name_raw TEXT NOT NULL DEFAULT '',
		name_target TEXT NOT NULL DEFAULT '',
		first_person TEXT NOT NULL DEFAULT '',
		speech_style TEXT NOT NULL DEFAULT '',
		appearance_blob BLOB
	);
	CREATE INDEX IF NOT EXISTS idx_chars_work ON characters(work_id);

	CREATE TABLE IF NOT EXISTS character_relations (
		work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
		from_char_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
		to_char_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
		honorific_target TEXT NOT NULL DEFAULT '',
		register_note TEXT NOT NULL DEFAULT '',
		PRIMARY KEY (work_id, from_char_id, to_char_id)
	);

	CREATE TABLE IF NOT EXISTS glossary_terms (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
		source_term TEXT NOT NULL,
		target_term TEXT NOT NULL,
		ruby_override TEXT NOT NULL DEFAULT '',
		notes TEXT NOT NULL DEFAULT '',
		UNIQUE(work_id, source_term)
	);

	CREATE TABLE IF NOT EXISTS comic_sources (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		kind TEXT NOT NULL,
		base_url TEXT NOT NULL,
		headers_json TEXT NOT NULL DEFAULT '{}',
		default_profile TEXT NOT NULL DEFAULT 'auto',
		created_at TEXT NOT NULL
	);
	`
	if _, err := s.db.Exec(ddl); err != nil {
		return fmt.Errorf("migrate schema: %w", err)
	}
	_, err := s.db.Exec(`INSERT OR REPLACE INTO prism_t_meta(key, value) VALUES('schema_version', '1')`)
	return err
}

const workSelectCols = `
	w.id, w.title, w.source_lang, w.target_lang, w.format_profile, w.created_at,
	(SELECT COUNT(*) FROM pages p WHERE p.work_id = w.id),
	(SELECT COUNT(*) FROM pages p WHERE p.work_id = w.id AND p.needs_review = 1),
	(SELECT COUNT(*) FROM pages p WHERE p.work_id = w.id AND p.status = 'reviewed'),
	(SELECT COUNT(DISTINCT p.chapter_title) FROM pages p WHERE p.work_id = w.id),
	COALESCE((SELECT p.id FROM pages p WHERE p.work_id = w.id ORDER BY p.chapter_title ASC, p.page_index ASC, p.id ASC LIMIT 1), 0),
	COALESCE((SELECT MAX(p.updated_at) FROM pages p WHERE p.work_id = w.id), w.created_at)
`

func scanWork(scanner interface{ Scan(dest ...any) error }) (Work, error) {
	var w Work
	err := scanner.Scan(
		&w.ID, &w.Title, &w.SourceLang, &w.TargetLang, &w.FormatProfile, &w.CreatedAt,
		&w.PageCount, &w.ReviewCount, &w.ReviewedCount, &w.ChapterCount, &w.CoverPageID, &w.UpdatedAt,
	)
	return w, err
}

func (s *Store) EnsureWork(ctx context.Context, title, sourceLang, targetLang, profile string) (Work, error) {
	if title == "" {
		title = "Untitled Work"
	}
	if sourceLang == "" {
		sourceLang = "auto"
	}
	if targetLang == "" {
		targetLang = "zh-CN"
	}
	if profile == "" {
		profile = "auto"
	}
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO works(title, source_lang, target_lang, format_profile, created_at)
		VALUES(?, ?, ?, ?, ?)
		ON CONFLICT(title) DO UPDATE SET target_lang = excluded.target_lang
	`, title, sourceLang, targetLang, profile, now)
	if err != nil {
		return Work{}, err
	}
	return scanWork(s.db.QueryRowContext(ctx, `SELECT `+workSelectCols+` FROM works w WHERE w.title = ?`, title))
}

func (s *Store) GetWork(ctx context.Context, workID int64) (Work, error) {
	return scanWork(s.db.QueryRowContext(ctx, `SELECT `+workSelectCols+` FROM works w WHERE w.id = ?`, workID))
}

func (s *Store) ListWorks(ctx context.Context) ([]Work, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT `+workSelectCols+` FROM works w ORDER BY w.id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Work
	for rows.Next() {
		w, err := scanWork(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, w)
	}
	return out, rows.Err()
}

func (s *Store) RenameWork(ctx context.Context, workID int64, title string) (Work, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		return Work{}, fmt.Errorf("work title cannot be empty")
	}
	res, err := s.db.ExecContext(ctx, `UPDATE works SET title = ? WHERE id = ?`, title, workID)
	if err != nil {
		return Work{}, err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return Work{}, sql.ErrNoRows
	}
	return s.GetWork(ctx, workID)
}

func (s *Store) DeleteWork(ctx context.Context, workID int64) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM works WHERE id = ?`, workID)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *Store) DeletePage(ctx context.Context, pageID int64) (int64, error) {
	var workID int64
	if err := s.db.QueryRowContext(ctx, `SELECT work_id FROM pages WHERE id = ?`, pageID).Scan(&workID); err != nil {
		return 0, err
	}
	if _, err := s.db.ExecContext(ctx, `DELETE FROM pages WHERE id = ?`, pageID); err != nil {
		return 0, err
	}
	return workID, nil
}

func (s *Store) ListComicSources(ctx context.Context) ([]ComicSource, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, name, kind, base_url, headers_json, default_profile, created_at
		FROM comic_sources ORDER BY id DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ComicSource
	for rows.Next() {
		var cs ComicSource
		var hdrJSON string
		if err := rows.Scan(&cs.ID, &cs.Name, &cs.Kind, &cs.BaseURL, &hdrJSON, &cs.DefaultProfile, &cs.CreatedAt); err != nil {
			return nil, err
		}
		if hdrJSON != "" && hdrJSON != "{}" {
			_ = json.Unmarshal([]byte(hdrJSON), &cs.Headers)
		}
		out = append(out, cs)
	}
	return out, rows.Err()
}

func (s *Store) CreateComicSource(ctx context.Context, cs ComicSource) (ComicSource, error) {
	cs.Name = strings.TrimSpace(cs.Name)
	cs.Kind = strings.TrimSpace(cs.Kind)
	cs.BaseURL = strings.TrimSpace(cs.BaseURL)
	if cs.Name == "" || cs.BaseURL == "" {
		return ComicSource{}, fmt.Errorf("source name and base_url are required")
	}
	if cs.Kind == "" {
		cs.Kind = "opds"
	}
	if cs.DefaultProfile == "" {
		cs.DefaultProfile = "auto"
	}
	hdrBytes := []byte("{}")
	if len(cs.Headers) > 0 {
		if b, err := json.Marshal(cs.Headers); err == nil {
			hdrBytes = b
		}
	}
	cs.CreatedAt = time.Now().UTC().Format(time.RFC3339)
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO comic_sources(name, kind, base_url, headers_json, default_profile, created_at)
		VALUES(?, ?, ?, ?, ?, ?)
	`, cs.Name, cs.Kind, cs.BaseURL, string(hdrBytes), cs.DefaultProfile, cs.CreatedAt)
	if err != nil {
		return ComicSource{}, err
	}
	cs.ID, _ = res.LastInsertId()
	return cs, nil
}

func (s *Store) DeleteComicSource(ctx context.Context, id int64) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM comic_sources WHERE id = ?`, id)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func computeNeedsReview(pageDOM dom.PageDOM, status string) int {
	if status == "reviewed" {
		return 0
	}
	for _, n := range pageDOM.Nodes {
		if n.Uncertainty.NeedsReview {
			return 1
		}
	}
	return 0
}

func (s *Store) SavePage(
	ctx context.Context,
	workID int64,
	chapterTitle string,
	pageIndex int,
	filename string,
	blake3Hash string,
	pageDOM dom.PageDOM,
	rawPNG []byte,
	compositePNG []byte,
	status string,
) (int64, error) {
	domBytes, err := json.Marshal(pageDOM)
	if err != nil {
		return 0, fmt.Errorf("marshal PageDOM: %w", err)
	}
	needsRev := computeNeedsReview(pageDOM, status)
	now := time.Now().UTC().Format(time.RFC3339)

	// Upsert if (work_id, chapter_title, filename) already exists
	var existingID int64
	qErr := s.db.QueryRowContext(ctx, `
		SELECT id FROM pages WHERE work_id = ? AND chapter_title = ? AND filename = ?
	`, workID, chapterTitle, filename).Scan(&existingID)

	if qErr == nil && existingID > 0 {
		_, err = s.db.ExecContext(ctx, `
			UPDATE pages SET
				page_index = ?, blake3_hash = ?, width = ?, height = ?, format_profile = ?,
				status = ?, needs_review = ?, total_nodes = ?, bit_exact_ratio = ?,
				dom_json = ?, raw_png = ?, composite_png = ?, updated_at = ?
			WHERE id = ?
		`, pageIndex, blake3Hash, pageDOM.Width, pageDOM.Height, pageDOM.FormatProfile,
			status, needsRev, len(pageDOM.Nodes), pageDOM.Metrics.BitExactUnmaskedRatio,
			string(domBytes), rawPNG, compositePNG, now, existingID)
		return existingID, err
	}

	res, err := s.db.ExecContext(ctx, `
		INSERT INTO pages(
			work_id, chapter_title, page_index, filename, blake3_hash,
			width, height, format_profile, status, needs_review, total_nodes,
			bit_exact_ratio, dom_json, raw_png, composite_png, updated_at
		) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, workID, chapterTitle, pageIndex, filename, blake3Hash,
		pageDOM.Width, pageDOM.Height, pageDOM.FormatProfile, status, needsRev, len(pageDOM.Nodes),
		pageDOM.Metrics.BitExactUnmaskedRatio, string(domBytes), rawPNG, compositePNG, now)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

func (s *Store) SaveTranslationProgress(ctx context.Context, workID int64, chapter string, pageIndex int, progress *dom.TranslationProgress) error {
	return s.savePageProgress(ctx, workID, chapter, pageIndex, "translation_progress", progress)
}

func (s *Store) SaveRecognitionProgress(ctx context.Context, workID int64, chapter string, pageIndex int, progress *dom.RecognitionProgress) error {
	return s.savePageProgress(ctx, workID, chapter, pageIndex, "recognition_progress", progress)
}

func (s *Store) savePageProgress(ctx context.Context, workID int64, chapter string, pageIndex int, field string, progress any) error {
	payload, err := json.Marshal(progress)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `UPDATE pages
		SET dom_json=json_set(dom_json,?,json(?)),updated_at=?
		WHERE work_id=? AND chapter_title=? AND page_index=?`, "$."+field, string(payload), time.Now().UTC().Format(time.RFC3339), workID, chapter, pageIndex)
	return err
}

func (s *Store) UpdatePageDOMAndStatus(
	ctx context.Context,
	pageID int64,
	pageDOM dom.PageDOM,
	compositePNG []byte,
	status string,
) error {
	domBytes, err := json.Marshal(pageDOM)
	if err != nil {
		return fmt.Errorf("marshal PageDOM: %w", err)
	}
	needsRev := computeNeedsReview(pageDOM, status)
	now := time.Now().UTC().Format(time.RFC3339)
	var res sql.Result
	if compositePNG != nil {
		res, err = s.db.ExecContext(ctx, `
			UPDATE pages SET
				width = ?, height = ?, format_profile = ?,
				status = ?, needs_review = ?, total_nodes = ?, bit_exact_ratio = ?,
				dom_json = ?, composite_png = ?, updated_at = ?
			WHERE id = ?
		`, pageDOM.Width, pageDOM.Height, pageDOM.FormatProfile,
			status, needsRev, len(pageDOM.Nodes), pageDOM.Metrics.BitExactUnmaskedRatio,
			string(domBytes), compositePNG, now, pageID)
	} else {
		res, err = s.db.ExecContext(ctx, `
			UPDATE pages SET
				width = ?, height = ?, format_profile = ?,
				status = ?, needs_review = ?, total_nodes = ?, bit_exact_ratio = ?,
				dom_json = ?, updated_at = ?
			WHERE id = ?
		`, pageDOM.Width, pageDOM.Height, pageDOM.FormatProfile,
			status, needsRev, len(pageDOM.Nodes), pageDOM.Metrics.BitExactUnmaskedRatio,
			string(domBytes), now, pageID)
	}
	if err != nil {
		return err
	}
	rows, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if rows == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *Store) UpdatePageRender(ctx context.Context, pageID int64, pageDOM dom.PageDOM, compositePNG []byte, markReviewed bool) error {
	domBytes, err := json.Marshal(pageDOM)
	if err != nil {
		return err
	}
	status := "translated"
	if markReviewed {
		status = "reviewed"
	}
	needsRev := computeNeedsReview(pageDOM, status)
	now := time.Now().UTC().Format(time.RFC3339)
	_, err = s.db.ExecContext(ctx, `
		UPDATE pages SET
			status = ?, needs_review = ?, total_nodes = ?, bit_exact_ratio = ?,
			dom_json = ?, composite_png = ?, updated_at = ?
		WHERE id = ?
	`, status, needsRev, len(pageDOM.Nodes), pageDOM.Metrics.BitExactUnmaskedRatio, string(domBytes), compositePNG, now, pageID)
	return err
}

func (s *Store) GetPage(ctx context.Context, pageID int64) (PageRecord, error) {
	var rec PageRecord
	var needsRev int
	var domStr string
	err := s.db.QueryRowContext(ctx, `
		SELECT id, work_id, chapter_title, page_index, filename, blake3_hash,
		       width, height, format_profile, status, needs_review, total_nodes,
		       bit_exact_ratio, dom_json, raw_png, composite_png, updated_at
		FROM pages WHERE id = ?
	`, pageID).Scan(
		&rec.Summary.ID, &rec.Summary.WorkID, &rec.Summary.ChapterTitle, &rec.Summary.PageIndex,
		&rec.Summary.Filename, &rec.Blake3Hash, &rec.Summary.Width, &rec.Summary.Height,
		&rec.Summary.FormatProfile, &rec.Summary.Status, &needsRev, &rec.Summary.TotalNodes,
		&rec.Summary.BitExactUnmaskedRatio, &domStr, &rec.RawPNG, &rec.CompositePNG, &rec.Summary.UpdatedAt,
	)
	if err != nil {
		return PageRecord{}, err
	}
	rec.Summary.NeedsReview = needsRev != 0
	if err := json.Unmarshal([]byte(domStr), &rec.DOM); err != nil {
		return PageRecord{}, fmt.Errorf("unmarshal PageDOM: %w", err)
	}
	return rec, nil
}

func (s *Store) ListPages(ctx context.Context, workID int64, onlyReview bool) ([]PageSummary, error) {
	query := `
		SELECT id, work_id, chapter_title, page_index, filename, width, height,
		       format_profile, status, needs_review, total_nodes, bit_exact_ratio, updated_at
		FROM pages WHERE 1=1
	`
	var args []any
	if workID > 0 {
		query += ` AND work_id = ?`
		args = append(args, workID)
	}
	if onlyReview {
		query += ` AND needs_review = 1`
	}
	query += ` ORDER BY work_id DESC, id ASC`

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []PageSummary
	for rows.Next() {
		var p PageSummary
		var nr int
		if err := rows.Scan(
			&p.ID, &p.WorkID, &p.ChapterTitle, &p.PageIndex, &p.Filename, &p.Width, &p.Height,
			&p.FormatProfile, &p.Status, &nr, &p.TotalNodes, &p.BitExactUnmaskedRatio, &p.UpdatedAt,
		); err != nil {
			return nil, err
		}
		p.NeedsReview = nr != 0
		out = append(out, p)
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.WorkID != b.WorkID {
			return a.WorkID > b.WorkID
		}
		if a.ChapterTitle != b.ChapterTitle {
			return archive.NaturalLess(a.ChapterTitle, b.ChapterTitle)
		}
		if a.PageIndex != b.PageIndex {
			return a.PageIndex < b.PageIndex
		}
		return a.ID < b.ID
	})
	return out, rows.Err()
}

// MatchOrRegisterSpeaker matches a 64D float32 appearance vector against the Work's persisted characters
// via cosine similarity (>= cos(pi/7) ~ 0.901), creating or updating the persistent Character state machine!
func (s *Store) MatchOrRegisterSpeaker(ctx context.Context, workID int64, appearanceB64 string) (Character, error) {
	s.speakerMu.Lock()
	defer s.speakerMu.Unlock()
	rawBytes, err := base64.StdEncoding.DecodeString(appearanceB64)
	if err != nil || len(rawBytes) < 16 || len(rawBytes)%4 != 0 {
		return Character{}, fmt.Errorf("invalid appearance vector")
	}
	dim := len(rawBytes) / 4
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		bits := binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4])
		vec[i] = math.Float32frombits(bits)
	}

	chars, err := s.ListCharacters(ctx, workID)
	if err != nil {
		return Character{}, err
	}

	bestIdx := -1
	bestSim := float64(-1.0)
	for idx, ch := range chars {
		if len(ch.Appearance) == dim {
			var dot float64
			for k := 0; k < dim; k++ {
				dot += float64(vec[k]) * float64(ch.Appearance[k])
			}
			if dot > bestSim {
				bestSim = dot
				bestIdx = idx
			}
		}
	}

	cosThresh := math.Cos(math.Pi / 7.0)
	if bestIdx >= 0 && bestSim >= cosThresh {
		matched := chars[bestIdx]
		// Exponential moving average update on unit hypersphere
		merged := make([]float32, dim)
		var normSq float64
		for k := 0; k < dim; k++ {
			v := 0.8*matched.Appearance[k] + 0.2*vec[k]
			merged[k] = v
			normSq += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Max(math.Sqrt(normSq), 1e-8))
		blob := make([]byte, dim*4)
		for k := 0; k < dim; k++ {
			merged[k] *= inv
			binary.LittleEndian.PutUint32(blob[k*4:(k+1)*4], math.Float32bits(merged[k]))
		}
		if _, uErr := s.db.ExecContext(ctx, `UPDATE characters SET appearance_blob = ? WHERE id = ?`, blob, matched.ID); uErr != nil {
			return Character{}, uErr
		}
		matched.Appearance = merged
		return matched, nil
	}

	nextOrdinal := 1
	for _, ch := range chars {
		if strings.HasPrefix(ch.ClusterKey, "char_") {
			if n, pErr := strconv.Atoi(strings.TrimPrefix(ch.ClusterKey, "char_")); pErr == nil && n >= nextOrdinal {
				nextOrdinal = n + 1
			}
		}
	}
	if len(chars)+1 > nextOrdinal {
		nextOrdinal = len(chars) + 1
	}
	clusterKey := fmt.Sprintf("char_%02d", nextOrdinal)
	defaultName := fmt.Sprintf("角色#%d", nextOrdinal)
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO characters(work_id, cluster_key, name_raw, name_target, first_person, speech_style, appearance_blob)
		VALUES(?, ?, '', ?, '', '', ?)
	`, workID, clusterKey, defaultName, rawBytes)
	if err != nil {
		return Character{}, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Character{}, err
	}
	return Character{
		ID:         id,
		WorkID:     workID,
		ClusterKey: clusterKey,
		NameTarget: defaultName,
		Appearance: vec,
	}, nil
}

func (s *Store) ListCharacters(ctx context.Context, workID int64) ([]Character, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, work_id, cluster_key, name_raw, name_target, first_person, speech_style, appearance_blob
		FROM characters WHERE work_id = ? ORDER BY id ASC
	`, workID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Character
	for rows.Next() {
		var c Character
		var blob []byte
		if err := rows.Scan(&c.ID, &c.WorkID, &c.ClusterKey, &c.NameRaw, &c.NameTarget, &c.FirstPerson, &c.SpeechStyle, &blob); err != nil {
			return nil, err
		}
		if len(blob) >= 4 && len(blob)%4 == 0 {
			dim := len(blob) / 4
			c.Appearance = make([]float32, dim)
			for k := 0; k < dim; k++ {
				c.Appearance[k] = math.Float32frombits(binary.LittleEndian.Uint32(blob[k*4 : (k+1)*4]))
			}
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (s *Store) UpsertCharacter(ctx context.Context, c Character) (int64, error) {
	if c.ID > 0 {
		_, err := s.db.ExecContext(ctx, `
			UPDATE characters SET name_raw = ?, name_target = ?, first_person = ?, speech_style = ?
			WHERE id = ? AND work_id = ?
		`, c.NameRaw, c.NameTarget, c.FirstPerson, c.SpeechStyle, c.ID, c.WorkID)
		return c.ID, err
	}
	if c.ClusterKey == "" {
		c.ClusterKey = fmt.Sprintf("manual_%d", time.Now().UnixNano()%100000)
	}
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO characters(work_id, cluster_key, name_raw, name_target, first_person, speech_style)
		VALUES(?, ?, ?, ?, ?, ?)
	`, c.WorkID, c.ClusterKey, c.NameRaw, c.NameTarget, c.FirstPerson, c.SpeechStyle)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

func (s *Store) UpsertGlossaryTerm(ctx context.Context, t GlossaryTerm) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO glossary_terms(work_id, source_term, target_term, ruby_override, notes)
		VALUES(?, ?, ?, ?, ?)
		ON CONFLICT(work_id, source_term) DO UPDATE SET
			target_term = excluded.target_term,
			ruby_override = excluded.ruby_override,
			notes = excluded.notes
	`, t.WorkID, t.SourceTerm, t.TargetTerm, t.RubyOverride, t.Notes)
	return err
}

func (s *Store) ListGlossary(ctx context.Context, workID int64) ([]GlossaryTerm, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, work_id, source_term, target_term, ruby_override, notes
		FROM glossary_terms WHERE work_id = ? ORDER BY length(source_term) DESC, id ASC
	`, workID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []GlossaryTerm
	for rows.Next() {
		var g GlossaryTerm
		if err := rows.Scan(&g.ID, &g.WorkID, &g.SourceTerm, &g.TargetTerm, &g.RubyOverride, &g.Notes); err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, rows.Err()
}

func (s *Store) UpsertRelation(ctx context.Context, r CharacterRelation) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO character_relations(work_id, from_char_id, to_char_id, honorific_target, register_note)
		VALUES(?, ?, ?, ?, ?)
		ON CONFLICT(work_id, from_char_id, to_char_id) DO UPDATE SET
			honorific_target = excluded.honorific_target,
			register_note = excluded.register_note
	`, r.WorkID, r.FromCharID, r.ToCharID, r.HonorificTarget, r.RegisterNote)
	return err
}

type DialogueTurn struct {
	PageIndex  int    `json:"page_index"`
	Speaker    string `json:"speaker"`
	SourceText string `json:"source_text"`
	TargetText string `json:"target_text"`
}

// GetRecentChapterDialogue retrieves up to `limit` preceding primary-track dialogue turns from the same
// work and chapter so cross-page sentences and speaker registers remain continuous across page boundaries.
func (s *Store) GetRecentChapterDialogue(ctx context.Context, workID int64, chapterTitle string, beforePageIdx int, limit int) ([]DialogueTurn, error) {
	if limit <= 0 {
		limit = 16
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT page_index, dom_json FROM pages
		WHERE work_id = ? AND chapter_title = ? AND page_index < ?
		ORDER BY page_index DESC LIMIT 6
	`, workID, chapterTitle, beforePageIdx)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type slimNode struct {
		Track            string `json:"track"`
		SpeakerClusterID string `json:"speaker_cluster_id"`
		SourceText       string `json:"source_text"`
		TargetText       string `json:"target_text"`
	}
	type slimDOM struct {
		Nodes []slimNode `json:"nodes"`
	}
	type pageChunk struct {
		idx   int
		nodes []slimNode
	}
	var chunks []pageChunk
	for rows.Next() {
		var pIdx int
		var dStr string
		if err := rows.Scan(&pIdx, &dStr); err != nil {
			return nil, err
		}
		var sd slimDOM
		if err := json.Unmarshal([]byte(dStr), &sd); err != nil {
			return nil, fmt.Errorf("unmarshal slim dialogue dom for page_index=%d: %w", pIdx, err)
		}
		chunks = append(chunks, pageChunk{idx: pIdx, nodes: sd.Nodes})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	var turns []DialogueTurn
	for i := len(chunks) - 1; i >= 0; i-- {
		ch := chunks[i]
		for _, n := range ch.nodes {
			if n.Track == "primary" && n.SourceText != "" && n.TargetText != "" {
				turns = append(turns, DialogueTurn{
					PageIndex:  ch.idx,
					Speaker:    n.SpeakerClusterID,
					SourceText: n.SourceText,
					TargetText: n.TargetText,
				})
			}
		}
	}
	if len(turns) > limit {
		turns = turns[len(turns)-limit:]
	}
	return turns, nil
}

func (s *Store) ListRelations(ctx context.Context, workID int64) ([]CharacterRelation, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT work_id, from_char_id, to_char_id, honorific_target, register_note
		FROM character_relations WHERE work_id = ?
	`, workID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []CharacterRelation
	for rows.Next() {
		var r CharacterRelation
		if err := rows.Scan(&r.WorkID, &r.FromCharID, &r.ToCharID, &r.HonorificTarget, &r.RegisterNote); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) GetStats(ctx context.Context) (Stats, error) {
	var st Stats
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM works`).Scan(&st.Works); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pages`).Scan(&st.Pages); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pages WHERE needs_review = 1`).Scan(&st.NeedsReview); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM characters`).Scan(&st.Characters); err != nil {
		return Stats{}, err
	}
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM glossary_terms`).Scan(&st.GlossarySize); err != nil {
		return Stats{}, err
	}
	return st, nil
}
