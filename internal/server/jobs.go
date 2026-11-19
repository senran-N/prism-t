package server

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"prism-t/internal/engine"
)

// JobSnapshot describes the live or completed state of an asynchronous translation batch job.
type JobSnapshot struct {
	ID              string  `json:"id"`
	Status          string  `json:"status"`
	SourceKind      string  `json:"source_kind,omitempty"`
	WorkTitle       string  `json:"work_title"`
	ChapterTitle    string  `json:"chapter_title"`
	WorkID          int64   `json:"work_id,omitempty"`
	TotalUnits      int     `json:"total_units"`
	DownloadedUnits int     `json:"downloaded_units,omitempty"`
	CompletedUnits  int     `json:"completed_units"`
	CurrentStage    string  `json:"current_stage"`
	CurrentFile     string  `json:"current_file,omitempty"`
	Message         string  `json:"message"`
	PageIDs         []int64 `json:"page_ids,omitempty"`
	Error           string  `json:"error,omitempty"`
	StartedAt       string  `json:"started_at"`
	UpdatedAt       string  `json:"updated_at"`
	ElapsedMS       int64   `json:"elapsed_ms"`
}

type trackedJob struct {
	snap    JobSnapshot
	started time.Time
	cancel  context.CancelFunc
}

type jobManager struct {
	mu    sync.Mutex
	seq   uint64
	order []string
	jobs  map[string]*trackedJob
}

func newJobManager() *jobManager {
	return &jobManager{
		jobs: make(map[string]*trackedJob),
	}
}

func (m *jobManager) HasActive() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, j := range m.jobs {
		if j.snap.Status == "running" {
			return true
		}
	}
	return false
}

func (m *jobManager) Start(workTitle, chapterTitle string, totalUnits int, cancel context.CancelFunc) JobSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seq++
	now := time.Now()
	id := fmt.Sprintf("job_%d_%d", now.UnixMilli(), m.seq)
	snap := JobSnapshot{
		ID:           id,
		Status:       "running",
		WorkTitle:    workTitle,
		ChapterTitle: chapterTitle,
		TotalUnits:   totalUnits,
		CurrentStage: "queued",
		Message:      fmt.Sprintf("任务已入队，共 %d 个输入单元", totalUnits),
		StartedAt:    now.Format(time.RFC3339),
		UpdatedAt:    now.Format(time.RFC3339),
	}
	m.jobs[id] = &trackedJob{snap: snap, started: now, cancel: cancel}
	m.order = append(m.order, id)
	m.pruneLocked()
	return cloneJobSnapshot(snap)
}

func (m *jobManager) Update(id string, ev engine.TranslateProgressEvent) {
	m.mu.Lock()
	defer m.mu.Unlock()
	tj, ok := m.jobs[id]
	if !ok || tj.snap.Status != "running" {
		return
	}
	now := time.Now()
	if ev.WorkID > 0 {
		tj.snap.WorkID = ev.WorkID
	}
	if ev.TotalUnits > 0 {
		tj.snap.TotalUnits = ev.TotalUnits
	}
	if ev.DownloadedUnits > tj.snap.DownloadedUnits {
		tj.snap.DownloadedUnits = ev.DownloadedUnits
	}
	if ev.CompletedUnits > tj.snap.CompletedUnits {
		tj.snap.CompletedUnits = ev.CompletedUnits
	}
	if ev.Stage != "" {
		tj.snap.CurrentStage = ev.Stage
	}
	if ev.Filename != "" {
		tj.snap.CurrentFile = ev.Filename
	}
	if ev.Message != "" {
		tj.snap.Message = ev.Message
	}
	if ev.PageID > 0 {
		seen := false
		for _, pid := range tj.snap.PageIDs {
			if pid == ev.PageID {
				seen = true
				break
			}
		}
		if !seen {
			tj.snap.PageIDs = append(tj.snap.PageIDs, ev.PageID)
		}
	}
	tj.snap.UpdatedAt = now.Format(time.RFC3339)
	tj.snap.ElapsedMS = now.Sub(tj.started).Milliseconds()
}

func (m *jobManager) SetSourceKind(id, kind string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if tj, ok := m.jobs[id]; ok {
		tj.snap.SourceKind = kind
	}
}

func (m *jobManager) MarkDownloaded(id string, downloaded, total int, filename string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	tj, ok := m.jobs[id]
	if !ok || tj.snap.Status != "running" {
		return
	}
	if total > tj.snap.TotalUnits {
		tj.snap.TotalUnits = total
	}
	if downloaded > tj.snap.DownloadedUnits {
		tj.snap.DownloadedUnits = downloaded
	}
	if tj.snap.CurrentStage == "queued" || tj.snap.CurrentStage == "downloading" {
		tj.snap.CurrentStage = "downloading"
		tj.snap.CurrentFile = filename
		tj.snap.Message = fmt.Sprintf("正在边下载边翻译：已下载 %d/%d，已译完 %d/%d", tj.snap.DownloadedUnits, tj.snap.TotalUnits, tj.snap.CompletedUnits, tj.snap.TotalUnits)
	}
	tj.snap.UpdatedAt = time.Now().Format(time.RFC3339)
	tj.snap.ElapsedMS = time.Since(tj.started).Milliseconds()
}

func (m *jobManager) Finish(id string, workID int64, pageIDs []int64, runErr error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	tj, ok := m.jobs[id]
	if !ok {
		return
	}
	now := time.Now()
	tj.cancel = nil
	if workID > 0 {
		tj.snap.WorkID = workID
	}
	if len(pageIDs) > 0 {
		tj.snap.PageIDs = append([]int64(nil), pageIDs...)
		tj.snap.CompletedUnits = len(pageIDs)
	}
	switch {
	case runErr == nil:
		tj.snap.Status = "completed"
		tj.snap.CurrentStage = "done"
		tj.snap.Error = ""
		tj.snap.Message = fmt.Sprintf("全部 %d 页处理完成", len(tj.snap.PageIDs))
	case errors.Is(runErr, context.Canceled):
		tj.snap.Status = "canceled"
		tj.snap.CurrentStage = "canceled"
		tj.snap.Error = runErr.Error()
		tj.snap.Message = "任务已中止"
	default:
		tj.snap.Status = "failed"
		tj.snap.CurrentStage = "failed"
		tj.snap.Error = runErr.Error()
		tj.snap.Message = runErr.Error()
	}
	tj.snap.UpdatedAt = now.Format(time.RFC3339)
	tj.snap.ElapsedMS = now.Sub(tj.started).Milliseconds()
}

func (m *jobManager) Cancel(id string) (JobSnapshot, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	tj, ok := m.jobs[id]
	if !ok {
		return JobSnapshot{}, false
	}
	if tj.snap.Status == "running" && tj.cancel != nil {
		tj.snap.Message = "正在中止任务..."
		tj.snap.UpdatedAt = time.Now().Format(time.RFC3339)
		tj.cancel()
	}
	return cloneJobSnapshot(tj.snap), true
}

func (m *jobManager) Get(id string) (JobSnapshot, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	tj, ok := m.jobs[id]
	if !ok {
		return JobSnapshot{}, false
	}
	snap := cloneJobSnapshot(tj.snap)
	if snap.Status == "running" {
		snap.ElapsedMS = time.Since(tj.started).Milliseconds()
	}
	return snap, true
}

func (m *jobManager) List() []JobSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]JobSnapshot, 0, min(20, len(m.order)))
	for i := len(m.order) - 1; i >= 0 && len(out) < 20; i-- {
		if tj, ok := m.jobs[m.order[i]]; ok {
			snap := cloneJobSnapshot(tj.snap)
			if snap.Status == "running" {
				snap.ElapsedMS = time.Since(tj.started).Milliseconds()
			}
			out = append(out, snap)
		}
	}
	return out
}

func (m *jobManager) pruneLocked() {
	const maxHistory = 50
	for len(m.order) > maxHistory {
		oldest := m.order[0]
		if tj, ok := m.jobs[oldest]; ok && tj.snap.Status == "running" {
			break
		}
		m.order = m.order[1:]
		delete(m.jobs, oldest)
	}
}

func cloneJobSnapshot(in JobSnapshot) JobSnapshot {
	out := in
	if in.PageIDs != nil {
		out.PageIDs = append([]int64(nil), in.PageIDs...)
	}
	return out
}
