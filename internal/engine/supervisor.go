package engine

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"prism-t/internal/config"
)

var ErrExternalEngineNotManaged = errors.New("prism-t-engine is externally managed; restart it manually to switch compute devices")

type Supervisor struct {
	mu        sync.Mutex
	restartMu sync.Mutex
	managed   bool
	cmd       *exec.Cmd
	client    *Client
	stopCh    chan struct{}
	doneCh    chan struct{}
	stopped   bool
}

func EnsureRunning(ctx context.Context, cfg config.Config, client *Client) (*Supervisor, error) {
	if _, err := config.ValidateDevice(cfg.Device); err != nil {
		return nil, err
	}

	hCtx, cancel := context.WithTimeout(ctx, 1500*time.Millisecond)
	_, hErr := client.Health(hCtx)
	cancel()
	if hErr == nil {
		return &Supervisor{client: client, managed: false}, nil
	}
	if !cfg.AutoEngine {
		return nil, fmt.Errorf("prism-t-engine unreachable at %s and auto-spawn disabled: %w", cfg.EngineURL, hErr)
	}

	sup := &Supervisor{client: client, managed: true}
	cmd, spawn, waitCh, err := startManagedEngine(ctx, cfg, client)
	if err != nil {
		return nil, err
	}
	sup.cmd = cmd
	stopCh := make(chan struct{})
	doneCh := make(chan struct{})
	sup.stopCh = stopCh
	sup.doneCh = doneCh
	go sup.watchLoop(spawn, waitCh, stopCh, doneCh)
	return sup, nil
}

// IsManaged reports whether prismtd spawned and supervises the local engine child process.
func (s *Supervisor) IsManaged() bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.managed
}

// Restart terminates the managed prism-t-engine child (if any) and relaunches it under updated device/concurrency config.
func (s *Supervisor) Restart(ctx context.Context, cfg config.Config) error {
	if s == nil {
		return ErrExternalEngineNotManaged
	}
	devSpec, err := config.ValidateDevice(cfg.Device)
	if err != nil {
		return err
	}
	s.restartMu.Lock()
	defer s.restartMu.Unlock()

	if !s.IsManaged() {
		hs, hErr := s.client.Health(ctx)
		if hErr == nil && hs.Hardware != nil && strings.EqualFold(hs.Hardware.ConfiguredDevice, devSpec) {
			return nil
		}
		return ErrExternalEngineNotManaged
	}

	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return errors.New("engine supervisor already stopped")
	}
	cmd := s.cmd
	stopCh := s.stopCh
	doneCh := s.doneCh
	s.cmd = nil
	s.stopCh = nil
	s.doneCh = nil
	s.mu.Unlock()

	if stopCh != nil {
		close(stopCh)
	}
	if cmd != nil && cmd.Process != nil {
		killEngineProcess(cmd)
	}
	if doneCh != nil {
		select {
		case <-doneCh:
		case <-time.After(3 * time.Second):
		}
	}

	if u, parseErr := url.Parse(cfg.EngineURL); parseErr == nil && u.Host != "" {
		waitForPortRelease(ctx, u.Host, 2500*time.Millisecond)
	}

	newCmd, spawn, waitCh, err := startManagedEngine(ctx, cfg, s.client)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		killEngineProcess(newCmd)
		return errors.New("engine supervisor stopped during restart")
	}
	s.cmd = newCmd
	stopCh = make(chan struct{})
	doneCh = make(chan struct{})
	s.stopCh = stopCh
	s.doneCh = doneCh
	go s.watchLoop(spawn, waitCh, stopCh, doneCh)
	return nil
}

func startManagedEngine(ctx context.Context, cfg config.Config, client *Client) (*exec.Cmd, func() (*exec.Cmd, <-chan error, error), <-chan error, error) {
	devSpec, err := config.ValidateDevice(cfg.Device)
	if err != nil {
		return nil, nil, nil, err
	}
	u, err := url.Parse(cfg.EngineURL)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("invalid engine URL %q: %w", cfg.EngineURL, err)
	}
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("engine URL %q must specify host:port: %w", cfg.EngineURL, err)
	}
	if host == "" {
		host = "127.0.0.1"
	}

	absEngineDir, err := filepath.Abs(cfg.EngineDir)
	if err != nil {
		return nil, nil, nil, err
	}

	spawn := func() (*exec.Cmd, <-chan error, error) {
		cmd, bErr := buildEngineCmd(absEngineDir, host, port)
		if bErr != nil {
			return nil, nil, bErr
		}
		cmd.Dir = absEngineDir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Env = withEngineEnv(os.Environ(), devSpec, cfg.MaxConcurrency)
		if sErr := cmd.Start(); sErr != nil {
			return nil, nil, sErr
		}
		waitCh := make(chan error, 1)
		go func() {
			waitCh <- cmd.Wait()
		}()
		return cmd, waitCh, nil
	}

	cmd, waitCh, err := spawn()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("spawn prism-t-engine: %w", err)
	}

	deadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			killEngineProcess(cmd)
			return nil, nil, nil, ctx.Err()
		case exitErr := <-waitCh:
			return nil, nil, nil, fmt.Errorf("prism-t-engine exited prematurely: %w", exitErr)
		default:
		}
		cCtx, cCancel := context.WithTimeout(ctx, 1200*time.Millisecond)
		_, okErr := client.Health(cCtx)
		cCancel()
		if okErr == nil {
			return cmd, spawn, waitCh, nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	killEngineProcess(cmd)
	return nil, nil, nil, fmt.Errorf("timed out waiting for prism-t-engine health at %s", cfg.EngineURL)
}

func withEngineEnv(base []string, devSpec string, maxConc int) []string {
	if maxConc < 1 {
		maxConc = 1
	}
	out := make([]string, 0, len(base)+2)
	for _, kv := range base {
		upper := strings.ToUpper(kv)
		if strings.HasPrefix(upper, "PRISM_DEVICE=") || strings.HasPrefix(upper, "PRISM_T_CONCURRENCY=") {
			continue
		}
		out = append(out, kv)
	}
	out = append(out, "PRISM_DEVICE="+devSpec, "PRISM_T_CONCURRENCY="+strconv.Itoa(maxConc))
	return out
}

func waitForPortRelease(ctx context.Context, addr string, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 150*time.Millisecond)
		if err != nil {
			return
		}
		_ = conn.Close()
		select {
		case <-ctx.Done():
			return
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func (s *Supervisor) watchLoop(spawn func() (*exec.Cmd, <-chan error, error), initialWait <-chan error, stopCh <-chan struct{}, doneCh chan<- struct{}) {
	defer close(doneCh)
	waitCh := initialWait
	for {
		select {
		case <-stopCh:
			return
		case err := <-waitCh:
			select {
			case <-stopCh:
				return
			default:
			}
			s.mu.Lock()
			if s.stopped || s.stopCh != stopCh {
				s.mu.Unlock()
				return
			}
			s.mu.Unlock()
			log.Printf("[prismtd] prism-t-engine exited (%v); restarting...", err)
			backoff := time.Second
			for {
				select {
				case <-stopCh:
					return
				case <-time.After(backoff):
				}
				newCmd, newWait, sErr := spawn()
				if sErr == nil {
					s.mu.Lock()
					if s.stopped || s.stopCh != stopCh {
						s.mu.Unlock()
						killEngineProcess(newCmd)
						return
					}
					s.cmd = newCmd
					s.mu.Unlock()
					waitCh = newWait
					break
				}
				log.Printf("[prismtd] restart failed (%v); retrying in %s", sErr, backoff)
				if backoff < 16*time.Second {
					backoff *= 2
				}
			}
		}
	}
}

func (s *Supervisor) Stop() {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	cmd := s.cmd
	stopCh := s.stopCh
	doneCh := s.doneCh
	s.mu.Unlock()

	if stopCh != nil {
		close(stopCh)
	}
	if cmd != nil && cmd.Process != nil {
		killEngineProcess(cmd)
	}
	if doneCh != nil {
		select {
		case <-doneCh:
		case <-time.After(3 * time.Second):
		}
	}
}

func killEngineProcess(cmd *exec.Cmd) {
	if runtime.GOOS == "windows" {
		// Windows venv python is a launcher; terminate its interpreter child too.
		_ = exec.Command("taskkill", "/PID", strconv.Itoa(cmd.Process.Pid), "/T", "/F").Run()
		return
	}
	_ = cmd.Process.Kill()
}

func buildEngineCmd(engineDir, host, port string) (*exec.Cmd, error) {
	var venvPy string
	if runtime.GOOS == "windows" {
		venvPy = filepath.Join(engineDir, ".venv", "Scripts", "python.exe")
	} else {
		venvPy = filepath.Join(engineDir, ".venv", "bin", "python")
	}
	if _, err := os.Stat(venvPy); err == nil {
		return exec.Command(
			venvPy, "-m", "uvicorn", "prism_t_engine.app:app",
			"--host", host, "--port", port, "--log-level", "info",
		), nil
	}
	if uvBin, err := exec.LookPath("uv"); err == nil {
		return exec.Command(
			uvBin, "run", "uvicorn", "prism_t_engine.app:app",
			"--host", host, "--port", port, "--log-level", "info",
		), nil
	}
	return nil, fmt.Errorf("neither %s nor `uv` found in PATH", venvPy)
}
