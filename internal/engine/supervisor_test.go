package engine

import (
	"errors"
	"os/exec"
	"sync/atomic"
	"testing"
	"time"
)

func TestSupervisorWatchLoopRetriesWithoutDeadlock(t *testing.T) {
	sup := &Supervisor{
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
	initialWait := make(chan error, 1)
	initialWait <- errors.New("simulated crash")

	var attempts int32
	recovered := make(chan struct{})

	spawn := func() (*exec.Cmd, <-chan error, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n == 1 {
			return nil, nil, errors.New("transient spawn failure")
		}
		select {
		case <-recovered:
		default:
			close(recovered)
		}
		return nil, make(chan error, 1), nil
	}

	go sup.watchLoop(spawn, initialWait, sup.stopCh, sup.doneCh)

	select {
	case <-recovered:
	case <-time.After(6 * time.Second):
		sup.Stop()
		t.Fatalf("watchLoop timed out after transient spawn failure (attempts=%d)", atomic.LoadInt32(&attempts))
	}

	sup.Stop()
}
