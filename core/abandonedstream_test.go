package bifrost

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/openai"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

// TestDrainAbandonedStream_UnblocksProducer covers the stream the caller never
// receives: the worker's 5s delivery timeout fires, or the client context is
// done, and the stream is dropped.
//
// The provider goroutine on the other end is still sending. GateSendChunk only
// escapes on ctx.Done(), so on the timeout path, where the context is very much
// alive, it blocks forever on a channel nobody reads. It never reaches its
// deferred ReleaseStreamingResponse, so its upstream connection is never
// returned and one slot of MaxConnsPerHost is burned permanently.
func TestDrainAbandonedStream_UnblocksProducer(t *testing.T) {
	t.Parallel()

	// Buffer smaller than the number of chunks, so the producer blocks unless
	// something consumes. This is the provider goroutine's shape.
	stream := make(chan *schemas.BifrostStreamChunk, 2)
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		defer close(stream)
		for range 10 {
			stream <- &schemas.BifrostStreamChunk{}
		}
	}()

	drainAbandonedStream(stream)

	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("provider goroutine still blocked on an abandoned stream; " +
			"its connection would never be released")
	}
}

// TestDrainAbandonedStream_NilIsSafe guards the delivery path, which can reach
// the abandonment branches with no stream to drain.
func TestDrainAbandonedStream_NilIsSafe(t *testing.T) {
	t.Parallel()
	drainAbandonedStream(nil)
}

// abandonedUpstreamProvider is a schemas.Provider whose ChatCompletion ignores the
// request context: it parks until the test releases it, then returns a result (or an
// error when fail is set). It models an upstream that completes after the caller has
// gone (#6972). Every other method comes from the embedded real OpenAI provider.
type abandonedUpstreamProvider struct {
	schemas.Provider
	started chan struct{} // one send per ChatCompletion entry
	release chan struct{} // ChatCompletion returns after one receive
	fail    bool
}

func (p *abandonedUpstreamProvider) ChatCompletion(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	p.started <- struct{}{}
	<-p.release
	if p.fail {
		return nil, &schemas.BifrostError{
			StatusCode: new(500),
			Error:      &schemas.ErrorField{Message: "upstream failed after the caller left"},
		}
	}
	return &schemas.BifrostChatResponse{
		ID:     "chatcmpl-abandoned",
		Object: "chat.completion",
		Model:  "gpt-4o-mini",
		Choices: []schemas.BifrostResponseChoice{{
			FinishReason: new("stop"),
			ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
				Message: &schemas.ChatMessage{
					Role:    schemas.ChatMessageRoleAssistant,
					Content: &schemas.ChatMessageContent{ContentStr: new("ok")},
				},
			},
		}},
		Usage: &schemas.BifrostLLMUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7},
	}, nil
}

// terminalHookCounter counts PostLLMHook invocations, split by result vs error, so a
// test can assert that every abandoned request was billed exactly once.
type terminalHookCounter struct {
	results atomic.Int64
	errors  atomic.Int64
}

func (c *terminalHookCounter) GetName() string { return "terminal-hook-counter" }
func (c *terminalHookCounter) Cleanup() error  { return nil }
func (c *terminalHookCounter) PreRequestHook(_ *schemas.BifrostContext, _ *schemas.BifrostRequest) error {
	return nil
}
func (c *terminalHookCounter) PreLLMHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}
func (c *terminalHookCounter) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if bifrostErr != nil {
		c.errors.Add(1)
	} else {
		c.results.Add(1)
	}
	return resp, bifrostErr, nil
}

// runAbandonedRequests drives the real requestWorker with an upstream that finishes
// after the caller's context is cancelled. Each iteration: enqueue, wait for the
// upstream call to start, cancel (the caller disconnects mid-flight), release the
// upstream. The worker's delivery select then has both a ready send (cap-1 channel,
// drained on acquire) and a ready ctx.Done(); Go picks uniformly among ready cases, so
// n iterations expose a missing pre-select ctx check with probability 1 - 2^-n.
func runAbandonedRequests(t *testing.T, n int, fail bool) *terminalHookCounter {
	t.Helper()

	counter := &terminalHookCounter{}
	account := NewMockAccount()
	account.AddProvider(schemas.OpenAI, 1, n)
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{{
		ID:     "abandoned-key",
		Value:  *schemas.NewSecretVar("sk-test"),
		Models: schemas.WhiteList{"*"},
		Weight: 100,
	}})
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account:    account,
		Logger:     NewNoOpLogger(),
		LLMPlugins: []schemas.LLMPlugin{counter},
	})
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	t.Cleanup(client.Shutdown)

	cfg, err := account.GetConfigForProvider(schemas.OpenAI)
	if err != nil {
		t.Fatalf("GetConfigForProvider: %v", err)
	}
	cfg.NetworkConfig.MaxRetries = 0

	upstream := &abandonedUpstreamProvider{
		Provider: openai.NewOpenAIProvider(cfg, NewNoOpLogger()),
		started:  make(chan struct{}),
		release:  make(chan struct{}),
		fail:     fail,
	}
	pq := &ProviderQueue{queue: make(chan *ChannelMessage, n), done: make(chan struct{})}
	var wg sync.WaitGroup
	wg.Add(1)
	go client.requestWorker(upstream, cfg, pq, &wg)

	for i := 0; i < n; i++ {
		ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
		// tryRequest stamps the tracer before enqueue; the retry loop refuses a context without one.
		ctx.SetValue(schemas.BifrostContextKeyTracer, client.getTracer())
		msg := client.getChannelMessage(schemas.BifrostRequest{
			RequestType: schemas.ChatCompletionRequest,
			ChatRequest: &schemas.BifrostChatRequest{
				Provider: schemas.OpenAI,
				Model:    "gpt-4o-mini",
				Input: []schemas.ChatMessage{{
					Role:    schemas.ChatMessageRoleUser,
					Content: &schemas.ChatMessageContent{ContentStr: new("hi")},
				}},
			},
		})
		msg.Context = ctx
		pq.queue <- msg

		select {
		case <-upstream.started: // the upstream call is in flight
		case e := <-msg.Err:
			t.Fatalf("iteration %d: worker replied with error before the provider: %+v / %+v", i, e, e.Error)
		case <-time.After(5 * time.Second):
			t.Fatalf("iteration %d: worker never reached the provider", i)
		}
		cancel()                       // the caller disconnects
		upstream.release <- struct{}{} // the upstream completes anyway
	}
	pq.signalClosing()
	wg.Wait() // every delivery is synchronous inside the worker loop
	return counter
}

// TestRequestWorkerBillsEveryAbandonedResult locks #6972 on the success-delivery path:
// a caller that disconnected must still get its completed upstream result billed and
// logged exactly once. Without a ctx check before the delivery select, roughly half of
// these sends win the race into a buffer nobody reads and the terminal hooks never run.
func TestRequestWorkerBillsEveryAbandonedResult(t *testing.T) {
	const n = 64
	c := runAbandonedRequests(t, n, false)
	if got := c.results.Load(); got != n {
		t.Fatalf("PostLLMHook ran for %d of %d abandoned results; every completed upstream call must be billed once", got, n)
	}
	if got := c.errors.Load(); got != 0 {
		t.Fatalf("PostLLMHook saw %d errors for successful upstream calls, want 0", got)
	}
}

// TestRequestWorkerBillsEveryAbandonedError is the same contract on the error-delivery
// path, which is the common production path once the transport cancels the context on
// a client socket close: the upstream call is cut with a 499 and that error must still
// reach the terminal hooks exactly once.
func TestRequestWorkerBillsEveryAbandonedError(t *testing.T) {
	const n = 64
	c := runAbandonedRequests(t, n, true)
	if got := c.errors.Load(); got != n {
		t.Fatalf("PostLLMHook ran for %d of %d abandoned errors; every failed upstream call must be logged once", got, n)
	}
	if got := c.results.Load(); got != 0 {
		t.Fatalf("PostLLMHook saw %d results for failed upstream calls, want 0", got)
	}
}
