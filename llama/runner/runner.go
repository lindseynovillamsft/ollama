package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llama"
)

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type input struct {
	token int

	// embd is an image embedding
	// important to note, embd contains a series of embeddings as a float*
	// TODO (jmorganca): change this to a slice of float32 instead of
	// using the llama.LlavaImageEmbed type
	embd *llama.LlavaImageEmbed
}

type Sequence struct {
	// number of tokens evaluated
	nPast int

	// batch index
	iBatch int

	// number of tokens predicted so far
	numPredicted int

	// prompt inputs left to evaluate
	inputs []input

	// channel to send responses over
	responses chan string

	// number of tokens to predict
	numPredict int

	samplingCtx *llama.SamplingContext

	// channel to send back the embedding if embedding only
	embedding chan []float32

	// stop sequences
	stop []string

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool

	doneReason string

	pieces []string

	// Metrics
	t_start_process_prompt time.Time
	t_start_genereration   time.Time
	n_decoded              int
	n_prompt_tokens        int
}

// prompt returns true if the prompt is still being processed
func (s *Sequence) isPromptProcessing() bool {
	var tokens int
	for _, i := range s.inputs {
		if i.embd == nil {
			tokens++
			continue
		}

		tokens += i.embd.Tokens()
	}

	return s.nPast < tokens-1
}

// inputs processes the prompt and images into a list of inputs
// by splitting the prompt on [img-<n>] tags, tokenizing text and
// generating image embeddings for each image
func (s *Server) inputs(prompt string, images []string) ([]input, error) {
	var inputs []input

	re := regexp.MustCompile(`\[img-(\d+)\]`)
	parts := re.Split(prompt, -1)
	matches := re.FindAllStringSubmatch(prompt, -1)

	for i, part := range parts {
		// text - tokenize
		if strings.TrimSpace(part) != "" {
			tokens, err := s.lc.Model().Tokenize(part, i == 0, true)
			if err != nil {
				return nil, err
			}
			for _, t := range tokens {
				inputs = append(inputs, input{token: t})
			}
		}

		// image - generate image embedding
		if i < len(matches) {
			n, _ := strconv.Atoi(matches[i][1])

			if n < 0 || n >= len(images) {
				return nil, fmt.Errorf("invalid image index: %d", n)
			}

			decoded, err := base64.StdEncoding.DecodeString(images[n])
			if err != nil {
				slog.Error("Failed to decode image", "error", err)
				return nil, err
			}

			// Vision models can not be accessed concurrently
			// TODO (jmorganca): store the actual float* embedding
			// returned by the vision model instead of the llama.LlavaImageEmbed
			s.clip.mu.Lock()
			embd := llama.NewLlavaImageEmbed(s.clip.cc, decoded)
			s.clip.mu.Unlock()

			inputs = append(inputs, input{embd: embd})
		}
	}

	return inputs, nil
}

func (s *Server) NewSequence(prompt string, images []string, numPredict int, stop []string, params *llama.SamplingParams, embedding bool) (*Sequence, error) {
	inputs, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	}

	var sc *llama.SamplingContext
	if params != nil {
		sc = llama.NewSamplingContext(*params)
		for _, t := range inputs {
			if t.embd == nil {
				sc.Accept(s.lc, t.token, false)
			}
		}
	}

	return &Sequence{
		inputs:          inputs,
		n_prompt_tokens: len(inputs),
		responses:       make(chan string, 1),
		embedding:       make(chan []float32, 1),
		samplingCtx:     sc,
		embeddingOnly:   embedding,
		stop:            stop,
	}, nil
}

type clip struct {
	cc *llama.ClipContext
	mu sync.Mutex
}

type Server struct {
	model *llama.Model
	lc    *llama.Context

	// required for image embeddings
	clip clip

	// batchSize is the number of tokens or image embeddings
	// to process in a batch
	batchSize int

	// parallel is the number of parallel requests to handle
	parallel int

	// seqs is the list of parallel sequences being evaluated
	// TODO (jmorganca): this can probably be moved into run()
	seqs []*Sequence

	// context window size
	numCtx int

	mu sync.Mutex

	cond *sync.Cond

	progress float32

	status string
}

// waiting returns true if there are no sequences to process
func (s *Server) waiting() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}

	return true
}

// processImage processes an image embedding if it's next in any sequence
// TODO (jmorganca): process image embeddings in parallel since this calls
// llama_decode separately from the main loop in run below
func (s *Server) processImage() bool {
	for _, seq := range s.seqs {
		if seq == nil {
			continue
		}

		if len(seq.inputs) > 0 && seq.inputs[0].embd != nil {
			// TODO (jmorganca): this only works with n_parallel=1
			llama.LlavaEvalImageEmbed(s.lc, seq.inputs[0].embd, s.batchSize, &seq.nPast)
			llama.LlavaImageEmbedFree(seq.inputs[0].embd)
			seq.iBatch = seq.inputs[0].embd.Tokens() - 1
			seq.inputs = seq.inputs[1:]
			return true
		}
	}

	return false
}

func (s *Server) cleanup(i int) {
	s.seqs[i].samplingCtx.Free()
	s.lc.KvCacheSeqRm(i, 0, -1)
	s.mu.Lock()
	s.seqs[i] = nil
	s.mu.Unlock()
}

// TODO (jmorganca): run should be simplified, removing:
// * sampling
// * stop token checking
// * metrics
// these should instead be handled by the handlers
// it should only be responsible for accepting tokens or embeddings and
// processing batches as fast as possible
func (s *Server) run(ctx context.Context) error {
	batch := llama.NewBatch(s.batchSize, 0, s.parallel)
	defer batch.Free()

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
			s.mu.Lock()
			for s.waiting() {
				s.cond.Wait()
			}
			s.mu.Unlock()

			// evaluate image embeddings if they are next in any sequence
			// TODO (jmorganca): ideally embeddings would be processed
			// alongside tokens as part of the same batch. this will
			// currently cause a pause in other sequences while the image
			// is processed
			if s.processImage() {
				continue
			}

			// create a token batch to process
			for i, seq := range s.seqs {
				if seq == nil {
					continue
				}

				hitLimit := seq.numPredict > 0 && seq.numPredicted > seq.numPredict

				// if past the num predict limit
				// TODO (jmorganca): should context shift
				if hitLimit || seq.nPast > s.numCtx {
					seq.doneReason = "limit"
					close(seq.responses)
					s.cleanup(i)
					continue
				}

				if seq.t_start_process_prompt.IsZero() {
					seq.t_start_process_prompt = time.Now()
				}

				var tokens int
				for j, t := range seq.inputs {
					if t.embd != nil {
						continue
					}

					if j > s.batchSize {
						break
					}

					batch.Add(t.token, seq.nPast, []int{i}, !seq.isPromptProcessing())
					tokens += 1
					seq.nPast++
				}

				seq.inputs = seq.inputs[1:]
				seq.iBatch = tokens - 1
			}

			if batch.NumTokens() > 0 {
				err := s.lc.Decode(batch)
				if err != nil {
					return err
				}
			}

			// sample and send responses
			for i, seq := range s.seqs {
				if seq == nil {
					continue
				}

				// don't sample while prompt processing
				if len(seq.inputs) > 0 {
					continue
				}

				// if done processing the prompt, send an embedding
				if seq.embeddingOnly {
					embd := s.lc.GetEmbeddingsSeq(i)
					if embd == nil {
						embd = s.lc.GetEmbeddingsIth(seq.iBatch)
					}

					seq.embedding <- embd
					close(seq.embedding)
					s.cleanup(i)
					continue
				}

				token := seq.samplingCtx.Sample(s.lc, nil, seq.iBatch)
				seq.samplingCtx.Accept(s.lc, token, true)
				seq.n_decoded += 1

				if seq.n_decoded == 1 {
					seq.t_start_genereration = time.Now()
				}
				piece := s.model.TokenToPiece(token)

				seq.numPredicted++

				slog.Debug("sampled", "piece", piece)

				// if it's an end of sequence token, break
				// TODO: just end this sequence
				if s.model.TokenIsEog(token) {
					// TODO (jmorganca): we should send this back
					// as it's important for the /api/generate context
					// seq.responses <- piece

					seq.doneReason = "stop"
					close(seq.responses)
					s.cleanup(i)
					continue
				}

				seq.inputs = []input{{token: token}}

				// look for stop sequences (or beginnings of one)
				seq.pieces = append(seq.pieces, piece)
				sequence := strings.Join(seq.pieces, "")
				if ok, stop := findStop(sequence, seq.stop); ok {
					truncated := truncateStop(seq.pieces, stop)

					for _, p := range truncated {
						seq.responses <- p
					}

					seq.doneReason = "stop"
					close(seq.responses)
					s.cleanup(i)
					continue
				}

				if containsStopSuffix(sequence, seq.stop) {
					continue
				}

				for _, p := range seq.pieces {
					seq.responses <- p
				}

				seq.pieces = []string{}
			}

			batch.Clear()
		}
	}
}

// TODO (jmorganca): use structs from the api package to avoid duplication
// this way the api acts as a proxy instead of using a different api for the
// runner
type CompletionRequest struct {
	Prompt  string   `json:"prompt"`
	Images  []string `json:"images"`
	Grammar string   `json:"grammar"`
	Stop    []string `json:"stop"`

	api.Options
}

type Timings struct {
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
}

type CompletionResponse struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`

	Model        string  `json:"model,omitempty"`
	Prompt       string  `json:"prompt,omitempty"`
	StoppedLimit bool    `json:"stopped_limit,omitempty"`
	PredictedN   int     `json:"predicted_n,omitempty"`
	PredictedMS  float64 `json:"predicted_ms,omitempty"`
	PromptN      int     `json:"prompt_n,omitempty"`
	PromptMS     float64 `json:"prompt_ms,omitempty"`

	Timings Timings `json:"timings"`
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	var req CompletionRequest
	req.Options = api.DefaultOptions()
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %s", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	var samplingParams llama.SamplingParams
	samplingParams.TopK = req.TopK
	samplingParams.TopP = req.TopP
	samplingParams.TfsZ = req.TFSZ
	samplingParams.TypicalP = req.TypicalP
	samplingParams.Temp = req.Temperature
	samplingParams.PenaltyRepeat = req.RepeatPenalty
	samplingParams.PenaltyFreq = req.FrequencyPenalty
	samplingParams.PenaltyPresent = req.PresencePenalty
	samplingParams.Mirostat = req.Mirostat
	samplingParams.MirostatTau = req.MirostatTau
	samplingParams.MirostatEta = req.MirostatEta
	samplingParams.PenalizeNl = req.PenalizeNewline
	samplingParams.Seed = uint32(req.Seed)
	samplingParams.Grammar = req.Grammar

	seq, err := s.NewSequence(req.Prompt, req.Images, req.NumPredict, req.Stop, &samplingParams, false)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// TODO (jmorganca): add to sequence queue instead of
	// failing if a slot isn't available
	var seqId int
	s.mu.Lock()
	for i, sq := range s.seqs {
		if sq == nil {
			seqId = i
			s.seqs[i] = seq
			s.cond.Signal()
			break
		}
	}
	s.mu.Unlock()

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "could not get flusher", http.StatusInternalServerError)
		return
	}

	for {
		select {
		case <-r.Context().Done():
			s.cleanup(seqId)
			return
		case content, ok := <-seq.responses:
			if !ok {
				// Send the final response
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Stop: true,
					Timings: Timings{
						PromptN:     seq.n_prompt_tokens,
						PromptMS:    float64(seq.t_start_genereration.Sub(seq.t_start_process_prompt).Milliseconds()),
						PredictedN:  seq.n_decoded,
						PredictedMS: float64(time.Since(seq.t_start_genereration).Milliseconds()),
					},
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
					return
				}
				flusher.Flush()

				return
			}

			if err := json.NewEncoder(w).Encode(&CompletionResponse{
				Content: content,
			}); err != nil {
				http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
				return
			}
			flusher.Flush()
		}
	}
}

type EmbeddingRequest struct {
	Content []string `json:"content"`
}

type EmbeddingResponse struct {
	Embedding [][]float32 `json:"embedding"`
}

// TODO (jmorganca): is it safe to do this concurrently with decoding?
func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	slog.Debug("embedding request", "content", req.Content)
	seqs := make([]*Sequence, len(req.Content))
	embeddings := make([][]float32, len(req.Content))
	var processed int
	var err error
	for i, content := range req.Content {
		seqs[i], err = s.NewSequence(content, nil, 0, nil, nil, true)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
			return
		}
	}

	// TODO - refactor to go routines to add seq's and drain the responses
	// so we don't stall until each set is iterated through
	for processed < len(seqs) {
		s.mu.Lock()
		for i, sq := range s.seqs {
			if processed >= len(seqs) {
				break
			}
			if sq == nil {
				s.seqs[i] = seqs[processed]
				processed += 1
			}
		}
		s.cond.Signal()
		s.mu.Unlock()

		for i := range processed {
			embeddings[i] = <-seqs[i].embedding
		}
	}

	if err := json.NewEncoder(w).Encode(&EmbeddingResponse{
		Embedding: embeddings,
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}
}

type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

// TODO (jmorganca): is it safe to do this concurrently with decoding?
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&HealthResponse{
		Status:   s.status,
		Progress: s.progress,
	}); err != nil {
		log.Println("Failed to encode result:", err)
		return
	}
}

func main() {
	mpath := flag.String("model", "", "Path to model binary file")
	ppath := flag.String("mmproj", "", "Path to projector binary file")
	parallel := flag.Int("parallel", 1, "Number of sequences to handle simultaneously")
	batchSize := flag.Int("batch-size", 512, "Batch size")
	nGpuLayers := flag.Int("n-gpu-layers", 0, "Number of layers to offload to GPU")
	mainGpu := flag.Int("main-gpu", 0, "Main GPU")
	flashAttention := flag.Bool("flash-attn", false, "Enable flash attention")
	numCtx := flag.Int("ctx-size", 2048, "Context (or KV cache) size")
	lpath := flag.String("lora", "", "Path to lora layer file")
	port := flag.Int("port", 8080, "Port to expose the server on")
	threads := flag.Int("threads", runtime.NumCPU(), "Number of threads to use during generation")

	// TODO not yet implemented but wired to keep the parsing aligned
	embedding := flag.Bool("embedding", false, "enable embedding vector output (default: disabled)")
	logDisable := flag.Bool("log-disable", false, "disables logging to a file")
	verbose := flag.Bool("verbose", false, "verbose output (default: disabled)")
	f32 := flag.Bool("memory-f32", false, "use f32 instead of f16 for memory key+value (default: disabled) not recommended: doubles context memory required and no measurable increase in quality")
	noMmap := flag.Bool("no-mmap", false, "do not memory-map model (slower load but may reduce pageouts if not using mlock)")
	mlock := flag.Bool("mlock", false, "force system to keep model in RAM rather than swapping or compressing")
	tensorSplit := flag.String("tensor-split", "", "fraction of the model to offload to each GPU, comma-separated list of proportions")

	if *parallel > 1 && *ppath != "" {
		log.Fatal("parallel mode not supported with projector")
	}

	flag.Parse()
	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	})
	slog.SetDefault(slog.New(handler))

	// TODO actually implement...
	if *embedding {
		slog.Warn("embeddings not yet support")
	}
	if *logDisable {
		slog.Info("ignoring --log-disable")
	}
	if *f32 {
		slog.Warn("memory-f32 not yet supported")
	}
	if *noMmap {
		slog.Warn("no-mmap not yet supported")
	}
	if *mlock {
		slog.Warn("mlock not yet supported")
	}
	if *tensorSplit != "" {
		slog.Warn("tensor-split not yet implemented")
	}

	server := &Server{
		numCtx:    *numCtx,
		batchSize: *batchSize,
		parallel:  *parallel,
		seqs:      make([]*Sequence, *parallel),
		status:    "loading",
	}

	// load the model
	llama.BackendInit()
	params := llama.NewModelParams(*nGpuLayers, *mainGpu, func(progress float32) {
		slog.Debug("Loading model", "progress %", math.Round(float64(progress*100)))
		server.progress = progress
	})
	server.model = llama.LoadModelFromFile(*mpath, params)

	if *lpath != "" {
		err := server.model.ApplyLoraFromFile(*lpath, 1.0, "", *threads)
		if err != nil {
			panic(err)
		}
	}

	ctxParams := llama.NewContextParams(*numCtx, *threads, *flashAttention)
	server.lc = llama.NewContextWithModel(server.model, ctxParams)

	if *ppath != "" {
		server.clip.cc = llama.NewClipContext(*ppath)
		defer server.clip.cc.Close()
	}

	server.cond = sync.NewCond(&server.mu)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		err := server.run(ctx)
		if err != nil {
			slog.Error("fatal error encountered", "error", err)
			os.Exit(1)
		}
	}()

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/embedding", server.embeddings)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/health", server.health)

	httpServer := http.Server{
		Handler: mux,
	}

	server.status = "ok"

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
	}

	cancel()
}
