package convert

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/llm"
)

type Parameters struct {
	Architectures []string `json:"architectures"`
	VocabSize     uint32   `json:"vocab_size"`
}

type AdapterParameters struct {
	LoraLayers     uint32 `json:"lora_layers"`
	LoraParameters struct {
		Rank  uint32  `json:"rank"`
		Alpha float32 `json:"alpha"`
		Scale float32 `json:"scale"`
	} `json:"lora_parameters"`
}

func (Parameters) KV(t *Tokenizer) llm.KV {
	kv := llm.KV{
		"general.file_type":            uint32(1),
		"general.quantization_version": uint32(2),
		"tokenizer.ggml.pre":           t.Pre,
		"tokenizer.ggml.model":         t.Vocabulary.Model,
		"tokenizer.ggml.tokens":        t.Vocabulary.Tokens,
		"tokenizer.ggml.scores":        t.Vocabulary.Scores,
		"tokenizer.ggml.token_type":    t.Vocabulary.Types,
	}

	if len(t.Merges) > 0 {
		kv["tokenizer.ggml.merges"] = t.Merges
	}

	if t.Template != "" {
		kv["tokenizer.chat_template"] = t.Template
	}

	for _, sv := range t.SpecialVocabulary {
		kv[fmt.Sprintf("tokenizer.ggml.%s_token_id", sv.Key())] = uint32(sv.ID)
		kv[fmt.Sprintf("tokenizer.ggml.add_%s_token", sv.Key())] = sv.AddToken
	}

	return kv
}

func (Parameters) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}

func (Parameters) writeFile(ws io.WriteSeeker, kv llm.KV, ts []llm.Tensor) error {
	return llm.WriteGGUF(ws, kv, ts)
}

func (p AdapterParameters) KV() llm.KV {
	kv := llm.KV{
		"adapter.lora.alpha": p.LoraParameters.Alpha,
		"adapter.type":       "lora",
		"general.file_type":  uint32(1),
		"general.type":       "adapter",
		"general.version":    "v0.2",
	}

	return kv
}

type Converter interface {
	// KV maps parameters to LLM key-values
	KV(*Tokenizer) llm.KV
	AdapterKV(llm.KV) llm.KV
	// Tensors maps input tensors to LLM tensors. Model specific modifications can be done here.
	Tensors([]Tensor) []llm.Tensor
	// Replacements returns a list of string pairs to replace in tensor names.
	// See [strings.Replacer](https://pkg.go.dev/strings#Replacer) for details
	Replacements() []string

	// specialTokenTypes returns any special token types the model uses
	specialTokenTypes() []string
	// writeFile writes the model to the provided io.WriteSeeker
	writeFile(io.WriteSeeker, llm.KV, []llm.Tensor) error
}

type moreParser interface {
	parseMore(fs.FS) error
}

func ConvertAdapter(fsys fs.FS, ws io.WriteSeeker, baseLayer *llm.GGML) error {
	bts, err := fs.ReadFile(fsys, "adapter_config.json")
	if err != nil {
		return err
	}

	var p AdapterParameters
	if err := json.Unmarshal(bts, &p); err != nil {
		return err
	}

	ts, err := parseTensors(fsys)
	if err != nil {
		return err
	}

	baseKV := baseLayer.KV()
	arch, ok := baseKV["general.architecture"]
	if !ok {
		return errors.New("architecture not set for the base model")
	}

	var conv Converter
	switch arch {
	case "llama":
		conv = &llama{}
	default:
		return errors.New("unsupported architecture")
	}

	if err := json.Unmarshal(bts, conv); err != nil {
		return err
	}

	return conv.writeFile(ws, conv.AdapterKV(baseKV), conv.Tensors(ts))
}

// Convert writes an Ollama compatible model to the provided io.WriteSeeker based on configurations
// and files it finds in the input path.
// Supported input model formats include safetensors.
// Supported input tokenizers files include tokenizer.json (preferred) and tokenizer.model.
func ConvertModel(fsys fs.FS, ws io.WriteSeeker) error {
	bts, err := fs.ReadFile(fsys, "config.json")
	if err != nil {
		return err
	}

	var p Parameters
	if err := json.Unmarshal(bts, &p); err != nil {
		return err
	}

	if len(p.Architectures) < 1 {
		return errors.New("unknown architecture")
	}

	var conv Converter
	switch p.Architectures[0] {
	case "LlamaForCausalLM", "MistralForCausalLM":
		conv = &llama{}
	case "MixtralForCausalLM":
		conv = &mixtral{}
	case "GemmaForCausalLM":
		conv = &gemma{}
	case "Gemma2ForCausalLM":
		conv = &gemma2{}
	case "Phi3ForCausalLM":
		conv = &phi3{}
	case "BertModel":
		conv = &bert{}
	default:
		return errors.New("unsupported architecture")
	}

	if err := json.Unmarshal(bts, conv); err != nil {
		return err
	}

	if t, ok := conv.(moreParser); ok {
		if err := t.parseMore(fsys); err != nil {
			return err
		}
	}

	t, err := parseTokenizer(fsys, conv.specialTokenTypes())
	if err != nil {
		return err
	}

	if vocabSize := int(p.VocabSize); vocabSize > len(t.Vocabulary.Tokens) {
		slog.Warn("vocabulary is smaller than expected, padding with dummy tokens", "expect", p.VocabSize, "actual", len(t.Vocabulary.Tokens))
		for i := range vocabSize - len(t.Vocabulary.Tokens) {
			t.Vocabulary.Tokens = append(t.Vocabulary.Tokens, fmt.Sprintf("[PAD%d]", i))
			t.Vocabulary.Scores = append(t.Vocabulary.Scores, -1)
			t.Vocabulary.Types = append(t.Vocabulary.Types, tokenTypeUserDefined)
		}
	} else {
		slog.Debug("vocabulary", "size", len(t.Vocabulary.Tokens))
	}

	ts, err := parseTensors(fsys, strings.NewReplacer(conv.Replacements()...))
	if err != nil {
		return err
	}

	return conv.writeFile(ws, conv.KV(t), conv.Tensors(ts))
}
