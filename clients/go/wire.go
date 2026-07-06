package llmleaf

// wire.go is the deliberate JSON serialization layer. The generated *pb types
// (camelCase json tags, UPPER enum names) do NOT match the OpenAI/OpenRouter
// wire (snake_case keys, lowercase enum tokens), so the transport never
// (de)serialises *pb structs directly. Instead these small wire DTOs mirror the
// proto messages with correct `json:"..."` tags and convert to/from *pb. The
// *pb types remain the SDK's public typed model; these DTOs are internal.

import (
	"encoding/json"
	"fmt"

	pb "github.com/codefionn/llmleaf/clients/go/llmleafpb"
)

// ---------------------------------------------------------------------------
// pointer helpers
// ---------------------------------------------------------------------------

func ptr[T any](v T) *T { return &v }

// rawValue parses a free-form JSON string (e.g. ChatRequest.extra,
// FunctionDef.parameters) into a json.RawMessage so it is spliced into the body
// verbatim as a JSON value, never double-encoded as a string. An empty / nil
// input yields nil (field omitted). Invalid JSON is surfaced as an error.
func rawValue(s *string) (json.RawMessage, error) {
	if s == nil || *s == "" {
		return nil, nil
	}
	raw := json.RawMessage(*s)
	if !json.Valid(raw) {
		return nil, fmt.Errorf("llmleaf: invalid free-form JSON: %q", *s)
	}
	return raw, nil
}

// rawString captures a decoded JSON sub-object back into a raw-JSON *string for
// the *pb model. A nil / "null" message yields nil.
func rawString(raw json.RawMessage) *string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	return ptr(string(raw))
}

// ---------------------------------------------------------------------------
// content parts
// ---------------------------------------------------------------------------

type wireTextPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type wireImageURL struct {
	URL    string  `json:"url"`
	Detail *string `json:"detail,omitempty"`
}

type wireImagePart struct {
	Type     string       `json:"type"`
	ImageURL wireImageURL `json:"image_url"`
}

// wireContentPart marshals to one of the OpenAI content-part shapes depending on
// which oneof arm is set.
type wireContentPart struct {
	text  *pb.TextPart
	image *pb.ImageUrlPart
}

func (p wireContentPart) MarshalJSON() ([]byte, error) {
	switch {
	case p.text != nil:
		return json.Marshal(wireTextPart{Type: "text", Text: p.text.GetText()})
	case p.image != nil:
		iu := wireImageURL{URL: p.image.GetUrl()}
		if p.image.Detail != nil {
			iu.Detail = p.image.Detail
		}
		return json.Marshal(wireImagePart{Type: "image_url", ImageURL: iu})
	default:
		return []byte("null"), nil
	}
}

func (p *wireContentPart) UnmarshalJSON(data []byte) error {
	var probe struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL    string  `json:"url"`
			Detail *string `json:"detail"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Type {
	case "image_url":
		if probe.ImageURL != nil {
			p.image = &pb.ImageUrlPart{Url: probe.ImageURL.URL, Detail: probe.ImageURL.Detail}
		}
	default: // "text" or unknown -> treat as text
		p.text = &pb.TextPart{Text: probe.Text}
	}
	return nil
}

func partToPB(p *wireContentPart) *pb.ContentPart {
	switch {
	case p.image != nil:
		return &pb.ContentPart{Part: &pb.ContentPart_ImageUrl{ImageUrl: p.image}}
	default:
		t := p.text
		if t == nil {
			t = &pb.TextPart{}
		}
		return &pb.ContentPart{Part: &pb.ContentPart_Text{Text: t}}
	}
}

func partFromPB(p *pb.ContentPart) wireContentPart {
	return wireContentPart{text: p.GetText(), image: p.GetImageUrl()}
}

// ---------------------------------------------------------------------------
// tool calls
// ---------------------------------------------------------------------------

type wireFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function wireFunctionCall `json:"function"`
}

func toolCallToWire(tc *pb.ToolCall) wireToolCall {
	out := wireToolCall{ID: tc.GetId(), Type: tc.GetType()}
	if fn := tc.GetFunction(); fn != nil {
		out.Function = wireFunctionCall{Name: fn.GetName(), Arguments: fn.GetArguments()}
	}
	return out
}

func toolCallFromWire(tc wireToolCall) *pb.ToolCall {
	return &pb.ToolCall{
		Id:       tc.ID,
		Type:     tc.Type,
		Function: &pb.FunctionCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments},
	}
}

type wireFunctionCallDelta struct {
	Name      *string `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

type wireToolCallDelta struct {
	Index    uint32                 `json:"index"`
	ID       *string                `json:"id,omitempty"`
	Type     *string                `json:"type,omitempty"`
	Function *wireFunctionCallDelta `json:"function,omitempty"`
}

func toolCallDeltaFromWire(d wireToolCallDelta) *pb.ToolCallDelta {
	out := &pb.ToolCallDelta{Index: d.Index, Id: d.ID, Type: d.Type}
	if d.Function != nil {
		out.Function = &pb.FunctionCallDelta{Name: d.Function.Name, Arguments: d.Function.Arguments}
	}
	return out
}

// ---------------------------------------------------------------------------
// reasoning ("thinking") blocks
// ---------------------------------------------------------------------------

// wireReasoningDetail mirrors one OpenRouter reasoning_details[] entry. `type` discriminates:
// "reasoning.text" -> Text (+ optional Signature) is OPEN; "reasoning.summary" -> Summary is OPEN;
// "reasoning.encrypted" -> Data is HIDDEN (redacted). Signature/Data are opaque and replayed verbatim.
type wireReasoningDetail struct {
	Type      string  `json:"type"`
	Text      *string `json:"text,omitempty"`
	Summary   *string `json:"summary,omitempty"`
	Data      *string `json:"data,omitempty"`
	Signature *string `json:"signature,omitempty"`
	ID        *string `json:"id,omitempty"`
	Format    *string `json:"format,omitempty"`
	Index     *uint32 `json:"index,omitempty"`
}

func reasoningDetailToWire(d *pb.ReasoningDetail) wireReasoningDetail {
	return wireReasoningDetail{
		Type:      d.GetType(),
		Text:      d.Text,
		Summary:   d.Summary,
		Data:      d.Data,
		Signature: d.Signature,
		ID:        d.Id,
		Format:    d.Format,
		Index:     d.Index,
	}
}

func reasoningDetailFromWire(d wireReasoningDetail) *pb.ReasoningDetail {
	return &pb.ReasoningDetail{
		Type:      d.Type,
		Text:      d.Text,
		Summary:   d.Summary,
		Data:      d.Data,
		Signature: d.Signature,
		Id:        d.ID,
		Format:    d.Format,
		Index:     d.Index,
	}
}

// ---------------------------------------------------------------------------
// chat message
// ---------------------------------------------------------------------------

type wireChatMessage struct {
	Role             string                `json:"role"`
	Content          json.RawMessage       `json:"content,omitempty"`
	Name             *string               `json:"name,omitempty"`
	ToolCalls        []wireToolCall        `json:"tool_calls,omitempty"`
	ToolCallID       *string               `json:"tool_call_id,omitempty"`
	Reasoning        *string               `json:"reasoning,omitempty"`
	ReasoningDetails []wireReasoningDetail `json:"reasoning_details,omitempty"`
}

func chatMessageToWire(m *pb.ChatMessage) (wireChatMessage, error) {
	out := wireChatMessage{
		Role:       enumToWire(pb.Role_name, int32(m.GetRole())),
		Name:       m.Name,
		ToolCallID: m.ToolCallId,
	}
	switch c := m.GetContent().(type) {
	case *pb.ChatMessage_Text:
		b, err := json.Marshal(c.Text)
		if err != nil {
			return out, err
		}
		out.Content = b
	case *pb.ChatMessage_Parts:
		parts := make([]wireContentPart, 0, len(c.Parts.GetItems()))
		for _, it := range c.Parts.GetItems() {
			parts = append(parts, partFromPB(it))
		}
		b, err := json.Marshal(parts)
		if err != nil {
			return out, err
		}
		out.Content = b
	}
	for _, tc := range m.GetToolCalls() {
		out.ToolCalls = append(out.ToolCalls, toolCallToWire(tc))
	}
	out.Reasoning = m.Reasoning
	for _, rd := range m.GetReasoningDetails() {
		out.ReasoningDetails = append(out.ReasoningDetails, reasoningDetailToWire(rd))
	}
	return out, nil
}

func chatMessageFromWire(m wireChatMessage) (*pb.ChatMessage, error) {
	out := &pb.ChatMessage{Name: m.Name, ToolCallId: m.ToolCallID, Reasoning: m.Reasoning}
	if role, ok := enumFromWire(pb.Role_value, m.Role); ok {
		out.Role = pb.Role(role)
	}
	if len(m.Content) > 0 && string(m.Content) != "null" {
		switch m.Content[0] {
		case '"':
			var s string
			if err := json.Unmarshal(m.Content, &s); err != nil {
				return nil, err
			}
			out.Content = &pb.ChatMessage_Text{Text: s}
		case '[':
			var parts []wireContentPart
			if err := json.Unmarshal(m.Content, &parts); err != nil {
				return nil, err
			}
			items := make([]*pb.ContentPart, 0, len(parts))
			for i := range parts {
				items = append(items, partToPB(&parts[i]))
			}
			out.Content = &pb.ChatMessage_Parts{Parts: &pb.ContentParts{Items: items}}
		}
	}
	for _, tc := range m.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, toolCallFromWire(tc))
	}
	for _, rd := range m.ReasoningDetails {
		out.ReasoningDetails = append(out.ReasoningDetails, reasoningDetailFromWire(rd))
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// tools / tool_choice / response_format
// ---------------------------------------------------------------------------

type wireFunctionDef struct {
	Name        string          `json:"name"`
	Description *string         `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type wireToolDef struct {
	Type     string          `json:"type"`
	Function wireFunctionDef `json:"function"`
}

func toolDefToWire(t *pb.ToolDef) (wireToolDef, error) {
	out := wireToolDef{Type: t.GetType()}
	if fn := t.GetFunction(); fn != nil {
		params, err := rawValue(fn.Parameters)
		if err != nil {
			return out, err
		}
		out.Function = wireFunctionDef{
			Name:        fn.GetName(),
			Description: fn.Description,
			Parameters:  params,
		}
	}
	return out, nil
}

// wireToolChoice serialises the oneof: a bare mode string or a named object.
type wireToolChoice struct {
	mode  string
	named *pb.NamedToolChoice
}

func (t wireToolChoice) MarshalJSON() ([]byte, error) {
	if t.named != nil {
		obj := struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}{Type: t.named.GetType()}
		if fn := t.named.GetFunction(); fn != nil {
			obj.Function.Name = fn.GetName()
		}
		return json.Marshal(obj)
	}
	return json.Marshal(t.mode)
}

func toolChoiceToWire(tc *pb.ToolChoice) *wireToolChoice {
	if tc == nil {
		return nil
	}
	switch c := tc.GetChoice().(type) {
	case *pb.ToolChoice_Mode:
		return &wireToolChoice{mode: c.Mode}
	case *pb.ToolChoice_Named:
		return &wireToolChoice{named: c.Named}
	default:
		return nil
	}
}

type wireResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

func responseFormatToWire(rf *pb.ResponseFormat) (*wireResponseFormat, error) {
	if rf == nil {
		return nil, nil
	}
	schema, err := rawValue(rf.JsonSchema)
	if err != nil {
		return nil, err
	}
	return &wireResponseFormat{Type: rf.GetType(), JSONSchema: schema}, nil
}

// ---------------------------------------------------------------------------
// chat request
// ---------------------------------------------------------------------------

// wireChatRequest carries every ChatRequest field except `extra`, which is
// merged at the top level by encodeChatRequest.
type wireChatRequest struct {
	Model               string              `json:"model"`
	Messages            []wireChatMessage   `json:"messages"`
	Stream              *bool               `json:"stream,omitempty"`
	Temperature         *float32            `json:"temperature,omitempty"`
	TopP                *float32            `json:"top_p,omitempty"`
	MaxTokens           *uint32             `json:"max_tokens,omitempty"`
	MaxCompletionTokens *uint32             `json:"max_completion_tokens,omitempty"`
	Stop                json.RawMessage     `json:"stop,omitempty"`
	N                   *uint32             `json:"n,omitempty"`
	Seed                *int64              `json:"seed,omitempty"`
	FrequencyPenalty    *float32            `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float32            `json:"presence_penalty,omitempty"`
	Tools               []wireToolDef       `json:"tools,omitempty"`
	ToolChoice          *wireToolChoice     `json:"tool_choice,omitempty"`
	ResponseFormat      *wireResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort     *string             `json:"reasoning_effort,omitempty"`
}

// encodeStop emits a bare string for a single element, else an array. nil for
// an empty list.
func encodeStop(stop []string) (json.RawMessage, error) {
	switch len(stop) {
	case 0:
		return nil, nil
	case 1:
		return json.Marshal(stop[0])
	default:
		return json.Marshal(stop)
	}
}

// chatRequestToWire converts the *pb model into the wire DTO plus the parsed
// `extra` object to merge at the top level. streamOverride, when non-nil,
// replaces the request's stream flag on the wire without mutating req.
func chatRequestToWire(req *pb.ChatRequest, streamOverride *bool) (wireChatRequest, json.RawMessage, error) {
	stream := req.Stream
	if streamOverride != nil {
		stream = streamOverride
	}
	w := wireChatRequest{
		Model:               req.GetModel(),
		Stream:              stream,
		Temperature:         req.Temperature,
		TopP:                req.TopP,
		MaxTokens:           req.MaxTokens,
		MaxCompletionTokens: req.MaxCompletionTokens,
		N:                   req.N,
		Seed:                req.Seed,
		FrequencyPenalty:    req.FrequencyPenalty,
		PresencePenalty:     req.PresencePenalty,
		ToolChoice:          toolChoiceToWire(req.ToolChoice),
		ReasoningEffort:     req.ReasoningEffort,
	}
	for _, m := range req.GetMessages() {
		wm, err := chatMessageToWire(m)
		if err != nil {
			return w, nil, err
		}
		w.Messages = append(w.Messages, wm)
	}
	if w.Messages == nil {
		w.Messages = []wireChatMessage{}
	}
	stop, err := encodeStop(req.GetStop())
	if err != nil {
		return w, nil, err
	}
	w.Stop = stop
	for _, t := range req.GetTools() {
		wt, err := toolDefToWire(t)
		if err != nil {
			return w, nil, err
		}
		w.Tools = append(w.Tools, wt)
	}
	rf, err := responseFormatToWire(req.ResponseFormat)
	if err != nil {
		return w, nil, err
	}
	w.ResponseFormat = rf
	extra, err := rawValue(req.Extra)
	if err != nil {
		return w, nil, err
	}
	return w, extra, nil
}

// encodeChatRequest produces the final request body, merging `extra` keys at the
// top level of the request object (per SPEC.md). Explicit ChatRequest fields win
// over extra keys of the same name. streamOverride, when non-nil, forces the
// stream flag without mutating req (nil leaves the caller's flag untouched).
func encodeChatRequest(req *pb.ChatRequest, streamOverride *bool) ([]byte, error) {
	w, extra, err := chatRequestToWire(req, streamOverride)
	if err != nil {
		return nil, err
	}
	return mergeExtra(w, extra)
}

// mergeExtra marshals v, then splices the keys of `extra` (a JSON object) into
// the top level without overwriting keys v already set.
func mergeExtra(v any, extra json.RawMessage) ([]byte, error) {
	base, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	if len(extra) == 0 {
		return base, nil
	}
	var extraObj map[string]json.RawMessage
	if err := json.Unmarshal(extra, &extraObj); err != nil {
		return nil, fmt.Errorf("llmleaf: `extra` must be a JSON object: %w", err)
	}
	var merged map[string]json.RawMessage
	if err := json.Unmarshal(base, &merged); err != nil {
		return nil, err
	}
	for k, val := range extraObj {
		if _, exists := merged[k]; !exists {
			merged[k] = val
		}
	}
	return json.Marshal(merged)
}

// ---------------------------------------------------------------------------
// chat response (decode)
// ---------------------------------------------------------------------------

type wireChoice struct {
	Index        uint32          `json:"index"`
	Message      wireChatMessage `json:"message"`
	FinishReason *string         `json:"finish_reason"`
}

// wirePromptTokensDetails mirrors usage.prompt_tokens_details — today just the cache-read (hit) share.
type wirePromptTokensDetails struct {
	CachedTokens *uint32 `json:"cached_tokens,omitempty"`
}

type wireUsage struct {
	PromptTokens        uint32                   `json:"prompt_tokens"`
	CompletionTokens    uint32                   `json:"completion_tokens"`
	TotalTokens         uint32                   `json:"total_tokens"`
	CostUSD             *float64                 `json:"cost_usd"`
	PromptTokensDetails *wirePromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	CacheCreationTokens *uint32                  `json:"cache_creation_tokens,omitempty"`
}

func usageFromWire(u *wireUsage) *pb.Usage {
	if u == nil {
		return nil
	}
	out := &pb.Usage{
		PromptTokens:        u.PromptTokens,
		CompletionTokens:    u.CompletionTokens,
		TotalTokens:         u.TotalTokens,
		CostUsd:             u.CostUSD,
		CacheCreationTokens: u.CacheCreationTokens,
	}
	if d := u.PromptTokensDetails; d != nil {
		out.PromptTokensDetails = &pb.PromptTokensDetails{CachedTokens: d.CachedTokens}
	}
	return out
}

type wireChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []wireChoice `json:"choices"`
	Usage   *wireUsage   `json:"usage"`
}

func finishReasonFromWire(s *string) *pb.FinishReason {
	if s == nil {
		return nil
	}
	if v, ok := enumFromWire(pb.FinishReason_value, *s); ok {
		return ptr(pb.FinishReason(v))
	}
	return nil
}

func (w *wireChatResponse) toPB() (*pb.ChatResponse, error) {
	out := &pb.ChatResponse{
		Id:      w.ID,
		Object:  w.Object,
		Created: w.Created,
		Model:   w.Model,
		Usage:   usageFromWire(w.Usage),
	}
	for _, c := range w.Choices {
		msg, err := chatMessageFromWire(c.Message)
		if err != nil {
			return nil, err
		}
		out.Choices = append(out.Choices, &pb.Choice{
			Index:        c.Index,
			Message:      msg,
			FinishReason: finishReasonFromWire(c.FinishReason),
		})
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// streaming chunk (decode)
// ---------------------------------------------------------------------------

type wireDelta struct {
	Role             *string               `json:"role"`
	Content          *string               `json:"content"`
	ToolCalls        []wireToolCallDelta   `json:"tool_calls"`
	Reasoning        *string               `json:"reasoning,omitempty"`
	ReasoningDetails []wireReasoningDetail `json:"reasoning_details,omitempty"`
}

type wireChunkChoice struct {
	Index        uint32    `json:"index"`
	Delta        wireDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"`
}

type wireChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []wireChunkChoice `json:"choices"`
	Usage   *wireUsage        `json:"usage"`
}

func (w *wireChunk) toPB() *pb.ChatCompletionChunk {
	out := &pb.ChatCompletionChunk{
		Id:      w.ID,
		Object:  w.Object,
		Created: w.Created,
		Model:   w.Model,
		Usage:   usageFromWire(w.Usage),
	}
	for _, c := range w.Choices {
		delta := &pb.Delta{Content: c.Delta.Content, Reasoning: c.Delta.Reasoning}
		if c.Delta.Role != nil {
			if v, ok := enumFromWire(pb.Role_value, *c.Delta.Role); ok {
				delta.Role = ptr(pb.Role(v))
			}
		}
		for _, tc := range c.Delta.ToolCalls {
			delta.ToolCalls = append(delta.ToolCalls, toolCallDeltaFromWire(tc))
		}
		for _, rd := range c.Delta.ReasoningDetails {
			delta.ReasoningDetails = append(delta.ReasoningDetails, reasoningDetailFromWire(rd))
		}
		out.Choices = append(out.Choices, &pb.ChunkChoice{
			Index:        c.Index,
			Delta:        delta,
			FinishReason: finishReasonFromWire(c.FinishReason),
		})
	}
	return out
}

// ---------------------------------------------------------------------------
// embeddings
// ---------------------------------------------------------------------------

type wireEmbeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	Dimensions     *uint32         `json:"dimensions,omitempty"`
	EncodingFormat *string         `json:"encoding_format,omitempty"`
}

func encodeInput(input []string) (json.RawMessage, error) {
	if len(input) == 1 {
		return json.Marshal(input[0])
	}
	if input == nil {
		input = []string{}
	}
	return json.Marshal(input)
}

func encodeEmbeddingRequest(req *pb.EmbeddingRequest) ([]byte, error) {
	in, err := encodeInput(req.GetInput())
	if err != nil {
		return nil, err
	}
	w := wireEmbeddingRequest{
		Model:          req.GetModel(),
		Input:          in,
		Dimensions:     req.Dimensions,
		EncodingFormat: req.EncodingFormat,
	}
	extra, err := rawValue(req.Extra)
	if err != nil {
		return nil, err
	}
	return mergeExtra(w, extra)
}

// wireEmbedding decodes `embedding` as either a float array or a base64 string
// of little-endian f32 bytes.
type wireEmbedding struct {
	Object    string          `json:"object"`
	Index     uint32          `json:"index"`
	Embedding json.RawMessage `json:"embedding"`
}

type wireEmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []wireEmbedding `json:"data"`
	Model  string          `json:"model"`
	Usage  *wireUsage      `json:"usage"`
}

func (w *wireEmbeddingResponse) toPB() (*pb.EmbeddingResponse, error) {
	out := &pb.EmbeddingResponse{
		Object: w.Object,
		Model:  w.Model,
		Usage:  usageFromWire(w.Usage),
	}
	for _, e := range w.Data {
		vec, err := decodeEmbeddingVector(e.Embedding)
		if err != nil {
			return nil, err
		}
		out.Data = append(out.Data, &pb.Embedding{
			Object:    e.Object,
			Index:     e.Index,
			Embedding: vec,
		})
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// speech / voices
// ---------------------------------------------------------------------------

type wireSpeechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice"`
	ResponseFormat *string  `json:"response_format,omitempty"`
	Speed          *float32 `json:"speed,omitempty"`
}

func encodeSpeechRequest(req *pb.SpeechRequest) ([]byte, error) {
	w := wireSpeechRequest{
		Model:          req.GetModel(),
		Input:          req.GetInput(),
		Voice:          req.GetVoice(),
		ResponseFormat: req.ResponseFormat,
		Speed:          req.Speed,
	}
	extra, err := rawValue(req.Extra)
	if err != nil {
		return nil, err
	}
	return mergeExtra(w, extra)
}

type wireVoice struct {
	ID        string   `json:"id"`
	Name      *string  `json:"name"`
	Languages []string `json:"languages"`
}

type wireVoicesResponse struct {
	Model  string      `json:"model"`
	Voices []wireVoice `json:"voices"`
}

func (w *wireVoicesResponse) toPB() *pb.VoicesResponse {
	out := &pb.VoicesResponse{Model: w.Model}
	for _, v := range w.Voices {
		out.Voices = append(out.Voices, &pb.Voice{
			Id:        v.ID,
			Name:      v.Name,
			Languages: v.Languages,
		})
	}
	return out
}

// ---------------------------------------------------------------------------
// transcription
// ---------------------------------------------------------------------------

type wireTranscriptionResponse struct {
	Text     string     `json:"text"`
	Task     *string    `json:"task"`
	Language *string    `json:"language"`
	Duration *float32   `json:"duration"`
	Usage    *wireUsage `json:"usage"`
}

func (w *wireTranscriptionResponse) toPB() *pb.TranscriptionResponse {
	return &pb.TranscriptionResponse{
		Text:     w.Text,
		Task:     w.Task,
		Language: w.Language,
		Duration: w.Duration,
		Usage:    usageFromWire(w.Usage),
	}
}

// ---------------------------------------------------------------------------
// models catalog
// ---------------------------------------------------------------------------

type wireArchitecture struct {
	InputModalities  []string `json:"input_modalities"`
	OutputModalities []string `json:"output_modalities"`
	Modality         *string  `json:"modality"`
	Tokenizer        string   `json:"tokenizer"`
	InstructType     *string  `json:"instruct_type"`
}

type wirePricing struct {
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
}

type wireTopProvider struct {
	ContextLength       *uint32 `json:"context_length"`
	MaxCompletionTokens *uint32 `json:"max_completion_tokens"`
	IsModerated         bool    `json:"is_moderated"`
	MaxThinkingTokens   *uint32 `json:"max_thinking_tokens"`
}

type wireModelEndpoint struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	Down     bool   `json:"down"`
	Source   string `json:"source"`
}

type wireModelEntry struct {
	ID                    string              `json:"id"`
	CanonicalSlug         string              `json:"canonical_slug"`
	Name                  string              `json:"name"`
	Created               int64               `json:"created"`
	Description           string              `json:"description"`
	ContextLength         *uint32             `json:"context_length"`
	Architecture          *wireArchitecture   `json:"architecture"`
	Pricing               *wirePricing        `json:"pricing"`
	TopProvider           *wireTopProvider    `json:"top_provider"`
	SupportedParameters   []string            `json:"supported_parameters"`
	UnsupportedParameters []string            `json:"unsupported_parameters"`
	DefaultParameters     json.RawMessage     `json:"default_parameters"`
	Endpoints             []wireModelEndpoint `json:"endpoints"`
}

type wireListModelsResponse struct {
	Data []wireModelEntry `json:"data"`
}

func (w *wireModelEntry) toPB() *pb.ModelEntry {
	out := &pb.ModelEntry{
		Id:                    w.ID,
		CanonicalSlug:         w.CanonicalSlug,
		Name:                  w.Name,
		Created:               w.Created,
		Description:           w.Description,
		ContextLength:         w.ContextLength,
		SupportedParameters:   w.SupportedParameters,
		UnsupportedParameters: w.UnsupportedParameters,
		DefaultParameters:     rawString(w.DefaultParameters),
	}
	if a := w.Architecture; a != nil {
		out.Architecture = &pb.Architecture{
			InputModalities:  a.InputModalities,
			OutputModalities: a.OutputModalities,
			Modality:         a.Modality,
			Tokenizer:        a.Tokenizer,
			InstructType:     a.InstructType,
		}
	}
	if p := w.Pricing; p != nil {
		out.Pricing = &pb.Pricing{Prompt: p.Prompt, Completion: p.Completion}
	}
	if tp := w.TopProvider; tp != nil {
		out.TopProvider = &pb.TopProvider{
			ContextLength:       tp.ContextLength,
			MaxCompletionTokens: tp.MaxCompletionTokens,
			IsModerated:         tp.IsModerated,
			MaxThinkingTokens:   tp.MaxThinkingTokens,
		}
	}
	for _, e := range w.Endpoints {
		out.Endpoints = append(out.Endpoints, &pb.ModelEndpoint{
			Provider: e.Provider,
			Model:    e.Model,
			Down:     e.Down,
			Source:   e.Source,
		})
	}
	return out
}

func (w *wireListModelsResponse) toPB() *pb.ListModelsResponse {
	out := &pb.ListModelsResponse{}
	for i := range w.Data {
		out.Data = append(out.Data, w.Data[i].toPB())
	}
	return out
}

// ---------------------------------------------------------------------------
// batches
// ---------------------------------------------------------------------------

type wireBatchRequestItem struct {
	CustomID string          `json:"custom_id"`
	Body     json.RawMessage `json:"body"`
}

type wireBatchCreateRequest struct {
	Requests []wireBatchRequestItem `json:"requests"`
}

func encodeBatchCreateRequest(req *pb.BatchCreateRequest) ([]byte, error) {
	w := wireBatchCreateRequest{}
	for _, item := range req.GetRequests() {
		body, err := encodeChatRequest(item.GetBody(), nil)
		if err != nil {
			return nil, err
		}
		w.Requests = append(w.Requests, wireBatchRequestItem{
			CustomID: item.GetCustomId(),
			Body:     body,
		})
	}
	if w.Requests == nil {
		w.Requests = []wireBatchRequestItem{}
	}
	return json.Marshal(w)
}

type wireBatchCounts struct {
	Total      uint64 `json:"total"`
	Processing uint64 `json:"processing"`
	Succeeded  uint64 `json:"succeeded"`
	Errored    uint64 `json:"errored"`
	Canceled   uint64 `json:"canceled"`
	Expired    uint64 `json:"expired"`
}

type wireBatchHandle struct {
	ID        string           `json:"id"`
	Status    string           `json:"status"`
	Counts    *wireBatchCounts `json:"counts"`
	CreatedAt *int64           `json:"created_at"`
	ExpiresAt *int64           `json:"expires_at"`
	EndedAt   *int64           `json:"ended_at"`
	Endpoint  *string          `json:"endpoint"`
}

func (w *wireBatchHandle) toPB() *pb.BatchHandle {
	out := &pb.BatchHandle{
		Id:        w.ID,
		CreatedAt: w.CreatedAt,
		ExpiresAt: w.ExpiresAt,
		EndedAt:   w.EndedAt,
		Endpoint:  w.Endpoint,
	}
	if v, ok := enumFromWire(pb.BatchStatus_value, w.Status); ok {
		out.Status = pb.BatchStatus(v)
	}
	if c := w.Counts; c != nil {
		out.Counts = &pb.BatchCounts{
			Total:      c.Total,
			Processing: c.Processing,
			Succeeded:  c.Succeeded,
			Errored:    c.Errored,
			Canceled:   c.Canceled,
			Expired:    c.Expired,
		}
	}
	return out
}

type wireBatchResponse struct {
	StatusCode uint32           `json:"status_code"`
	Body       wireChatResponse `json:"body"`
}

type wireBatchError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type wireBatchResultLine struct {
	CustomID string             `json:"custom_id"`
	Response *wireBatchResponse `json:"response"`
	Error    *wireBatchError    `json:"error"`
}

func (w *wireBatchResultLine) toPB() (*pb.BatchResultLine, error) {
	out := &pb.BatchResultLine{CustomId: w.CustomID}
	if w.Response != nil {
		body, err := w.Response.Body.toPB()
		if err != nil {
			return nil, err
		}
		out.Response = &pb.BatchResponse{StatusCode: w.Response.StatusCode, Body: body}
	}
	if w.Error != nil {
		out.Error = &pb.BatchError{Code: w.Error.Code, Message: w.Error.Message}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// error body (shared by the Responses `error` field / "error" stream event)
// ---------------------------------------------------------------------------

type wireErrorBody struct {
	Message string  `json:"message"`
	Type    *string `json:"type,omitempty"`
	Code    *string `json:"code,omitempty"`
}

func (w *wireErrorBody) toPB() *pb.ErrorBody {
	if w == nil {
		return nil
	}
	return &pb.ErrorBody{Message: w.Message, Type: w.Type, Code: w.Code}
}

// ---------------------------------------------------------------------------
// Responses dialect — content parts
//
// Unlike the chat dialect these carry input_text / input_image / output_text
// tokens; input_image.image_url is a plain STRING (not the chat `{url}` object)
// and a constructed output_text part emits "annotations":[].
// ---------------------------------------------------------------------------

type wireResponseInputText struct {
	Type string `json:"type"` // "input_text"
	Text string `json:"text"`
}

type wireResponseInputImage struct {
	Type     string  `json:"type"` // "input_image"
	ImageURL string  `json:"image_url"`
	Detail   *string `json:"detail,omitempty"`
}

type wireResponseOutputText struct {
	Type        string            `json:"type"` // "output_text"
	Text        string            `json:"text"`
	Annotations []json.RawMessage `json:"annotations"` // emitted as [] when constructing
}

// wireResponseContentPart marshals to one of the Responses content-part shapes
// depending on which oneof arm is set.
type wireResponseContentPart struct {
	inputText  *pb.ResponseTextPart
	inputImage *pb.ResponseInputImagePart
	outputText *pb.ResponseTextPart
}

func (p wireResponseContentPart) MarshalJSON() ([]byte, error) {
	switch {
	case p.inputImage != nil:
		return json.Marshal(wireResponseInputImage{
			Type:     "input_image",
			ImageURL: p.inputImage.GetImageUrl(),
			Detail:   p.inputImage.Detail,
		})
	case p.outputText != nil:
		return json.Marshal(wireResponseOutputText{
			Type:        "output_text",
			Text:        p.outputText.GetText(),
			Annotations: []json.RawMessage{},
		})
	case p.inputText != nil:
		return json.Marshal(wireResponseInputText{Type: "input_text", Text: p.inputText.GetText()})
	default:
		return []byte("null"), nil
	}
}

func (p *wireResponseContentPart) UnmarshalJSON(data []byte) error {
	var probe struct {
		Type     string  `json:"type"`
		Text     string  `json:"text"`
		ImageURL string  `json:"image_url"`
		Detail   *string `json:"detail"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Type {
	case "input_image":
		p.inputImage = &pb.ResponseInputImagePart{ImageUrl: probe.ImageURL, Detail: probe.Detail}
	case "output_text":
		p.outputText = &pb.ResponseTextPart{Text: probe.Text}
	default: // "input_text" or unknown -> treat as input text
		p.inputText = &pb.ResponseTextPart{Text: probe.Text}
	}
	return nil
}

func responseContentPartToPB(p *wireResponseContentPart) *pb.ResponseContentPart {
	switch {
	case p.inputImage != nil:
		return &pb.ResponseContentPart{Part: &pb.ResponseContentPart_InputImage{InputImage: p.inputImage}}
	case p.outputText != nil:
		return &pb.ResponseContentPart{Part: &pb.ResponseContentPart_OutputText{OutputText: p.outputText}}
	default:
		t := p.inputText
		if t == nil {
			t = &pb.ResponseTextPart{}
		}
		return &pb.ResponseContentPart{Part: &pb.ResponseContentPart_InputText{InputText: t}}
	}
}

func responseContentPartFromPB(p *pb.ResponseContentPart) wireResponseContentPart {
	return wireResponseContentPart{
		inputText:  p.GetInputText(),
		inputImage: p.GetInputImage(),
		outputText: p.GetOutputText(),
	}
}

// ---------------------------------------------------------------------------
// Responses dialect — response items (the input / output oneof)
// ---------------------------------------------------------------------------

// wireResponseMessageItem is a role-keyed message item. On the wire it carries
// NO "type" token (plain messages are role-keyed objects); the item dispatcher
// routes to it for `"type":"message"` or an absent type.
type wireResponseMessageItem struct {
	ID      *string         `json:"id,omitempty"`
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content,omitempty"`
	Status  *string         `json:"status,omitempty"`
}

type wireResponseFunctionCallItem struct {
	Type      string  `json:"type"` // "function_call"
	ID        *string `json:"id,omitempty"`
	CallID    string  `json:"call_id"`
	Name      string  `json:"name"`
	Arguments string  `json:"arguments"`
	Status    *string `json:"status,omitempty"`
}

type wireResponseFunctionCallOutputItem struct {
	Type   string  `json:"type"` // "function_call_output"
	ID     *string `json:"id,omitempty"`
	CallID string  `json:"call_id"`
	Output string  `json:"output"`
}

// wireResponseReasoningEntry is one summary[]/content[] entry; its "type" token
// ("summary_text" vs "reasoning_text") is decided by the list it lives in.
type wireResponseReasoningEntry struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type wireResponseReasoningItem struct {
	Type             string                       `json:"type"` // "reasoning"
	ID               *string                      `json:"id,omitempty"`
	Summary          []wireResponseReasoningEntry `json:"summary"`
	Content          []wireResponseReasoningEntry `json:"content,omitempty"`
	EncryptedContent *string                      `json:"encrypted_content,omitempty"`
}

func reasoningItemToWire(r *pb.ResponseReasoningItem) wireResponseReasoningItem {
	out := wireResponseReasoningItem{Type: "reasoning", ID: r.Id, EncryptedContent: r.EncryptedContent}
	out.Summary = make([]wireResponseReasoningEntry, 0, len(r.GetSummary()))
	for _, s := range r.GetSummary() {
		out.Summary = append(out.Summary, wireResponseReasoningEntry{Type: "summary_text", Text: s.GetText()})
	}
	for _, c := range r.GetContent() {
		out.Content = append(out.Content, wireResponseReasoningEntry{Type: "reasoning_text", Text: c.GetText()})
	}
	return out
}

func reasoningItemFromWire(r *wireResponseReasoningItem) *pb.ResponseReasoningItem {
	out := &pb.ResponseReasoningItem{Id: r.ID, EncryptedContent: r.EncryptedContent}
	for _, s := range r.Summary {
		out.Summary = append(out.Summary, &pb.ResponseReasoningText{Text: s.Text})
	}
	for _, c := range r.Content {
		out.Content = append(out.Content, &pb.ResponseReasoningText{Text: c.Text})
	}
	return out
}

// wireResponseItem serialises the ResponseItem oneof: message items are
// role-keyed (no "type"); the rest carry their "type" discriminator. On decode
// the "type" field selects the arm ("message" or absent -> message).
type wireResponseItem struct {
	message            *pb.ResponseMessageItem
	functionCall       *pb.ResponseFunctionCallItem
	functionCallOutput *pb.ResponseFunctionCallOutputItem
	reasoning          *pb.ResponseReasoningItem
}

func (i wireResponseItem) MarshalJSON() ([]byte, error) {
	switch {
	case i.functionCall != nil:
		fc := i.functionCall
		return json.Marshal(wireResponseFunctionCallItem{
			Type:      "function_call",
			ID:        fc.Id,
			CallID:    fc.GetCallId(),
			Name:      fc.GetName(),
			Arguments: fc.GetArguments(),
			Status:    fc.Status,
		})
	case i.functionCallOutput != nil:
		fo := i.functionCallOutput
		return json.Marshal(wireResponseFunctionCallOutputItem{
			Type:   "function_call_output",
			ID:     fo.Id,
			CallID: fo.GetCallId(),
			Output: fo.GetOutput(),
		})
	case i.reasoning != nil:
		return json.Marshal(reasoningItemToWire(i.reasoning))
	case i.message != nil:
		return marshalResponseMessageItem(i.message)
	default:
		return []byte("null"), nil
	}
}

func (i *wireResponseItem) UnmarshalJSON(data []byte) error {
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch probe.Type {
	case "function_call":
		var fc wireResponseFunctionCallItem
		if err := json.Unmarshal(data, &fc); err != nil {
			return err
		}
		i.functionCall = &pb.ResponseFunctionCallItem{
			Id: fc.ID, CallId: fc.CallID, Name: fc.Name, Arguments: fc.Arguments, Status: fc.Status,
		}
	case "function_call_output":
		var fo wireResponseFunctionCallOutputItem
		if err := json.Unmarshal(data, &fo); err != nil {
			return err
		}
		i.functionCallOutput = &pb.ResponseFunctionCallOutputItem{Id: fo.ID, CallId: fo.CallID, Output: fo.Output}
	case "reasoning":
		var r wireResponseReasoningItem
		if err := json.Unmarshal(data, &r); err != nil {
			return err
		}
		i.reasoning = reasoningItemFromWire(&r)
	default: // "message" or absent -> role-keyed message
		m, err := unmarshalResponseMessageItem(data)
		if err != nil {
			return err
		}
		i.message = m
	}
	return nil
}

func (i *wireResponseItem) toPB() *pb.ResponseItem {
	switch {
	case i.functionCall != nil:
		return &pb.ResponseItem{Item: &pb.ResponseItem_FunctionCall{FunctionCall: i.functionCall}}
	case i.functionCallOutput != nil:
		return &pb.ResponseItem{Item: &pb.ResponseItem_FunctionCallOutput{FunctionCallOutput: i.functionCallOutput}}
	case i.reasoning != nil:
		return &pb.ResponseItem{Item: &pb.ResponseItem_Reasoning{Reasoning: i.reasoning}}
	default:
		msg := i.message
		if msg == nil {
			msg = &pb.ResponseMessageItem{}
		}
		return &pb.ResponseItem{Item: &pb.ResponseItem_Message{Message: msg}}
	}
}

func responseItemFromPB(it *pb.ResponseItem) wireResponseItem {
	return wireResponseItem{
		message:            it.GetMessage(),
		functionCall:       it.GetFunctionCall(),
		functionCallOutput: it.GetFunctionCallOutput(),
		reasoning:          it.GetReasoning(),
	}
}

func marshalResponseMessageItem(m *pb.ResponseMessageItem) ([]byte, error) {
	out := wireResponseMessageItem{ID: m.Id, Role: m.GetRole(), Status: m.Status}
	switch c := m.GetContent().(type) {
	case *pb.ResponseMessageItem_Text:
		b, err := json.Marshal(c.Text)
		if err != nil {
			return nil, err
		}
		out.Content = b
	case *pb.ResponseMessageItem_Parts:
		parts := make([]wireResponseContentPart, 0, len(c.Parts.GetItems()))
		for _, it := range c.Parts.GetItems() {
			parts = append(parts, responseContentPartFromPB(it))
		}
		b, err := json.Marshal(parts)
		if err != nil {
			return nil, err
		}
		out.Content = b
	}
	return json.Marshal(out)
}

func unmarshalResponseMessageItem(data []byte) (*pb.ResponseMessageItem, error) {
	var m wireResponseMessageItem
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	out := &pb.ResponseMessageItem{Id: m.ID, Role: m.Role, Status: m.Status}
	if len(m.Content) > 0 && string(m.Content) != "null" {
		switch m.Content[0] {
		case '"':
			var s string
			if err := json.Unmarshal(m.Content, &s); err != nil {
				return nil, err
			}
			out.Content = &pb.ResponseMessageItem_Text{Text: s}
		case '[':
			var parts []wireResponseContentPart
			if err := json.Unmarshal(m.Content, &parts); err != nil {
				return nil, err
			}
			items := make([]*pb.ResponseContentPart, 0, len(parts))
			for i := range parts {
				items = append(items, responseContentPartToPB(&parts[i]))
			}
			out.Content = &pb.ResponseMessageItem_Parts{Parts: &pb.ResponseContentParts{Items: items}}
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Responses dialect — tools / tool_choice / reasoning (FLAT in this dialect)
// ---------------------------------------------------------------------------

type wireResponsesToolDef struct {
	Type        string          `json:"type"` // "function"
	Name        string          `json:"name"`
	Description *string         `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

func responsesToolDefToWire(t *pb.ResponsesToolDef) (wireResponsesToolDef, error) {
	params, err := rawValue(t.Parameters)
	if err != nil {
		return wireResponsesToolDef{}, err
	}
	return wireResponsesToolDef{
		Type:        t.GetType(),
		Name:        t.GetName(),
		Description: t.Description,
		Parameters:  params,
		Strict:      t.Strict,
	}, nil
}

// wireResponsesToolChoice serialises the oneof: a bare mode string or the FLAT
// named object {"type":"function","name":"..."} (no nested `function`).
type wireResponsesToolChoice struct {
	mode  string
	named *pb.ResponsesNamedToolChoice
}

func (t wireResponsesToolChoice) MarshalJSON() ([]byte, error) {
	if t.named != nil {
		return json.Marshal(struct {
			Type string `json:"type"`
			Name string `json:"name"`
		}{Type: t.named.GetType(), Name: t.named.GetName()})
	}
	return json.Marshal(t.mode)
}

func responsesToolChoiceToWire(tc *pb.ResponsesToolChoice) *wireResponsesToolChoice {
	if tc == nil {
		return nil
	}
	switch c := tc.GetChoice().(type) {
	case *pb.ResponsesToolChoice_Mode:
		return &wireResponsesToolChoice{mode: c.Mode}
	case *pb.ResponsesToolChoice_Named:
		return &wireResponsesToolChoice{named: c.Named}
	default:
		return nil
	}
}

type wireResponsesReasoning struct {
	Effort  *string `json:"effort,omitempty"`
	Summary *string `json:"summary,omitempty"`
}

func responsesReasoningToWire(r *pb.ResponsesReasoning) *wireResponsesReasoning {
	if r == nil {
		return nil
	}
	return &wireResponsesReasoning{Effort: r.Effort, Summary: r.Summary}
}

func responsesReasoningFromWire(r *wireResponsesReasoning) *pb.ResponsesReasoning {
	if r == nil {
		return nil
	}
	return &pb.ResponsesReasoning{Effort: r.Effort, Summary: r.Summary}
}

// ---------------------------------------------------------------------------
// Responses dialect — request (encode)
// ---------------------------------------------------------------------------

// wireResponsesRequest carries every ResponsesRequest field except `extra`,
// which is merged at the top level by encodeResponsesRequest.
type wireResponsesRequest struct {
	Model           string                   `json:"model"`
	Input           json.RawMessage          `json:"input"`
	Instructions    *string                  `json:"instructions,omitempty"`
	Stream          *bool                    `json:"stream,omitempty"`
	Temperature     *float32                 `json:"temperature,omitempty"`
	TopP            *float32                 `json:"top_p,omitempty"`
	MaxOutputTokens *uint32                  `json:"max_output_tokens,omitempty"`
	Tools           []wireResponsesToolDef   `json:"tools,omitempty"`
	ToolChoice      *wireResponsesToolChoice `json:"tool_choice,omitempty"`
	Reasoning       *wireResponsesReasoning  `json:"reasoning,omitempty"`
	Store           *bool                    `json:"store,omitempty"`
}

// encodeResponsesInput serialises the `input` oneof: a bare string for a single
// user message, or an array of items.
func encodeResponsesInput(req *pb.ResponsesRequest) (json.RawMessage, error) {
	switch in := req.GetInput().(type) {
	case *pb.ResponsesRequest_Text:
		return json.Marshal(in.Text)
	case *pb.ResponsesRequest_Items:
		items := make([]wireResponseItem, 0, len(in.Items.GetItems()))
		for _, it := range in.Items.GetItems() {
			items = append(items, responseItemFromPB(it))
		}
		return json.Marshal(items)
	default:
		return json.Marshal([]wireResponseItem{})
	}
}

func responsesRequestToWire(req *pb.ResponsesRequest, streamOverride *bool) (wireResponsesRequest, json.RawMessage, error) {
	stream := req.Stream
	if streamOverride != nil {
		stream = streamOverride
	}
	w := wireResponsesRequest{
		Model:           req.GetModel(),
		Instructions:    req.Instructions,
		Stream:          stream,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		MaxOutputTokens: req.MaxOutputTokens,
		ToolChoice:      responsesToolChoiceToWire(req.ToolChoice),
		Reasoning:       responsesReasoningToWire(req.GetReasoning()),
		Store:           req.Store,
	}
	in, err := encodeResponsesInput(req)
	if err != nil {
		return w, nil, err
	}
	w.Input = in
	for _, t := range req.GetTools() {
		wt, err := responsesToolDefToWire(t)
		if err != nil {
			return w, nil, err
		}
		w.Tools = append(w.Tools, wt)
	}
	extra, err := rawValue(req.Extra)
	if err != nil {
		return w, nil, err
	}
	return w, extra, nil
}

// encodeResponsesRequest produces the final request body, merging `extra` keys
// at the top level (per SPEC.md), exactly like encodeChatRequest.
func encodeResponsesRequest(req *pb.ResponsesRequest, streamOverride *bool) ([]byte, error) {
	w, extra, err := responsesRequestToWire(req, streamOverride)
	if err != nil {
		return nil, err
	}
	return mergeExtra(w, extra)
}

// ---------------------------------------------------------------------------
// Responses dialect — response / usage (decode)
// ---------------------------------------------------------------------------

type wireResponsesInputTokensDetails struct {
	CachedTokens *uint32 `json:"cached_tokens,omitempty"`
}

type wireResponsesOutputTokensDetails struct {
	ReasoningTokens *uint32 `json:"reasoning_tokens,omitempty"`
}

type wireResponsesUsage struct {
	InputTokens         uint32                            `json:"input_tokens"`
	InputTokensDetails  *wireResponsesInputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokens        uint32                            `json:"output_tokens"`
	OutputTokensDetails *wireResponsesOutputTokensDetails `json:"output_tokens_details,omitempty"`
	TotalTokens         uint32                            `json:"total_tokens"`
}

func responsesUsageFromWire(u *wireResponsesUsage) *pb.ResponsesUsage {
	if u == nil {
		return nil
	}
	out := &pb.ResponsesUsage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
	if d := u.InputTokensDetails; d != nil {
		out.InputTokensDetails = &pb.ResponsesInputTokensDetails{CachedTokens: d.CachedTokens}
	}
	if d := u.OutputTokensDetails; d != nil {
		out.OutputTokensDetails = &pb.ResponsesOutputTokensDetails{ReasoningTokens: d.ReasoningTokens}
	}
	return out
}

type wireResponsesIncompleteDetails struct {
	Reason string `json:"reason"`
}

type wireResponsesResponse struct {
	ID                string                          `json:"id"`
	Object            string                          `json:"object"`
	CreatedAt         int64                           `json:"created_at"`
	Status            string                          `json:"status"`
	IncompleteDetails *wireResponsesIncompleteDetails `json:"incomplete_details"`
	Error             *wireErrorBody                  `json:"error"`
	Model             string                          `json:"model"`
	Output            []wireResponseItem              `json:"output"`
	Usage             *wireResponsesUsage             `json:"usage"`
	Store             *bool                           `json:"store"`
	Instructions      *string                         `json:"instructions"`
	MaxOutputTokens   *uint32                         `json:"max_output_tokens"`
	Temperature       *float32                        `json:"temperature"`
	TopP              *float32                        `json:"top_p"`
	Reasoning         *wireResponsesReasoning         `json:"reasoning"`
}

func (w *wireResponsesResponse) toPB() (*pb.ResponsesResponse, error) {
	out := &pb.ResponsesResponse{
		Id:              w.ID,
		Object:          w.Object,
		CreatedAt:       w.CreatedAt,
		Status:          w.Status,
		Error:           w.Error.toPB(),
		Model:           w.Model,
		Usage:           responsesUsageFromWire(w.Usage),
		Store:           w.Store,
		Instructions:    w.Instructions,
		MaxOutputTokens: w.MaxOutputTokens,
		Temperature:     w.Temperature,
		TopP:            w.TopP,
		Reasoning:       responsesReasoningFromWire(w.Reasoning),
	}
	if d := w.IncompleteDetails; d != nil {
		out.IncompleteDetails = &pb.ResponsesIncompleteDetails{Reason: d.Reason}
	}
	for i := range w.Output {
		out.Output = append(out.Output, w.Output[i].toPB())
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Responses dialect — streaming event (decode)
// ---------------------------------------------------------------------------

type wireResponsesStreamEvent struct {
	Type           string                   `json:"type"`
	SequenceNumber uint64                   `json:"sequence_number"`
	Response       *wireResponsesResponse   `json:"response"`
	OutputIndex    *uint32                  `json:"output_index"`
	ItemID         *string                  `json:"item_id"`
	ContentIndex   *uint32                  `json:"content_index"`
	Item           *wireResponseItem        `json:"item"`
	Part           *wireResponseContentPart `json:"part"`
	Delta          *string                  `json:"delta"`
	Text           *string                  `json:"text"`
	Arguments      *string                  `json:"arguments"`
	Message        *string                  `json:"message"`
}

func (w *wireResponsesStreamEvent) toPB() (*pb.ResponsesStreamEvent, error) {
	out := &pb.ResponsesStreamEvent{
		Type:           w.Type,
		SequenceNumber: w.SequenceNumber,
		OutputIndex:    w.OutputIndex,
		ItemId:         w.ItemID,
		ContentIndex:   w.ContentIndex,
		Delta:          w.Delta,
		Text:           w.Text,
		Arguments:      w.Arguments,
		Message:        w.Message,
	}
	if w.Response != nil {
		r, err := w.Response.toPB()
		if err != nil {
			return nil, err
		}
		out.Response = r
	}
	if w.Item != nil {
		out.Item = w.Item.toPB()
	}
	if w.Part != nil {
		out.Part = responseContentPartToPB(w.Part)
	}
	return out, nil
}
