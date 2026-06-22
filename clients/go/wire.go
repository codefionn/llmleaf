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
