// Agent event contract — TypeScript mirror of the TEAM backend's SSE stream.
//
// The backend (`backend/app/services/chat_service.py` -> `ml/agent/events.py`)
// serialises each Pydantic event to an SSE frame `event: <type>\ndata: <json>`.
// `useChat` consumes the stream with fetch + a reader and reduces each event
// into the Pinia chat store. Keep this file in sync with `ml/agent/events.py`
// and the `_sse_event` serialisation in `chat_service.py` (sources of truth).
//
// Notes on the real wire shapes (verified in the backend code):
//  - `perceiver_observation`: the SERVICE emits `{observation, prompt_block}`
//    (chat_service.py), while `ml/agent/events.py` documents `{text}`. We read
//    `text ?? prompt_block` so both shapes render.
//  - `tool_call`: Gemini does NOT supply `call_id` (it is `null`); we synthesise
//    a stable id in the store to correlate the following `tool_result`.
//  - `tool_result`: carries `{name, result, ok}` and NO `call_id`; `result` is a
//    generic dict (ParcelList / AoiStats / TimeSeries / ClassificationResult /
//    `{error}`). It NEVER carries parcel geometry.
//  - `text_delta`: incremental answer chunks.
//  - `done` / `error`: terminal events.

/** Provenance of a figure or claim surfaced in a finding card. */
export interface Citation {
  /** Tool name that produced the data (the backend has no call_id). */
  tool_call_id: string;
  /** Human-readable origin, e.g. "list_parcels" or "XGBoost+AlphaEarth". */
  source: string;
  scene_id?: string | null;
  parcel_id?: number | null;
  aoi_id?: number | null;
  /** ISO dates backing the claim. */
  dates?: string[] | null;
}

/** A single structured finding rendered as a card.
 *
 * Geometry is OPTIONAL and currently absent: the team's `tool_result` payloads
 * do not carry parcel boundaries, so the map paints the drawn AOI instead of
 * real parcels (see APORTE PENDIENTE in useChat.ts). The field is kept so the
 * map/demo preview keep working and future geometry plugs in without churn.
 */
export interface Finding {
  parcel_id: number;
  crop_class?: string | null;
  /** Confidence in [0, 1]. */
  confidence?: number | null;
  area_ha?: number | null;
  ndvi_mean?: number | null;
  metrics: Record<string, number>;
  /** Parcel boundary as a GeoJSON Polygon, when available (for the map). */
  geometry?: { type: "Polygon"; coordinates: number[][][] } | null;
  citation: Citation;
}

// ---------------------------------------------------------------------------
// Wire events — discriminated by `type` (mirror of ml/agent/events.py).
// ---------------------------------------------------------------------------

/** The perceiver's initial TEXT observation, injected before the reasoner. */
export interface PerceiverObservationEvent {
  type: "perceiver_observation";
  /** Documented shape in events.py. */
  text?: string;
  /** Real shape emitted by chat_service.py (rendered grounding block). */
  prompt_block?: string;
  /** Real shape emitted by chat_service.py (structured fields). */
  observation?: Record<string, unknown>;
}

/** The reasoner decided to call a tool (one event per requested call). */
export interface ToolCallEvent {
  type: "tool_call";
  name: string;
  arguments: Record<string, unknown>;
  /** Provider-supplied id; `null` for Gemini. */
  call_id?: string | null;
}

/** A tool finished and returned a (generic) result mapping. */
export interface ToolResultEvent {
  type: "tool_result";
  name: string;
  /** Output model dumped to a JSON mapping, or `{error: ...}` on failure. */
  result: Record<string, unknown>;
  ok: boolean;
}

/** An incremental chunk of the reasoner's final answer. */
export interface TextDeltaEvent {
  type: "text_delta";
  text: string;
}

/** Terminal event: the reasoner produced its final answer. */
export interface DoneEvent {
  type: "done";
}

/** Terminal event: the stream aborted with an error. */
export interface AgentErrorEvent {
  type: "error";
  message: string;
  /** Optional detail emitted by the perceiver-failure branch of the service. */
  detail?: string;
}

/** Discriminated union of every event the team's `/chat` stream can emit. */
export type AgentEvent =
  | PerceiverObservationEvent
  | ToolCallEvent
  | ToolResultEvent
  | TextDeltaEvent
  | DoneEvent
  | AgentErrorEvent;

/** LLM backend shown in the A/B switch.
 *
 * APORTE PENDIENTE: the team's `/chat` does NOT accept `llm_variant` per request
 * (the backend reads `settings.llm_variant_default`). The switch stays visible
 * but disabled; this type only drives the UI label/state.
 */
export type LlmVariant = "gemini" | "qwen35";

// ---------------------------------------------------------------------------
// Backend request/response payloads (team contract).
// ---------------------------------------------------------------------------

/** A GeoJSON geometry as accepted by the agent (mirror of GeoJSONGeometry). */
export interface GeoJSONGeometry {
  type: "Polygon" | "MultiPolygon";
  coordinates: number[][][] | number[][][][];
}

/** A single chat turn sent to the backend (mirror of ChatMessage). */
export interface ChatTurn {
  role: "user" | "assistant" | "system";
  content: string;
}

/** Body of POST /chat (mirror of ChatRequest). */
export interface ChatRequest {
  messages: ChatTurn[];
  session_id?: string;
  parcel_id?: number | null;
  aoi?: GeoJSONGeometry | null;
  year?: number;
}
