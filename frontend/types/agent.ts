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
  /** Canonical PASTIS parcel id `"{patch}_{local}"` when the source carries one.
   *  `parcel_id` above is numeric (it feeds `activeParcelId`), so the local part
   *  alone is ambiguous across patches; this keeps the real, unique id. */
  canonical_parcel_id?: string | null;
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
  /** Ground-truth crop, when known (prediction demo only). Lets the map toggle
   *  between predicted (`crop_class`) and true crop, and compute hits/errors. */
  true_class?: string | null;
  /** Whether the prediction matched the ground truth (prediction demo only). */
  correct?: boolean | null;
  /** Per-class posterior probabilities in [0, 1], keyed by crop label
   *  (mirror of `ClassificationResult.class_probabilities`). Drives the
   *  probability bar chart in the finding card. */
  class_probabilities?: Record<string, number> | null;
  /** Identifier of the model that produced the classification, e.g.
   *  "Voting-3" (shown as a small chip in the finding card). */
  served_model?: string | null;
}

// ---------------------------------------------------------------------------
// Wire events — discriminated by `type` (mirror of ml/agent/events.py).
// ---------------------------------------------------------------------------

/** A live per-cell crop segment painted on the map (prediction overlay). */
export interface MapSegment {
  crop_class?: string | null;
  confidence?: number | null;
  area_ha?: number | null;
  geometry?: { type: "Polygon"; coordinates: number[][][] } | null;
}

/** The perceiver's initial TEXT observation, injected before the reasoner. */
export interface PerceiverObservationEvent {
  type: "perceiver_observation";
  /** Documented shape in events.py. */
  text?: string;
  /** Real shape emitted by chat_service.py (rendered grounding block). */
  prompt_block?: string;
  /** Real shape emitted by chat_service.py (structured fields). Carries
   *  `map_segments` for an AOI: the live per-cell crop polygons to paint. */
  observation?: Record<string, unknown> & { map_segments?: MapSegment[] };
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

/** LLM backend the per-session reasoner switch can activate.
 *
 * These three strings are EXACTLY the persisted variant tags the backend accepts
 * (`chat_sessions.llm_model` CHECK / `ml.agent.llm_routing.VARIANTS` / the
 * `POST /llm/switch` `Literal`): `gemini` (cloud), `qwen-onprem` (on-prem Qwen
 * text vLLM) and `qwen-vl` (on-prem multimodal Qwen3.6-VL). The switch POSTs the
 * chosen tag to `/llm/switch`, which persists it; the next `/chat` reads it back
 * and builds the matching backend. The two on-prem variants are reachable only
 * behind the demo VM tunnel (`make demo-vm`); when their host is down the backend
 * degrades the request to `gemini` (availability-aware routing) rather than
 * failing, and the switch surfaces a transient error toast if the POST itself
 * cannot be persisted. The hosted `qwen-api` / `gemma` variants exist server-side
 * but are intentionally NOT exposed in the UI (3-option product decision, E12).
 *
 * Declared as a runtime array so the persisted store can VALIDATE a rehydrated
 * value against it (the tags changed in E12: the pre-E12 `qwen35` still lives in
 * returning users' localStorage). The type is derived from the array, so the two
 * can never drift.
 */
export const LLM_VARIANTS = ["gemini", "qwen-onprem", "qwen-vl"] as const;

export type LlmVariant = (typeof LLM_VARIANTS)[number];

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

/** Crop-classification model the user can pin for `classify_new_parcel`
 *  (mirror of `ml/agent/schemas.py` `CropModel`). When set, the backend injects a
 *  system turn telling the reasoner to forward it to the tool. Runtime array +
 *  derived type, for the same rehydration guard as `LLM_VARIANTS`. */
export const CROP_MODELS = ["voting3", "xgb", "stacking5"] as const;

export type CropModel = (typeof CROP_MODELS)[number];

/** Body of POST /chat (mirror of ChatRequest). */
export interface ChatRequest {
  messages: ChatTurn[];
  session_id?: string;
  parcel_id?: number | null;
  aoi?: GeoJSONGeometry | null;
  year?: number;
  /** Crop-classification model pinned by the user, if any (else the LLM/tool
   *  default of `voting3` applies). Omitted from the body when not selected. */
  crop_model?: CropModel;
}
