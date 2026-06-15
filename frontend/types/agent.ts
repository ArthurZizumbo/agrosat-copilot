// Agent event contract — TypeScript mirror of ml/agent/events.py.
//
// The backend serialises each Pydantic AgentEvent to JSON and streams it over
// WebSocket. These types keep the frontend reducer aligned with the wire
// format. Keep this file in sync with ml/agent/events.py (source of truth).

/** Provenance of a figure or claim surfaced in the final answer. */
export interface Citation {
  /** Id of the tool_call that produced the data. */
  tool_call_id: string;
  /** Human-readable origin, e.g. "XGBoost+AlphaEarth". */
  source: string;
  scene_id?: string | null;
  parcel_id?: number | null;
  aoi_id?: number | null;
  /** ISO dates backing the claim. */
  dates?: string[] | null;
}

/** A single structured observation produced by the vision agent. */
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

/** The orchestrator published its plan before acting. */
export interface PlanCreatedEvent {
  type: "plan_created";
  steps: string[];
}

/** The orchestrator (or the vision agent) is invoking a tool. */
export interface ToolCallEvent {
  type: "tool_call";
  call_id: string;
  tool: string;
  args: Record<string, unknown>;
  agent: "orchestrator" | "vision";
}

/** A tool finished; `summary` is safe to show, `data` is structured. */
export interface ToolResultEvent {
  type: "tool_result";
  call_id: string;
  tool: string;
  ok: boolean;
  summary: string;
  duration_ms: number;
  data: Record<string, unknown>;
  findings: Finding[];
}

/** A partial chunk of the final answer (optional token streaming). */
export interface TokenEvent {
  type: "token";
  text: string;
}

/** The orchestrator's final natural-language answer with citations. */
export interface FinalAnswerEvent {
  type: "final_answer";
  text: string;
  citations: Citation[];
}

/** A recoverable error surfaced to the client. */
export interface AgentErrorEvent {
  type: "error";
  code: string;
  message: string;
}

/** Terminal event: the job finished (success or after an error). */
export interface DoneEvent {
  type: "done";
  job_id: string;
}

/** Discriminated union of every event the orchestrator can emit. */
export type AgentEvent =
  | PlanCreatedEvent
  | ToolCallEvent
  | ToolResultEvent
  | TokenEvent
  | FinalAnswerEvent
  | AgentErrorEvent
  | DoneEvent;

/** LLM backend selectable from the UI (A/B switch). */
export type LlmVariant = "gemini" | "qwen35";

// ---------------------------------------------------------------------------
// Backend request/response payloads.
// ---------------------------------------------------------------------------

/** Response of POST /sessions. */
export interface CreateSessionResponse {
  session_id: string;
  user_id?: string;
}

/** Response of POST /chat — the job was dispatched. */
export interface ChatDispatchResponse {
  job_id: string;
  ws_url: string;
}

/** Response of POST /llm/switch. */
export interface LlmSwitchResponse {
  session_id: string;
  llm_variant: LlmVariant;
}
