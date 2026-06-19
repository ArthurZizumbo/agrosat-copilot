// UI-facing chat state types. These wrap the wire-level AgentEvent shapes from
// types/agent.ts into a shape convenient for rendering in ChatDock/MapCanvas.

import type { Citation, Finding, LlmVariant } from "./agent";

/** A turn shown in the conversation transcript. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  /** Rendered text. For the assistant turn it grows while deltas stream. */
  text: string;
  /** Citations attached to a final assistant answer. */
  citations?: Citation[];
  /** Epoch ms when the turn was created (client clock). */
  createdAt: number;
}

/** A tool invocation tracked in the live activity panel.
 *
 * The team backend has no `call_id` (Gemini) and no per-call `agent` tag, so we
 * synthesise a local `call_id` and track only the tool `name`, its `status` and
 * an optional human-readable `summary` derived from the result payload.
 */
export interface TrackedToolCall {
  call_id: string;
  tool: string;
  args: Record<string, unknown>;
  status: "running" | "ok" | "failed";
  summary?: string;
}

/** What the perceiver "saw" before the reasoner spoke (Be My Eyes). */
export interface PerceiverNote {
  id: string;
  /** Rendered grounding text (from `text` or `prompt_block`). */
  text: string;
}

/** Lifecycle of the current turn. */
export type ChatStatus = "idle" | "dispatching" | "streaming" | "error";

export type { Citation, Finding, LlmVariant };
