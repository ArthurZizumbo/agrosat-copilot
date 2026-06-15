// UI-facing chat state types. These wrap the wire-level AgentEvent shapes from
// types/agent.ts into a shape convenient for rendering in ChatPanel/MapView.

import type { Citation, Finding, LlmVariant } from "./agent";

/** A turn shown in the conversation transcript. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  /** Rendered text. For the assistant turn it grows while tokens stream. */
  text: string;
  /** Citations attached to a final assistant answer. */
  citations?: Citation[];
  /** Epoch ms when the turn was created (client clock). */
  createdAt: number;
}

/** A tool invocation tracked in the live activity panel. */
export interface TrackedToolCall {
  call_id: string;
  tool: string;
  agent: "orchestrator" | "vision";
  args: Record<string, unknown>;
  status: "running" | "ok" | "failed";
  summary?: string;
  duration_ms?: number;
}

/** Lifecycle of the current job/turn. */
export type ChatStatus = "idle" | "dispatching" | "streaming" | "error";

export type { Citation, Finding, LlmVariant };
