// Pinia store: reduces the TEAM backend's SSE AgentEvent stream into renderable
// chat state.
//
// The reducer (`applyEvent`) is pure with respect to the store state: feeding a
// sequence of events produces a deterministic state, which is what the unit
// test asserts. The composable `useChat` owns the transport (POST + fetch SSE)
// and calls these actions; ChatDock/MapCanvas consume `findings`, `toolCalls`,
// `perceiverNotes` and `messages`.
//
// Event mapping (team contract — see types/agent.ts):
//   perceiver_observation -> a `PerceiverNote` ("what the agent saw")
//   tool_call             -> a running `TrackedToolCall` (id synthesised)
//   tool_result           -> resolves the matching call + parses `result` into
//                            FindingCards (ParcelList) or a summary
//                            (AoiStats / TimeSeries / ClassificationResult)
//   text_delta            -> appended to the active assistant message
//   done / error          -> terminal state

import { defineStore } from "pinia";
import type {
  AgentEvent,
  Finding,
  LlmVariant,
  ToolResultEvent,
} from "~/types/agent";
import type {
  ChatMessage,
  ChatStatus,
  PerceiverNote,
  TrackedToolCall,
} from "~/types/chat";

let messageSeq = 0;

function nextMessageId(prefix: string): string {
  messageSeq += 1;
  return `${prefix}-${messageSeq}`;
}

/** Build FindingCards + a one-line summary from a generic `tool_result.result`.
 *
 * The team's tool outputs are generic dicts (no geometry). We recognise the
 * known shapes from `ml/agent/schemas.py`:
 *   - ParcelList `{parcels:[{parcel_id,crop_class,confidence}], count}`
 *   - AoiStats   `{area_ha, dominant_crop, crop_fractions, n_parcels}`
 *   - TimeSeries `{parcel_id, index, dates, values}`
 *   - ClassificationResult `{crop_class, confidence, class_probabilities}`
 * Anything else yields no findings and a minimal summary.
 */
function parseToolResult(event: ToolResultEvent): {
  findings: Finding[];
  summary: string;
} {
  const r = event.result ?? {};
  const source = event.name;

  if (!event.ok) {
    const err = typeof r.error === "string" ? r.error : "error";
    return { findings: [], summary: err };
  }

  // ParcelList -> one card per parcel (no geometry available).
  if (Array.isArray(r.parcels)) {
    const parcels = r.parcels as Array<{
      parcel_id?: unknown;
      crop_class?: unknown;
      confidence?: unknown;
    }>;
    const findings: Finding[] = parcels
      .filter((p) => typeof p.parcel_id === "number")
      .map((p) => ({
        parcel_id: p.parcel_id as number,
        crop_class: typeof p.crop_class === "string" ? p.crop_class : null,
        confidence: typeof p.confidence === "number" ? p.confidence : null,
        area_ha: null,
        ndvi_mean: null,
        metrics: {},
        geometry: null,
        citation: {
          tool_call_id: source,
          source,
          parcel_id: p.parcel_id as number,
        },
      }));
    const count = typeof r.count === "number" ? r.count : findings.length;
    return { findings, summary: `${count}` };
  }

  // AoiStats -> summary only (dominant crop + parcel count).
  if (typeof r.dominant_crop === "string" && typeof r.n_parcels === "number") {
    return {
      findings: [],
      summary: `${r.dominant_crop} · ${r.n_parcels}`,
    };
  }

  // TimeSeries -> summary only (index + sample count).
  if (Array.isArray(r.dates) && Array.isArray(r.values)) {
    const index = typeof r.index === "string" ? r.index : "index";
    return { findings: [], summary: `${index} · ${r.values.length}` };
  }

  // ClassificationResult -> one card (no parcel id; use 0 as a sentinel).
  if (typeof r.crop_class === "string" && typeof r.confidence === "number") {
    return {
      findings: [
        {
          parcel_id: 0,
          crop_class: r.crop_class,
          confidence: r.confidence,
          area_ha: null,
          ndvi_mean: null,
          metrics: {},
          geometry: null,
          citation: { tool_call_id: source, source },
        },
      ],
      summary: r.crop_class,
    };
  }

  return { findings: [], summary: "ok" };
}

interface ChatState {
  messages: ChatMessage[];
  /** What the perceiver "saw" before reasoning (Be My Eyes). */
  perceiverNotes: PerceiverNote[];
  toolCalls: TrackedToolCall[];
  findings: Finding[];
  /** Display-only; the backend ignores per-request variant (server config). */
  llmVariant: LlmVariant;
  status: ChatStatus;
  /** Id of the assistant turn currently being streamed, if any. */
  activeAssistantId: string | null;
  errorMessage: string | null;
  /** Monotonic counter to synthesise tool_call ids (backend has none). */
  toolSeq: number;
}

export const useChatStore = defineStore("chat", {
  state: (): ChatState => ({
    messages: [],
    perceiverNotes: [],
    toolCalls: [],
    findings: [],
    llmVariant: "gemini",
    status: "idle",
    activeAssistantId: null,
    errorMessage: null,
    toolSeq: 0,
  }),

  getters: {
    runningToolCalls: (state): TrackedToolCall[] =>
      state.toolCalls.filter((c) => c.status === "running"),
    isBusy: (state): boolean =>
      state.status === "dispatching" || state.status === "streaming",
    /** Latest assistant message, if any (handy for tests/UI). */
    lastAssistant: (state): ChatMessage | undefined =>
      [...state.messages].reverse().find((m) => m.role === "assistant"),
    /** Conversation history in the backend's `{role, content}` shape. */
    historyForRequest: (
      state,
    ): Array<{ role: "user" | "assistant"; content: string }> =>
      state.messages
        .filter((m) => m.text.length > 0)
        .map((m) => ({ role: m.role, content: m.text })),
  },

  actions: {
    /** Display-only; transport does not send this (see useChat.switchLlm). */
    setLlmVariant(variant: LlmVariant) {
      this.llmVariant = variant;
    },

    /** Append the user's turn and prepare the per-turn state. */
    startUserTurn(text: string) {
      this.messages.push({
        id: nextMessageId("user"),
        role: "user",
        text,
        createdAt: Date.now(),
      });
      this.perceiverNotes = [];
      this.toolCalls = [];
      this.findings = [];
      this.activeAssistantId = null;
      this.errorMessage = null;
      this.status = "dispatching";
    },

    markStreaming() {
      this.status = "streaming";
    },

    /** Ensure there is an assistant turn to stream into; returns its id. */
    ensureAssistantTurn(): string {
      if (this.activeAssistantId) return this.activeAssistantId;
      const id = nextMessageId("assistant");
      this.messages.push({
        id,
        role: "assistant",
        text: "",
        citations: [],
        createdAt: Date.now(),
      });
      this.activeAssistantId = id;
      return id;
    },

    /** Reducer: fold a single AgentEvent into the store state. */
    applyEvent(event: AgentEvent) {
      switch (event.type) {
        case "perceiver_observation": {
          this.markStreaming();
          const text = event.text ?? event.prompt_block ?? "";
          if (text.trim().length > 0) {
            this.toolSeq += 1;
            this.perceiverNotes.push({ id: `note-${this.toolSeq}`, text });
          }
          break;
        }
        case "tool_call": {
          this.markStreaming();
          this.toolSeq += 1;
          this.toolCalls.push({
            call_id: `tc-${this.toolSeq}`,
            tool: event.name,
            args: event.arguments ?? {},
            status: "running",
          });
          break;
        }
        case "tool_result": {
          // No call_id on the wire: correlate to the most recent running call
          // of the same tool name; otherwise append a synthetic resolved call.
          const tracked = [...this.toolCalls]
            .reverse()
            .find((c) => c.tool === event.name && c.status === "running");
          const { findings, summary } = parseToolResult(event);
          if (tracked) {
            tracked.status = event.ok ? "ok" : "failed";
            tracked.summary = summary;
            tracked.result = event.result;
          } else {
            this.toolSeq += 1;
            this.toolCalls.push({
              call_id: `tc-${this.toolSeq}`,
              tool: event.name,
              args: {},
              status: event.ok ? "ok" : "failed",
              summary,
              result: event.result,
            });
          }
          if (findings.length > 0) this.findings.push(...findings);
          break;
        }
        case "text_delta": {
          this.markStreaming();
          const id = this.ensureAssistantTurn();
          const msg = this.messages.find((m) => m.id === id);
          if (msg) msg.text += event.text;
          break;
        }
        case "error": {
          this.status = "error";
          this.errorMessage = event.message;
          this.activeAssistantId = null;
          break;
        }
        case "done": {
          // Terminal event. Only settle if not already in an error state.
          if (this.status !== "error") this.status = "idle";
          this.activeAssistantId = null;
          // Any tool still flagged running never reported back; mark failed.
          for (const c of this.toolCalls) {
            if (c.status === "running") c.status = "failed";
          }
          break;
        }
        default: {
          // Exhaustiveness guard: unknown event types are ignored safely.
          break;
        }
      }
    },

    /** Reset transient turn state after a fatal transport failure. */
    failTransport(message: string) {
      this.status = "error";
      this.errorMessage = message;
      this.activeAssistantId = null;
      for (const c of this.toolCalls) {
        if (c.status === "running") c.status = "failed";
      }
    },

    reset() {
      this.messages = [];
      this.perceiverNotes = [];
      this.toolCalls = [];
      this.findings = [];
      this.status = "idle";
      this.activeAssistantId = null;
      this.errorMessage = null;
    },

    /**
     * Load a local sample assistant turn + findings (no transport). Used by the
     * empty-state "see example" button to showcase the FindingCard design. The
     * reducer and event contract are untouched; this only seeds renderable
     * state directly and is clearly flagged as a preview by the UI.
     */
    loadPreview(answer: string, findings: Finding[]) {
      this.reset();
      this.messages.push({
        id: nextMessageId("assistant"),
        role: "assistant",
        text: answer,
        citations: findings.map((f) => f.citation),
        createdAt: Date.now(),
      });
      this.findings = [...findings];
      this.status = "idle";
    },

    /**
     * Seed only the parcel findings (no assistant message), used by the
     * "demo area" tool so the map paints parcels for the seeded AOI. Keeps the
     * chat transcript empty so the user can still ask their own question.
     */
    loadDemoParcels(findings: Finding[]) {
      this.findings = [...findings];
    },
  },

  // Persist only durable conversation state. Transient per-turn state
  // (toolCalls, perceiverNotes, findings, status, activeAssistantId,
  // errorMessage, toolSeq) is intentionally excluded: rehydrating a "running"
  // tool or a half-streamed turn would be incorrect. `session_id` is NOT
  // duplicated here — it already persists via the `agrosat-session-id` cookie
  // in useSession. SSR-safe: `localStorage()` is a no-op on the server, so the
  // store hydrates from localStorage on the client only (transcript wrapped in
  // <ClientOnly> in ChatDock to avoid a hydration mismatch).
  persist: {
    storage: piniaPluginPersistedstate.localStorage(),
    pick: ["messages", "llmVariant"],
  },
});
