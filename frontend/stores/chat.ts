// Pinia store: reduces the AgentEvent stream into renderable chat state.
//
// The reducer (`applyEvent`) is pure with respect to the store state: feeding a
// sequence of events produces a deterministic state, which is what the unit
// test asserts. The composable useChat owns the transport (POST + WebSocket)
// and calls these actions; MapView consumes `findings`.

import { defineStore } from "pinia";
import type { AgentEvent, Finding, LlmVariant } from "~/types/agent";
import type { ChatMessage, ChatStatus, TrackedToolCall } from "~/types/chat";

let messageSeq = 0;

function nextMessageId(prefix: string): string {
  messageSeq += 1;
  return `${prefix}-${messageSeq}`;
}

interface ChatState {
  messages: ChatMessage[];
  currentPlan: string[];
  toolCalls: TrackedToolCall[];
  findings: Finding[];
  llmVariant: LlmVariant;
  status: ChatStatus;
  /** Id of the assistant turn currently being streamed, if any. */
  activeAssistantId: string | null;
  errorMessage: string | null;
}

export const useChatStore = defineStore("chat", {
  state: (): ChatState => ({
    messages: [],
    currentPlan: [],
    toolCalls: [],
    findings: [],
    llmVariant: "gemini",
    status: "idle",
    activeAssistantId: null,
    errorMessage: null,
  }),

  getters: {
    runningToolCalls: (state): TrackedToolCall[] =>
      state.toolCalls.filter((c) => c.status === "running"),
    isBusy: (state): boolean =>
      state.status === "dispatching" || state.status === "streaming",
    /** Latest assistant message, if any (handy for tests/UI). */
    lastAssistant: (state): ChatMessage | undefined =>
      [...state.messages].reverse().find((m) => m.role === "assistant"),
  },

  actions: {
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
      this.currentPlan = [];
      this.toolCalls = [];
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
        case "plan_created": {
          this.currentPlan = [...event.steps];
          this.markStreaming();
          break;
        }
        case "tool_call": {
          this.markStreaming();
          this.toolCalls.push({
            call_id: event.call_id,
            tool: event.tool,
            agent: event.agent,
            args: event.args,
            status: "running",
          });
          break;
        }
        case "tool_result": {
          const tracked = this.toolCalls.find(
            (c) => c.call_id === event.call_id,
          );
          if (tracked) {
            tracked.status = event.ok ? "ok" : "failed";
            tracked.summary = event.summary;
            tracked.duration_ms = event.duration_ms;
          } else {
            // Result without a prior tool_call (out-of-order/backlog): track it.
            this.toolCalls.push({
              call_id: event.call_id,
              tool: event.tool,
              agent: "vision",
              args: {},
              status: event.ok ? "ok" : "failed",
              summary: event.summary,
              duration_ms: event.duration_ms,
            });
          }
          if (event.findings.length > 0) {
            this.findings.push(...event.findings);
          }
          break;
        }
        case "token": {
          const id = this.ensureAssistantTurn();
          const msg = this.messages.find((m) => m.id === id);
          if (msg) msg.text += event.text;
          break;
        }
        case "final_answer": {
          const id = this.ensureAssistantTurn();
          const msg = this.messages.find((m) => m.id === id);
          if (msg) {
            msg.text = event.text;
            msg.citations = [...event.citations];
          }
          break;
        }
        case "error": {
          this.status = "error";
          this.errorMessage = event.message;
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
    },

    reset() {
      this.messages = [];
      this.currentPlan = [];
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
});
