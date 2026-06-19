import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useChatStore } from "~/stores/chat";
import type { AgentEvent } from "~/types/agent";

// Exercises the AgentEvent reducer against the TEAM backend's SSE contract:
// perceiver_observation -> tool_call -> tool_result -> text_delta -> done, plus
// the error / out-of-order / no-call_id edge cases the stream can deliver.

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("chatStore reducer (team SSE contract)", () => {
  it("reduces a full happy-path turn into renderable state", () => {
    const store = useChatStore();
    store.startUserTurn("How are my parcels doing?");

    const sequence: AgentEvent[] = [
      {
        type: "perceiver_observation",
        prompt_block: "AOI over Tuscany: dominant vineyard, high vigour.",
        observation: { dominant_crop: "Vineyard" },
      },
      {
        type: "tool_call",
        name: "list_parcels",
        arguments: { aoi: { type: "Polygon", coordinates: [] } },
        call_id: null,
      },
      {
        type: "tool_result",
        name: "list_parcels",
        ok: true,
        result: {
          parcels: [
            { parcel_id: 10, crop_class: "Meadow", confidence: 0.91 },
            { parcel_id: 11, crop_class: "Vineyard", confidence: 0.84 },
          ],
          count: 2,
        },
      },
      { type: "text_delta", text: "The AOI has 2 parcels: " },
      { type: "text_delta", text: "meadow and vineyard predominate." },
      { type: "done" },
    ];

    for (const event of sequence) store.applyEvent(event);

    // Perceiver observation captured ("what the agent saw").
    expect(store.perceiverNotes).toHaveLength(1);
    expect(store.perceiverNotes[0]?.text).toContain("Tuscany");

    // Tool call resolved (correlated by name, no call_id on the wire).
    expect(store.toolCalls).toHaveLength(1);
    expect(store.toolCalls[0]).toMatchObject({
      tool: "list_parcels",
      status: "ok",
    });

    // ParcelList parsed into one finding per parcel (no geometry).
    expect(store.findings).toHaveLength(2);
    expect(store.findings[0]?.crop_class).toBe("Meadow");
    expect(store.findings[0]?.geometry ?? null).toBeNull();

    // Transcript: one user + one assistant turn; deltas accumulated.
    expect(store.messages).toHaveLength(2);
    const assistant = store.lastAssistant;
    expect(assistant?.role).toBe("assistant");
    expect(assistant?.text).toBe(
      "The AOI has 2 parcels: meadow and vineyard predominate.",
    );

    // Terminal state.
    expect(store.status).toBe("idle");
    expect(store.activeAssistantId).toBeNull();
  });

  it("summarises an AoiStats tool_result without findings", () => {
    const store = useChatStore();
    store.startUserTurn("stats");
    store.applyEvent({ type: "tool_call", name: "get_aoi_stats", arguments: {} });
    store.applyEvent({
      type: "tool_result",
      name: "get_aoi_stats",
      ok: true,
      result: {
        area_ha: 42.0,
        dominant_crop: "Olive grove",
        crop_fractions: { "Olive grove": 0.6, Vineyard: 0.4 },
        n_parcels: 7,
      },
    });

    expect(store.findings).toHaveLength(0);
    expect(store.toolCalls[0]?.status).toBe("ok");
    expect(store.toolCalls[0]?.summary).toContain("Olive grove");
  });

  it("marks a still-running tool as failed when `done` arrives early", () => {
    const store = useChatStore();
    store.startUserTurn("trigger");
    store.applyEvent({ type: "tool_call", name: "get_parcel_timeseries", arguments: {} });
    store.applyEvent({ type: "done" });

    expect(store.toolCalls[0]?.status).toBe("failed");
    expect(store.status).toBe("idle");
  });

  it("captures an error event and keeps the error status through done", () => {
    const store = useChatStore();
    store.startUserTurn("trigger");
    store.applyEvent({ type: "error", message: "perceiver_observation_failed" });
    store.applyEvent({ type: "done" });

    expect(store.status).toBe("error");
    expect(store.errorMessage).toBe("perceiver_observation_failed");
  });

  it("tolerates a tool_result with no preceding tool_call (out-of-order)", () => {
    const store = useChatStore();
    store.startUserTurn("trigger");
    store.applyEvent({
      type: "tool_result",
      name: "classify_new_parcel",
      ok: false,
      result: { error: "no embedding" },
    });

    expect(store.toolCalls).toHaveLength(1);
    expect(store.toolCalls[0]?.status).toBe("failed");
    expect(store.toolCalls[0]?.summary).toBe("no embedding");
  });

  it("builds backend history from the message transcript", () => {
    const store = useChatStore();
    store.startUserTurn("hello");
    store.applyEvent({ type: "text_delta", text: "hi there" });
    store.applyEvent({ type: "done" });

    expect(store.historyForRequest).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there" },
    ]);
  });
});
