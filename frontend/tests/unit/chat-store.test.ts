import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useChatStore } from "~/stores/chat";
import type { AgentEvent } from "~/types/agent";

// Exercises the AgentEvent reducer end to end: a full happy-path turn plus the
// out-of-order / error edge cases the WebSocket transport can deliver.

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("chatStore reducer", () => {
  it("reduces a full happy-path turn into renderable state", () => {
    const store = useChatStore();
    store.startUserTurn("How are my parcels doing?");

    const sequence: AgentEvent[] = [
      {
        type: "plan_created",
        steps: ["List parcels", "Classify crop", "Compute NDVI", "Synthesize"],
      },
      {
        type: "tool_call",
        call_id: "c1",
        tool: "classify_parcel",
        args: { aoi_id: 1 },
        agent: "vision",
      },
      {
        type: "tool_result",
        call_id: "c1",
        tool: "classify_parcel",
        ok: true,
        summary: "3 parcels classified",
        duration_ms: 142,
        data: {},
        findings: [
          {
            parcel_id: 10,
            crop_class: "Meadow",
            confidence: 0.91,
            area_ha: 4.2,
            metrics: {},
            citation: {
              tool_call_id: "c1",
              source: "XGBoost+AlphaEarth",
              parcel_id: 10,
            },
          },
        ],
      },
      { type: "token", text: "The AOI has 3 parcels: " },
      { type: "token", text: "meadow predominates." },
      {
        type: "final_answer",
        text: "The AOI has 3 parcels: meadow predominates (4.2 ha, conf. 0.91).",
        citations: [
          { tool_call_id: "c1", source: "XGBoost+AlphaEarth", parcel_id: 10 },
        ],
      },
      { type: "done", job_id: "job_abc" },
    ];

    for (const event of sequence) store.applyEvent(event);

    // Plan captured.
    expect(store.currentPlan).toEqual([
      "List parcels",
      "Classify crop",
      "Compute NDVI",
      "Synthesize",
    ]);

    // Tool call resolved with summary + duration.
    expect(store.toolCalls).toHaveLength(1);
    expect(store.toolCalls[0]).toMatchObject({
      call_id: "c1",
      tool: "classify_parcel",
      status: "ok",
      summary: "3 parcels classified",
      duration_ms: 142,
    });

    // Findings stored for the map.
    expect(store.findings).toHaveLength(1);
    expect(store.findings[0]?.crop_class).toBe("Meadow");

    // Transcript: one user + one assistant turn; final_answer overrides tokens.
    expect(store.messages).toHaveLength(2);
    const assistant = store.lastAssistant;
    expect(assistant?.role).toBe("assistant");
    expect(assistant?.text).toBe(
      "The AOI has 3 parcels: meadow predominates (4.2 ha, conf. 0.91).",
    );
    expect(assistant?.citations).toHaveLength(1);
    expect(assistant?.citations?.[0]?.parcel_id).toBe(10);

    // Terminal state.
    expect(store.status).toBe("idle");
    expect(store.activeAssistantId).toBeNull();
  });

  it("marks a still-running tool as failed when `done` arrives early", () => {
    const store = useChatStore();
    store.startUserTurn("trigger");
    store.applyEvent({
      type: "tool_call",
      call_id: "cX",
      tool: "compute_ndvi",
      args: {},
      agent: "vision",
    });
    store.applyEvent({ type: "done", job_id: "job_1" });

    expect(store.toolCalls[0]?.status).toBe("failed");
    expect(store.status).toBe("idle");
  });

  it("captures an error event and keeps the error status through done", () => {
    const store = useChatStore();
    store.startUserTurn("trigger");
    store.applyEvent({
      type: "error",
      code: "vision_unavailable",
      message: "vision agent timed out",
    });
    store.applyEvent({ type: "done", job_id: "job_2" });

    expect(store.status).toBe("error");
    expect(store.errorMessage).toBe("vision agent timed out");
  });

  it("tolerates a tool_result with no preceding tool_call (backlog/out-of-order)", () => {
    const store = useChatStore();
    store.startUserTurn("trigger");
    store.applyEvent({
      type: "tool_result",
      call_id: "orphan",
      tool: "segment_scene",
      ok: false,
      summary: "no scene",
      duration_ms: 5,
      data: {},
      findings: [],
    });

    expect(store.toolCalls).toHaveLength(1);
    expect(store.toolCalls[0]?.status).toBe("failed");
  });
});
