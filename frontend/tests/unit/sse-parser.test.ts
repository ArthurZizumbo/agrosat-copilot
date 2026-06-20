import { describe, expect, it } from "vitest";
import { parseSseFrame } from "~/composables/useChat";

// Exercises the REAL SSE frame parser (`parseSseFrame`, exported from useChat)
// against frames byte-for-byte as the TEAM backend emits them.
//
// Wire contract (backend/app/services/chat_service.py `_sse_event`):
//   f"event: {event}\ndata: {payload}\n\n"
// where `payload = json.dumps(data, separators=(",", ":"))` (compact, no spaces).
// REGLA ARTHUR: the frames below are the exact bytes that serialisation
// produces for each ml/agent/events.py event — not invented shapes.

describe("parseSseFrame (real backend SSE contract)", () => {
  it("parses a perceiver_observation frame ({observation, prompt_block})", () => {
    // Exactly what chat_service.py yields: observation.model_dump + prompt_block.
    const frame =
      "event: perceiver_observation\n" +
      'data: {"observation":{"dominant_crop":"Vineyard","vigor":"high"},' +
      '"prompt_block":"AOI over Tuscany: dominant vineyard, high vigour."}';
    const event = parseSseFrame(frame);
    expect(event).not.toBeNull();
    expect(event?.type).toBe("perceiver_observation");
    if (event?.type === "perceiver_observation") {
      expect(event.prompt_block).toContain("Tuscany");
      expect(event.observation).toEqual({
        dominant_crop: "Vineyard",
        vigor: "high",
      });
    }
  });

  it("parses a tool_call frame with call_id null (Gemini omits the id)", () => {
    const frame =
      "event: tool_call\n" +
      'data: {"name":"list_parcels","arguments":{"year":2019},"call_id":null}';
    const event = parseSseFrame(frame);
    expect(event?.type).toBe("tool_call");
    if (event?.type === "tool_call") {
      expect(event.name).toBe("list_parcels");
      expect(event.arguments).toEqual({ year: 2019 });
      expect(event.call_id).toBeNull();
    }
  });

  it("parses a tool_result frame ({name, result, ok})", () => {
    const frame =
      "event: tool_result\n" +
      'data: {"name":"list_parcels","result":{"parcels":[{"parcel_id":10,' +
      '"crop_class":"Meadow","confidence":0.91}],"count":1},"ok":true}';
    const event = parseSseFrame(frame);
    expect(event?.type).toBe("tool_result");
    if (event?.type === "tool_result") {
      expect(event.ok).toBe(true);
      expect(event.result.count).toBe(1);
    }
  });

  it("parses a text_delta frame preserving UTF-8 (no ensure_ascii)", () => {
    // chat_service.py dumps with ensure_ascii=False, so accents stay literal.
    const frame = "event: text_delta\n" + 'data: {"text":"Predomina viñedo y olivar."}';
    const event = parseSseFrame(frame);
    expect(event?.type).toBe("text_delta");
    if (event?.type === "text_delta") {
      expect(event.text).toBe("Predomina viñedo y olivar.");
    }
  });

  it("parses the terminal done and error frames", () => {
    expect(parseSseFrame("event: done\ndata: {}")?.type).toBe("done");
    const err = parseSseFrame(
      'event: error\ndata: {"message":"perceiver_observation_failed",' +
        '"detail":"boom"}',
    );
    expect(err?.type).toBe("error");
    if (err?.type === "error") expect(err.message).toBe("perceiver_observation_failed");
  });

  it("ignores SSE heartbeat comments and malformed JSON", () => {
    // A keep-alive comment line (starts with ':') yields no event.
    expect(parseSseFrame(": keep-alive")).toBeNull();
    // A frame with an event line but unparseable data returns null (not a throw).
    expect(parseSseFrame("event: text_delta\ndata: {not json")).toBeNull();
    // A bare data line with no event type is ignored.
    expect(parseSseFrame('data: {"text":"orphan"}')).toBeNull();
  });

  it("tolerates the `data: ` space the backend writes after the colon", () => {
    // _sse_event writes `data: {json}` (space after colon); the parser trims it.
    const event = parseSseFrame('event: text_delta\ndata: {"text":"hi"}');
    expect(event?.type).toBe("text_delta");
    if (event?.type === "text_delta") expect(event.text).toBe("hi");
  });
});
