import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";
import { createApp } from "vue";
import { createPinia, setActivePinia } from "pinia";
import { useChatStore } from "~/stores/chat";

// Exercises the REAL `useChat` transport: fetch + SSE reader + retry/backoff,
// reducing genuine contract frames into the real Pinia chat store.
//
// The SSE bytes below are exactly what the TEAM backend emits
// (`backend/app/services/chat_service.py` `_sse_event`):
//   f"event: {type}\ndata: {compact_json}\n\n"
// REGLA ARTHUR: real frames, not invented ones.
//
// Nuxt auto-imports used by useChat (`useSession`, `useI18n`) are stubbed as
// globals; `useChatStore`/`useMapStore` are real (imported by the composable).
// `import.meta.client` is forced true so the client-only guard takes the browser
// path. fetch is mocked to return SSE streams or to fail transiently; timers are
// faked so the exponential backoff resolves instantly.

import { useChat } from "~/composables/useChat";

// --- Real contract frames (one terminated SSE frame per array entry). -------
const FR_PERCEIVER =
  "event: perceiver_observation\n" +
  'data: {"observation":{"dominant_crop":"Vineyard"},' +
  '"prompt_block":"AOI over Tuscany: dominant vineyard, high vigour."}\n\n';
const FR_TOOL_CALL =
  "event: tool_call\n" +
  'data: {"name":"list_parcels","arguments":{"year":2019},"call_id":null}\n\n';
const FR_TOOL_RESULT =
  "event: tool_result\n" +
  'data: {"name":"list_parcels","result":{"parcels":[{"parcel_id":10,' +
  '"crop_class":"Meadow","confidence":0.91}],"count":1},"ok":true}\n\n';
const FR_DELTA_1 = 'event: text_delta\ndata: {"text":"The AOI has 1 parcel: "}\n\n';
const FR_DELTA_2 = 'event: text_delta\ndata: {"text":"a meadow."}\n\n';
const FR_DONE = "event: done\ndata: {}\n\n";

const HAPPY_PATH = [
  FR_PERCEIVER,
  FR_TOOL_CALL,
  FR_TOOL_RESULT,
  FR_DELTA_1,
  FR_DELTA_2,
  FR_DONE,
];

/** Build a streaming `Response` whose body yields the given SSE frames. */
function sseResponse(frames: string[], opts?: { chunkBytes?: boolean }): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) {
        const bytes = encoder.encode(frame);
        if (opts?.chunkBytes) {
          // Split each frame mid-way to prove the buffer reassembles frames
          // across read() boundaries (the backend does not align chunks).
          const mid = Math.floor(bytes.length / 2);
          controller.enqueue(bytes.slice(0, mid));
          controller.enqueue(bytes.slice(mid));
        } else {
          controller.enqueue(bytes);
        }
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

/** A `Response` whose body delivers `frames` then cuts (no terminal event).
 *
 * Pull-based so the consumer fully reads (and parses) each frame on its own
 * `read()` before the stream errors on the next pull — this guarantees the
 * delta reaches the store BEFORE the cut, which is what the idempotency guard
 * needs to observe (cut AFTER a delta -> not retried).
 */
function truncatedSseResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < frames.length) {
        controller.enqueue(encoder.encode(frames[i]));
        i += 1;
      } else {
        // All frames delivered (and read): now simulate a mid-stream drop.
        controller.error(new TypeError("network error"));
      }
    },
  });
  return new Response(stream, { status: 200 });
}

let app: ReturnType<typeof createApp>;

beforeEach(() => {
  // `import.meta.client` is replaced with `true` by the vitest `define` config,
  // so useChat's client-only guard takes the browser path here.

  // Real store on an app-installed pinia.
  app = createApp({ render: () => null });
  const pinia = createPinia();
  app.use(pinia);
  setActivePinia(pinia);

  // Stub the Nuxt auto-imports the composable reads as globals.
  vi.stubGlobal("useSession", () => ({
    ensureSession: () => "11111111-2222-4333-8444-555555555555",
    apiUrl: (path: string) => `http://localhost:8000${path}`,
  }));
  vi.stubGlobal("useI18n", () => ({ locale: { value: "es" } }));

  // Deterministic backoff + fake timers so retries do not actually sleep.
  vi.spyOn(Math, "random").mockReturnValue(0.5);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useChat transport (real SSE frames + retry)", () => {
  it("streams a happy-path turn into the store and sends the active locale", async () => {
    const fetchMock: Mock = vi.fn(async () => sseResponse(HAPPY_PATH));
    vi.stubGlobal("fetch", fetchMock);

    const store = useChatStore();
    const { sendMessage } = useChat();
    await sendMessage("How is my parcel?");

    // One fetch, with the contract headers + locale in the body.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const call = fetchMock.mock.calls[0] as [string, RequestInit];
    const url = call[0];
    const init = call[1];
    expect(url).toBe("http://localhost:8000/chat");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Session-ID"]).toBe("11111111-2222-4333-8444-555555555555");
    const body = JSON.parse(init.body as string);
    expect(body.locale).toBe("es");
    expect(body.messages[0]).toEqual({ role: "user", content: "How is my parcel?" });

    // The real reducer folded every frame.
    expect(store.perceiverNotes[0]?.text).toContain("Tuscany");
    expect(store.toolCalls[0]).toMatchObject({ tool: "list_parcels", status: "ok" });
    expect(store.findings).toHaveLength(1);
    expect(store.lastAssistant?.text).toBe("The AOI has 1 parcel: a meadow.");
    expect(store.status).toBe("idle");
  });

  it("reassembles frames split across read() chunk boundaries", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => sseResponse(HAPPY_PATH, { chunkBytes: true })));

    const store = useChatStore();
    await useChat().sendMessage("split test");

    expect(store.lastAssistant?.text).toBe("The AOI has 1 parcel: a meadow.");
    expect(store.status).toBe("idle");
  });

  it("retries a transient network error then succeeds (idempotent: no delta yet)", async () => {
    const fetchMock: Mock = vi
      .fn()
      // First attempt: fetch itself rejects with a network TypeError.
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      // Second attempt: full happy path.
      .mockResolvedValueOnce(sseResponse(HAPPY_PATH));
    vi.stubGlobal("fetch", fetchMock);

    const store = useChatStore();
    const promise = useChat().sendMessage("retry me");
    // Let the backoff timer (faked) elapse so the retry fires.
    await vi.runAllTimersAsync();
    await promise;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(store.lastAssistant?.text).toBe("The AOI has 1 parcel: a meadow.");
    expect(store.status).toBe("idle");
  });

  it("does NOT retry after a delta already streamed (idempotency guard)", async () => {
    // First attempt streams a delta, then the stream errors (cut after delta).
    const fetchMock: Mock = vi
      .fn()
      .mockResolvedValueOnce(truncatedSseResponse([FR_DELTA_1]))
      .mockResolvedValueOnce(sseResponse(HAPPY_PATH));
    vi.stubGlobal("fetch", fetchMock);

    const store = useChatStore();
    const promise = useChat().sendMessage("cut after delta");
    await vi.runAllTimersAsync();
    await promise;

    // The cut happened AFTER a delta, so retrying would duplicate text: fatal.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(store.status).toBe("error");
    expect(store.errorMessage).toBe("stream_interrupted");
  });

  it("does NOT retry a 4xx contract error (e.g. 422 extra=forbid)", async () => {
    const fetchMock: Mock = vi.fn(async () => new Response("bad", { status: 422 }));
    vi.stubGlobal("fetch", fetchMock);

    const store = useChatStore();
    await useChat().sendMessage("trigger 422");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(store.status).toBe("error");
    expect(store.errorMessage).toBe("http_422");
  });

  it("gives up after MAX_RETRIES transient failures and fails the transport", async () => {
    // Every attempt is a 5xx (transient): 1 initial + 3 retries = 4 calls.
    const fetchMock: Mock = vi.fn(async () => new Response("boom", { status: 503 }));
    vi.stubGlobal("fetch", fetchMock);

    const store = useChatStore();
    const promise = useChat().sendMessage("always 503");
    await vi.runAllTimersAsync();
    await promise;

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(store.status).toBe("error");
    expect(store.errorMessage).toBe("network_error");
  });
});
