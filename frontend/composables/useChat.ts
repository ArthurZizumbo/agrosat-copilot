// Chat orchestration composable — TEAM backend (SSE over POST).
//
// Flow: POST /chat (header `X-Session-ID: <uuid>`, body = ChatRequest) ->
// `text/event-stream` response consumed with `response.body.getReader()` ->
// parse `event: <type>\ndata: <json>\n\n` frames -> reduce each event into the
// Pinia chatStore. Closes on `done`/`error`. SSR-safe: the request only ever
// runs on the client (guarded), so the stream reader never touches the server.
//
// Why not EventSource: the endpoint is POST (it carries the message history +
// AOI), and the browser `EventSource` is GET-only. We therefore stream manually
// with fetch + a UTF-8 decoder and a frame splitter on the blank-line boundary.

import type { AgentEvent, ChatRequest, LlmVariant } from "~/types/agent";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";

/** Parse one SSE frame (`event:`/`data:` lines) into an AgentEvent.
 *
 * The backend always sends both an `event:` line (the type) and a `data:` line
 * (the JSON payload without the type). We merge `{type}` back in so the payload
 * matches the discriminated `AgentEvent` union. Returns `null` for keep-alive
 * comments or malformed frames.
 */
function parseSseFrame(frame: string): AgentEvent | null {
  let eventType: string | null = null;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // SSE comment / heartbeat
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (!eventType || dataLines.length === 0) return null;
  try {
    const payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    return { type: eventType, ...payload } as AgentEvent;
  } catch {
    return null;
  }
}

export function useChat() {
  const store = useChatStore();
  const mapStore = useMapStore();
  const { ensureSession, apiUrl } = useSession();

  let abort: AbortController | null = null;

  function stop() {
    if (abort) {
      abort.abort();
      abort = null;
    }
  }

  /** Send a user message and stream the agent's SSE response. */
  async function sendMessage(text: string): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed || store.isBusy) return;
    if (!import.meta.client) return;

    store.startUserTurn(trimmed);

    const sessionId = ensureSession();

    // History already includes the just-added user turn (assistant turns are
    // appended lazily once deltas stream, so they are not in the request yet).
    const messages = store.historyForRequest;

    // Send the active drawn AOI inline (GeoJSONGeometry); the backend's
    // perceiver observes it. Demo/drawn AOIs are pure geometry, so no id needed.
    const aoi = mapStore.activeAoi?.geometry ?? null;

    const body: ChatRequest = {
      messages,
      session_id: sessionId,
      aoi,
      year: 2019,
    };

    abort = new AbortController();
    try {
      const res = await fetch(apiUrl("/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          "X-Session-ID": sessionId,
        },
        body: JSON.stringify(body),
        signal: abort.signal,
      });

      if (!res.ok || !res.body) {
        store.failTransport(`http_${res.status}`);
        return;
      }

      store.markStreaming();
      await readStream(res.body);
    } catch (err) {
      // Aborts are intentional (dispose/stop); anything else is a transport
      // failure surfaced to the UI.
      if ((err as Error)?.name === "AbortError") return;
      store.failTransport("network_error");
    } finally {
      abort = null;
    }
  }

  /** Read the SSE body to completion, dispatching each frame to the store. */
  async function readStream(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let terminal = false;

    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Frames are separated by a blank line. Normalise CRLF to LF first.
        buffer = buffer.replace(/\r\n/g, "\n");
        let sep = buffer.indexOf("\n\n");
        while (sep !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const event = parseSseFrame(frame);
          if (event) {
            store.applyEvent(event);
            if (event.type === "done" || event.type === "error") {
              terminal = true;
            }
          }
          sep = buffer.indexOf("\n\n");
        }
        if (terminal) break;
      }

      // Flush a trailing frame that was not blank-line terminated.
      const tail = buffer.trim();
      if (!terminal && tail.length > 0) {
        const event = parseSseFrame(tail);
        if (event) store.applyEvent(event);
      }

      // The stream ended without a terminal event: settle so the UI unblocks.
      if (!terminal && store.status === "streaming") {
        store.applyEvent({ type: "done" });
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Switch the active LLM backend (display only).
   *
   * APORTE PENDIENTE: the team's `/chat` does NOT accept a per-request
   * `llm_variant`; the reasoner backend is fixed by `settings.llm_variant_default`
   * on the server. We keep the store flag in sync so the segmented control
   * reflects a choice, but no request is sent. The LlmSwitch is rendered
   * disabled with a "server configuration" tooltip.
   */
  function switchLlm(variant: LlmVariant): void {
    store.setLlmVariant(variant);
  }

  /** Tear down any in-flight stream (call from onBeforeUnmount). */
  function dispose() {
    stop();
  }

  return {
    sendMessage,
    switchLlm,
    dispose,
  };
}
