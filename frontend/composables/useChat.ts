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
//
// Resilience (US-057, D3): `streamWithRetry` wraps the whole fetch + readStream
// attempt and retries on TRANSIENT failures (network TypeError, 5xx, an early
// reader cut before any terminal event) with exponential backoff + full jitter,
// capped. It does NOT retry FATAL failures (AbortError, 4xx contract errors, a
// backend `error` event, or a stream that already emitted `done`). Idempotency:
// once a `text_delta` reached the store, a later cut is NOT retried (that would
// duplicate text); we `failTransport("stream_interrupted")` instead.

import type { AgentEvent, ChatRequest, LlmVariant } from "~/types/agent";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";
import { useSessionsStore } from "~/stores/sessions";

/** Retry tuning (US-057 D3). */
const MAX_RETRIES = 3;
const BASE_DELAY_MS = 500;
const MAX_DELAY_MS = 8000;

/** ChatRequest plus the per-request `locale` (US-057 D4).
 *
 * The wire contract in `types/agent.ts` does not yet declare `locale` (it lands
 * in a sibling change together with the backend `ChatRequest.locale` field).
 * We type the body locally so this file typechecks and sends the active locale
 * the moment the backend accepts it. Until then the backend ignores it; see the
 * handoff note "BACKEND PENDING (locale)".
 */
type ChatRequestWithLocale = ChatRequest & { locale?: "it" | "es" | "en" };

/** Marker error: the reader was cut before a terminal event (transient). */
class StreamCutError extends Error {
  /** True once at least one `text_delta` reached the store this turn. */
  readonly afterDelta: boolean;
  constructor(afterDelta: boolean) {
    super("stream_cut");
    this.name = "StreamCutError";
    this.afterDelta = afterDelta;
  }
}

/** Marker error: the backend returned a non-OK HTTP status. */
class HttpStatusError extends Error {
  readonly status: number;
  constructor(status: number) {
    super(`http_${status}`);
    this.name = "HttpStatusError";
    this.status = status;
  }
}

/** Outcome of a single stream read. */
interface ReadOutcome {
  /** True if a `done`/`error` terminal event was observed. */
  terminal: boolean;
  /** True if any `text_delta` reached the store (idempotency guard). */
  receivedDelta: boolean;
}

/** Parse one SSE frame (`event:`/`data:` lines) into an AgentEvent.
 *
 * The backend always sends both an `event:` line (the type) and a `data:` line
 * (the JSON payload without the type). We merge `{type}` back in so the payload
 * matches the discriminated `AgentEvent` union. Returns `null` for keep-alive
 * comments or malformed frames.
 *
 * Exported so the SSE-parser unit tests exercise the real implementation
 * instead of a reimplementation (US-057 §6, REGLA ARTHUR: real contract frames).
 */
export function parseSseFrame(frame: string): AgentEvent | null {
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

/** Sleep `ms` milliseconds (cancellable via `signal`). */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Exponential backoff with full jitter, capped (US-057 D3).
 *
 * `base * 2**attempt`, clamped to `MAX_DELAY_MS`, then full jitter
 * `delay * (0.5 + random*0.5)` so retries spread out and avoid thundering herd.
 */
function backoffDelayMs(attempt: number): number {
  const exp = Math.min(BASE_DELAY_MS * 2 ** attempt, MAX_DELAY_MS);
  return Math.round(exp * (0.5 + Math.random() * 0.5));
}

export function useChat() {
  const store = useChatStore();
  const mapStore = useMapStore();
  const sessionsStore = useSessionsStore();
  const { apiUrl } = useSession();
  const { locale, t } = useI18n();

  let abort: AbortController | null = null;

  function stop() {
    if (abort) {
      abort.abort();
      abort = null;
    }
  }

  /** Send a user message and stream the agent's SSE response (with retry). */
  async function sendMessage(text: string): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed || store.isBusy) return;
    if (!import.meta.client) return;

    // The active chat tab IS the backend session (US-080). It is created on app
    // open by useSessions.ensureActiveSession, so it is always set here.
    const sessionId = sessionsStore.activeId;
    if (!sessionId) return;

    // Title an untitled tab from its first user turn (local + server, so the
    // session list shows it from any browser). The PATCH is best-effort.
    if (store.messages.length === 0) {
      const snippet = trimmed.length > 32 ? `${trimmed.slice(0, 32)}…` : trimmed;
      sessionsStore.renameTab(sessionId, snippet);
      // Best-effort server PATCH so the session list shows the title from any
      // browser. Referenced via globalThis so the unit tests (no Nuxt $fetch)
      // simply skip it instead of throwing.
      const patchFetch = (
        globalThis as unknown as {
          $fetch?: (url: string, opts: unknown) => Promise<unknown>;
        }
      ).$fetch;
      if (typeof patchFetch === "function") {
        void patchFetch(apiUrl(`/sessions/${sessionId}`), {
          method: "PATCH",
          headers: { "X-Session-ID": sessionId },
          body: { title: snippet },
        }).catch(() => {});
      }
    }

    store.startUserTurn(trimmed);

    // History already includes the just-added user turn (assistant turns are
    // appended lazily once deltas stream, so they are not in the request yet).
    const messages = store.historyForRequest;

    // Send the active drawn AOI inline (GeoJSONGeometry); the backend's
    // perceiver observes it. Demo/drawn AOIs are pure geometry, so no id needed.
    const aoi = mapStore.activeAoi?.geometry ?? null;

    // Active UI locale (US-057 D4) so the reasoner replies in the user's
    // language. BACKEND PENDING: ChatRequest is `extra="forbid"`; until the
    // backend adds `locale`, it is ignored (or 422s). Sent only on the client.
    const activeLocale = locale.value as "it" | "es" | "en";

    // Active parcel clicked on the map (US-058 link parcel->chat). REAL id from
    // the rendered feature; null when no parcel is selected. The contract
    // `ChatRequest.parcel_id` already declares it (types/agent.ts).
    const parcelId = store.activeParcelId ?? null;

    const body: ChatRequestWithLocale = {
      messages,
      session_id: sessionId,
      aoi,
      parcel_id: parcelId,
      year: 2019,
      locale: activeLocale,
    };

    // Crop-classification model the user pinned (US-081). Sent only when set so
    // the body stays minimal and the backend's `voting3` tool default applies
    // otherwise. The reasoner forwards it to `classify_new_parcel` (see the
    // system instruction injected by chat_service.py).
    if (store.cropModel) body.crop_model = store.cropModel;

    abort = new AbortController();
    const signal = abort.signal;
    try {
      await streamWithRetry(body, signal);
    } finally {
      abort = null;
    }
  }

  /** Run one fetch + readStream attempt, retrying transient failures (D3).
   *
   * `receivedDeltaEver` tracks, across attempts, whether any `text_delta` has
   * already reached the store this turn: once it has, a later cut is NOT a
   * retry candidate (idempotency — retrying would duplicate streamed text).
   */
  async function streamWithRetry(
    body: ChatRequestWithLocale,
    signal: AbortSignal,
  ): Promise<void> {
    let receivedDeltaEver = false;

    for (let attempt = 0; ; attempt += 1) {
      try {
        const outcome = await runStreamAttempt(body, signal, receivedDeltaEver);
        // Success: the attempt completed (terminal event or a clean settle).
        return void outcome;
      } catch (err) {
        receivedDeltaEver = receivedDeltaEver || deltaSeen(err);

        // Fatal: never retry — abort, 4xx contract error, or post-delta cut.
        if (isAbort(err)) return; // intentional stop/dispose: stay silent.

        if (err instanceof HttpStatusError && err.status < 500) {
          // 4xx (incl. 422 = contract mismatch like extra="forbid"): fatal.
          store.failTransport(`http_${err.status}`);
          return;
        }

        if (err instanceof StreamCutError && err.afterDelta) {
          // Idempotency: a delta already streamed; retrying duplicates text.
          console.warn("[useChat] stream cut after delta; not retrying");
          store.failTransport("stream_interrupted");
          return;
        }

        // Transient: network TypeError, 5xx, or an early (pre-delta) cut.
        if (attempt >= MAX_RETRIES) {
          console.warn(
            `[useChat] transport failed after ${attempt + 1} attempts:`,
            (err as Error)?.message,
          );
          store.failTransport("network_error");
          return;
        }

        const wait = backoffDelayMs(attempt);
        console.warn(
          `[useChat] transient transport error (attempt ${attempt + 1}/` +
            `${MAX_RETRIES + 1}); retrying in ${wait}ms:`,
          (err as Error)?.message,
        );
        try {
          await delay(wait, signal);
        } catch {
          return; // aborted while backing off.
        }
      }
    }
  }

  /** A single fetch + readStream. Throws a classified error on failure. */
  async function runStreamAttempt(
    body: ChatRequestWithLocale,
    signal: AbortSignal,
    receivedDeltaBefore: boolean,
  ): Promise<ReadOutcome> {
    const sessionId = body.session_id ?? sessionsStore.activeId ?? "";

    const res = await fetch(apiUrl("/chat"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        "X-Session-ID": sessionId,
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!res.ok || !res.body) {
      // Non-OK status: classify (4xx fatal, 5xx transient) in the retry loop.
      throw new HttpStatusError(res.status);
    }

    store.markStreaming();
    return readStream(res.body, receivedDeltaBefore);
  }

  /** Read the SSE body to completion, dispatching each frame to the store.
   *
   * Returns whether a terminal event was observed and whether any `text_delta`
   * reached the store. Throws `StreamCutError` if `reader.read()` fails (or the
   * stream ends) before a terminal event — the retry loop decides what to do.
   */
  async function readStream(
    body: ReadableStream<Uint8Array>,
    receivedDeltaBefore: boolean,
  ): Promise<ReadOutcome> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let terminal = false;
    let receivedDelta = receivedDeltaBefore;

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
            if (event.type === "text_delta") receivedDelta = true;
            if (event.type === "done" || event.type === "error") {
              terminal = true;
            }
          }
          sep = buffer.indexOf("\n\n");
        }
        if (terminal) break;
      }

      // Flush a trailing frame that was not blank-line terminated.
      if (!terminal) {
        const tail = buffer.trim();
        if (tail.length > 0) {
          const event = parseSseFrame(tail);
          if (event) {
            store.applyEvent(event);
            if (event.type === "text_delta") receivedDelta = true;
            if (event.type === "done" || event.type === "error") {
              terminal = true;
            }
          }
        }
      }
    } catch (err) {
      // The reader cut mid-stream (network drop). Abort is intentional and
      // re-thrown unchanged so the retry loop stays silent; any other read
      // failure becomes a (possibly retryable) StreamCutError.
      if (isAbort(err)) throw err;
      throw new StreamCutError(receivedDelta);
    } finally {
      reader.releaseLock();
    }

    // The stream ended cleanly but without a terminal event. If a delta had
    // streamed, settle the turn (the answer is already in the store). If not,
    // treat the early end as a transient cut so the retry loop can recover.
    if (!terminal) {
      if (!receivedDelta) throw new StreamCutError(false);
      if (store.status === "streaming") store.applyEvent({ type: "done" });
    }

    return { terminal, receivedDelta };
  }

  /**
   * Switch the active per-session reasoner backend (E12, real).
   *
   * Persists the choice on the session via `POST /llm/switch` (header
   * `X-Session-ID`): the backend writes it to `chat_sessions.llm_model` and the
   * NEXT `/chat` reads it back and builds the matching backend. We update the
   * store OPTIMISTICALLY so the segmented control reacts instantly, then revert
   * (and surface a transient toast) if the POST fails — the on-prem variants
   * (`qwen-onprem` / `qwen-vl`) are only reachable behind the demo VM tunnel, so
   * a failed switch must degrade gracefully, never break the chat. (When the host
   * is up but a later `/chat` finds it down, the backend itself degrades to
   * `gemini` via availability-aware routing; that path is not surfaced here.)
   *
   * SSR-safe: runs only on the client (the switch is a user gesture). The unit
   * tests have no `fetch`/Nuxt session, so a missing session id or `import.meta`
   * client guard simply mirrors the variant locally without a network call.
   */
  async function switchLlm(variant: LlmVariant): Promise<void> {
    const previous = store.llmVariant;
    if (variant === previous) return;

    // Optimistic: reflect the choice immediately and clear any stale notice.
    store.setLlmVariant(variant);
    store.setLlmSwitchError(null);

    if (!import.meta.client) return;
    const sessionId = sessionsStore.activeId;
    if (!sessionId) return; // no session yet: keep the local mirror only.

    try {
      const res = await fetch(apiUrl("/llm/switch"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId,
        },
        body: JSON.stringify({ model: variant }),
      });
      if (!res.ok) throw new HttpStatusError(res.status);
    } catch (err) {
      // Revert the optimistic change and surface a non-blocking notice. The
      // chat stays fully usable on the previous (working) backend.
      store.setLlmVariant(previous);
      store.setLlmSwitchError(t("llm.switch_failed"));
      console.warn(
        "[useChat] llm switch failed; reverted to",
        previous,
        (err as Error)?.message,
      );
    }
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

/** True if the error is an intentional abort (stop/dispose). */
function isAbort(err: unknown): boolean {
  return (err as Error)?.name === "AbortError";
}

/** True if the (classified) error indicates a delta already streamed. */
function deltaSeen(err: unknown): boolean {
  return err instanceof StreamCutError && err.afterDelta;
}
