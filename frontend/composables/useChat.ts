// Chat orchestration composable.
//
// Flow (ADR-011): POST /chat -> 202 {job_id, ws_url} -> open WebSocket
// /ws/chat/{session_id}?job_id=... -> parse AgentEvent JSON -> reduce into the
// Pinia chatStore. Closes on `done`/`error`. SSR-safe: the WebSocket only ever
// opens on the client.

import type {
  AgentEvent,
  ChatDispatchResponse,
  LlmSwitchResponse,
  LlmVariant,
} from "~/types/agent";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";

const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BASE_DELAY_MS = 750;

export function useChat() {
  const store = useChatStore();
  const mapStore = useMapStore();
  const { ensureSession, apiFetch, apiUrl, sessionId } = useSession();

  let socket: WebSocket | null = null;
  let reconnectAttempts = 0;
  let currentWsUrl: string | null = null;
  let closedByUs = false;

  /** Resolve a possibly-relative ws_url from the backend to an absolute URL. */
  function resolveWsUrl(rawWsUrl: string, jobId: string): string {
    let url: URL;
    if (/^wss?:\/\//i.test(rawWsUrl)) {
      url = new URL(rawWsUrl);
    } else {
      // Derive ws(s):// from the http(s) API base.
      const httpBase = new URL(apiUrl(rawWsUrl));
      httpBase.protocol = httpBase.protocol === "https:" ? "wss:" : "ws:";
      url = httpBase;
    }
    if (!url.searchParams.has("job_id")) {
      url.searchParams.set("job_id", jobId);
    }
    return url.toString();
  }

  function closeSocket() {
    closedByUs = true;
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      try {
        socket.close();
      } catch {
        // ignore close races
      }
      socket = null;
    }
  }

  function handleEvent(event: AgentEvent) {
    store.applyEvent(event);
    if (event.type === "done" || event.type === "error") {
      closeSocket();
    }
  }

  function openSocket(wsUrl: string) {
    if (!import.meta.client) return;
    closedByUs = false;

    const ws = new WebSocket(wsUrl);
    socket = ws;

    ws.onopen = () => {
      reconnectAttempts = 0;
      store.markStreaming();
    };

    ws.onmessage = (msg) => {
      // Each frame is one JSON-encoded AgentEvent (newline-delimited possible).
      const raw = typeof msg.data === "string" ? msg.data : "";
      for (const line of raw.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const event = JSON.parse(trimmed) as AgentEvent;
          handleEvent(event);
        } catch {
          // Malformed frame: surface a recoverable error but keep listening.
          store.applyEvent({
            type: "error",
            code: "bad_frame",
            message: "malformed_event_frame",
          });
        }
      }
    };

    ws.onerror = () => {
      // onclose handles reconnection; nothing to do here beyond logging.
    };

    ws.onclose = () => {
      socket = null;
      if (closedByUs) return;
      const terminal = store.status === "idle" || store.status === "error";
      if (terminal) return;
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS && currentWsUrl) {
        reconnectAttempts += 1;
        const delay = RECONNECT_BASE_DELAY_MS * reconnectAttempts;
        window.setTimeout(() => {
          if (currentWsUrl) openSocket(currentWsUrl);
        }, delay);
      } else {
        store.failTransport("connection_lost");
      }
    };
  }

  /** Send a user message and stream the agent's response. */
  async function sendMessage(text: string): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed || store.isBusy) return;

    store.startUserTurn(trimmed);

    try {
      const session = await ensureSession();
      // Scope the analysis to the selected AOI only when it is a real (persisted)
      // id; locally-drawn/demo AOIs use negative ids and fall back to session scope.
      const activeAoiId = mapStore.activeAoi?.id;
      const aoiId = typeof activeAoiId === "number" && activeAoiId > 0 ? activeAoiId : null;
      const res = await apiFetch<ChatDispatchResponse>("/chat", {
        method: "POST",
        body: JSON.stringify({
          session_id: session,
          message: trimmed,
          llm_variant: store.llmVariant,
          aoi_id: aoiId,
        }),
      });
      currentWsUrl = resolveWsUrl(res.ws_url, res.job_id);
      reconnectAttempts = 0;
      openSocket(currentWsUrl);
    } catch {
      store.failTransport("dispatch_failed");
    }
  }

  /** Switch the active LLM backend for this session. */
  async function switchLlm(variant: LlmVariant): Promise<void> {
    store.setLlmVariant(variant);
    if (!sessionId.value) return;
    try {
      await apiFetch<LlmSwitchResponse>("/llm/switch", {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId.value,
          llm_variant: variant,
        }),
      });
    } catch {
      // Non-fatal: the UI already reflects the chosen variant; the backend
      // will also receive it on the next /chat dispatch.
    }
  }

  /** Tear down the socket (call from onBeforeUnmount). */
  function dispose() {
    closeSocket();
    currentWsUrl = null;
  }

  return {
    sendMessage,
    switchLlm,
    dispose,
  };
}
