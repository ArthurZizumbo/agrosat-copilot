import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createApp, nextTick } from "vue";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import { createPersistedState } from "pinia-plugin-persistedstate";
import { useChatStore } from "~/stores/chat";

// Exercises the REAL persistence path of the chat store. The Nuxt module wires
// `pinia-plugin-persistedstate` onto Pinia at runtime; here we install the same
// plugin (`createPersistedState`) and mount Pinia on a Vue app so its store
// plugins actually run.
//
// US-080: the transcript is NO LONGER persisted to localStorage -- it lives in
// Postgres (chat_messages) and is reloaded per active session via
// `loadMessages`. Only the display-only `llmVariant` is persisted client-side.
// These tests assert that contract (messages are NOT written/rehydrated) and
// that `loadMessages` restores a server transcript into the store.

const STORE_KEY = "chat"; // defineStore("chat", ...) -> default persist key.

/** Build a Pinia installed on a throwaway Vue app so store plugins execute. */
function mountPinia(): Pinia {
  const app = createApp({ render: () => null });
  const pinia = createPinia();
  pinia.use(createPersistedState());
  app.use(pinia);
  setActivePinia(pinia);
  return pinia;
}

beforeEach(() => {
  globalThis.localStorage.clear();
});

afterEach(() => {
  globalThis.localStorage.clear();
});

describe("chatStore persistence (pick: llmVariant only; transcript is server-side)", () => {
  it("rehydrates llmVariant only; messages in storage are ignored", () => {
    // Seed storage with both a (legacy) messages array and llmVariant. Only the
    // picked key (llmVariant) must hydrate; messages must NOT come back.
    globalThis.localStorage.setItem(
      STORE_KEY,
      JSON.stringify({
        messages: [{ id: "user-1", role: "user", text: "stale", createdAt: 1 }],
        llmVariant: "qwen-onprem",
      }),
    );

    mountPinia();
    const store = useChatStore();

    expect(store.llmVariant).toBe("qwen-onprem");
    // The transcript is no longer persisted -> not rehydrated from storage.
    expect(store.messages).toHaveLength(0);
  });

  it("persists only llmVariant after a turn (messages + transient excluded)", async () => {
    mountPinia();
    const store = useChatStore();
    store.startUserTurn("how are my parcels?");
    store.applyEvent({ type: "tool_call", name: "list_parcels", arguments: {} });
    store.applyEvent({ type: "text_delta", text: "All good." });
    store.applyEvent({ type: "done" });

    // The plugin writes on the scheduler microtask.
    await nextTick();

    const raw = globalThis.localStorage.getItem(STORE_KEY);
    expect(raw).not.toBeNull();
    const saved = JSON.parse(raw as string);

    // Only the display-only variant is persisted now.
    expect(saved.llmVariant).toBe("gemini");
    // The transcript and transient per-turn state are NOT persisted.
    expect(saved).not.toHaveProperty("messages");
    expect(saved).not.toHaveProperty("toolCalls");
    expect(saved).not.toHaveProperty("findings");
    expect(saved).not.toHaveProperty("status");
    expect(saved).not.toHaveProperty("activeAssistantId");
  });

  it("drops a stale pre-E12 llmVariant from storage instead of rehydrating it", () => {
    // Regression: the reasoner tags were renamed in E12 ("qwen35" ->
    // "qwen-onprem" / "qwen-vl"). A user who opened the app before that deploy
    // still has the old tag in localStorage; rehydrated verbatim it is not in
    // LlmVariant, so no LlmSwitch option matches `variant === opt.value` and the
    // segmented control renders with NOTHING selected. The afterHydrate guard
    // must fold an unknown tag back to the always-resolvable default.
    globalThis.localStorage.setItem(
      STORE_KEY,
      JSON.stringify({ llmVariant: "qwen35", cropModel: "voting3" }),
    );

    mountPinia();
    const store = useChatStore();

    expect(store.llmVariant).toBe("gemini");
  });

  it("keeps a still-valid persisted variant and cropModel untouched", () => {
    globalThis.localStorage.setItem(
      STORE_KEY,
      JSON.stringify({ llmVariant: "qwen-vl", cropModel: "xgb" }),
    );

    mountPinia();
    const store = useChatStore();

    expect(store.llmVariant).toBe("qwen-vl");
    expect(store.cropModel).toBe("xgb");
  });

  it("folds an unknown persisted cropModel back to the champion default", () => {
    globalThis.localStorage.setItem(
      STORE_KEY,
      JSON.stringify({ llmVariant: "gemini", cropModel: "bogus-model" }),
    );

    mountPinia();
    const store = useChatStore();

    expect(store.cropModel).toBe("voting3");
  });

  it("loadMessages restores a server transcript into the store", () => {
    mountPinia();
    const store = useChatStore();
    store.loadMessages([
      { id: 1, role: "user", content: "ping", created_at: "2026-06-28T10:00:00Z" },
      {
        id: 2,
        role: "assistant",
        content: "pong",
        created_at: "2026-06-28T10:00:05Z",
      },
      // A persisted system/grounding row (if any) is not shown as a turn.
      { id: 3, role: "system", content: "grounding", created_at: "2026-06-28T10:00:06Z" },
    ]);

    expect(store.messages).toHaveLength(2);
    expect(store.messages[0]).toMatchObject({ role: "user", text: "ping" });
    expect(store.lastAssistant?.text).toBe("pong");
    expect(store.status).toBe("idle");
  });
});
