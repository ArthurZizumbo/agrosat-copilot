import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createApp, nextTick } from "vue";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import { createPersistedState } from "pinia-plugin-persistedstate";
import { useChatStore } from "~/stores/chat";

// Exercises the REAL persistence path of the chat store. The Nuxt module wires
// `pinia-plugin-persistedstate` onto Pinia at runtime; here we install the same
// plugin (`createPersistedState`) and mount Pinia on a Vue app so its store
// plugins actually run (plugins only fire for stores created on an app-installed
// pinia). The store's own `persist` block (pick: ["messages","llmVariant"],
// localStorage backend) then runs unchanged. The storage adapter comes from the
// `piniaPluginPersistedstate` global shim (tests/setup), which reads/writes the
// in-memory `localStorage` — a genuine serialise -> store -> read -> hydrate
// round-trip, not a stubbed behaviour.
//
// Writes are flushed by the plugin's detached `$subscribe`, which runs on the
// Vue scheduler microtask, so the write assertions `await nextTick()` first.

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

describe("chatStore persistence (pick: messages, llmVariant)", () => {
  it("rehydrates messages + llmVariant from localStorage on creation", () => {
    // Seed storage as the plugin serialises it: only the picked keys.
    globalThis.localStorage.setItem(
      STORE_KEY,
      JSON.stringify({
        messages: [
          { id: "user-1", role: "user", text: "hello", createdAt: 1 },
          {
            id: "assistant-1",
            role: "assistant",
            text: "hi there",
            citations: [],
            createdAt: 2,
          },
        ],
        llmVariant: "qwen35",
      }),
    );

    mountPinia();
    // Creating the store triggers hydration from the seeded storage.
    const store = useChatStore();

    expect(store.messages).toHaveLength(2);
    expect(store.messages[1]?.text).toBe("hi there");
    expect(store.llmVariant).toBe("qwen35");
    // Transient state is NOT persisted, so it starts clean.
    expect(store.toolCalls).toHaveLength(0);
    expect(store.status).toBe("idle");
  });

  it("persists only the picked keys after a turn (not transient state)", async () => {
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

    // Durable conversation state is written.
    expect(saved.messages).toHaveLength(2);
    expect(saved.llmVariant).toBe("gemini");
    // Transient per-turn state is excluded (would be wrong to rehydrate).
    expect(saved).not.toHaveProperty("toolCalls");
    expect(saved).not.toHaveProperty("findings");
    expect(saved).not.toHaveProperty("status");
    expect(saved).not.toHaveProperty("activeAssistantId");
  });

  it("round-trips: a saved store rehydrates into a fresh store instance", async () => {
    mountPinia();
    const first = useChatStore();
    first.startUserTurn("ping");
    first.applyEvent({ type: "text_delta", text: "pong" });
    first.applyEvent({ type: "done" });
    first.setLlmVariant("qwen35");
    await nextTick();

    // A fresh app+pinia reading the same localStorage simulates a page reload.
    mountPinia();
    const reloaded = useChatStore();
    expect(reloaded.lastAssistant?.text).toBe("pong");
    expect(reloaded.llmVariant).toBe("qwen35");
  });
});
