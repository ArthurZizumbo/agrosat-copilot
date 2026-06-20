// Vitest setup: shim the Nuxt auto-import globals that the chat store and
// composables reference but that are NOT present outside the Nuxt runtime.
//
// The store's `persist` block calls `piniaPluginPersistedstate.localStorage()`
// (a global injected by the `pinia-plugin-persistedstate/nuxt` module — see
// nuxt.config.ts). At store-definition time vitest has no such global, so the
// store module throws `ReferenceError: piniaPluginPersistedstate is not defined`
// on import. We reproduce the module's real factory: `.localStorage()` returns a
// storage adapter backed by `window.localStorage`, matching what Nuxt provides
// in the browser.
//
// jsdom in this toolchain ships a no-op `localStorage` (it warns about
// `--localstorage-file`), so we install a real in-memory `Storage` first. The
// persist/rehydrate test then exercises a genuine serialise -> store -> read ->
// hydrate round-trip through the SAME storage the store writes to — not a stub
// of the behaviour, just a working backing store.

/** A spec-faithful in-memory Storage (getItem/setItem/removeItem/clear/key). */
class MemoryStorage implements Storage {
  private map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.has(key) ? (this.map.get(key) as string) : null;
  }
  key(index: number): string | null {
    return [...this.map.keys()][index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, String(value));
  }
}

// Install working web Storage globals (jsdom's are inert in this setup).
const localStorageImpl = new MemoryStorage();
const sessionStorageImpl = new MemoryStorage();
Object.defineProperty(globalThis, "localStorage", {
  value: localStorageImpl,
  configurable: true,
  writable: true,
});
Object.defineProperty(globalThis, "sessionStorage", {
  value: sessionStorageImpl,
  configurable: true,
  writable: true,
});

interface PersistStorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** Mirror of the Nuxt module's `piniaPluginPersistedstate` global helper. */
const piniaPluginPersistedstate = {
  /** Storage adapter over `window.localStorage` (as the real plugin uses). */
  localStorage(): PersistStorageLike {
    return {
      getItem: (key) => globalThis.localStorage.getItem(key),
      setItem: (key, value) => globalThis.localStorage.setItem(key, value),
      removeItem: (key) => globalThis.localStorage.removeItem(key),
    };
  },
  /** Session-storage variant (unused by the chat store, kept for parity). */
  sessionStorage(): PersistStorageLike {
    return {
      getItem: (key) => globalThis.sessionStorage.getItem(key),
      setItem: (key, value) => globalThis.sessionStorage.setItem(key, value),
      removeItem: (key) => globalThis.sessionStorage.removeItem(key),
    };
  },
};

// Expose as a real global so `stores/chat.ts` resolves it at import time, exactly
// like the Nuxt auto-import does at runtime.
(globalThis as Record<string, unknown>).piniaPluginPersistedstate =
  piniaPluginPersistedstate;
