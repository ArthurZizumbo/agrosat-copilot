// useSessions: lifecycle of the in-app chat tabs (US-080).
//
// Each tab is a backend session. This composable owns the API calls
// (POST /sessions, GET /sessions/{id}/messages, PATCH, DELETE) and the
// orchestration of a tab switch: cache the current tab's map AOI, swap the
// active id, restore the target tab's AOI, and reload its transcript from the
// server into the chat store. Isolation between tabs (chat + map) lives here.

import { useSessionsStore } from "~/stores/sessions";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";

interface ServerMessage {
  id: number;
  role: string;
  content: string;
  created_at: string;
}

interface ServerSession {
  id: string;
  title: string | null;
  llm_model: string;
}

export function useSessions() {
  const store = useSessionsStore();
  const chat = useChatStore();
  const map = useMapStore();
  const { apiUrl } = useSession();
  const { t } = useI18n();

  function defaultTitle(): string {
    return t("chat.tab_default", { n: store.tabs.length + 1 });
  }

  /** Create a session on the server, open it as a tab, and switch to it. */
  async function createSession(title?: string): Promise<string> {
    const userId = store.ensureUserId();
    const created = await $fetch<ServerSession>(apiUrl("/sessions"), {
      method: "POST",
      headers: { "X-User-ID": userId },
      body: { title: title ?? null },
    });
    store.addTab({ id: created.id, title: created.title || defaultTitle() });
    await switchSession(created.id);
    return created.id;
  }

  /** Switch the active tab: cache/restore the map AOI and reload the
   *  transcript. Returns false when the session is gone on the server (stale
   *  localStorage, e.g. after a DB reset) so the caller can recover. */
  async function switchSession(id: string): Promise<boolean> {
    if (store.activeId && store.activeId !== id) {
      store.setAoiFor(store.activeId, map.activeAoi);
    }
    store.setActive(id);
    map.clearSelectedParcel();
    map.setActiveAoi(store.aoiBySession[id] ?? null);
    chat.reset();
    try {
      const messages = await $fetch<ServerMessage[]>(
        apiUrl(`/sessions/${id}/messages`),
        { headers: { "X-Session-ID": id } },
      );
      chat.loadMessages(messages);
      return true;
    } catch {
      // 403/404 (unknown session) or network error: leave the tab empty and
      // report failure so ensureActiveSession can drop a stale id.
      return false;
    }
  }

  /** Rename a tab (local title + best-effort server PATCH). */
  async function renameSession(id: string, title: string): Promise<void> {
    const clean = title.trim();
    if (!clean) return;
    store.renameTab(id, clean);
    try {
      await $fetch(apiUrl(`/sessions/${id}`), {
        method: "PATCH",
        headers: { "X-Session-ID": id },
        body: { title: clean },
      });
    } catch {
      // Title also lives in the persisted tab list; a failed PATCH is non-fatal.
    }
  }

  /** Close a tab (delete on the server), then activate another or create one. */
  async function closeSession(id: string): Promise<void> {
    try {
      await $fetch(apiUrl(`/sessions/${id}`), {
        method: "DELETE",
        headers: { "X-Session-ID": id },
      });
    } catch {
      // Best effort: drop it locally regardless.
    }
    const wasActive = store.activeId === id;
    store.removeTab(id);
    if (!store.activeId) {
      await createSession();
    } else if (wasActive) {
      await switchSession(store.activeId);
    }
  }

  /** Fetch the user's sessions from the server, or null if it is unreachable
   *  (so the caller can tell "no sessions" from "offline"). */
  async function fetchServerSessions(userId: string): Promise<ServerSession[] | null> {
    try {
      return await $fetch<ServerSession[]>(apiUrl("/sessions"), {
        headers: { "X-User-ID": userId },
      });
    } catch {
      return null;
    }
  }

  /** Rebuild the tab list from the server (source of truth), oldest-first for a
   *  stable left-to-right order, keeping a locally-set title when the server has
   *  none. */
  function rebuildTabsFromServer(list: ServerSession[]): void {
    const localTitle = new Map(store.tabs.map((tab) => [tab.id, tab.title]));
    const ordered = [...list].reverse(); // server returns newest-first
    store.replaceTabs(
      ordered.map((s, i) => ({
        id: s.id,
        title: s.title || localTitle.get(s.id) || t("chat.tab_default", { n: i + 1 }),
      })),
    );
  }

  /** Ensure there is a valid active session on app open. The server session
   *  list (by stable X-User-ID, no auth) is the source of truth: it rebuilds the
   *  tabs so they restore from any browser. Falls back to the locally-persisted
   *  tabs when the server is unreachable, and creates the first session when
   *  there are genuinely none. */
  async function ensureActiveSession(): Promise<void> {
    const userId = store.ensureUserId();
    const list = await fetchServerSessions(userId);

    if (list && list.length > 0) {
      rebuildTabsFromServer(list);
      const ids = new Set(list.map((s) => s.id));
      const target =
        store.activeId && ids.has(store.activeId)
          ? store.activeId
          : (store.tabs[store.tabs.length - 1]?.id ?? store.tabs[0]?.id);
      if (target) {
        await switchSession(target);
        return;
      }
    }

    if (list && list.length === 0) {
      // Server reachable, genuinely no sessions -> start the first one.
      await createSession();
      return;
    }

    // Server unreachable (list === null): fall back to the persisted tabs.
    const target = store.activeId ?? store.tabs[0]?.id;
    if (target && (await switchSession(target))) return;
    await createSession();
  }

  return {
    createSession,
    switchSession,
    renameSession,
    closeSession,
    ensureActiveSession,
  };
}
