// Pinia store: the open chat tabs and which one is active (US-080).
//
// Each chat tab is its own backend session (own transcript, persisted in
// Postgres) AND its own map AOI (kept here, keyed by session id, so marking a
// zone in Chat 1 never leaks into Chat 2). The tab list + active id + the
// per-session AOI cache persist in localStorage so the workspace survives a
// reload; the transcript itself is reloaded from the server on switch.

import { defineStore } from "pinia";
import type { Aoi } from "~/types/map";

/** One open chat tab. `id` is the backend session UUID. */
export interface SessionTab {
  id: string;
  title: string;
}

// No auth (yet): use a STABLE shared id so the server's session list is the same
// from any browser/tab of this deployment. Swap for the real user id when Clerk
// lands and per-user isolation is wanted.
const DEFAULT_USER_ID = "local-user";

interface SessionsState {
  /** Open chat tabs, left-to-right. */
  tabs: SessionTab[];
  /** Currently selected tab id, or null before the first session exists. */
  activeId: string | null;
  /** Anonymous per-browser id (X-User-ID) until Clerk auth lands. */
  userId: string;
  /** Per-session drawn AOI so each chat keeps its own map selection. */
  aoiBySession: Record<string, Aoi | null>;
}

export const useSessionsStore = defineStore("sessions", {
  state: (): SessionsState => ({
    tabs: [],
    activeId: null,
    userId: "",
    aoiBySession: {},
  }),

  getters: {
    hasSessions: (s): boolean => s.tabs.length > 0,
    activeTab: (s): SessionTab | null =>
      s.tabs.find((t) => t.id === s.activeId) ?? null,
  },

  actions: {
    /** Return the (stable, no-auth) id sent as X-User-ID. Forced to the shared
     *  constant so the server session list is identical from any browser. */
    ensureUserId(): string {
      this.userId = DEFAULT_USER_ID;
      return this.userId;
    },
    addTab(tab: SessionTab) {
      if (!this.tabs.some((t) => t.id === tab.id)) this.tabs.push(tab);
    },
    /** Replace the whole tab list (used to rebuild it from the server). */
    replaceTabs(tabs: SessionTab[]) {
      this.tabs = tabs;
    },
    setActive(id: string) {
      this.activeId = id;
    },
    renameTab(id: string, title: string) {
      const tab = this.tabs.find((t) => t.id === id);
      if (tab) tab.title = title;
    },
    removeTab(id: string) {
      this.tabs = this.tabs.filter((t) => t.id !== id);
      delete this.aoiBySession[id];
      if (this.activeId === id) this.activeId = this.tabs[0]?.id ?? null;
    },
    setAoiFor(id: string, aoi: Aoi | null) {
      this.aoiBySession[id] = aoi;
    },
  },

  // Persist the workspace (tabs + active + anon user + per-session AOIs). The
  // transcript is NOT persisted here — it lives in Postgres and is reloaded on
  // switch — so there is no stale-message duplication.
  persist: {
    storage: piniaPluginPersistedstate.localStorage(),
    pick: ["tabs", "activeId", "userId", "aoiBySession"],
  },
});
