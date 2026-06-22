// Multiple chats per browser: the chat LIST + per-chat transcripts.
//
// Browser-local model (see `useSession`): the browser owns N chats; each is a
// backend `chat_sessions` row. This composable owns the CLIENT side of that —
// the list the user sees and switches between, plus each chat's transcript —
// while `useSession` owns identity + creating/deleting the backend row.
//
// Persistence (client-only, localStorage):
//   - `agrosat-chats`       -> ChatMeta[] (the ordered list, newest first)
//   - `agrosat-transcripts` -> Record<chatId, ChatMessage[]> (each chat's turns)
// The chat STORE holds only the CURRENT chat's transcript (its `messages`); on a
// switch we save the outgoing transcript here and load the incoming one.

import { useChatStore } from "~/stores/chat";
import type { ChatMessage } from "~/types/chat";

/** A chat entry in the browser's list. */
export interface ChatMeta {
  id: string;
  title: string;
  createdAt: number;
}

const CHATS_KEY = "agrosat-chats";
const TRANSCRIPTS_KEY = "agrosat-transcripts";
const AUTOSAVE_MS = 400;

// Wire the autosave watcher at most once per client runtime.
let autosaveWired = false;

function readJson<T>(key: string, fallback: T): T {
  if (!import.meta.client) return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown): void {
  if (!import.meta.client) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Quota / private-mode: a lost transcript is acceptable, never fatal.
  }
}

export function useChats() {
  const store = useChatStore();
  const {
    sessionId,
    newUuid,
    ensureOwner,
    setCurrentChatId,
    ensureChatRow,
    deleteChatRow,
  } = useSession();
  const { t } = useI18n();

  const chats = useState<ChatMeta[]>("agrosat-chats-list", () => []);

  function persistChats() {
    writeJson(CHATS_KEY, chats.value);
  }

  function readTranscripts(): Record<string, ChatMessage[]> {
    return readJson<Record<string, ChatMessage[]>>(TRANSCRIPTS_KEY, {});
  }

  /** Save the CURRENT chat's visible transcript into the localStorage map. */
  function saveCurrentTranscript() {
    const id = sessionId.value;
    if (!id) return;
    const all = readTranscripts();
    all[id] = store.messages;
    writeJson(TRANSCRIPTS_KEY, all);
  }

  function loadTranscriptFor(id: string) {
    store.loadTranscript(readTranscripts()[id] ?? []);
  }

  function defaultTitle(): string {
    return `${t("chat.untitled")} ${chats.value.length + 1}`;
  }

  /** Create a new chat (backend row + list entry) and make it current. */
  async function createChat(title?: string): Promise<string> {
    ensureOwner();
    const id = newUuid();
    await ensureChatRow(id);
    chats.value = [
      { id, title: title?.trim() || defaultTitle(), createdAt: Date.now() },
      ...chats.value,
    ];
    persistChats();
    setCurrentChatId(id);
    store.loadTranscript([]);
    return id;
  }

  /** Switch to an existing chat, preserving both transcripts. */
  function switchChat(id: string) {
    if (id === sessionId.value) return;
    saveCurrentTranscript();
    setCurrentChatId(id);
    loadTranscriptFor(id);
  }

  /** Delete a chat (backend row + list entry + transcript). */
  async function deleteChat(id: string) {
    await deleteChatRow(id);
    chats.value = chats.value.filter((c) => c.id !== id);
    persistChats();
    const all = readTranscripts();
    delete all[id];
    writeJson(TRANSCRIPTS_KEY, all);

    if (id === sessionId.value) {
      // Deleted the active chat: fall back to the newest remaining, or a fresh
      // one when the list is now empty.
      const next = chats.value[0];
      if (next) {
        setCurrentChatId(next.id);
        loadTranscriptFor(next.id);
      } else {
        await createChat();
      }
    }
  }

  /** Rename a chat in the list. */
  function renameChat(id: string, title: string) {
    const trimmed = title.trim();
    if (!trimmed) return;
    chats.value = chats.value.map((c) =>
      c.id === id ? { ...c, title: trimmed } : c,
    );
    persistChats();
  }

  /** Auto-title the current chat from its first user message while it is still
   *  untitled (its title is the generated "Nuevo chat N" placeholder). */
  function titleCurrentFromMessages() {
    const id = sessionId.value;
    if (!id) return;
    const entry = chats.value.find((c) => c.id === id);
    if (!entry || !entry.title.startsWith(t("chat.untitled"))) return;
    const firstUser = store.messages.find((m) => m.role === "user");
    const snippet = firstUser?.text.trim().slice(0, 40);
    if (snippet) renameChat(id, snippet);
  }

  /**
   * Initialise the browser's chat state on load (client only). Ensures an owner,
   * hydrates the list, guarantees a current chat exists (and its backend row),
   * and loads its transcript. This is what closes the session-creation gap.
   */
  async function ensureReady(): Promise<void> {
    if (!import.meta.client) return;
    ensureOwner();
    chats.value = readJson<ChatMeta[]>(CHATS_KEY, []);

    const current = sessionId.value;
    const known = current && chats.value.some((c) => c.id === current);
    if (!chats.value.length || !current || !known) {
      if (current && !known) {
        // A current id exists (cookie) but is not in the list (first run after
        // the multi-chat upgrade): adopt it as the first chat.
        chats.value = [
          { id: current, title: defaultTitle(), createdAt: Date.now() },
          ...chats.value,
        ];
        persistChats();
        await ensureChatRow(current);
        loadTranscriptFor(current);
      } else {
        await createChat();
      }
    } else {
      await ensureChatRow(current);
      loadTranscriptFor(current);
    }
  }

  /** Wire a debounced watcher saving the current transcript as it changes. */
  function startAutosave() {
    if (!import.meta.client || autosaveWired) return;
    autosaveWired = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    watch(
      () => store.messages,
      () => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => {
          titleCurrentFromMessages();
          saveCurrentTranscript();
        }, AUTOSAVE_MS);
      },
      { deep: true },
    );
  }

  return {
    chats,
    currentChatId: sessionId,
    createChat,
    switchChat,
    deleteChat,
    renameChat,
    ensureReady,
    startAutosave,
  };
}
