// Browser identity + chat-session lifecycle for the TEAM backend.
//
// Model (browser-local, pre-account): the browser is an OWNER (a stable UUID in
// a cookie). It has N chats; each chat is one backend `chat_sessions` row whose
// id travels as the `X-Session-ID` header and is the multi-tenant key. The OWNER
// id is `chat_sessions.user_id`, grouping a browser's chats (used once account
// auth lands to migrate browser chats to a user).
//
// This composable owns identity + the SERVER side (create the row so requests
// stop 403-ing). The chat LIST + per-chat transcripts live in `useChats`.
//
// SSR-safe: cookies are readable on server and client; minting + network calls
// happen on the client only.

const SESSION_COOKIE = "agrosat-session-id";
const OWNER_COOKIE = "agrosat-owner-id";

// Chat ids whose backend row we already created this page life, so we POST
// /sessions at most once per chat. Module-scoped: shared across useSession()
// calls within the same client runtime.
const ensuredRows = new Set<string>();

/** Generate an RFC 4122 v4 UUID, with a fallback for older runtimes. */
function newUuid(): string {
  const c = globalThis.crypto as Crypto | undefined;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    const v = ch === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function useSession() {
  const config = useRuntimeConfig();
  const apiBaseUrl = config.public.apiBaseUrl as string;

  // One-year cookies, readable on server + client.
  const cookieOpts = { maxAge: 60 * 60 * 24 * 365, sameSite: "lax" as const };
  const sessionCookie = useCookie<string | null>(SESSION_COOKIE, {
    ...cookieOpts,
    default: () => null,
  });
  const ownerCookie = useCookie<string | null>(OWNER_COOKIE, {
    ...cookieOpts,
    default: () => null,
  });

  // `sessionId` is the CURRENT chat id (the X-Session-ID for every request).
  const sessionId = useState<string | null>(
    "agrosat-session-id",
    () => sessionCookie.value,
  );
  const ownerId = useState<string | null>(
    "agrosat-owner-id",
    () => ownerCookie.value,
  );

  /** Build an absolute backend URL from a path. */
  function apiUrl(path: string): string {
    const base = apiBaseUrl.replace(/\/$/, "");
    const suffix = path.startsWith("/") ? path : `/${path}`;
    return `${base}${suffix}`;
  }

  /** Ensure the browser owner id exists; mint + persist it on the client. */
  function ensureOwner(): string {
    if (ownerId.value) return ownerId.value;
    const id = newUuid();
    ownerId.value = id;
    ownerCookie.value = id;
    return id;
  }

  /**
   * Ensure a CURRENT chat id exists; mint + persist one on the client when
   * missing. No network call (the backend row is created by `ensureChatRow` /
   * `useChats`). Sync, so existing callers (`useChat`, `useMap`) stay unchanged.
   */
  function ensureSession(): string {
    if (sessionId.value) return sessionId.value;
    const id = newUuid();
    setCurrentChatId(id);
    return id;
  }

  /** Point the browser at `id` as its current chat (header + cookie + state). */
  function setCurrentChatId(id: string) {
    sessionId.value = id;
    sessionCookie.value = id;
  }

  /**
   * Create the backend `chat_sessions` row for `id` (idempotent). This is what
   * closes the session-creation gap: every chat id must be POSTed once before
   * the data endpoints will serve it (else 403). Safe to call repeatedly.
   */
  async function ensureChatRow(id: string): Promise<void> {
    if (!import.meta.client || ensuredRows.has(id)) return;
    const owner = ensureOwner();
    try {
      const res = await fetch(apiUrl("/sessions"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": id,
        },
        body: JSON.stringify({ user_id: owner }),
      });
      // 201 (created) or 200 (already existed) both mean the row is present.
      if (res.ok) ensuredRows.add(id);
    } catch {
      // Offline / backend down: leave it un-ensured so a later call retries.
    }
  }

  /** Ensure the CURRENT chat's backend row exists (mint the id if needed). */
  async function ensureCurrentChatRow(): Promise<string> {
    const id = ensureSession();
    await ensureChatRow(id);
    return id;
  }

  /** Delete a chat's backend row (and its data, via CASCADE). Best-effort. */
  async function deleteChatRow(id: string): Promise<void> {
    if (!import.meta.client) return;
    ensuredRows.delete(id);
    try {
      await fetch(apiUrl(`/sessions/${id}`), {
        method: "DELETE",
        headers: { "X-Session-ID": id },
      });
    } catch {
      // Best-effort: a failed delete leaves an orphan row (harmless).
    }
  }

  return {
    sessionId,
    ownerId,
    apiBaseUrl,
    apiUrl,
    newUuid,
    ensureOwner,
    ensureSession,
    setCurrentChatId,
    ensureChatRow,
    ensureCurrentChatRow,
    deleteChatRow,
  };
}
