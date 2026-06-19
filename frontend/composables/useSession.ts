// Session lifecycle for the TEAM backend.
//
// The team's `/chat` is multi-tenant by a client-supplied `session_id` (UUID),
// passed via the `X-Session-ID` header. There is NO `POST /sessions` endpoint,
// so we MINT the UUID on the client (`crypto.randomUUID()`) and persist it in a
// cookie + `useState`, reusing it across turns.
//
// SSR-safe: the cookie is readable on server and client; minting only happens on
// the client (where `crypto.randomUUID` and cookie persistence are reliable).

const SESSION_COOKIE = "agrosat-session-id";

export function useSession() {
  const config = useRuntimeConfig();
  const apiBaseUrl = config.public.apiBaseUrl as string;

  // Cookie is readable on both server and client; one year max-age.
  const sessionCookie = useCookie<string | null>(SESSION_COOKIE, {
    maxAge: 60 * 60 * 24 * 365,
    sameSite: "lax",
    default: () => null,
  });

  const sessionId = useState<string | null>(
    "agrosat-session-id",
    () => sessionCookie.value,
  );

  /** Build an absolute backend URL from a path. */
  function apiUrl(path: string): string {
    const base = apiBaseUrl.replace(/\/$/, "");
    const suffix = path.startsWith("/") ? path : `/${path}`;
    return `${base}${suffix}`;
  }

  /** Generate an RFC 4122 v4 UUID, with a fallback for older runtimes. */
  function newUuid(): string {
    const c = globalThis.crypto as Crypto | undefined;
    if (c && typeof c.randomUUID === "function") return c.randomUUID();
    // Fallback (should not run in modern browsers/Node 20+): RFC4122-ish.
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (ch) => {
      const r = (Math.random() * 16) | 0;
      const v = ch === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  /**
   * Ensure a session UUID exists; mint + persist one on the client when missing.
   * No network call: the backend accepts whatever UUID we generate.
   */
  function ensureSession(): string {
    if (sessionId.value) return sessionId.value;
    const id = newUuid();
    sessionId.value = id;
    sessionCookie.value = id;
    return id;
  }

  return {
    sessionId,
    apiBaseUrl,
    apiUrl,
    ensureSession,
  };
}
