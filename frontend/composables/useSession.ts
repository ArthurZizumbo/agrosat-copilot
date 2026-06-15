// Session lifecycle: create/recover a backend session_id and expose a typed
// fetch helper bound to runtimeConfig.public.apiBaseUrl.
//
// SSR-safe: the persisted id lives in a cookie (works on server and client).
// We only POST /sessions to mint a new id on the client, where we can persist
// it; on the server we just read whatever cookie was sent.

import type { CreateSessionResponse } from "~/types/agent";

const SESSION_COOKIE = "agrosat-session-id";

/** Narrow fetch options compatible with ofetch ($fetch). */
type ApiFetchInit = {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: string;
  headers?: Record<string, string>;
};

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

  /** Typed fetch against the backend, JSON in/out. */
  async function apiFetch<T>(path: string, init?: ApiFetchInit): Promise<T> {
    return await $fetch<T>(apiUrl(path), {
      method: init?.method,
      body: init?.body,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  }

  /** Ensure a session exists; mint one on the client when missing. */
  async function ensureSession(): Promise<string> {
    if (sessionId.value) return sessionId.value;

    // Only mint on the client so we can persist the id reliably.
    if (!import.meta.client) {
      throw new Error("session_not_initialised_on_server");
    }

    const res = await apiFetch<CreateSessionResponse>("/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    });
    sessionId.value = res.session_id;
    sessionCookie.value = res.session_id;
    return res.session_id;
  }

  return {
    sessionId,
    apiBaseUrl,
    apiUrl,
    apiFetch,
    ensureSession,
  };
}
