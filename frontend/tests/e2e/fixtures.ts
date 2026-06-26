// Shared E2E fixtures (B-E9-1).
//
// A FIXED session UUID used by the chat spec. `global-setup.ts` seeds a matching
// chat_sessions row so the `/chat` guard (verify_chat_session -> 403 on unknown
// session) authorises the request. Specs set this as the `agrosat-session-id`
// cookie before navigating, so the SSE chat hits a REAL, authorised session.

/** Deterministic session id seeded in chat_sessions for the chat E2E. */
export const E2E_SESSION_ID = "e2e00000-0000-4000-8000-000000000001";

/** Backend base URL (the FastAPI app the SSE chat streams from). */
export const BACKEND_URL = process.env.E2E_API_BASE_URL ?? "http://localhost:8000";

/** Postgres container name used to seed the session via `docker exec psql`. */
export const PG_CONTAINER =
  process.env.E2E_PG_CONTAINER ?? "agro_sat_copilot-postgres-1";
