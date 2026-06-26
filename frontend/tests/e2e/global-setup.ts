// Playwright global setup (B-E9-1): seed the E2E session and assert the live stack.
//
// 1. Seeds a deterministic chat_sessions row (E2E_SESSION_ID) via `docker exec
//    psql` so the `/chat` guard authorises the SSE request (it 403s on an unknown
//    session). Idempotent: ON CONFLICT DO NOTHING.
// 2. Verifies the backend /healthz and the frontend root are up before any spec
//    runs, failing fast with a clear message if the runbook forgot to boot them.
//
// We shell out to the Postgres container (no node pg client is a dependency of
// the frontend); the DB is local-dev only, so the dev password is acceptable.

import { execFileSync } from "node:child_process";
import { E2E_SESSION_ID, BACKEND_URL, PG_CONTAINER } from "./fixtures";

const FRONTEND_URL = process.env.E2E_BASE_URL ?? "http://localhost:3000";

function seedSession(): void {
  const sql = `
    INSERT INTO chat_sessions (id, user_id, llm_model)
    VALUES ('${E2E_SESSION_ID}', 'e2e@agrosat.dev', 'gemini')
    ON CONFLICT (id) DO NOTHING;
  `;
  execFileSync(
    "docker",
    ["exec", PG_CONTAINER, "psql", "-U", "agrosat", "-d", "agrosat", "-v", "ON_ERROR_STOP=1", "-c", sql],
    { stdio: "pipe" },
  );
}

async function waitForHttp(url: string, label: string, expectStatuses: number[]): Promise<void> {
  const deadline = Date.now() + 90_000;
  let lastErr = "";
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url, { redirect: "manual" });
      if (expectStatuses.includes(res.status)) return;
      lastErr = `status ${res.status}`;
    } catch (e) {
      lastErr = (e as Error).message;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error(
    `${label} not reachable at ${url} (${lastErr}). ` +
      "Boot the stack first (backend :8000, frontend :3000, Postgres :55432).",
  );
}

export default async function globalSetup(): Promise<void> {
  seedSession();
  await waitForHttp(`${BACKEND_URL}/healthz`, "Backend", [200]);
  await waitForHttp(FRONTEND_URL, "Frontend", [200, 301, 302]);
}
