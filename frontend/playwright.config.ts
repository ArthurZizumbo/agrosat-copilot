// Playwright E2E config — AgroSatCopilot (B-E9-1 live validation).
//
// Validates the real app end-to-end against an ALREADY-RUNNING stack:
//   - Nuxt SSR dev server on http://localhost:3000 (pnpm dev)
//   - FastAPI backend on http://localhost:8000 (poetry run uvicorn ... :8000)
//   - Postgres+PostGIS on :55432 with a seeded chat_sessions row
//
// We do NOT spawn the servers here (`webServer` omitted): the validation runbook
// boots backend + frontend + DB first, then runs `pnpm test:e2e`. This keeps the
// suite a pure black-box check of the live system (REGLA ARTHUR: no simular).
//
// The chat flow needs a session that EXISTS in chat_sessions (the `/chat` guard
// returns 403 for an unknown session). `tests/e2e/global-setup.ts` seeds a fixed
// UUID and the spec sets it as the `agrosat-session-id` cookie before navigating.

import { defineConfig, devices } from "@playwright/test";

const BASE_URL = process.env.E2E_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./tests/e2e",
  globalSetup: "./tests/e2e/global-setup.ts",
  // First Nuxt SSR render compiles on demand and can take several seconds.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: [["list"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  outputDir: "test-results",
  use: {
    baseURL: BASE_URL,
    // Desktop viewport so the chat dock renders as a permanent column (>=lg) and
    // the map fills the content area (layouts/default.vue responsive rules).
    viewport: { width: 1440, height: 900 },
    actionTimeout: 30_000,
    navigationTimeout: 90_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
});
