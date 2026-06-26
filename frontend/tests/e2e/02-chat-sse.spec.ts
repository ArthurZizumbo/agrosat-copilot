// E2E 2 (B-E9-1): the chat answers a real message over SSE (live Gemini).
//
// Validates (b) the critical conversational flow: typing a message and pressing
// send POSTs /chat with `X-Session-ID`, the backend streams `text_delta` frames
// from the real reasoner, and the assistant transcript fills in. NOT mocked: this
// hits the running FastAPI + ADK agent + Gemini.
//
// The `/chat` guard 403s on an unknown session, so we set the `agrosat-session-id`
// cookie to the E2E_SESSION_ID that global-setup seeded into chat_sessions.

import { test, expect } from "@playwright/test";
import { E2E_SESSION_ID } from "./fixtures";

test("chat streams a real assistant reply over SSE", async ({ page, context }) => {
  // Authorise the session: the cookie drives useSession -> X-Session-ID.
  await context.addCookies([
    { name: "agrosat-session-id", value: E2E_SESSION_ID, domain: "localhost", path: "/" },
  ]);

  // Wait for full client hydration (networkidle): with only domcontentloaded the
  // textarea's `input` events fire before Vue binds v-model, so `draft` never
  // updates and the send button stays disabled.
  await page.goto("/", { waitUntil: "networkidle" });

  // The composer textarea (id=chat-input). Type with real key events (small delay)
  // so Vue's v-model updates `draft` and the send button (disabled while empty)
  // enables.
  const input = page.locator("#chat-input").first();
  await expect(input).toBeVisible();
  await input.click();
  await input.pressSequentially("In one short sentence: what is NDVI?", { delay: 25 });

  // Send button is labelled with t('chat.send') (Send/Invia/Enviar); it enables
  // once the draft is non-empty.
  const send = page.getByRole("button", { name: /Invia|Enviar|Send/ }).first();
  await expect(send).toBeEnabled();
  await send.click();

  // The transcript log (role=log) receives the streamed assistant reply.
  const log = page.getByRole("log");
  await expect(log).toContainText(/NDVI|vegetation|vegetazione|vegetaci/i, {
    timeout: 90_000,
  });

  // The reply is substantive, not a stub: assert a reasonable accumulated length.
  await expect
    .poll(async () => (await log.innerText()).length, { timeout: 90_000 })
    .toBeGreaterThan(40);

  await page.screenshot({ path: "test-results/02-chat-sse.png", fullPage: true });
});
