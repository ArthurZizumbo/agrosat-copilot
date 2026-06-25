// E2E 1 (B-E9-1): the app shell loads and the MapLibre canvas renders.
//
// Validates (a) the page loads end-to-end (SSR + client hydration) and (c) the
// MapLibre map mounts a real <canvas> inside its container. The map section is
// labelled with the i18n key `map.label` ("Mappa delle particelle" in the it
// default locale); MapLibre injects a `.maplibregl-canvas` once it initialises.

import { test, expect } from "@playwright/test";

test("app shell loads and MapLibre canvas renders", async ({ page }) => {
  await page.goto("/", { waitUntil: "domcontentloaded" });

  // The map content region (aria-label = t('map.label')) is present.
  const mapRegion = page.getByRole("region", { name: /particelle|parcels|parcelas/i });
  await expect(mapRegion).toBeVisible();

  // The chat dock (permanent column on desktop) is present: its panel is labelled
  // with t('chat.panel_label') and the assistant title is visible.
  await expect(page.getByRole("heading", { level: 2 })).toContainText(
    /Assistente|Asistente|Assistant/,
  );

  // MapLibre GL mounts a real canvas once initialised (import.meta.client path).
  const mapCanvas = page.locator("canvas.maplibregl-canvas");
  await expect(mapCanvas).toBeVisible({ timeout: 30_000 });

  await page.screenshot({ path: "test-results/01-page-and-map.png", fullPage: true });
});
