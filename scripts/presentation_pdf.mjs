// Imprime la presentacion (modo ?print-pdf de Reveal.js) a PDF usando la API
// CDP de Playwright (page.pdf con preferCSSPageSize), que respeta el tamano de
// pagina 1600x900 que inyecta Reveal. El print-to-pdf del CLI de Edge NO lo
// respeta (pagina el deck en ~4 hojas por lamina), por eso este camino.
// Reutiliza @playwright/test del frontend y el canal msedge del sistema
// (sin descargar navegadores). Lo invoca scripts/presentation_pdf.ps1.
//
// Uso: node scripts/presentation_pdf.mjs <url-print-pdf> <salida.pdf>
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const requireFrontend = createRequire(path.join(here, "..", "frontend", "package.json"));
const { chromium } = requireFrontend("@playwright/test");

const [url, out] = process.argv.slice(2);
if (!url || !out) {
  console.error("uso: node presentation_pdf.mjs <url> <salida.pdf>");
  process.exit(2);
}

let browser;
try {
  // Chromium de Playwright o Chrome del sistema. NUNCA msedge: el printToPDF
  // de Edge ignora el tamano de pagina CSS y pagina cada lamina en ~4 hojas.
  browser = await chromium.launch().catch(() => chromium.launch({ channel: "chrome" }));
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.goto(url, { waitUntil: "networkidle" });
  // print.html marca data-print-ready cuando fuentes + imagenes cargaron.
  await page.waitForSelector("html[data-print-ready='1']", { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(600);
  await page.pdf({ path: out, preferCSSPageSize: true, printBackground: true });
  console.log(`OK: ${out}`);
} finally {
  if (browser) await browser.close();
}
