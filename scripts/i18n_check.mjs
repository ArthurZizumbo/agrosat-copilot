#!/usr/bin/env node
// Valida paridad de claves entre frontend/locales/{it,es,en}.json.
// Replica el script Python inline de .github/workflows/ci.yml para ejecucion
// local via `pnpm i18n:check` (Makefile target `i18n-check`).
//
// Exit codes:
//   0 — los 3 locales tienen el mismo conjunto de claves
//   1 — falta uno de los 3 archivos, o hay claves missing/extra entre locales

import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");

const candidates = [
  resolve(repoRoot, "frontend/i18n/locales"),
  resolve(repoRoot, "frontend/locales"),
];
const localesDir = candidates.find((p) => existsSync(p));

if (!localesDir) {
  console.log("no locales dir yet (frontend/{i18n/locales,locales}), skipping");
  process.exit(0);
}

const codes = ["it", "es", "en"];
const locales = {};
for (const code of codes) {
  const fp = resolve(localesDir, `${code}.json`);
  if (!existsSync(fp)) {
    console.error(`missing ${fp}`);
    process.exit(1);
  }
  locales[code] = JSON.parse(readFileSync(fp, "utf8"));
}

function* flatten(obj, prefix = "") {
  for (const [k, v] of Object.entries(obj)) {
    const key = `${prefix}${k}`;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      yield* flatten(v, `${key}.`);
    } else {
      yield key;
    }
  }
}

const sets = Object.fromEntries(
  codes.map((c) => [c, new Set(flatten(locales[c]))]),
);
const ref = sets.en;
let fail = false;
for (const c of codes) {
  const missing = [...ref].filter((k) => !sets[c].has(k)).sort();
  const extra = [...sets[c]].filter((k) => !ref.has(k)).sort();
  if (missing.length || extra.length) {
    console.error(
      `${c}: missing=${JSON.stringify(missing)} extra=${JSON.stringify(extra)}`,
    );
    fail = true;
  }
}

if (fail) {
  process.exit(1);
}
console.log(`i18n parity OK across ${codes.join(", ")}`);
