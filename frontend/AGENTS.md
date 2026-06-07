# Frontend — AgroSatCopilot

> Sub-guía del orquestador. Las reglas NON-NEGOTIABLE viven en [`../CLAUDE.md`](../CLAUDE.md) — aquí no se repiten, solo lo operativo de `frontend/`.

Web app **Nuxt 4 SSR** trilingüe (it/es/en). Mapa MapLibre + deck.gl, chat SSE y switch A/B LLM son el destino del producto, **aún no implementados**.

## Estado

**SKELETON.** Solo existen archivos reales: `app.vue`, `pages/index.vue`, los 3 locales `i18n/locales/{it,es,en}.json`, `assets/css/main.css` y configs (`nuxt.config.ts`, `tailwind.config.ts`, `tsconfig.json`, `package.json`).

`components/`, `composables/`, `stores/`, `plugins/`, `middleware/`, `server/`, `types/` son **directorios `.gitkeep` vacíos** — placeholders, sin código todavía. No asumas que existen `ChatPanel`, `MapView`, `useChat`, `useSSE`, `chatStore`, `clerk.client.ts` ni rutas como `chat/[id].vue`: no están escritos.

## Comandos

```bash
pnpm dev          # Nuxt dev server :3000
pnpm build        # SSR build → .output/
pnpm lint         # eslint (.vue,.ts,.tsx)
pnpm typecheck    # nuxt typecheck (vue-tsc)
pnpm test         # vitest run  (sin tests aún)
pnpm test:e2e     # playwright  (sin tests aún)
pnpm i18n:check   # paridad de claves it/es/en

make i18n-check     # == pnpm i18n:check (entra a frontend/)
make test-frontend  # == pnpm test
make test-e2e       # == pnpm test:e2e
make bootstrap      # poetry install + (cd frontend && pnpm install)
```

`pnpm` exclusivo (`pnpm>=10`, `node>=20`). Nunca npm/yarn.

## Stack local

| Capa | Lib (real en `package.json`) |
|------|------------------------------|
| Framework | Nuxt 4 SSR (Vue 3 Composition) |
| UI | `@nuxt/ui-pro` |
| i18n | `@nuxtjs/i18n` (it default, prefix_except_default) |
| Estado | `@pinia/nuxt` + `pinia` |
| CSS | Tailwind **v4** — tema vía `@theme` en `assets/css/main.css`, NO config v3 |
| Mapa | `maplibre-gl` 5.24 + `deck.gl` 9.3 — instalados, **sin usar todavía** |
| Chat | `@ai-sdk/vue` — instalado, sin usar |
| Test | `vitest` + `@playwright/test` — instalados, sin config |

`tailwind.config.ts` existe solo para tooling legacy; la fuente de verdad del tema es `@theme` en `main.css`.

## Convenciones (✅/❌)

- ✅ Componentes con `<script setup lang="ts">` y `const { t } = useI18n()` para todo texto visible → `t('key')`.
- ❌ Strings de UI hardcodeados en template o script.
- ✅ Al agregar una clave i18n, añadirla a `it.json` **y** `es.json` **y** `en.json` simultáneamente (lo valida `scripts/i18n_check.mjs` comparando claves aplanadas; **no** es un plugin de eslint).
- ✅ SSR-safe: `import.meta.client` antes de tocar `window`/browser APIs.
- ✅ Secretos solo en `runtimeConfig` privado server-side, nunca en `runtimeConfig.public`.
- ❌ Inferencia ML, Vertex AI / vLLM / GEE o llamadas a modelos desde el cliente — todo va por `/chat` SSE al backend.

## No tocar

- `pnpm-lock.yaml` — solo cambia vía `pnpm add`/`pnpm install`.
- `.nuxt/`, `.output/`, `node_modules/` — generados; nunca editar ni commitear.
- Nunca agregar una clave i18n a un solo locale: rompe `i18n:check` y bloquea el merge.
- No referencias `vue-echarts` ni `pinia-plugin-persistedstate`: **no están** en `package.json`; agregarlos requiere `pnpm add` y acuerdo de equipo.

## Tests

Andamiaje presente (`vitest`, `@playwright/test` en devDependencies) pero **vacío**: no hay `vitest.config.ts` ni `playwright.config.ts`, ni archivos de test. `pnpm test` / `pnpm test:e2e` corren sin casos.

**TODO**: agregar configs y primeros tests al implementar el primer componente real. Cobertura objetivo ≥50 % frontend (ver checklist root).

## Skills

| Acción | Skill |
|--------|-------|
| Componente / página / layout Vue | `agrosat-frontend-components` |
| Composable, Pinia store, SSE, middleware | `agrosat-frontend-composables` |
| MapLibre / deck.gl / AOI / overlay COG | `agrosat-maplibre-geo` (+ `agrosat-titiler-cog`) |
| Auth Clerk, role guard, CSP | `agrosat-security` |
| Tests Vitest / Playwright | `agrosat-testing` |
