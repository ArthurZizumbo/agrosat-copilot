# Frontend Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root cuando haya conflicto en contexto frontend.

**Rol**: Web app Nuxt 4 SSR pura (sin PWA, sin Tauri) bilingüe/trilingüe (italiano + español + inglés). Mapa interactivo MapLibre + deck.gl, chat con streaming SSE, switch A/B LLM, panel de resultados con ECharts.

## Skills References

- [agrosat-frontend-components](../.claude/skills/agrosat-frontend-components/SKILL.md) — Vue 3 Composition, Nuxt UI Pro, chat UI
- [agrosat-frontend-composables](../.claude/skills/agrosat-frontend-composables/SKILL.md) — Composables, Pinia, SSE
- [agrosat-maplibre-geo](../.claude/skills/agrosat-maplibre-geo/SKILL.md) — MapLibre GL + deck.gl + draw-polygon
- [agrosat-security](../.claude/skills/agrosat-security/SKILL.md) — Auth Clerk, role guard, CSP
- [agrosat-testing](../.claude/skills/agrosat-testing/SKILL.md) — Vitest, Playwright

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Crear componente Vue/Nuxt | `agrosat-frontend-components` |
| Crear composable o Pinia store | `agrosat-frontend-composables` |
| Crear página con `definePageMeta` | `agrosat-frontend-components` |
| Integrar MapLibre / deck.gl | `agrosat-maplibre-geo` |
| Dibujar AOI con maplibre-gl-draw | `agrosat-maplibre-geo` |
| Mostrar overlay COG (NDVI/NDWI) en mapa | `agrosat-maplibre-geo` + `agrosat-titiler-cog` |
| Conectar streaming chat SSE | `agrosat-frontend-composables` |
| Switch A/B LLM en UI | `agrosat-frontend-components` |
| i18n it/es/en | `agrosat-frontend-components` |
| Gráfica ECharts (NDVI timeseries) | `agrosat-frontend-components` |
| Role guard middleware | `agrosat-security` |
| Test E2E Playwright | `agrosat-testing` |

## Critical Rules

- **ALWAYS**: Texto visible al usuario debe existir en `i18n/locales/{it,es,en}.json` SIMULTÁNEAMENTE
- **ALWAYS**: Usar `useI18n()` + `t('key')`, jamás strings hardcodeados
- **ALWAYS**: Estado global en Pinia store, jamás `reactive()` exportado entre archivos
- **ALWAYS**: SSR-safe — usar `useFetch`, `useAsyncData` para fetching server-side
- **ALWAYS**: Composables que tocan `window` / browser APIs deben verificar `import.meta.client`
- **ALWAYS**: TypeScript estricto; tipos importados desde `types/` o auto-generados
- **ALWAYS**: TailwindCSS utility-first, evitar CSS inline o `<style scoped>` salvo overrides necesarios
- **NEVER**: Llamar a Vertex AI / vLLM / GEE desde el cliente
- **NEVER**: Inferencia ML en el browser (todo va por `/chat` SSE)
- **NEVER**: Almacenar secretos en `runtimeConfig.public` (usar `runtimeConfig` privado server-side)
- **NEVER**: Strings hardcodeados sin `t()`
- **NEVER**: npm/yarn — solo `pnpm add`

## Project Structure

```
frontend/
├── pages/                # Routing file-based (index.vue, chat/[id].vue, results/[aoi_id].vue)
├── components/           # Componentes por dominio: ChatPanel.vue, MapView.vue, LLMSwitch.vue, AOIDrawer.vue
├── composables/          # useChat.ts, useSSE.ts, useMap.ts, useAOI.ts, useLLMVariant.ts
├── stores/               # Pinia: chatStore.ts, aoiStore.ts, llmStore.ts, sessionStore.ts
├── layouts/              # default.vue, dashboard.vue
├── middleware/           # auth.ts, role-guard.ts
├── plugins/              # clerk.client.ts, maplibre.client.ts
├── server/               # Nitro server: api/, middleware/ (proxy a FastAPI si aplica)
├── i18n/locales/         # it.json, es.json, en.json — TODO en sync
├── types/                # api.d.ts, geo.d.ts, llm.d.ts
├── public/               # icons, favicon, OG image
├── assets/               # css, fonts, hero images (procesados por Vite)
├── nuxt.config.ts
├── tsconfig.json
└── package.json          # pnpm exclusivo
```

## Stack frontend canónico

| Capa | Lib |
|------|-----|
| Framework | Nuxt 4 SSR (Vue 3 Composition API) |
| UI Library | Nuxt UI Pro + TailwindCSS v4 |
| Mapa | MapLibre GL JS + deck.gl |
| Dibujo AOI | maplibre-gl-draw wrapper Vue |
| Chat streaming | `@ai-sdk/vue` con `useChat()` |
| Charts | Apache ECharts via `vue-echarts` |
| Estado | Pinia + `pinia-plugin-persistedstate` |
| i18n | `@nuxtjs/i18n` (it/es/en con rutas localizadas) |
| Auth | Clerk Nuxt module (free tier) |
| Testing | Vitest (unit) + Playwright (E2E) |

## Decision Trees

```
¿Estado compartido entre componentes?
  Reactivo en una vista solo  → ref/reactive local
  Compartido entre páginas    → Pinia store
  Persistente entre sesiones  → Pinia + pinia-plugin-persistedstate

¿Componente nuevo?
  Reutilizable transversal    → components/common/
  Específico de chat          → components/chat/
  Específico de mapa          → components/map/
  Específico de resultados    → components/results/

¿Cómo fetch desde el componente?
  En setup() SSR-ready        → useFetch / useAsyncData
  Reactivo on event           → $fetch dentro de método
  Streaming SSE               → useChat() de @ai-sdk/vue → composable useSSE.ts

¿Renderizar overlay raster?
  COG NDVI/NDWI en mapa       → MapLibre raster source apuntando a TiTiler endpoint
  Histograma de banda         → ECharts bar con datos del backend
  Serie temporal              → ECharts line con datos de /timeseries
```

## i18n — Patrón Obligatorio

```typescript
// component.vue
<template>
  <UButton>{{ t('chat.send') }}</UButton>
</template>

<script setup lang="ts">
const { t } = useI18n()
</script>
```

```json
// i18n/locales/it.json
{ "chat": { "send": "Invia" } }
// i18n/locales/es.json
{ "chat": { "send": "Enviar" } }
// i18n/locales/en.json
{ "chat": { "send": "Send" } }
```

**Toda PR que agregue una key en uno solo de los 3 archivos será bloqueada por el linter `eslint-plugin-i18n-keys`.**

## Commands

```bash
pnpm dev                    # Nuxt dev server :3000
pnpm build                  # SSR build → .output/
pnpm preview                # Preview build local
pnpm lint                   # eslint
pnpm typecheck              # vue-tsc
pnpm test                   # vitest
pnpm test:e2e               # playwright
pnpm i18n:check             # valida que it/es/en estén en sync
```

## QA Checklist Frontend

- [ ] Todas las keys i18n existen en `it.json`, `es.json`, `en.json`
- [ ] Componentes Vue con `<script setup lang="ts">` y type hints
- [ ] Estado global en Pinia, no `reactive()` exportado
- [ ] SSR-safe: `import.meta.client` antes de tocar `window`
- [ ] Mapa MapLibre con cleanup en `onBeforeUnmount`
- [ ] SSE chat con reconexión automática y error handling
- [ ] Switch A/B LLM persistido en Pinia + localStorage
- [ ] Skeletons de loading en chat y mapa
- [ ] Dark mode soportado via Nuxt color-mode
- [ ] Accesibilidad WCAG AA básica (focus visible, alt text, aria-labels)
- [ ] Tests Vitest cobertura ≥50%
- [ ] Playwright E2E del flujo: login → dibujar AOI → chat → resultado
- [ ] `pnpm lint` y `pnpm typecheck` limpios
- [ ] Lighthouse Performance ≥80, Accessibility ≥90
