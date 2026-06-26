# Blockers EPIC 9 (Frontend Nuxt 4) — validacion 2026-06-25

> Pasada de validacion autonoma US-057 (ChatPanel) + US-058 (MapView). El
> frontend esta VERDE en todos los gates automatizables. Los blockers son una
> deuda de documentacion y la demo en vivo pendiente de entorno.

## B-E9-1 — Validacion E2E en vivo (Playwright) pendiente de backend/.env.local, severidad MEDIA

**Que**: la validacion E2E con navegador real (Playwright) NO se ejecuto en esta
pasada.

**Causa**: levantar la app completa (backend uvicorn + frontend `pnpm dev`)
requiere `backend/.env.local` (DB URL + credenciales LLM Gemini/Vertex) que esta
**AUSENTE**, mas una sesion sembrada en Postgres con parcelas que tengan
`alphaearth_embedding`.

**Impacto**: NO es un gap de calidad del codigo. El frontend se valido por sus
gates reales (todos verdes): vitest 53/7, typecheck 0 err, i18n parity OK
(it/es/en), eslint limpio, build de produccion completo (17.5 MB / 3.46 MB gzip).

**Accion recomendada (Arthur)**: crear `backend/.env.local` con los secretos,
sembrar la sesion demo (`scripts/seed.py` o equivalente) y correr los flujos de
`docs/manual-test/us-057.md` / `us-058.md` seccion "Validacion E2E en vivo":
chat trilingue, markdown + tool cards, switch A/B LLM, draw AOI, link
parcela->chat. El perceiver champion (E7) se vera en vivo solo si las parcelas
sembradas tienen embedding persistido.

## B-E9-2 — Comentario fantasma `<ClientOnly>` en ChatDock, severidad BAJA

**Que**: el comentario en `frontend/stores/chat.ts:358` y el manual-test afirman
que el transcript del chat va envuelto en `<ClientOnly>`, pero `ChatDock.vue` lo
renderiza bajo `<template v-else>` SIN ese wrapper. El unico `<ClientOnly>` real
del repo esta en `AppHeader.vue` (toggle dark mode).

**Impacto**: ninguno funcional. La SSR-safety se sostiene por el `localStorage()`
no-op de `pinia-plugin-persistedstate` en servidor, no por el wrapper. Los 53
tests pasan y no hay hydration mismatch reportado.

**Accion recomendada**: alinear el comentario de `chat.ts:358` con la realidad
(quitar la mencion al wrapper) o anadir el `<ClientOnly>` si se quiere esa
defensa explicita. Cosmetico.

## B-E9-3 — Documentacion desactualizada, severidad INFORMATIVA

**Que**:
- `frontend/CLAUDE.md` dice "SKELETON" y que `pinia-plugin-persistedstate` no esta
  en package.json — AMBOS desactualizados (el frontend esta completo y la dep SI
  esta instalada).
- El handoff US-058 listaba `map-chat-link.test.ts` como archivo NUEVO; en
  realidad esta consolidado en `map-store.test.ts` + `use-map.test.ts`. Corregido
  en el us-resolved.
- El handoff US-057 citaba 39 tests vitest; el real es 53. Corregido.

**Impacto**: ninguno funcional — inexactitudes de documentacion, ya corregidas en
los us-resolved con los datos reales.

**Accion recomendada**: actualizar `frontend/CLAUDE.md` (quitar "SKELETON",
reflejar deps reales) en un pase de mantenimiento de docs.
