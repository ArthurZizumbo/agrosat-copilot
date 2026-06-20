# US-057 fe/C — Handoff: markdown + tool cards colapsables

**Rama**: `feature/E9-US-057-chatpanel` · **Write-set**: `frontend/components/chat/MessageBubble.vue`, `frontend/components/chat/ToolActivity.vue`, `frontend/utils/markdown.ts` (nuevo), `frontend/types/chat.ts`, `frontend/stores/chat.ts` (1 linea en el case `tool_result`), `frontend/i18n/locales/{it,es,en}.json` (3 claves nuevas en sync).
**Estado**: HECHO. `pnpm typecheck` limpio (exit 0, sin `error TS`). `pnpm i18n:check` verde. XSS verificado en runtime.

## Que se implemento

### 1. `utils/markdown.ts` — renderMarkdown (D1, seguridad load-bearing)
- `renderMarkdown(src)` = `DOMPurify.sanitize(marked.parse(src, { async: false }))`.
- `marked.use({ gfm: true, breaks: true })` -> tablas + code blocks GFM, sincrono.
- `isomorphic-dompurify` -> sanitiza en server Node (SSR-safe, sin `window is not defined`) y cliente; HTML deterministico -> no rompe hidratacion.
- `marked` ya NO sanitiza (opcion `sanitize` eliminada v12+); DOMPurify externo es obligatorio.
- Guard `if (!src) return ""`.

### 2. `MessageBubble.vue` — markdown solo en assistant
- Turno **assistant**: `<div v-html="renderedHtml" class="markdown-body">` donde `renderedHtml = renderMarkdown(message.text)` (computed). `v-html` SOLO recibe HTML ya sanitizado, nunca el raw del LLM.
- Turno **user**: sigue `<p class="whitespace-pre-wrap">{{ message.text }}</p>` (texto propio, sin markdown -> reduce superficie XSS).
- `<style scoped>` con `:deep()` para tablas (scroll-x), code/pre, listas, blockquote, links, hr, img — theme-aware via tokens `var(--color-*)` + `:global(.dark)` para links; no asume `prose` de Tailwind instalado.
- A11y: contenedor con `:aria-label="t('chat.assistant')"`.

### 3. `ToolActivity.vue` — filas colapsables (D1)
- Cada `<li>` pasa a `<details class="group">` + `<summary>` (toggle teclado Enter/Space, foco y semantica expanded/collapsed nativos, gratis).
- `<summary>`: fila actual (chevron + status icon+label nunca color-only + nombre mono + summary) con `focus-visible:ring-*` y `:aria-label="t('chat.tool_details', { tool })"`.
- Expandido: dos bloques `<pre>` con `JSON.stringify(call.args)` (input) y `JSON.stringify(call.result)` (output), labels i18n `chat.tool_input`/`chat.tool_output`. Si no hay `result` (p.ej. running) muestra el status label en italic.
- Chevron rota con `group-open:rotate-90`, transicion 150ms con `@media (prefers-reduced-motion: reduce)` que la desactiva. Marker nativo oculto.
- `formatJson()` y `hasContent()` helpers locales.

### 4. `types/chat.ts` — `result?` en TrackedToolCall
- `result?: Record<string, unknown>` (raw `tool_result.result`, `undefined` mientras running).

### 5. `stores/chat.ts` — reducer (1 linea por rama)
- En el case `tool_result`: `tracked.result = event.result` (rama correlacionada) y `result: event.result` (rama sintetica). Sin tocar el bloque `persist` (dueno: fe/B; regiones disjuntas del archivo).

### 6. i18n (3 claves nuevas, it+es+en en sync)
- `chat.tool_input` (Input/Entrada), `chat.tool_output` (Output/Salida), `chat.tool_details` (aria con interpolacion `{tool}`).
- `pnpm i18n:check` verde. (Las claves de fe/A `chat.reconnecting`/`errors.network`/`errors.stream_interrupted` y `chat.expand_tool`/`collapse_tool` del plan NO eran necesarias aqui: el toggle usa `<details>` nativo con un solo `tool_details`; si fe/A o el subtask i18n las agrega, no colisionan.)

## Verificacion XSS (load-bearing, runtime Node)
```
render('<img src=x onerror=alert(1)>')  -> NO contiene `onerror`
render('<script>alert(1)</script>')     -> NO contiene `<script`
render(tabla GFM)                        -> contiene `<table>`
render('```python ...```')               -> contiene `<pre><code`
```
Todos PASS. El test de componente jsdom (tests/component/MessageBubble.test.ts del plan §6) debe montar MessageBubble con un mensaje assistant y assertear que el DOM montado no contiene `onerror` ni `<script>`.

## Pendientes / coordinacion
- Tests Vitest del plan §6 (parser SSE, retry, render markdown+XSS, ToolActivity colapsable) son write-set aparte; este handoff deja la implementacion lista para testear con frames REALES del contrato.
- `pnpm lint` (ESLint v9) sigue roto por bug pre-existente sin `eslint.config.js` (memoria; no es US-057).
- Warnings IDE de "canonical classes" (`rounded-[var(--radius-sm)]` -> `rounded-sm`): se mantuvo el estilo del archivo original (mismas clases que ya usaba `ToolActivity.vue`); son sugerencias, no errores.
