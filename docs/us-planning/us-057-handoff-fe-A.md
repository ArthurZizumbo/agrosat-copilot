# US-057 fe/A — Handoff: retry+backoff+locale en `useChat`

**Rama**: `feature/E9-US-057-chatpanel` · **Write-set**: `frontend/composables/useChat.ts` (solo este archivo).
**Estado**: HECHO. `pnpm typecheck` limpio (exit 0, sin `error TS`).

## Que se implemento

### 1. Retry + backoff exponencial con full jitter (D3)
- Helper interno `streamWithRetry(body, signal)` envuelve `runStreamAttempt` (fetch + `readStream`).
- Constantes: `MAX_RETRIES=3`, `BASE_DELAY_MS=500`, `MAX_DELAY_MS=8000`.
- `backoffDelayMs(attempt)`: `min(base * 2**attempt, cap)` y full jitter `delay*(0.5 + random*0.5)`.
- `delay(ms, signal)` cancelable por `AbortSignal` (el backoff aborta si el usuario hace stop/dispose).
- Logging con `console.warn` (no hay structlog en frontend); mensajes por intento y al agotar reintentos.

### 2. Clasificacion transitorio vs fatal
- **Transitorio (reintenta)**:
  - `TypeError` / network error del `fetch` (no es `HttpStatusError` ni `AbortError`, cae al branch transitorio).
  - `res.status >= 500` (502/503/504): se lanza `HttpStatusError(status)`; el loop reintenta si `status >= 500`.
  - Corte del reader (`reader.read()` lanza) ANTES de cualquier evento terminal -> `StreamCutError(afterDelta=false)`.
  - Stream que termina limpio sin terminal y SIN deltas -> `StreamCutError(false)` (fallo temprano).
- **Fatal (no reintenta)**:
  - `AbortError` (stop/dispose) -> retorno silencioso, sin marcar error.
  - `res.status` 4xx (incluido 422 contrato `extra="forbid"`) -> `failTransport("http_<status>")`.
  - Evento `error` del backend -> es respuesta valida del agente; `readStream` lo trata como terminal, NO lanza, no reintenta.
  - Stream que ya emitio `done` -> terminal, retorna normal.

### 3. Idempotencia (D4 / R4)
- Flag `receivedDelta` rastreado dentro de `readStream` y propagado entre intentos via `receivedDeltaEver` en `streamWithRetry`.
- Si llego >=1 `text_delta` y luego se corta el stream -> `StreamCutError(afterDelta=true)` -> NO se reintenta (evita texto duplicado) -> `store.failTransport("stream_interrupted")`.
- Si el stream termina limpio post-delta sin terminal -> se emite `{type:"done"}` para asentar el turno (la respuesta ya esta en el store).

### 4. Parser SSE conservado y exportado
- `parseSseFrame` y el splitter de frames (`\n\n`, normalizacion CRLF, flush de cola) **intactos** — es el contrato real testeado.
- `parseSseFrame` ahora es **named export** para que `tests/unit/sse-parser.test.ts` use la implementacion real (no reimplementacion). REGLA ARTHUR: frames reales del contrato.

### 5. Locale en el payload (D4)
- `const { locale } = useI18n()` en `useChat()`.
- `body.locale = locale.value` (`'it'|'es'|'en'`), enviado solo en cliente (`import.meta.client` ya guarda `sendMessage`).
- Tipado local `ChatRequestWithLocale = ChatRequest & { locale?: 'it'|'es'|'en' }` para no tocar `types/agent.ts` (write-set disjunto; fe que corresponda agrega `locale?` a la interfaz `ChatRequest`).

## BACKEND PENDING (cambio requerido, R5) — `locale_needs_backend_change=true`
`ChatRequest` en `backend/app/services/chat_service.py` tiene `model_config = ConfigDict(extra="forbid")`.
**Hoy enviar `locale` provoca HTTP 422.** Se requiere:
1. Agregar a `ChatRequest`: `locale: Literal["it","es","en"] | None = None`.
2. Inyectar instruccion de idioma al reasoner en `_agent_messages` (o el prompt del reasoner).
3. (Frontend hermano) agregar `locale?: "it"|"es"|"en"` a la interfaz `ChatRequest` en `frontend/types/agent.ts`.

Mientras tanto: el front YA envia `locale`. Si el backend aun no lo acepta, **devolvera 422**, que esta clasificado como FATAL (no reintenta, `failTransport("http_422")`). Por eso el cambio backend debe ir en el MISMO PR (es ~3 lineas) o desactivar el envio de `locale` hasta aplicarlo. Recomendado: incluir el cambio backend en el PR.

## Notas para tests (fe que escribe tests)
- Mock `globalThis.fetch`: 1er intento `throw new TypeError("network")`, 2do un `Response` con `ReadableStream` de frames reales (`text_delta`+`done`) -> reintenta y completa.
- 4 fallos -> `failTransport("network_error")`, `status==="error"`.
- `res.status===422` -> `failTransport("http_422")` inmediato (sin retry).
- `AbortError` -> sin retry, sin error.
- Corte tras `text_delta` -> `failTransport("stream_interrupted")` (sin retry).
- Backoff: fake timers, delays crecientes con cap 8000ms y jitter.
