---
name: agrosat-frontend-composables
description: Create composables, Pinia stores, SSE clients, and middleware for AgroSatCopilot Nuxt 4 frontend. Use when implementing useChat, useSSE, useMap, useAOI, useLLMVariant, Pinia stores, route middleware, or auth guards.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Frontend Composables Skill

## Rules — NON-NEGOTIABLE

- Composables return reactive state + actions, never raw `ref` exports
- Pinia stores for cross-component state
- SSE composable handles reconnection, error, abort
- Auth middleware via Clerk Nuxt module
- TypeScript estricto; tipos en `types/`
- `import.meta.client` antes de browser APIs

## useChat with SSE Streaming

```typescript
// composables/useChat.ts
import type { ChatMessage, ChatEvent } from '~/types/chat'

export function useChat(sessionId: string) {
  const messages = ref<ChatMessage[]>([])
  const isStreaming = ref(false)
  const error = ref<string | null>(null)

  async function send(query: string, aoi: GeoJSON.Feature) {
    isStreaming.value = true
    error.value = null
    const llmStore = useLLMStore()

    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: query }
    messages.value.push(userMsg)
    const assistantMsg: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', content: '', toolCalls: [] }
    messages.value.push(assistantMsg)

    try {
      await fetchEventSource('/api/v1/chat', {
        method: 'POST',
        body: JSON.stringify({ query, aoi_geojson: aoi, llm_variant: llmStore.variant, session_id: sessionId }),
        onmessage(ev) {
          const event = JSON.parse(ev.data) as ChatEvent
          if (event.type === 'tool_call') assistantMsg.toolCalls!.push(event.data)
          else if (event.type === 'final_answer') assistantMsg.content = event.data.text
        },
        onerror(err) { error.value = err.message; throw err },
      })
    } finally {
      isStreaming.value = false
    }
  }

  return { messages, isStreaming, error, send }
}
```

## Pinia Stores

```typescript
// stores/llmStore.ts
import { defineStore } from 'pinia'

export const useLLMStore = defineStore('llm', {
  state: () => ({
    variant: 'gemini' as 'gemini' | 'qwen35',
    health: { gemini: 'ok', qwen35: 'unknown' },
  }),
  actions: {
    async switchTo(v: 'gemini' | 'qwen35') {
      await $fetch('/api/v1/llm/switch', { method: 'POST', body: { variant: v } })
      this.variant = v
    },
    async checkHealth() {
      this.health = await $fetch('/api/v1/llm/health')
    },
  },
  persist: true,
})
```

```typescript
// stores/aoiStore.ts
export const useAOIStore = defineStore('aoi', {
  state: () => ({
    current: null as GeoJSON.Feature | null,
    history: [] as GeoJSON.Feature[],
  }),
  actions: {
    setCurrent(aoi: GeoJSON.Feature) {
      this.current = aoi
      this.history.unshift(aoi)
    },
  },
})
```

## useMap (MapLibre wrapper)

```typescript
// composables/useMap.ts
import maplibregl from 'maplibre-gl'

export function useMap(containerId: string) {
  const map = shallowRef<maplibregl.Map | null>(null)
  const isReady = ref(false)

  onMounted(() => {
    if (!import.meta.client) return
    map.value = new maplibregl.Map({
      container: containerId,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [11.25, 43.5],  // Toscana
      zoom: 8,
    })
    map.value.on('load', () => { isReady.value = true })
  })

  onBeforeUnmount(() => {
    map.value?.remove()
  })

  return { map, isReady }
}
```

## Auth Middleware (Clerk)

```typescript
// middleware/auth.ts
export default defineNuxtRouteMiddleware((to) => {
  const { isSignedIn } = useUser()
  if (!isSignedIn.value && to.path !== '/login') {
    return navigateTo('/login')
  }
})
```

## SSE Helper

Usar `@microsoft/fetch-event-source` o implementación nativa:

```typescript
import { fetchEventSource } from '@microsoft/fetch-event-source'
```

## QA Checklist

- [ ] Composables retornan estado reactivo + acciones
- [ ] Pinia para estado cross-component
- [ ] SSE con reconnect + error handling
- [ ] Middleware auth con Clerk
- [ ] Tests con Vitest + happy-dom
