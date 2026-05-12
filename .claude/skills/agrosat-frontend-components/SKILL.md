---
name: agrosat-frontend-components
description: Create or modify Vue 3 / Nuxt 4 components for AgroSatCopilot. Use when building UI components (ChatPanel, MapView, LLMSwitch, AOIDrawer, ResultPanel, TimeSeriesChart), pages, layouts, Nuxt UI Pro elements, i18n-aware components, dark mode, or A11y.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot Frontend Components Skill

## Rules — NON-NEGOTIABLE

- `<script setup lang="ts">` with type hints obligatorios
- Texto visible al usuario solo via `t('key')` con `useI18n()` — sin strings hardcodeados
- Estado compartido en Pinia, jamás `reactive()` exportado entre archivos
- SSR-safe: `import.meta.client` antes de tocar `window`/`document`
- TailwindCSS utility-first; evitar `<style scoped>` salvo overrides
- Nuxt UI Pro como base (UButton, UCard, UModal, etc.)
- Dark mode soportado via `useColorMode()`
- A11y: focus visible, aria-labels, alt en imágenes

## Componentes Principales por Dominio

```
components/
├── chat/
│   ├── ChatPanel.vue            # Layout principal del chat
│   ├── ChatMessage.vue          # Burbuja individual con tool calls colapsables
│   ├── ChatInput.vue            # Input con submit, file attach
│   └── ToolCallTrace.vue        # Render de tool_call + tool_result events
├── map/
│   ├── MapView.vue              # MapLibre + deck.gl wrapper
│   ├── AOIDrawer.vue            # maplibre-gl-draw integration
│   └── LayerSwitcher.vue        # toggle Sentinel / NDVI / segmentación
├── llm/
│   └── LLMSwitch.vue            # Switch A (Gemini) / B (Qwen3.5)
├── results/
│   ├── ResultPanel.vue          # Panel derecho con clasificación
│   ├── TimeSeriesChart.vue      # ECharts NDVI por parcela
│   └── ParcelTable.vue          # Tabla con hectáreas, clase, confianza
└── common/
    ├── LangSwitcher.vue
    └── LoadingSkeleton.vue
```

## Component Pattern

```vue
<template>
  <UCard>
    <template #header>
      <h2 class="text-lg font-semibold">{{ t('chat.title') }}</h2>
    </template>
    <ChatMessage
      v-for="msg in messages"
      :key="msg.id"
      :message="msg"
      :tool-calls="msg.toolCalls"
    />
    <template #footer>
      <ChatInput @send="handleSend" :disabled="isStreaming" />
    </template>
  </UCard>
</template>

<script setup lang="ts">
import type { ChatMessage as ChatMsg } from '~/types/chat'

defineProps<{
  messages: ChatMsg[]
  isStreaming: boolean
}>()

const emit = defineEmits<{
  send: [text: string]
}>()

const { t } = useI18n()

function handleSend(text: string) {
  emit('send', text)
}
</script>
```

## Switch A/B LLM (Variante)

```vue
<template>
  <USelectMenu
    v-model="llmStore.variant"
    :options="variants"
    :ui="{ width: 'w-40' }"
    @change="handleSwitch"
  >
    <template #label>{{ t(`llm.variant.${llmStore.variant}`) }}</template>
  </USelectMenu>
</template>

<script setup lang="ts">
const llmStore = useLLMStore()
const { t } = useI18n()
const variants = [
  { value: 'gemini', label: 'Gemini 3.1 Pro (cloud)' },
  { value: 'qwen35', label: 'Qwen3.5-35B-A3B (on-prem)' },
]

async function handleSwitch(v: 'gemini' | 'qwen35') {
  await $fetch('/api/v1/llm/switch', { method: 'POST', body: { variant: v } })
}
</script>
```

## TimeSeriesChart with ECharts

```vue
<template>
  <VChart class="h-64" :option="option" autoresize />
</template>

<script setup lang="ts">
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent])

const props = defineProps<{
  data: { date: string; value: number }[]
  index: string  // NDVI, NDWI, EVI
}>()

const { t } = useI18n()

const option = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'time' },
  yAxis: { type: 'value', name: t(`charts.${props.index}`) },
  series: [{
    type: 'line',
    smooth: true,
    data: props.data.map(d => [d.date, d.value]),
  }],
}))
</script>
```

## i18n Keys Estructura

```
i18n/locales/{it,es,en}.json
{
  "chat": {
    "title": "...",
    "send": "...",
    "placeholder": "Ask about your parcel..."
  },
  "llm": {
    "variant": { "gemini": "...", "qwen35": "..." }
  },
  "charts": {
    "NDVI": "Indice di vegetazione",
    "NDWI": "Indice di acqua"
  }
}
```

**Las 3 locales (it/es/en) deben tener TODAS las keys idénticas.**
