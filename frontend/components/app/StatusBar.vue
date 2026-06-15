<script setup lang="ts">
// Bottom status bar: session id (mono, truncated), connection state, active
// LLM, cursor coords, parcel count. All numbers use tabular-nums.

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";

const { t } = useI18n();
const chatStore = useChatStore();
const mapStore = useMapStore();
const { status, llmVariant } = storeToRefs(chatStore);
const { cursorCoords, parcelCount } = storeToRefs(mapStore);
const { sessionId } = useSession();

const conn = computed(() => {
  if (status.value === "error") return { cls: "bg-state-danger", key: "ws.error" };
  if (status.value === "streaming") return { cls: "bg-state-success", key: "ws.live" };
  if (status.value === "dispatching") return { cls: "bg-amber-cta-500", key: "ws.connecting" };
  return { cls: "bg-[var(--color-border-strong)]", key: "ws.idle" };
});

const shortSession = computed(() => {
  const sid = sessionId.value;
  if (!sid) return "—";
  return sid.length > 12 ? `${sid.slice(0, 8)}…${sid.slice(-4)}` : sid;
});

const llmLabel = computed(() => (llmVariant.value === "gemini" ? "Gemini" : "Qwen"));
</script>

<template>
  <footer
    class="flex h-8 items-center gap-3 overflow-hidden border-t border-[var(--color-border)] bg-[var(--color-surface)] px-3 text-[0.6875rem] text-[var(--color-muted-fg)]"
    :aria-label="t('status.label')"
  >
    <span class="flex items-center gap-1.5">
      <UIcon name="i-lucide-hash" class="size-3" aria-hidden="true" />
      <span class="font-mono tabular-nums">{{ shortSession }}</span>
    </span>

    <span class="hidden items-center gap-1.5 sm:flex">
      <span class="size-1.5 rounded-full" :class="conn.cls" aria-hidden="true" />
      {{ t(conn.key) }}
    </span>

    <span class="hidden items-center gap-1.5 md:flex">
      <UIcon name="i-lucide-cpu" class="size-3" aria-hidden="true" />
      {{ llmLabel }}
    </span>

    <span class="ml-auto flex items-center gap-1.5">
      <UIcon name="i-lucide-map-pin" class="size-3" aria-hidden="true" />
      <span class="font-mono tabular-nums">{{ parcelCount }}</span>
      <span class="hidden sm:inline">{{ t("status.parcels") }}</span>
    </span>

    <span v-if="cursorCoords" class="hidden items-center gap-1.5 lg:flex">
      <UIcon name="i-lucide-crosshair" class="size-3" aria-hidden="true" />
      <span class="font-mono tabular-nums">
        {{ cursorCoords.lat.toFixed(3) }}, {{ cursorCoords.lng.toFixed(3) }}
      </span>
    </span>
  </footer>
</template>
