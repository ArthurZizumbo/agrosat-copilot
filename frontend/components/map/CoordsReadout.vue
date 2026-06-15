<script setup lang="ts">
// Floating cursor coordinate readout (lat, lng) in mono tabular-nums.

import { storeToRefs } from "pinia";
import { useMapStore } from "~/stores/map";

const { t } = useI18n();
const store = useMapStore();
const { cursorCoords } = storeToRefs(store);
</script>

<template>
  <div
    v-if="cursorCoords"
    class="pointer-events-none rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)]/90 px-2 py-1 font-mono text-[0.6875rem] tabular-nums text-[var(--color-muted-fg)] shadow-[var(--shadow-panel)] backdrop-blur"
  >
    <span class="sr-only">{{ t("map.coords") }}: </span>
    {{ cursorCoords.lat.toFixed(4) }}, {{ cursorCoords.lng.toFixed(4) }}
  </div>
</template>
