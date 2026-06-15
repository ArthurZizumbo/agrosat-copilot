<script setup lang="ts">
// Floating draw hint shown over the map while AOI-draw mode is armed.

import { storeToRefs } from "pinia";
import { useMapStore } from "~/stores/map";

const { t } = useI18n();
const store = useMapStore();
const { drawMode } = storeToRefs(store);
</script>

<template>
  <div
    v-if="drawMode"
    class="pointer-events-auto flex items-center gap-2 rounded-full border border-amber-cta-600/40 bg-amber-cta-50 px-3 py-1.5 text-xs font-medium text-amber-cta-700 shadow-[var(--shadow-pop)] dark:bg-amber-cta-600/15 dark:text-amber-cta-300"
    role="status"
  >
    <UIcon name="i-lucide-square-dashed-mouse-pointer" class="size-4" aria-hidden="true" />
    {{ t("tools.draw_hint") }}
    <button
      type="button"
      class="ml-1 inline-flex items-center gap-1 rounded-full bg-amber-cta-600/20 px-2 py-0.5"
      @click="store.setDrawMode(false)"
    >
      {{ t("tools.cancel") }}
      <kbd class="font-mono text-[0.625rem]">Esc</kbd>
    </button>
  </div>
</template>
