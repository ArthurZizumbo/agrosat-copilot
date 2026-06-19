<script setup lang="ts">
// Segmented basemap switcher (Satellite / Streets / Topo). Reads + writes the
// map store basemap; the MapCanvas watches the store and calls setStyle.

import { storeToRefs } from "pinia";
import { useMapStore } from "~/stores/map";
import type { BasemapId } from "~/types/map";

const { t } = useI18n();
const store = useMapStore();
const { basemap } = storeToRefs(store);

const options: { value: BasemapId; icon: string; labelKey: string }[] = [
  { value: "satellite", icon: "i-lucide-satellite", labelKey: "basemap.satellite" },
  { value: "streets", icon: "i-lucide-map", labelKey: "basemap.streets" },
  { value: "topo", icon: "i-lucide-mountain", labelKey: "basemap.topo" },
];
</script>

<template>
  <div
    class="inline-flex items-center gap-0.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] p-0.5"
    role="group"
    :aria-label="t('basemap.label')"
  >
    <button
      v-for="opt in options"
      :key="opt.value"
      type="button"
      class="inline-flex min-h-9 items-center gap-1.5 rounded-[var(--radius-sm)] px-2.5 text-xs font-medium transition-colors duration-150"
      :class="
        basemap === opt.value
          ? 'bg-[var(--color-surface)] text-[var(--color-fg)] shadow-[var(--shadow-panel)]'
          : 'text-[var(--color-muted-fg)] hover:text-[var(--color-fg)]'
      "
      :aria-pressed="basemap === opt.value"
      :title="t(opt.labelKey)"
      @click="store.setBasemap(opt.value)"
    >
      <UIcon :name="opt.icon" class="size-4" aria-hidden="true" />
      <span class="hidden sm:inline">{{ t(opt.labelKey) }}</span>
    </button>
  </div>
</template>
