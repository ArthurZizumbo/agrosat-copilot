<script setup lang="ts">
// Chip showing the chat's active area context. Amber when a zone is selected,
// muted when none. Optional dismiss clears the selection.

import { storeToRefs } from "pinia";
import { useMapStore } from "~/stores/map";

defineProps<{ dismissible?: boolean }>();

const { t } = useI18n();
const store = useMapStore();
const { activeAoi } = storeToRefs(store);

const label = computed(() => {
  const aoi = activeAoi.value;
  if (!aoi) return t("area.none");
  return aoi.label ?? `${t("aoi.title")} #${Math.abs(aoi.id)}`;
});
</script>

<template>
  <div
    class="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium"
    :class="
      activeAoi
        ? 'border-amber-cta-600/40 bg-amber-cta-50 text-amber-cta-700 dark:bg-amber-cta-600/15 dark:text-amber-cta-300'
        : 'border-[var(--color-border)] bg-[var(--color-surface-2)] text-[var(--color-muted-fg)]'
    "
  >
    <UIcon
      :name="activeAoi ? 'i-lucide-map-pin' : 'i-lucide-map-pin-off'"
      class="size-3.5"
      aria-hidden="true"
    />
    <span class="truncate max-w-[12rem]">{{ t("area.active", { area: label }) }}</span>
    <span v-if="activeAoi?.area_ha != null" class="font-mono tabular-nums opacity-80">
      {{ activeAoi.area_ha.toFixed(1) }} ha
    </span>
    <button
      v-if="dismissible && activeAoi"
      type="button"
      class="ml-0.5 inline-flex size-4 items-center justify-center rounded-full hover:bg-amber-cta-600/20"
      :aria-label="t('tools.clear')"
      @click="store.clearSelection()"
    >
      <UIcon name="i-lucide-x" class="size-3" aria-hidden="true" />
    </button>
  </div>
</template>
