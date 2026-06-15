<script setup lang="ts">
// Rich finding card: crop name + colour dot, confidence as a thin progress bar,
// NDVI / area in mono tabular-nums, and a "view on map" link that emits the
// parcel id for the map to fly to.

import type { Finding } from "~/types/agent";
import { colorForCrop } from "~/utils/cropPalette";

const props = defineProps<{ finding: Finding }>();
const emit = defineEmits<{ (e: "locate", parcelId: number): void }>();

const { t } = useI18n();

const cropColor = computed(() => colorForCrop(props.finding.crop_class));
const confidencePct = computed(() =>
  props.finding.confidence != null
    ? Math.round(props.finding.confidence * 100)
    : null,
);
const cropLabel = computed(
  () => props.finding.crop_class ?? t("map.crop_unknown"),
);
</script>

<template>
  <article
    class="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 shadow-[var(--shadow-panel)]"
  >
    <header class="flex items-center justify-between gap-2">
      <div class="flex min-w-0 items-center gap-2">
        <span
          class="size-3 shrink-0 rounded-full"
          :style="{ backgroundColor: cropColor }"
          aria-hidden="true"
        />
        <span class="truncate text-sm font-semibold text-[var(--color-fg)]">
          {{ cropLabel }}
        </span>
      </div>
      <span
        class="shrink-0 font-mono text-[0.6875rem] tabular-nums text-[var(--color-muted-fg)]"
      >
        {{ t("chat.parcel") }} #{{ finding.parcel_id }}
      </span>
    </header>

    <!-- Confidence -->
    <div v-if="confidencePct != null" class="mt-2.5">
      <div class="mb-1 flex items-center justify-between text-[0.6875rem]">
        <span class="text-[var(--color-muted-fg)]">{{ t("map.confidence") }}</span>
        <span class="font-mono tabular-nums font-medium text-[var(--color-fg)]">
          {{ confidencePct }}%
        </span>
      </div>
      <div
        class="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-2)]"
        role="progressbar"
        :aria-valuenow="confidencePct"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-label="t('map.confidence')"
      >
        <div
          class="h-full rounded-full bg-agro-600 transition-[width] duration-300"
          :style="{ width: `${confidencePct}%` }"
        />
      </div>
    </div>

    <!-- Metrics -->
    <dl class="mt-2.5 grid grid-cols-2 gap-2 text-xs">
      <div class="flex flex-col">
        <dt class="text-[0.6875rem] text-[var(--color-muted-fg)]">{{ t("map.ndvi") }}</dt>
        <dd class="font-mono tabular-nums font-medium text-[var(--color-fg)]">
          {{ finding.ndvi_mean != null ? finding.ndvi_mean.toFixed(2) : "—" }}
        </dd>
      </div>
      <div class="flex flex-col">
        <dt class="text-[0.6875rem] text-[var(--color-muted-fg)]">{{ t("map.area") }}</dt>
        <dd class="font-mono tabular-nums font-medium text-[var(--color-fg)]">
          {{ finding.area_ha != null ? `${finding.area_ha.toFixed(1)} ha` : "—" }}
        </dd>
      </div>
    </dl>

    <footer
      class="mt-2.5 flex items-center justify-between gap-2 border-t border-[var(--color-border)] pt-2"
    >
      <span class="truncate text-[0.625rem] text-[var(--color-muted-fg)]">
        {{ finding.citation.source }}
      </span>
      <button
        type="button"
        class="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-agro-700 hover:underline dark:text-agro-400"
        @click="emit('locate', finding.parcel_id)"
      >
        <UIcon name="i-lucide-locate" class="size-3.5" aria-hidden="true" />
        {{ t("chat.view_on_map") }}
      </button>
    </footer>
  </article>
</template>
