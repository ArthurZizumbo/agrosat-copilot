<script setup lang="ts">
// Floating, collapsible crop legend. Reads distinct crops from chat findings
// and renders swatches via the shared palette util.
//
// In the prediction demo (findings carry ground truth) it also exposes a
// predicted / true / hits-errors toggle and the parcel accuracy, so the map
// mirrors the notebook panels (predicted map, ground-truth map, error map).

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";
import {
  buildCropLegend,
  CORRECT_COLOR,
  ERROR_COLOR,
  type DemoView,
} from "~/utils/cropPalette";

const { t } = useI18n();
const store = useChatStore();
const mapStore = useMapStore();
const { findings, hasPrediction } = storeToRefs(store);
const { demoView, predictionAccuracy } = storeToRefs(mapStore);

const open = ref(true);

const views = computed<{ id: DemoView; label: string }[]>(() => [
  { id: "pred", label: t("map.view_predicted") },
  { id: "truth", label: t("map.view_true") },
  { id: "errors", label: t("map.view_errors") },
]);

const entries = computed(() => {
  if (hasPrediction.value && demoView.value === "errors") {
    return [
      { crop: t("map.correct"), color: CORRECT_COLOR },
      { crop: t("map.error"), color: ERROR_COLOR },
    ];
  }
  const useTrue = hasPrediction.value && demoView.value === "truth";
  return buildCropLegend(
    findings.value.map((f) => (useTrue ? f.true_class : f.crop_class)),
  );
});

const accuracyPct = computed(() =>
  predictionAccuracy.value != null
    ? `${Math.round(predictionAccuracy.value * 100)}%`
    : null,
);
</script>

<template>
  <div
    v-if="entries.length > 0"
    class="pointer-events-auto overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)]/95 shadow-[var(--shadow-pop)] backdrop-blur"
  >
    <button
      type="button"
      class="flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-xs font-semibold text-[var(--color-fg)]"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span class="flex items-center gap-1.5">
        <UIcon name="i-lucide-palette" class="size-3.5" aria-hidden="true" />
        {{ hasPrediction ? t("map.prediction_legend") : t("map.legend") }}
      </span>
      <UIcon
        :name="open ? 'i-lucide-chevron-down' : 'i-lucide-chevron-up'"
        class="size-4 text-[var(--color-muted-fg)]"
        aria-hidden="true"
      />
    </button>

    <div v-if="open" class="px-2.5 pb-2">
      <!-- Prediction demo: view toggle + accuracy -->
      <template v-if="hasPrediction">
        <div
          class="mb-2 inline-flex rounded-[var(--radius-sm)] border border-[var(--color-border)] p-0.5 text-[11px]"
          role="group"
          :aria-label="t('map.view_label')"
        >
          <button
            v-for="v in views"
            :key="v.id"
            type="button"
            class="rounded-[calc(var(--radius-sm)-2px)] px-1.5 py-0.5 transition-colors"
            :class="
              demoView === v.id
                ? 'bg-[var(--color-primary)] text-white'
                : 'text-[var(--color-muted-fg)] hover:text-[var(--color-fg)]'
            "
            :aria-pressed="demoView === v.id"
            @click="mapStore.setDemoView(v.id)"
          >
            {{ v.label }}
          </button>
        </div>
        <p
          v-if="accuracyPct"
          class="mb-1.5 text-[11px] text-[var(--color-muted-fg)]"
        >
          {{ t("map.accuracy") }}:
          <strong class="text-[var(--color-fg)] tabular-nums">{{ accuracyPct }}</strong>
        </p>
      </template>

      <ul class="space-y-1">
        <li
          v-for="entry in entries"
          :key="entry.crop"
          class="flex items-center gap-2 text-xs text-[var(--color-fg)]"
        >
          <span
            class="size-3 shrink-0 rounded-[3px]"
            :style="{ backgroundColor: entry.color }"
            aria-hidden="true"
          />
          <span class="truncate">{{ entry.crop }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>
