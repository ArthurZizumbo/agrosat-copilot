<script setup lang="ts">
// Horizontal bar chart of per-class crop probabilities (top-8, descending).
//
// Renders with vue-echarts (VChart) registering ONLY the modules we need
// (BarChart + Grid + Tooltip + Canvas) to keep the bundle small. The predicted
// class (`cropClass`) is highlighted with the agro accent; the ground-truth
// class (`trueClass`), when present, is marked with a "truth" annotation and a
// bordered bar so the user can see at a glance whether the model was right.
//
// SSR-safe: ECharts touches `window`, so the chart is only mounted on the
// client (guarded by `import.meta.client` plus a <ClientOnly> wrapper) and the
// module registration happens at module scope on the client only.

import { use } from "echarts/core";
import { BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import VChart from "vue-echarts";
import type { EChartsOption } from "echarts";
import { colorForCrop } from "~/utils/cropPalette";

// Register the minimal set of ECharts modules (client-side only). On the server
// `import.meta.client` is false, so we skip registration and rely on
// <ClientOnly> to avoid mounting VChart during SSR.
if (import.meta.client) {
  use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);
}

const props = defineProps<{
  /** Per-class posterior probabilities in [0, 1], keyed by crop label. */
  classProbabilities: Record<string, number>;
  /** The predicted class (highlighted in the chart). */
  cropClass?: string | null;
  /** The ground-truth class, when known (annotated as "truth"). */
  trueClass?: string | null;
}>();

const { t } = useI18n();

const MAX_BARS = 8;

/** Resolve the agro accent colour from the design-system CSS variable, with a
 *  static fallback for SSR/first paint where the variable is not yet readable. */
function accentColor(): string {
  if (import.meta.client) {
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue("--color-agro-600")
      .trim();
    if (value) return value;
  }
  return "#16a34a";
}

/** Top-N classes by probability, ascending so ECharts paints the largest on
 *  top (its y-axis grows upward). */
const ranked = computed(() => {
  const entries = Object.entries(props.classProbabilities ?? {})
    .filter(([, p]) => typeof p === "number" && Number.isFinite(p))
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_BARS);
  // Ascending for the y-axis (bottom-to-top render order).
  return entries.reverse();
});

const hasData = computed(() => ranked.value.length > 0);

const chartOption = computed<EChartsOption>(() => {
  const accent = accentColor();
  const muted = "#94a3a0";
  const labels = ranked.value.map(([crop]) => crop);
  const values = ranked.value.map(([, p]) => p);

  return {
    grid: { left: 4, right: 36, top: 4, bottom: 4, containLabel: true },
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params: unknown) => {
        const p = params as { name: string; value: number };
        const pct = (p.value * 100).toFixed(1);
        const isTrue = props.trueClass != null && p.name === props.trueClass;
        const truthTag = isTrue ? ` (${t("map.true_class")})` : "";
        return `${p.name}${truthTag}: ${pct}%`;
      },
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      show: false,
    },
    yAxis: {
      type: "category",
      data: labels,
      axisTick: { show: false },
      axisLine: { show: false },
      axisLabel: {
        color: muted,
        fontSize: 11,
        fontFamily: "Inter, ui-sans-serif, sans-serif",
        width: 96,
        overflow: "truncate",
      },
    },
    series: [
      {
        type: "bar",
        data: values.map((value, i) => {
          const crop = labels[i];
          const isPred =
            props.cropClass != null && crop === props.cropClass;
          const isTrue =
            props.trueClass != null && crop === props.trueClass;
          return {
            value,
            itemStyle: {
              // Predicted class uses the accent; others use the crop's own
              // palette colour (kept subtle via opacity).
              color: isPred ? accent : colorForCrop(crop),
              opacity: isPred ? 1 : 0.55,
              borderRadius: [0, 3, 3, 0],
              // Ground-truth class gets a dashed border so a wrong prediction
              // is visible at a glance (the truth bar is outlined).
              borderColor: isTrue ? accent : "transparent",
              borderWidth: isTrue ? 1.5 : 0,
              borderType: "dashed",
            },
          };
        }),
        barWidth: "62%",
        label: {
          show: true,
          position: "right",
          formatter: (params: unknown) => {
            const p = params as { value: number };
            return `${Math.round(p.value * 100)}%`;
          },
          color: muted,
          fontSize: 10,
          fontFamily: "JetBrains Mono, ui-monospace, monospace",
        },
      },
    ],
  };
});

/** Chart height scales with the number of bars (compact when few classes). */
const chartHeight = computed(() => `${Math.max(64, ranked.value.length * 22)}px`);
</script>

<template>
  <div v-if="hasData" class="w-full">
    <ClientOnly>
      <VChart
        class="w-full"
        :style="{ height: chartHeight }"
        :option="chartOption"
        :autoresize="true"
        role="img"
        :aria-label="t('chat.probabilities')"
      />
      <template #fallback>
        <div
          class="animate-pulse rounded-[var(--radius-sm)] bg-[var(--color-surface-2)]"
          :style="{ height: chartHeight }"
          aria-hidden="true"
        />
      </template>
    </ClientOnly>
  </div>
</template>
