<script setup lang="ts">
// Floating, collapsible crop legend. Reads distinct crops from chat findings
// and renders swatches via the shared palette util.

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import { buildCropLegend } from "~/utils/cropPalette";

const { t } = useI18n();
const store = useChatStore();
const { findings } = storeToRefs(store);

const open = ref(true);
const entries = computed(() =>
  buildCropLegend(findings.value.map((f) => f.crop_class)),
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
        {{ t("map.legend") }}
      </span>
      <UIcon
        :name="open ? 'i-lucide-chevron-down' : 'i-lucide-chevron-up'"
        class="size-4 text-[var(--color-muted-fg)]"
        aria-hidden="true"
      />
    </button>
    <ul v-if="open" class="space-y-1 px-2.5 pb-2">
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
</template>
