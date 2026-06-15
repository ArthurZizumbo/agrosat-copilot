<script setup lang="ts">
// Left sidebar: Tools (draw AOI, demo AOI, clear), Layers (parcels toggle +
// crop legend), AOIs (list from GET /aois). Collapsible to an icon rail >=xl.
//
// Emits "demo" so the page can ask the map to fly to the seeded Toscana AOI.

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";
import { buildCropLegend } from "~/utils/cropPalette";

const props = defineProps<{ collapsed?: boolean }>();
const emit = defineEmits<{
  (e: "demo"): void;
  (e: "toggle-collapse"): void;
}>();

const { t } = useI18n();
const mapStore = useMapStore();
const chatStore = useChatStore();
const { aois, activeAoi, drawMode, parcelsVisible } = storeToRefs(mapStore);
const { findings } = storeToRefs(chatStore);
const { listAois } = useAoi();

const legend = computed(() => buildCropLegend(findings.value.map((f) => f.crop_class)));
const loadingAois = ref(false);

async function refreshAois() {
  loadingAois.value = true;
  await listAois();
  loadingAois.value = false;
}

function startDraw() {
  mapStore.setDrawMode(!drawMode.value);
}
function selectAoi(id: number) {
  const aoi = aois.value.find((a) => a.id === id);
  if (aoi) mapStore.setActiveAoi(aoi);
}

onMounted(() => {
  if (import.meta.client) refreshAois();
});

const rail = computed(() => props.collapsed);
</script>

<template>
  <aside
    class="flex h-full flex-col border-r border-[var(--color-border)] bg-[var(--color-surface)]"
    :class="rail ? 'w-14 items-center' : 'w-[17.5rem]'"
    :aria-label="t('nav.tools')"
  >
    <!-- Collapse toggle (xl only) -->
    <div class="hidden items-center justify-end p-2 xl:flex" :class="rail ? 'justify-center' : ''">
      <button
        type="button"
        class="inline-flex size-8 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)]"
        :aria-label="t('nav.collapse')"
        @click="emit('toggle-collapse')"
      >
        <UIcon
          :name="rail ? 'i-lucide-chevrons-right' : 'i-lucide-chevrons-left'"
          class="size-4"
          aria-hidden="true"
        />
      </button>
    </div>

    <div class="flex-1 space-y-4 overflow-y-auto p-2">
      <!-- Tools -->
      <section>
        <p v-if="!rail" class="px-1.5 pb-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-[var(--color-muted-fg)]">
          {{ t("tools.title") }}
        </p>
        <div class="space-y-1">
          <button
            type="button"
            class="flex w-full items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm font-medium transition-colors"
            :class="
              drawMode
                ? 'bg-amber-cta-50 text-amber-cta-700 dark:bg-amber-cta-600/15 dark:text-amber-cta-300'
                : 'text-[var(--color-fg)] hover:bg-[var(--color-surface-2)]'
            "
            :title="t('tools.draw')"
            :aria-pressed="drawMode"
            @click="startDraw"
          >
            <UIcon name="i-lucide-square-dashed-mouse-pointer" class="size-4 shrink-0" aria-hidden="true" />
            <span v-if="!rail">{{ t("tools.draw") }}</span>
          </button>
          <button
            type="button"
            class="flex w-full items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm font-medium text-[var(--color-fg)] transition-colors hover:bg-[var(--color-surface-2)]"
            :title="t('tools.demo')"
            @click="emit('demo')"
          >
            <UIcon name="i-lucide-map-pinned" class="size-4 shrink-0" aria-hidden="true" />
            <span v-if="!rail">{{ t("tools.demo") }}</span>
          </button>
          <button
            type="button"
            class="flex w-full items-center gap-2.5 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm font-medium text-[var(--color-fg)] transition-colors hover:bg-[var(--color-surface-2)] disabled:opacity-40"
            :title="t('tools.clear')"
            :disabled="!activeAoi"
            @click="mapStore.clearSelection()"
          >
            <UIcon name="i-lucide-eraser" class="size-4 shrink-0" aria-hidden="true" />
            <span v-if="!rail">{{ t("tools.clear") }}</span>
          </button>
        </div>
      </section>

      <!-- Layers -->
      <section v-if="!rail">
        <p class="px-1.5 pb-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-[var(--color-muted-fg)]">
          {{ t("layers.title") }}
        </p>
        <button
          type="button"
          class="flex w-full items-center justify-between gap-2 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm text-[var(--color-fg)] hover:bg-[var(--color-surface-2)]"
          :aria-pressed="parcelsVisible"
          @click="mapStore.toggleParcels()"
        >
          <span class="flex items-center gap-2.5">
            <UIcon name="i-lucide-layers" class="size-4" aria-hidden="true" />
            {{ t("layers.parcels") }}
          </span>
          <UIcon
            :name="parcelsVisible ? 'i-lucide-eye' : 'i-lucide-eye-off'"
            class="size-4 text-[var(--color-muted-fg)]"
            aria-hidden="true"
          />
        </button>
        <ul v-if="legend.length > 0" class="mt-1.5 space-y-1 px-2.5">
          <li
            v-for="entry in legend"
            :key="entry.crop"
            class="flex items-center gap-2 text-xs text-[var(--color-muted-fg)]"
          >
            <span class="size-3 shrink-0 rounded-[3px]" :style="{ backgroundColor: entry.color }" aria-hidden="true" />
            <span class="truncate">{{ entry.crop }}</span>
          </li>
        </ul>
      </section>

      <!-- AOIs -->
      <section v-if="!rail">
        <div class="flex items-center justify-between px-1.5 pb-1.5">
          <p class="text-[0.6875rem] font-semibold uppercase tracking-wide text-[var(--color-muted-fg)]">
            {{ t("aoi.title") }}
          </p>
          <button
            type="button"
            class="inline-flex size-6 items-center justify-center rounded-[var(--radius-xs)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)]"
            :aria-label="t('aoi.refresh')"
            @click="refreshAois"
          >
            <UIcon name="i-lucide-refresh-cw" class="size-3.5" :class="loadingAois ? 'animate-spin' : ''" aria-hidden="true" />
          </button>
        </div>

        <!-- Skeletons -->
        <div v-if="loadingAois && aois.length === 0" class="space-y-1.5 px-1.5" aria-hidden="true">
          <div class="h-9 animate-pulse rounded-[var(--radius-sm)] bg-[var(--color-surface-2)]" />
          <div class="h-9 animate-pulse rounded-[var(--radius-sm)] bg-[var(--color-surface-2)]" />
        </div>

        <p
          v-else-if="aois.length === 0"
          class="px-2.5 py-2 text-xs text-[var(--color-muted-fg)]"
        >
          {{ t("aoi.empty") }}
        </p>

        <ul v-else class="space-y-1">
          <li v-for="aoi in aois" :key="aoi.id">
            <button
              type="button"
              class="flex w-full items-center justify-between gap-2 rounded-[var(--radius-sm)] px-2.5 py-2 text-left text-sm transition-colors"
              :class="
                activeAoi?.id === aoi.id
                  ? 'bg-agro-50 text-agro-800 dark:bg-agro-900/40 dark:text-agro-300'
                  : 'text-[var(--color-fg)] hover:bg-[var(--color-surface-2)]'
              "
              :aria-pressed="activeAoi?.id === aoi.id"
              @click="selectAoi(aoi.id)"
            >
              <span class="flex min-w-0 items-center gap-2">
                <UIcon name="i-lucide-map-pin" class="size-3.5 shrink-0" aria-hidden="true" />
                <span class="truncate">{{ aoi.label ?? `${t("aoi.title")} #${Math.abs(aoi.id)}` }}</span>
              </span>
              <span v-if="aoi.area_ha != null" class="shrink-0 font-mono text-[0.6875rem] tabular-nums text-[var(--color-muted-fg)]">
                {{ aoi.area_ha.toFixed(1) }} ha
              </span>
            </button>
          </li>
        </ul>
      </section>
    </div>
  </aside>
</template>
