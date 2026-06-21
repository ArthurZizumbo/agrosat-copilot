<script setup lang="ts">
// MapLibre canvas — thin component shell around the useMap composable (US-058).
//
// The imperative map logic (init/destroy, overlay sources/layers, parcel
// interactions, rectangle draw, basemap switch, flyTo helpers) lives in
// composables/useMap.ts. This component only:
//  - owns the <template> and the chip overlays (toolbar/legend/coords)
//  - draws the rubber-band rectangle from useMap's `drawRect` ref
//  - drives the map lifecycle (initMap in onMounted, destroyMap in onBeforeUnmount)
//  - registers the imperative API (flyToDemoAoi/locateParcel) for the layout
//
// SSR-safe: initMap dynamically imports maplibre-gl and is called only under
// import.meta.client; cleanup via useMap.destroyMap in onBeforeUnmount.

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";

const { t } = useI18n();
const chatStore = useChatStore();
const mapStore = useMapStore();
const { findings } = storeToRefs(chatStore);
const { activeAoi, selectedParcel } = storeToRefs(mapStore);

// Clear the selected-parcel highlight/chip AND the chat-side active parcel id
// together (they are set together on click; keep them in lockstep).
function clearParcelSelection() {
  mapStore.clearSelectedParcel();
  chatStore.setActiveParcelId(null);
}

// MapCanvas registers its imperative API into the ref provided by the layout
// (see layouts/default.vue) so the sidebar "demo area" and FindingCard
// "view on map" can drive the map across the component tree.
interface MapApi {
  flyToDemoAoi: () => void;
  locateParcel: (id: number) => void;
}
const mapApiRef = inject<Ref<MapApi | null> | null>("mapCanvas", null);

const mapContainer = ref<HTMLDivElement | null>(null);

// Cross-store link (US-058): a parcel click publishes its REAL parcel_id to two
// stores. The map store keeps the highlight/chip; the chat store keeps the id
// that the next POST /chat sends as `ChatRequest.parcel_id` (via useChat).
// useMap stays agnostic of the chat store (testable via this callback); the
// map-store<->chat-store coupling is decided here, in the component.
const { initMap, destroyMap, flyToDemoAoi, locateParcel, drawRect } = useMap({
  onParcelSelect: (parcelId, props) => {
    const cropClass = typeof props.crop_class === "string" ? props.crop_class : null;
    mapStore.setSelectedParcel({ parcel_id: parcelId, crop_class: cropClass });
    chatStore.setActiveParcelId(parcelId);
  },
});

defineExpose({ flyToDemoAoi, locateParcel });

// Register the imperative API for the layout/sidebar. Functions guard on the
// internal map ref, so registering during setup (before mount) is safe.
if (mapApiRef) mapApiRef.value = { flyToDemoAoi, locateParcel };

onMounted(async () => {
  if (import.meta.client && mapContainer.value) await initMap(mapContainer.value);
});

onBeforeUnmount(() => {
  destroyMap();
  if (mapApiRef) mapApiRef.value = null;
});
</script>

<template>
  <section class="relative h-full w-full" :aria-label="t('map.label')">
    <div ref="mapContainer" class="h-full w-full" />

    <!-- Rubber-band rectangle while drawing -->
    <div
      v-if="drawRect"
      class="pointer-events-none absolute rounded-[2px] border-2 border-dashed border-amber-cta-600 bg-amber-cta-500/15"
      :style="{
        left: `${drawRect.x}px`,
        top: `${drawRect.y}px`,
        width: `${drawRect.w}px`,
        height: `${drawRect.h}px`,
      }"
      aria-hidden="true"
    />

    <!-- Top-center: draw hint -->
    <div class="pointer-events-none absolute left-1/2 top-3 z-10 -translate-x-1/2">
      <MapDrawToolbar />
    </div>

    <!-- Top-left: legend -->
    <div class="pointer-events-none absolute left-3 top-3 z-10 max-w-[14rem]">
      <MapCropLegend />
    </div>

    <!-- Bottom-right: cursor coords -->
    <div class="pointer-events-none absolute bottom-9 right-3 z-10">
      <MapCoordsReadout />
    </div>

    <!-- Top-right (below nav controls): selected parcel -> chat context -->
    <div
      v-if="selectedParcel"
      class="absolute right-3 top-24 z-10 flex max-w-[15rem] items-center gap-2 rounded-full border border-[var(--color-border)] bg-[var(--color-surface)]/90 px-3 py-1.5 text-xs text-[var(--color-fg)] shadow-[var(--shadow-panel)] backdrop-blur"
      :title="t('map.parcel_in_chat')"
    >
      <span class="truncate">
        <strong>{{ t("map.selected_parcel") }}:</strong>
        {{ t("chat.parcel") }} {{ selectedParcel.parcel_id }}
        <span v-if="selectedParcel.crop_class" class="text-[var(--color-muted-fg)]">
          · {{ selectedParcel.crop_class }}
        </span>
      </span>
      <button
        type="button"
        class="shrink-0 rounded-full px-1.5 text-[var(--color-muted-fg)] hover:text-[var(--color-fg)]"
        :aria-label="t('map.clear_selection')"
        :title="t('map.clear_selection')"
        @click="clearParcelSelection"
      >
        ✕
      </button>
    </div>

    <!-- Empty hint -->
    <div
      v-if="findings.length === 0 && !activeAoi"
      class="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2 rounded-full border border-[var(--color-border)] bg-[var(--color-surface)]/90 px-3 py-1.5 text-xs text-[var(--color-muted-fg)] shadow-[var(--shadow-panel)] backdrop-blur"
    >
      {{ t("map.empty") }}
    </div>
  </section>
</template>
