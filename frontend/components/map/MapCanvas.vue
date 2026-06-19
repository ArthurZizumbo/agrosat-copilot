<script setup lang="ts">
// MapLibre canvas — refactor of the legacy MapView.vue.
//
// Responsibilities:
//  - render a keyless raster basemap (satellite/streets/topo), switchable live
//  - paint parcels from chat `findings`, coloured by crop, hover-highlight + popup
//  - draw a rectangle AOI (mousedown->mousemove->mouseup) -> active chat zone
//  - render the active AOI (amber dashed border + translucent fill)
//  - publish cursor coords + parcel count to the map store
//  - expose flyTo helpers (demo AOI, locate a parcel) to the parent
//
// SSR-safe: maplibre-gl is dynamically imported inside onMounted (client only);
// cleanup in onBeforeUnmount. Crop colours come from the shared palette util.

import { storeToRefs } from "pinia";
import type {
  Map as MlMap,
  GeoJSONSource,
  MapGeoJSONFeature,
  MapMouseEvent,
} from "maplibre-gl";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";
import type { Finding } from "~/types/agent";
import type { Aoi, AoiPolygon, LngLat } from "~/types/map";
import { colorForCrop } from "~/utils/cropPalette";
import { buildBasemapStyle } from "~/composables/useBasemap";
import { DEMO_AOI_BBOX, demoAoiPolygon, demoFindings } from "~/utils/demoPreview";

const { t } = useI18n();
const chatStore = useChatStore();
const mapStore = useMapStore();
const { findings } = storeToRefs(chatStore);
const { basemap, drawMode, parcelsVisible, activeAoi } = storeToRefs(mapStore);
const { selectDrawnAoi, rectToPolygon } = useAoi();

// MapCanvas registers its imperative API into the ref provided by the layout
// (see layouts/default.vue) so the sidebar "demo area" and FindingCard
// "view on map" can drive the map across the component tree.
interface MapApi {
  flyToDemoAoi: () => void;
  locateParcel: (id: number) => void;
}
const mapApiRef = inject<Ref<MapApi | null> | null>("mapCanvas", null);

const mapContainer = ref<HTMLDivElement | null>(null);
const drawRect = ref<{ x: number; y: number; w: number; h: number } | null>(null);

const FINDINGS_SOURCE = "findings";
const FINDINGS_FILL = "findings-fill";
const FINDINGS_LINE = "findings-line";
const FINDINGS_HL = "findings-highlight";
const AOI_SOURCE = "active-aoi";
const AOI_FILL = "active-aoi-fill";
const AOI_LINE = "active-aoi-line";

let map: MlMap | null = null;
let maplibre: typeof import("maplibre-gl") | null = null;
let popupCtor: typeof import("maplibre-gl").Popup | null = null;
let ready = false;
let hovered: string | number | null = null;

// Rectangle draw state (screen-space).
let drawStart: { lng: number; lat: number; px: number; py: number } | null = null;

function findingsToGeoJSON(items: Finding[]): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  for (const f of items) {
    const geometry = (f as unknown as { geometry?: GeoJSON.Geometry }).geometry;
    if (!geometry) continue;
    features.push({
      type: "Feature",
      id: f.parcel_id,
      geometry,
      properties: {
        parcel_id: f.parcel_id,
        crop_class: f.crop_class ?? null,
        confidence: f.confidence ?? null,
        area_ha: f.area_ha ?? null,
        ndvi_mean: f.ndvi_mean ?? null,
        source: f.citation?.source ?? null,
        color: colorForCrop(f.crop_class),
      },
    });
  }
  return { type: "FeatureCollection", features };
}

function aoiFeatureCollection(geom: AoiPolygon | null): GeoJSON.FeatureCollection {
  if (!geom) return { type: "FeatureCollection", features: [] };
  return {
    type: "FeatureCollection",
    features: [{ type: "Feature", geometry: geom, properties: {} }],
  };
}

function syncFindings(fit = true) {
  if (!map || !ready) return;
  const data = findingsToGeoJSON(findings.value);
  const source = map.getSource(FINDINGS_SOURCE) as GeoJSONSource | undefined;
  if (source) source.setData(data);
  mapStore.setParcelCount(data.features.length);
  if (fit && data.features.length > 0) fitToFeatures(data);
}

function syncAoi() {
  if (!map || !ready) return;
  const source = map.getSource(AOI_SOURCE) as GeoJSONSource | undefined;
  if (source) source.setData(aoiFeatureCollection(activeAoi.value?.geometry ?? null));
}

function fitToFeatures(data: GeoJSON.FeatureCollection) {
  if (!map || !maplibre) return;
  const bounds = new maplibre.LngLatBounds();
  let any = false;
  const visit = (coords: GeoJSON.Position[]) => {
    for (const pos of coords) {
      const x = pos[0];
      const y = pos[1];
      if (x === undefined || y === undefined) continue;
      bounds.extend([x, y]);
      any = true;
    }
  };
  for (const feat of data.features) {
    const g = feat.geometry;
    if (g.type === "Polygon") g.coordinates.forEach(visit);
    else if (g.type === "MultiPolygon") g.coordinates.forEach((p) => p.forEach(visit));
  }
  if (any) map.fitBounds(bounds, { padding: 60, maxZoom: 14, duration: 600 });
}

function addOverlayLayers() {
  if (!map) return;
  map.addSource(FINDINGS_SOURCE, {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: FINDINGS_FILL,
    type: "fill",
    source: FINDINGS_SOURCE,
    paint: { "fill-color": ["get", "color"], "fill-opacity": 0.45 },
  });
  map.addLayer({
    id: FINDINGS_HL,
    type: "fill",
    source: FINDINGS_SOURCE,
    paint: { "fill-color": ["get", "color"], "fill-opacity": 0.7 },
    filter: ["==", ["get", "parcel_id"], -1],
  });
  map.addLayer({
    id: FINDINGS_LINE,
    type: "line",
    source: FINDINGS_SOURCE,
    paint: { "line-color": ["get", "color"], "line-width": 1.5 },
  });

  map.addSource(AOI_SOURCE, {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: AOI_FILL,
    type: "fill",
    source: AOI_SOURCE,
    paint: { "fill-color": "#d97706", "fill-opacity": 0.12 },
  });
  map.addLayer({
    id: AOI_LINE,
    type: "line",
    source: AOI_SOURCE,
    paint: { "line-color": "#d97706", "line-width": 2, "line-dasharray": [2, 1.5] },
  });
}

function wireParcelInteractions() {
  if (!map) return;
  map.on("click", FINDINGS_FILL, (e) => {
    const feature = e.features?.[0] as MapGeoJSONFeature | undefined;
    if (!feature || !map || !popupCtor) return;
    const p = feature.properties ?? {};
    const conf = p.confidence != null ? `${Math.round(Number(p.confidence) * 100)}%` : "—";
    const ndvi = p.ndvi_mean != null ? Number(p.ndvi_mean).toFixed(2) : "—";
    const area = p.area_ha != null ? `${Number(p.area_ha).toFixed(1)} ha` : "—";
    const html = `
      <div style="font-size:12px;line-height:1.5;min-width:150px">
        <strong>${t("chat.parcel")} ${p.parcel_id ?? "—"}</strong><br/>
        ${t("map.crop")}: ${p.crop_class ?? "—"}<br/>
        ${t("map.confidence")}: <span style="font-variant-numeric:tabular-nums">${conf}</span><br/>
        ${t("map.ndvi")}: <span style="font-variant-numeric:tabular-nums">${ndvi}</span><br/>
        ${t("map.area")}: <span style="font-variant-numeric:tabular-nums">${area}</span><br/>
        <em>${t("map.source")}: ${p.source ?? "—"}</em>
      </div>`;
    new popupCtor({ closeButton: true }).setLngLat(e.lngLat).setHTML(html).addTo(map);
  });

  map.on("mousemove", FINDINGS_FILL, (e) => {
    if (!map || drawMode.value) return;
    map.getCanvas().style.cursor = "pointer";
    const id = e.features?.[0]?.properties?.parcel_id;
    if (id != null && id !== hovered) {
      hovered = id;
      map.setFilter(FINDINGS_HL, ["==", ["get", "parcel_id"], id]);
    }
  });
  map.on("mouseleave", FINDINGS_FILL, () => {
    if (!map || drawMode.value) return;
    map.getCanvas().style.cursor = "";
    hovered = null;
    map.setFilter(FINDINGS_HL, ["==", ["get", "parcel_id"], -1]);
  });

  map.on("mousemove", (e) => {
    mapStore.setCursorCoords({ lng: e.lngLat.lng, lat: e.lngLat.lat });
  });
}

// --- Rectangle draw -------------------------------------------------------
function onCanvasDown(e: MapMouseEvent) {
  if (!drawMode.value || !map) return;
  e.preventDefault();
  drawStart = { lng: e.lngLat.lng, lat: e.lngLat.lat, px: e.point.x, py: e.point.y };
  drawRect.value = { x: e.point.x, y: e.point.y, w: 0, h: 0 };
  map.dragPan.disable();
}
function onCanvasMove(e: MapMouseEvent) {
  if (!drawStart) return;
  const x = Math.min(drawStart.px, e.point.x);
  const y = Math.min(drawStart.py, e.point.y);
  drawRect.value = {
    x,
    y,
    w: Math.abs(e.point.x - drawStart.px),
    h: Math.abs(e.point.y - drawStart.py),
  };
}
async function onCanvasUp(e: MapMouseEvent) {
  if (!drawStart || !map) return;
  const start: LngLat = { lng: drawStart.lng, lat: drawStart.lat };
  const end: LngLat = { lng: e.lngLat.lng, lat: e.lngLat.lat };
  const dragged = drawRect.value && (drawRect.value.w > 6 || drawRect.value.h > 6);
  drawStart = null;
  drawRect.value = null;
  map.dragPan.enable();
  if (!dragged) {
    mapStore.setDrawMode(false);
    return;
  }
  const polygon = rectToPolygon(start, end);
  // selectDrawnAoi is synchronous (no /aois persistence): the polygon lives in
  // the map store and is sent inline as `aoi` on the next POST /chat.
  selectDrawnAoi(polygon, t("aoi.drawn_label"));
}

function onKeydown(ev: KeyboardEvent) {
  if (ev.key === "Escape" && drawMode.value) {
    drawStart = null;
    drawRect.value = null;
    if (map) map.dragPan.enable();
    mapStore.setDrawMode(false);
  }
}

// --- Public helpers (exposed to parent) -----------------------------------
function flyToDemoAoi() {
  if (!map) return;
  // Select the seeded demo AOI and paint its parcels so the action has clear,
  // visible feedback (active-area chip + coloured parcels + legend), then fly.
  const demoAoi: Aoi = {
    id: -1,
    label: t("tools.demo"),
    area_ha: 1.0,
    geometry: demoAoiPolygon(),
  };
  mapStore.setActiveAoi(demoAoi);
  mapStore.setPreviewActive(true);
  chatStore.loadDemoParcels(demoFindings());
  map.fitBounds(
    [
      [DEMO_AOI_BBOX.minLng, DEMO_AOI_BBOX.minLat],
      [DEMO_AOI_BBOX.maxLng, DEMO_AOI_BBOX.maxLat],
    ],
    { padding: 80, duration: 900, maxZoom: 15 },
  );
}
function locateParcel(parcelId: number) {
  if (!map) return;
  const f = findings.value.find((x) => x.parcel_id === parcelId);
  const geom = (f as unknown as { geometry?: GeoJSON.Geometry } | undefined)?.geometry;
  if (!geom) return;
  fitToFeatures({ type: "FeatureCollection", features: [{ type: "Feature", geometry: geom, properties: {} }] });
  if (map) map.setFilter(FINDINGS_HL, ["==", ["get", "parcel_id"], parcelId]);
}

defineExpose({ flyToDemoAoi, locateParcel });

// Register the imperative API for the layout/sidebar. Functions are hoisted and
// guard on `map`, so registering during setup (before mount) is safe.
if (mapApiRef) mapApiRef.value = { flyToDemoAoi, locateParcel };
onBeforeUnmount(() => {
  if (mapApiRef) mapApiRef.value = null;
});

onMounted(async () => {
  if (!import.meta.client || !mapContainer.value) return;

  maplibre = await import("maplibre-gl");
  await import("maplibre-gl/dist/maplibre-gl.css");
  popupCtor = maplibre.Popup;

  map = new maplibre.Map({
    container: mapContainer.value,
    style: buildBasemapStyle(basemap.value),
    center: [11.105, 43.305],
    zoom: 6,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibre.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new maplibre.ScaleControl({ unit: "metric" }), "bottom-left");

  map.on("load", () => {
    if (!map) return;
    addOverlayLayers();
    wireParcelInteractions();
    map.on("mousedown", onCanvasDown);
    map.on("mousemove", onCanvasMove);
    map.on("mouseup", onCanvasUp);
    ready = true;
    syncFindings();
    syncAoi();
  });

  window.addEventListener("keydown", onKeydown);
});

// Live basemap switch: setStyle wipes layers, so re-add overlays on styledata.
watch(basemap, (id) => {
  if (!map) return;
  ready = false;
  map.setStyle(buildBasemapStyle(id));
  map.once("styledata", () => {
    if (!map) return;
    addOverlayLayers();
    wireParcelInteractions();
    ready = true;
    syncFindings(false);
    syncAoi();
  });
});

watch(findings, () => syncFindings(), { deep: true });
watch(activeAoi, () => syncAoi(), { deep: true });
watch(parcelsVisible, (v) => {
  if (!map || !ready) return;
  const vis = v ? "visible" : "none";
  for (const id of [FINDINGS_FILL, FINDINGS_LINE, FINDINGS_HL]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis);
  }
});
watch(drawMode, (on) => {
  if (!map) return;
  map.getCanvas().style.cursor = on ? "crosshair" : "";
});

onBeforeUnmount(() => {
  if (import.meta.client) window.removeEventListener("keydown", onKeydown);
  if (map) {
    map.remove();
    map = null;
  }
  ready = false;
  popupCtor = null;
  maplibre = null;
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

    <!-- Empty hint -->
    <div
      v-if="findings.length === 0 && !activeAoi"
      class="pointer-events-none absolute bottom-3 left-1/2 z-10 -translate-x-1/2 rounded-full border border-[var(--color-border)] bg-[var(--color-surface)]/90 px-3 py-1.5 text-xs text-[var(--color-muted-fg)] shadow-[var(--shadow-panel)] backdrop-blur"
    >
      {{ t("map.empty") }}
    </div>
  </section>
</template>
