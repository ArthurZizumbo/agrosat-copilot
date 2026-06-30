// Pinia store: shared geospatial UI state for the dashboard shell.
//
// Holds the active AOI (selected zone the chat is scoped to), the basemap
// choice, draw mode, layer visibility, live cursor coordinates and parcel
// count. Map components and the chat dock both read from here; this is the
// bridge between "select a zone" and "chat about that zone".

import { defineStore } from "pinia";
import type { Aoi, BasemapId, LngLat } from "~/types/map";
import type { DemoView } from "~/utils/cropPalette";

/** Parcel the user clicked on the map (US-058 link parcel->chat).
 *
 * Map-side state for highlight + "selected parcel" chip. It is complementary to
 * `chatStore.activeParcelId` (which is what travels to the backend on the next
 * POST /chat), not a duplicate. `parcel_id` is REAL (from the rendered feature,
 * a backend or demo id); geometry is NOT stored here (the dense parcel geometry
 * is FUTURE — tool_result carries no boundaries, see types/agent.ts).
 */
export interface SelectedParcel {
  parcel_id: number;
  crop_class?: string | null;
}

/** Visible map extent as a flat bbox `[minLng, minLat, maxLng, maxLat]`. */
export type VisibleBbox = [number, number, number, number];

interface MapState {
  /** AOI currently selected; null = no zone chosen. */
  activeAoi: Aoi | null;
  /** AOIs known for this session (from GET /aois + locally drawn). */
  aois: Aoi[];
  /** Active basemap; satellite by default (this is a satellite app). */
  basemap: BasemapId;
  /** Whether the rectangle-draw interaction is armed. */
  drawMode: boolean;
  /** Whether the parcels layer is visible. */
  parcelsVisible: boolean;
  /** Latest cursor coordinate over the map (client only). */
  cursorCoords: LngLat | null;
  /** Number of parcels currently painted on the map. */
  parcelCount: number;
  /** True while a preview (demo) answer is shown instead of live data. */
  previewActive: boolean;
  /** Which crop the prediction demo paints (predicted / true / hits-errors). */
  demoView: DemoView;
  /** Parcel accuracy of the prediction demo, when active; null otherwise. */
  predictionAccuracy: number | null;
  /** Parcel currently selected by clicking the map; null = none. */
  selectedParcel: SelectedParcel | null;
  /** Latest visible map extent `[minLng, minLat, maxLng, maxLat]`; null until
   *  the map first reports a `moveend`. Kept for future spatial scoping. */
  visibleBbox: VisibleBbox | null;
}

export const useMapStore = defineStore("map", {
  state: (): MapState => ({
    activeAoi: null,
    aois: [],
    basemap: "satellite",
    drawMode: false,
    parcelsVisible: true,
    cursorCoords: null,
    parcelCount: 0,
    previewActive: false,
    demoView: "pred",
    predictionAccuracy: null,
    selectedParcel: null,
    visibleBbox: null,
  }),

  getters: {
    hasActiveAoi: (state): boolean => state.activeAoi !== null,
    activeAoiLabel: (state): string | null =>
      state.activeAoi?.label ?? null,
    hasSelectedParcel: (state): boolean => state.selectedParcel !== null,
  },

  actions: {
    setBasemap(id: BasemapId) {
      this.basemap = id;
    },
    setDrawMode(on: boolean) {
      this.drawMode = on;
    },
    toggleParcels() {
      this.parcelsVisible = !this.parcelsVisible;
    },
    setCursorCoords(coords: LngLat | null) {
      this.cursorCoords = coords;
    },
    setParcelCount(n: number) {
      this.parcelCount = n;
    },
    setActiveAoi(aoi: Aoi | null) {
      this.activeAoi = aoi;
    },
    setAois(items: Aoi[]) {
      this.aois = items;
    },
    upsertAoi(aoi: Aoi) {
      const idx = this.aois.findIndex((a) => a.id === aoi.id);
      if (idx >= 0) this.aois[idx] = aoi;
      else this.aois.unshift(aoi);
    },
    clearSelection() {
      this.activeAoi = null;
      this.drawMode = false;
    },
    setPreviewActive(on: boolean) {
      this.previewActive = on;
    },
    setDemoView(view: DemoView) {
      this.demoView = view;
    },
    setPredictionAccuracy(value: number | null) {
      this.predictionAccuracy = value;
    },
    /** Mark a parcel as selected (map highlight + chip). Complement to
     *  chatStore.setActiveParcelId, which the click handler also calls. */
    setSelectedParcel(parcel: SelectedParcel) {
      this.selectedParcel = parcel;
    },
    clearSelectedParcel() {
      this.selectedParcel = null;
    },
    setVisibleBbox(bbox: VisibleBbox) {
      this.visibleBbox = bbox;
    },
  },
});
