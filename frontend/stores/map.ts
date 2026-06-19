// Pinia store: shared geospatial UI state for the dashboard shell.
//
// Holds the active AOI (selected zone the chat is scoped to), the basemap
// choice, draw mode, layer visibility, live cursor coordinates and parcel
// count. Map components and the chat dock both read from here; this is the
// bridge between "select a zone" and "chat about that zone".

import { defineStore } from "pinia";
import type { Aoi, BasemapId, LngLat } from "~/types/map";

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
  }),

  getters: {
    hasActiveAoi: (state): boolean => state.activeAoi !== null,
    activeAoiLabel: (state): string | null =>
      state.activeAoi?.label ?? null,
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
  },
});
