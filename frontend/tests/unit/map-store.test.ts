import { beforeEach, describe, expect, it } from "vitest";
import { createApp } from "vue";
import { createPinia, setActivePinia } from "pinia";
import { useMapStore } from "~/stores/map";
import { useChatStore } from "~/stores/chat";
import { rectToPolygon } from "~/composables/useAoi";
import { buildCropLegend, colorForCrop } from "~/utils/cropPalette";
import { demoFindings } from "~/utils/demoPreview";
import type { Aoi } from "~/types/map";

// Covers the geospatial UI store, the rectangle->polygon helper and the shared
// crop palette util (single source of truth for map + chat colours).

beforeEach(() => {
  // App-installed pinia so the chat store's `persist` plugin resolves (the
  // cross-store link test touches both stores). nuxt-globals setup provides the
  // `piniaPluginPersistedstate` global the chat store reads at import time.
  const app = createApp({ render: () => null });
  const pinia = createPinia();
  app.use(pinia);
  setActivePinia(pinia);
});

describe("mapStore", () => {
  it("defaults to satellite, parcels visible, no active AOI", () => {
    const store = useMapStore();
    expect(store.basemap).toBe("satellite");
    expect(store.parcelsVisible).toBe(true);
    expect(store.hasActiveAoi).toBe(false);
  });

  it("selecting an AOI sets it active and exposes its label", () => {
    const store = useMapStore();
    const aoi: Aoi = {
      id: 7,
      label: "Toscana",
      area_ha: 12.5,
      geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] },
    };
    store.setActiveAoi(aoi);
    expect(store.hasActiveAoi).toBe(true);
    expect(store.activeAoiLabel).toBe("Toscana");
  });

  it("clearSelection drops the active AOI and disarms draw mode", () => {
    const store = useMapStore();
    store.setDrawMode(true);
    store.setActiveAoi({ id: 1, geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] } });
    store.clearSelection();
    expect(store.activeAoi).toBeNull();
    expect(store.drawMode).toBe(false);
  });

  it("upsertAoi inserts then updates by id", () => {
    const store = useMapStore();
    const base: Aoi = { id: 3, label: "A", geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] } };
    store.upsertAoi(base);
    expect(store.aois).toHaveLength(1);
    store.upsertAoi({ ...base, label: "B" });
    expect(store.aois).toHaveLength(1);
    expect(store.aois[0]?.label).toBe("B");
  });
});

describe("mapStore selected parcel (US-058 link parcel->chat)", () => {
  it("no parcel selected by default", () => {
    const store = useMapStore();
    expect(store.selectedParcel).toBeNull();
    expect(store.hasSelectedParcel).toBe(false);
  });

  it("setSelectedParcel stores the REAL feature id + crop and flips the getter", () => {
    const store = useMapStore();
    // REAL rendered demo feature (parcel 101, Vineyard), not an invented id.
    const real = demoFindings()[0]!;
    store.setSelectedParcel({ parcel_id: real.parcel_id, crop_class: real.crop_class });
    expect(store.hasSelectedParcel).toBe(true);
    expect(store.selectedParcel).toEqual({ parcel_id: 101, crop_class: "Vineyard" });
  });

  it("clearSelectedParcel resets it", () => {
    const store = useMapStore();
    store.setSelectedParcel({ parcel_id: 102, crop_class: "Olive grove" });
    store.clearSelectedParcel();
    expect(store.selectedParcel).toBeNull();
    expect(store.hasSelectedParcel).toBe(false);
  });
});

describe("mapStore visible bbox", () => {
  it("is null until the map reports a moveend, then stores the flat bbox", () => {
    const store = useMapStore();
    expect(store.visibleBbox).toBeNull();
    store.setVisibleBbox([11.0, 43.2, 11.2, 43.4]);
    expect(store.visibleBbox).toEqual([11.0, 43.2, 11.2, 43.4]);
  });
});

describe("cross-store link: map selectedParcel <-> chat activeParcelId", () => {
  it("selecting a real parcel feeds the map chip AND the chat request id together", () => {
    const mapStore = useMapStore();
    const chatStore = useChatStore();

    // Simulate the MapCanvas onParcelSelect fan-out with a REAL feature id.
    const real = demoFindings()[1]!; // parcel 102, Olive grove
    mapStore.setSelectedParcel({ parcel_id: real.parcel_id, crop_class: real.crop_class });
    chatStore.setActiveParcelId(real.parcel_id);

    expect(mapStore.selectedParcel?.parcel_id).toBe(102);
    expect(chatStore.activeParcelId).toBe(102);
    expect(chatStore.activeParcelId).toBe(real.parcel_id);
  });

  it("chat reset() clears the active parcel id (no stale parcel across turns)", () => {
    const chatStore = useChatStore();
    chatStore.setActiveParcelId(103);
    expect(chatStore.activeParcelId).toBe(103);
    chatStore.reset();
    expect(chatStore.activeParcelId).toBeNull();
  });
});

describe("rectToPolygon", () => {
  it("builds a closed, normalized rectangle ring from two corners", () => {
    const poly = rectToPolygon({ lng: 11.11, lat: 43.31 }, { lng: 11.1, lat: 43.3 });
    const ring = poly.coordinates[0]!;
    expect(poly.type).toBe("Polygon");
    expect(ring).toHaveLength(5);
    expect(ring[0]).toEqual(ring[4]); // closed
    expect(ring[0]).toEqual([11.1, 43.3]); // min corner first
  });
});

describe("cropPalette", () => {
  it("is deterministic per crop label", () => {
    expect(colorForCrop("Vineyard")).toBe(colorForCrop("Vineyard"));
  });

  it("returns the unknown colour for null/empty", () => {
    expect(colorForCrop(null)).toBe("#6b7280");
  });

  it("builds a deduped, sorted legend", () => {
    const legend = buildCropLegend(["Wheat", "Vineyard", "Wheat", null, undefined]);
    expect(legend.map((e) => e.crop)).toEqual(["Vineyard", "Wheat"]);
    expect(legend[0]?.color).toBe(colorForCrop("Vineyard"));
  });
});
