import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useMapStore } from "~/stores/map";
import { rectToPolygon } from "~/composables/useAoi";
import { buildCropLegend, colorForCrop } from "~/utils/cropPalette";
import type { Aoi } from "~/types/map";

// Covers the geospatial UI store, the rectangle->polygon helper and the shared
// crop palette util (single source of truth for map + chat colours).

beforeEach(() => {
  setActivePinia(createPinia());
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
