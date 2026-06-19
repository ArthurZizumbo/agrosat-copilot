// Basemap definitions + helper to build a MapLibre raster StyleSpecification.
//
// All sources are keyless raster tiles (no API key, no new pnpm deps):
//   satellite -> Esri World Imagery
//   streets   -> OpenStreetMap raster
//   topo      -> OpenTopoMap
// Switching at runtime is done with `map.setStyle(buildBasemapStyle(id))`.

import type { StyleSpecification } from "maplibre-gl";
import type { BasemapId } from "~/types/map";

interface BasemapDef {
  id: BasemapId;
  tiles: string[];
  attribution: string;
  maxzoom: number;
}

const BASEMAPS: Record<BasemapId, BasemapDef> = {
  satellite: {
    id: "satellite",
    tiles: [
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    attribution:
      "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    maxzoom: 19,
  },
  streets: {
    id: "streets",
    tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: "&copy; OpenStreetMap contributors",
    maxzoom: 19,
  },
  topo: {
    id: "topo",
    tiles: ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png"],
    attribution:
      "&copy; OpenTopoMap (CC-BY-SA) — &copy; OpenStreetMap contributors",
    maxzoom: 17,
  },
};

/** Build a single-raster-layer MapLibre style for the given basemap. */
export function buildBasemapStyle(id: BasemapId): StyleSpecification {
  const def = BASEMAPS[id];
  return {
    version: 8,
    sources: {
      basemap: {
        type: "raster",
        tiles: def.tiles,
        tileSize: 256,
        maxzoom: def.maxzoom,
        attribution: def.attribution,
      },
    },
    layers: [
      {
        id: "basemap",
        type: "raster",
        source: "basemap",
      },
    ],
  };
}

/** Ordered list of basemaps for segmented switchers. */
export function useBasemap() {
  const order: BasemapId[] = ["satellite", "streets", "topo"];
  return { order, buildBasemapStyle };
}
