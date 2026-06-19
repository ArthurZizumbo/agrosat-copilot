// Map / AOI / basemap UI types.
//
// The team backend exposes NO `/aois` endpoint: a drawn AOI is NOT persisted.
// It lives only in the map store and is sent inline as `aoi` (GeoJSONGeometry)
// on each POST /chat. These shapes therefore describe LOCAL UI state only.

/** A GeoJSON Polygon drawn on the map. */
export interface AoiPolygon {
  type: "Polygon";
  coordinates: number[][][];
}

/** A locally-held Area Of Interest (never persisted to the backend). */
export interface Aoi {
  /** Local id (negative epoch ms); not a backend primary key. */
  id: number;
  label?: string | null;
  area_ha?: number | null;
  geometry: AoiPolygon;
}

/** Selectable basemap layers (all keyless raster sources). */
export type BasemapId = "satellite" | "streets" | "topo";

/** A geographic coordinate read off the cursor. */
export interface LngLat {
  lng: number;
  lat: number;
}
