// Map / AOI / basemap UI types. The AOI shapes mirror the backend contract in
// backend/app/api/schemas.py (AoiResponse / CreateAoiRequest / AoiListResponse).

/** A GeoJSON Polygon as accepted/returned by the backend AOI API. */
export interface AoiPolygon {
  type: "Polygon";
  coordinates: number[][][];
}

/** Public projection of an AOI (mirror of AoiResponse). */
export interface Aoi {
  id: number;
  session_id?: string;
  label?: string | null;
  area_ha?: number | null;
  geometry: AoiPolygon;
}

/** Response of GET /aois. */
export interface AoiListResponse {
  items: Aoi[];
}

/** Body of POST /aois. */
export interface CreateAoiRequest {
  geometry: AoiPolygon;
  label?: string | null;
}

/** Selectable basemap layers (all keyless raster sources). */
export type BasemapId = "satellite" | "streets" | "topo";

/** A geographic coordinate read off the cursor. */
export interface LngLat {
  lng: number;
  lat: number;
}
