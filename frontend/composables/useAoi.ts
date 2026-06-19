// AOI composable: turn a drawn rectangle into the active zone the chat is
// scoped to.
//
// The team backend has NO `/aois` endpoint, so the AOI is NEVER persisted: the
// drawn polygon is held in the map store and sent inline as `aoi`
// (GeoJSONGeometry) on the next POST /chat (see useChat.ts). This composable
// owns the data side (rect -> polygon, area, store update) so it stays SSR-safe
// and reusable. No new pnpm deps.

import type { Aoi, AoiPolygon, LngLat } from "~/types/map";
import { useMapStore } from "~/stores/map";

/** Shoelace area (ha) of a lon/lat ring via an equirectangular approx. */
function ringAreaHa(ring: number[][]): number {
  if (ring.length < 4) return 0;
  const R = 6378137;
  let lat0 = 0;
  for (const [, lat] of ring) lat0 += lat ?? 0;
  lat0 = (lat0 / ring.length) * (Math.PI / 180);
  const mPerDegLng = (Math.PI / 180) * R * Math.cos(lat0);
  const mPerDegLat = (Math.PI / 180) * R;
  let area2 = 0;
  for (let i = 0; i < ring.length - 1; i += 1) {
    const a = ring[i] as number[];
    const b = ring[i + 1] as number[];
    const ax = (a[0] ?? 0) * mPerDegLng;
    const ay = (a[1] ?? 0) * mPerDegLat;
    const bx = (b[0] ?? 0) * mPerDegLng;
    const by = (b[1] ?? 0) * mPerDegLat;
    area2 += ax * by - bx * ay;
  }
  return Math.abs(area2) / 2 / 10000;
}

/** Build a closed rectangle Polygon from two opposite corners. */
export function rectToPolygon(a: LngLat, b: LngLat): AoiPolygon {
  const minLng = Math.min(a.lng, b.lng);
  const maxLng = Math.max(a.lng, b.lng);
  const minLat = Math.min(a.lat, b.lat);
  const maxLat = Math.max(a.lat, b.lat);
  return {
    type: "Polygon",
    coordinates: [
      [
        [minLng, minLat],
        [maxLng, minLat],
        [maxLng, maxLat],
        [minLng, maxLat],
        [minLng, minLat],
      ],
    ],
  };
}

export function useAoi() {
  const store = useMapStore();

  /** Build a local AOI (negative id; not a backend key) from a polygon. */
  function makeLocalAoi(geometry: AoiPolygon, label?: string | null): Aoi {
    const ring = geometry.coordinates[0] ?? [];
    return {
      id: -Date.now(),
      label: label ?? null,
      area_ha: Number(ringAreaHa(ring).toFixed(2)),
      geometry,
    };
  }

  /** Select a drawn polygon as the active chat-scoped zone (no persistence). */
  function selectDrawnAoi(geometry: AoiPolygon, label?: string | null): Aoi {
    const aoi = makeLocalAoi(geometry, label);
    store.setActiveAoi(aoi);
    store.setDrawMode(false);
    return aoi;
  }

  return { selectDrawnAoi, rectToPolygon, makeLocalAoi };
}
