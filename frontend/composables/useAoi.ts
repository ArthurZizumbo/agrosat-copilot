// AOI composable: list/persist AOIs and turn a drawn rectangle into the active
// zone the chat is scoped to.
//
// - listAois():   GET /aois?session_id=...   (best-effort; empty on failure)
// - persistAoi(): POST /aois?session_id=...   (best-effort; returns null on err)
// - rectToPolygon(): two corners -> closed GeoJSON Polygon ring
//
// The rectangle DRAW interaction itself lives in MapCanvas (mousedown ->
// mousemove -> mouseup on the map canvas); this composable owns the data side
// so it stays SSR-safe and reusable. No new pnpm deps.

import type { Aoi, AoiListResponse, AoiPolygon, CreateAoiRequest, LngLat } from "~/types/map";
import { useMapStore } from "~/stores/map";

/** Shoelace area (m^2) of a lon/lat ring via an equirectangular approx. */
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
  const { ensureSession, apiFetch, sessionId } = useSession();

  /** Fetch AOIs for the active session. Best-effort: never throws. */
  async function listAois(): Promise<Aoi[]> {
    const sid = sessionId.value;
    if (!sid) return [];
    try {
      const res = await apiFetch<AoiListResponse>(
        `/aois?session_id=${encodeURIComponent(sid)}`,
      );
      store.setAois(res.items);
      return res.items;
    } catch {
      return [];
    }
  }

  /**
   * Persist a polygon as an AOI. Best-effort: if the backend rejects it (auth,
   * offline), we still surface a local AOI so the chat can be scoped to it.
   */
  async function persistAoi(
    geometry: AoiPolygon,
    label?: string | null,
  ): Promise<Aoi> {
    const ring = geometry.coordinates[0] ?? [];
    const localAoi: Aoi = {
      id: -Date.now(),
      label: label ?? null,
      area_ha: Number(ringAreaHa(ring).toFixed(2)),
      geometry,
    };
    try {
      const sid = await ensureSession();
      const body: CreateAoiRequest = { geometry, label: label ?? null };
      const saved = await apiFetch<Aoi>(
        `/aois?session_id=${encodeURIComponent(sid)}`,
        { method: "POST", body: JSON.stringify(body) },
      );
      store.upsertAoi(saved);
      return saved;
    } catch {
      store.upsertAoi(localAoi);
      return localAoi;
    }
  }

  /** Persist + select a polygon as the active chat-scoped zone. */
  async function selectDrawnAoi(
    geometry: AoiPolygon,
    label?: string | null,
  ): Promise<Aoi> {
    const aoi = await persistAoi(geometry, label);
    store.setActiveAoi(aoi);
    store.setDrawMode(false);
    return aoi;
  }

  return { listAois, persistAoi, selectDrawnAoi, rectToPolygon };
}
