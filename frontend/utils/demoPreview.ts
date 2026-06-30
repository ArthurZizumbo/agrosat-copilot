// Local sample answer + findings used by the chat empty-state "Ver ejemplo"
// button. This is a client-only preview to showcase the FindingCard design
// without a backend round-trip; it never hits the network and is clearly
// labelled as an example in the UI.

import type { Finding } from "~/types/agent";

/** Minimal GeoJSON Polygon shape (avoids depending on @types/geojson here). */
interface PolygonGeometry {
  type: "Polygon";
  coordinates: number[][][];
}

/** Demo AOI bbox over Toscana (matches the seeded demo AOI). */
export const DEMO_AOI_BBOX = {
  minLng: 11.1,
  minLat: 43.3,
  maxLng: 11.11,
  maxLat: 43.31,
} as const;

/** A polygon ring covering the demo AOI bbox (closed). */
export function demoAoiPolygon(): PolygonGeometry {
  const { minLng, minLat, maxLng, maxLat } = DEMO_AOI_BBOX;
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

/** REAL demo parcels loaded from the build artifact (ground truth). */
export interface RealParcels {
  /** `[minLng, minLat, maxLng, maxLat]` of the demo area. */
  bbox: [number, number, number, number];
  findings: Finding[];
  aoiPolygon: PolygonGeometry;
}

/**
 * Load REAL PASTIS-R parcels (real boundaries + real crop labels) for the map
 * demo, replacing the synthetic Tuscany rectangles. The GeoJSON is the artifact
 * of `scripts/build_demo_parcels_real.py`
 * (`frontend/public/demo/parcelas_reales_francia.geojson`). Returns null if the
 * asset is missing so the caller can degrade gracefully.
 */
export async function loadRealParcels(): Promise<RealParcels | null> {
  interface RealFeature {
    geometry: PolygonGeometry;
    properties: { parcel_id?: number; crop_class?: string | null };
  }
  let fc: { bbox: [number, number, number, number]; features: RealFeature[] };
  try {
    const res = await fetch("/demo/parcelas_reales_francia.geojson");
    if (!res.ok) return null;
    fc = await res.json();
  } catch {
    return null;
  }
  const findings = fc.features.map((f, i) => ({
    parcel_id: f.properties.parcel_id ?? i,
    crop_class: f.properties.crop_class ?? null,
    confidence: null,
    area_ha: null,
    ndvi_mean: null,
    metrics: {},
    citation: {
      tool_call_id: "demo-real",
      source: "PASTIS-R (verdad de campo)",
      parcel_id: f.properties.parcel_id ?? i,
    },
    geometry: f.geometry,
  })) as Finding[];
  const [minLng, minLat, maxLng, maxLat] = fc.bbox;
  const aoiPolygon: PolygonGeometry = {
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
  return { bbox: fc.bbox, findings, aoiPolygon };
}

/** REAL parcels carrying the MODEL'S PREDICTION (held-out fold). */
export interface PredictionParcels extends RealParcels {
  /** Parcel accuracy over the painted set (predicted vs ground truth). */
  accuracy: number | null;
}

/**
 * Load REAL PASTIS-R parcels annotated with the MODEL'S PREDICTION (out-of-sample
 * fold-5 patch): each parcel carries the predicted crop (`crop_class`), the true
 * crop (`true_class`) and whether the prediction was correct, so the map can
 * toggle predicted / true / hits-errors. Artifact of
 * `scripts/build_demo_parcels_prediction.py`. Returns null if the asset is
 * missing so the caller can fall back to the ground-truth demo.
 */
export async function loadPredictionParcels(): Promise<PredictionParcels | null> {
  interface PredFeature {
    geometry: PolygonGeometry;
    properties: {
      parcel_id?: number | string;
      crop_class?: string | null;
      pred_class?: string | null;
      true_class?: string | null;
      correct?: boolean | null;
      confidence?: number | null;
    };
  }
  let fc: {
    bbox: [number, number, number, number];
    metadata?: { accuracy?: number | null };
    features: PredFeature[];
  };
  try {
    const res = await fetch("/demo/parcelas_prediccion_francia.geojson");
    if (!res.ok) return null;
    fc = await res.json();
  } catch {
    return null;
  }
  const findings = fc.features.map((f, i) => {
    const pid =
      typeof f.properties.parcel_id === "number" ? f.properties.parcel_id : i;
    return {
      parcel_id: pid,
      crop_class: f.properties.pred_class ?? f.properties.crop_class ?? null,
      confidence: f.properties.confidence ?? null,
      area_ha: null,
      ndvi_mean: null,
      metrics: {},
      true_class: f.properties.true_class ?? null,
      correct: f.properties.correct ?? null,
      citation: {
        tool_call_id: "demo-pred",
        source: "XGBoost+AlphaEarth (prediccion, fold reservado)",
        parcel_id: pid,
      },
      geometry: f.geometry,
    };
  }) as Finding[];
  const [minLng, minLat, maxLng, maxLat] = fc.bbox;
  const aoiPolygon: PolygonGeometry = {
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
  return { bbox: fc.bbox, findings, aoiPolygon, accuracy: fc.metadata?.accuracy ?? null };
}

/** Three illustrative findings with geometry, for the preview only. */
export function demoFindings(): Finding[] {
  return [
    {
      parcel_id: 101,
      crop_class: "Vineyard",
      confidence: 0.94,
      area_ha: 6.3,
      ndvi_mean: 0.71,
      metrics: {},
      citation: {
        tool_call_id: "demo-1",
        source: "XGBoost+AlphaEarth",
        parcel_id: 101,
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [11.101, 43.301],
            [11.104, 43.301],
            [11.104, 43.304],
            [11.101, 43.304],
            [11.101, 43.301],
          ],
        ],
      },
    } as Finding,
    {
      parcel_id: 102,
      crop_class: "Olive grove",
      confidence: 0.88,
      area_ha: 4.1,
      ndvi_mean: 0.58,
      metrics: {},
      citation: {
        tool_call_id: "demo-1",
        source: "XGBoost+AlphaEarth",
        parcel_id: 102,
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [11.105, 43.302],
            [11.108, 43.302],
            [11.108, 43.305],
            [11.105, 43.305],
            [11.105, 43.302],
          ],
        ],
      },
    } as Finding,
    {
      parcel_id: 103,
      crop_class: "Wheat",
      confidence: 0.79,
      area_ha: 9.7,
      ndvi_mean: 0.49,
      metrics: {},
      citation: {
        tool_call_id: "demo-1",
        source: "XGBoost+AlphaEarth",
        parcel_id: 103,
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [11.1015, 43.3055],
            [11.1045, 43.3055],
            [11.1045, 43.3085],
            [11.1015, 43.3085],
            [11.1015, 43.3055],
          ],
        ],
      },
    } as Finding,
  ];
}
