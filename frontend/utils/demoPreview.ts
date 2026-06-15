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
