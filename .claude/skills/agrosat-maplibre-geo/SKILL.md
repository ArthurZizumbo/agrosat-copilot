---
name: agrosat-maplibre-geo
description: Integrate MapLibre GL JS and deck.gl for interactive geospatial visualization in AgroSatCopilot. Use when adding map sources, raster overlays (COG via TiTiler), draw-polygon AOI tools, deck.gl GeoJsonLayer, or layer switching.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot MapLibre + deck.gl Skill

## Rules — NON-NEGOTIABLE

- MapLibre GL JS para basemap + raster (no Mapbox)
- deck.gl para layers complejos (heatmaps, hex-bins, 3D)
- maplibre-gl-draw para dibujar AOI polígonos
- Cleanup en `onBeforeUnmount` (memory leaks)
- SRID 4326 (WGS84) en frontend; backend convierte si necesita

## Setup en componente

```vue
<template>
  <div id="map-container" class="w-full h-full" />
</template>

<script setup lang="ts">
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import MapboxDraw from '@mapbox/mapbox-gl-draw'
import 'maplibre-gl-draw/dist/mapbox-gl-draw.css'

const { map, isReady } = useMap('map-container')
const aoiStore = useAOIStore()

watch(isReady, (ready) => {
  if (!ready || !map.value) return
  const draw = new MapboxDraw({ displayControlsDefault: false, controls: { polygon: true, trash: true } })
  map.value.addControl(draw as unknown as maplibregl.IControl)
  map.value.on('draw.create', (e) => {
    aoiStore.setCurrent(e.features[0])
  })
})
</script>
```

## Raster Overlay (NDVI via TiTiler)

```typescript
function addNDVILayer(sceneId: string) {
  const tilesUrl = `/api/v1/tiles/cog/{z}/{x}/{y}.png?url=gs://agrosat-data/raw/s2/${sceneId}/&rescale=0,1&colormap_name=viridis`
  map.value!.addSource('ndvi', {
    type: 'raster',
    tiles: [tilesUrl],
    tileSize: 256,
  })
  map.value!.addLayer({
    id: 'ndvi-layer',
    type: 'raster',
    source: 'ndvi',
    paint: { 'raster-opacity': 0.7 },
  })
}
```

## deck.gl GeoJsonLayer (segmentación)

```typescript
import { MapboxOverlay } from '@deck.gl/mapbox'
import { GeoJsonLayer } from '@deck.gl/layers'

const overlay = new MapboxOverlay({
  interleaved: true,
  layers: [
    new GeoJsonLayer({
      id: 'segmentation',
      data: segmentationFeatureCollection,
      filled: true,
      stroked: true,
      getFillColor: (f) => CROP_COLORS[f.properties.crop_class],
      getLineColor: [255, 255, 255],
      lineWidthMinPixels: 1,
      pickable: true,
      autoHighlight: true,
    }),
  ],
})

map.value!.addControl(overlay as unknown as maplibregl.IControl)
```

## Geocoding + Search

Usar Nominatim (OSM, gratuito) o servicio propio:

```typescript
async function geocode(query: string) {
  const r = await $fetch<GeocodeResult[]>('https://nominatim.openstreetmap.org/search', {
    params: { q: query, format: 'json', limit: 5, countrycodes: 'it,es,fr' }
  })
  return r
}
```

## QA Checklist

- [ ] Map cleanup en onBeforeUnmount
- [ ] AOI draw produce GeoJSON Feature válido
- [ ] Raster overlay con TiTiler endpoint
- [ ] deck.gl interleaved con MapLibre
- [ ] Performance: <60 features sin clustering
