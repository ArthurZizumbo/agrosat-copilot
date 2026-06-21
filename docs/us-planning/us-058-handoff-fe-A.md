# US-058 fe/A — Handoff: extraer `useMap.ts` de MapCanvas + remover deck.gl

**Rama**: `feature/E9-US-058-mapview` · **Write-set**: `frontend/composables/useMap.ts` (NUEVO), `frontend/components/map/MapCanvas.vue` (consumir), `frontend/package.json` + `frontend/nuxt.config.ts` (D3 deck.gl), `frontend/CLAUDE.md` + `frontend/AGENTS.md` (nota deck.gl, espejos sincronizados).
**Estado**: HECHO. `pnpm typecheck` limpio (exit 0, sin `error TS`). `deck.gl` removido (`pnpm remove deck.gl`, lockfile regenerado).

## Que se implemento

### 1. `composables/useMap.ts` (NUEVO) — firma D1
```ts
export function useMap(opts?: UseMapOptions): UseMapHandle
// UseMapOptions  = { onParcelSelect?: (parcelId: number, props: Record<string, unknown>) => void }
// UseMapHandle   = { initMap(container): Promise<void>; destroyMap(): void;
//                    flyToDemoAoi(); locateParcel(id); isReady: Ref<boolean>;
//                    drawRect: Ref<{x,y,w,h}|null> }
```
Logica imperativa MOVIDA desde MapCanvas (paridad funcional, sin reescribir):
- **init** (`initMap`): `import("maplibre-gl")` + CSS DENTRO de la funcion (SSR-safe), `new Map`, `NavigationControl`/`ScaleControl`, `buildBasemapStyle(basemap.value)`.
- **overlays** (`addOverlayLayers`): findings `fill`/`line`/`highlight` + AOI `fill`/`line`; ids constantes (`findings`, `findings-fill`, `findings-line`, `findings-highlight`, `active-aoi`, `active-aoi-fill`, `active-aoi-line`).
- **sync** (`syncFindings`/`syncAoi`): `setData` + `setParcelCount` + `fitToFeatures`.
- **interactions** (`wireParcelInteractions`): click (popup i18n) + mousemove/mouseleave (hover-highlight) + cursor coords.
- **draw-rect**: `onCanvasDown/Move/Up` + `onKeydown` (Escape); el rubber-band visual se queda en MapCanvas (lee `drawRect`).
- **flyTo**: `flyToDemoAoi` (carga demo AOI + parcelas demo), `locateParcel`.
- **basemap watcher**: `setStyle` + re-add overlays en `styledata`.
- **destroy** (`destroyMap`): `map.remove()` + `removeEventListener('keydown')` + stop de todos los watchers + reset de refs (`isReady`, `drawRect`, etc.).

**SSR-safe / sin singletons**: estado del mapa (`map`, `maplibre`, `popupCtor`, `hovered`, `drawStart`, `stopHandles`) vive DENTRO de `useMap()`, no a nivel de modulo. Solo constantes de ids (`FINDINGS_*`, `AOI_*`) son module-level. `import("maplibre-gl")` solo se ejecuta cuando MapCanvas llama `initMap` bajo `import.meta.client`.

**Watchers movidos a `initMap`**: los `watch(basemap|findings|activeAoi|parcelsVisible|drawMode)` antes eran top-level del setup de MapCanvas; ahora se registran dentro de `initMap` y se guardan sus stop-handles en `stopHandles[]`, que `destroyMap` invoca. Asi el cleanup entre navegaciones desconecta todo.

### 2. `MapCanvas.vue` — pasa a CONSUMIR useMap
- Borrada toda la logica imperativa (init/destroy/layers/handlers/flyTo/watchers).
- CONSERVADO: `<template>` completo, chips (`MapDrawToolbar`/`MapCropLegend`/`MapCoordsReadout`), rubber-band (`v-if="drawRect"` ahora desde `useMap`), `defineExpose({flyToDemoAoi, locateParcel})`, registro de `mapApiRef` (inject `"mapCanvas"`, acoplamiento al layout).
- Lifecycle nuevo:
  ```ts
  const { initMap, destroyMap, flyToDemoAoi, locateParcel, drawRect } = useMap();
  onMounted(async () => { if (import.meta.client && mapContainer.value) await initMap(mapContainer.value); });
  onBeforeUnmount(() => { destroyMap(); if (mapApiRef) mapApiRef.value = null; });
  ```
- El componente solo lee `findings` y `activeAoi` (del store, para el `v-if` del empty-hint); ya NO lee `basemap`/`drawMode`/`parcelsVisible` (los consume useMap).

### 3. deck.gl REMOVIDO (D3)
- `pnpm remove deck.gl` (lockfile regenerado por la via correcta; `-304` paquetes).
- `package.json`: fuera `"deck.gl": "^9.3.2"`.
- `nuxt.config.ts` linea 3: comentario actualizado (quitada mencion deck.gl).
- `frontend/CLAUDE.md` **y** `frontend/AGENTS.md` (espejos): tabla stack marca `deck.gl` como **removido** (0 usos reales; reañadir con `pnpm add` + acuerdo de equipo si una US futura necesita densidad alta). Se conserva la fila de la skill `agrosat-maplibre-geo` que nombra deck.gl (es nombre de skill, no dep).
- Evidencia: `grep -i deckgl` en `frontend/` (excl. `pnpm-lock.yaml`) = 0 usos de codigo; solo menciones en docs/skill-table.

## Para fe/B (link parcela->chat, SEGUNDO, sobre este useMap)
- El hook ya esta cableado: `wireParcelInteractions` invoca `opts.onParcelSelect?.(parcelId, props)` en el click sobre `FINDINGS_FILL`, con `parcel_id` REAL parseado de `e.features[0].properties.parcel_id`. **useMap NO importa `stores/chat`** — el cross-store lo decide MapCanvas pasando el callback.
- fe/B solo debe: (a) `useMap({ onParcelSelect: (id) => { mapStore.setSelectedParcel({parcel_id:id}); chatStore.setActiveParcelId(id); } })` en MapCanvas; (b) `stores/map.ts` (+`selectedParcel`/`visibleBbox`); (c) `stores/chat.ts` (+`activeParcelId`); (d) `useChat` body `parcel_id`; (e) i18n D5; (f) `map.on("moveend", ...)` -> `setVisibleBbox` (anadir dentro de `initMap`, junto a los otros `map.on`).
- NO se anadio `moveend`/`setVisibleBbox` aqui (es fe/B + requiere la action del store que aun no existe).

## Criterio de no-romper (paridad funcional)
Identico a antes: basemap Esri keyless, parcelas por cultivo + hover + popup i18n, draw-rect AOI -> `selectDrawnAoi`, switch basemap live (re-add overlays), `flyToDemoAoi`/`locateParcel` via `mapApiRef`/`defineExpose`, cleanup en `onBeforeUnmount`. Verificar en `pnpm dev` (R3): mapa, parcelas demo, draw, popup, hover, switch, navegar fuera (cleanup).

## Gate
- `pnpm typecheck`: exit 0, limpio.
- `pnpm lint`: NO ejecutado aqui (MEMORY: ESLint v9 puede romper en frontend por config pre-existente, ajeno a esta US).
