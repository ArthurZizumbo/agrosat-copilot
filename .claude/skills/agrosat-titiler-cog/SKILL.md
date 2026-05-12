---
name: agrosat-titiler-cog
description: Serve dynamic Cloud-Optimized GeoTIFF (COG) tiles via TiTiler for NDVI, NDWI, EVI overlays on MapLibre maps. Use when configuring tile endpoints, building MosaicJSON over Sentinel-2 / AlphaEarth, applying colormaps, or caching tiles in Redis.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot TiTiler COG Skill

## Rules — NON-NEGOTIABLE

- COGs stored as `gs://agrosat-data/{collection}/{roi}/{date}/{band}.tif`
- Read via `rasterio` VSI GCS (`/vsigs/agrosat-data/...`)
- Tile cache Redis key `tile:{z}:{x}:{y}:{cog_hash}:{rescale}:{cmap}` TTL 1h
- MosaicJSON cache TTL 24h
- Rescale always explicit (`rescale=0,1` for normalized indices)
- Colormap: `viridis` NDVI, `RdBu_r` NDWI, `RdYlGn` vigor

## Setup

```python
from titiler.core.factory import TilerFactory
from titiler.mosaic.factory import MosaicTilerFactory

cog_tiler = TilerFactory(router_prefix="/tiles/cog")
mosaic_tiler = MosaicTilerFactory(router_prefix="/tiles/mosaic")
```

## NDVI from Sentinel-2 (B04, B08)

```python
from rio_tiler.io import COGReader
from fastapi.responses import Response

@router.get("/ndvi/{z}/{x}/{y}.png")
async def ndvi_tile(z, x, y, scene_id):
    cog_url = f"gs://agrosat-data/raw/s2/{scene_id}/"
    cache_key = f"ndvi:{scene_id}:{z}:{x}:{y}"
    if cached := await redis.get(cache_key):
        return Response(cached, media_type="image/png")

    with COGReader(cog_url + "B04.tif") as red, COGReader(cog_url + "B08.tif") as nir:
        r = red.tile(x, y, z)
        n = nir.tile(x, y, z)
    ndvi = (n.data - r.data) / (n.data + r.data + 1e-6)
    png = render(ndvi, colormap="viridis", rescale=(-0.2, 0.9))
    await redis.setex(cache_key, 3600, png)
    return Response(png, media_type="image/png")
```

## Frontend (MapLibre) Consumer

```typescript
map.addSource('ndvi', {
  type: 'raster',
  tiles: [`/api/v1/tiles/cog/{z}/{x}/{y}.png?url=${cogUrl}&rescale=0,1&colormap_name=viridis`],
  tileSize: 256,
})
map.addLayer({ id: 'ndvi-layer', type: 'raster', source: 'ndvi', paint: { 'raster-opacity': 0.7 } })
```

## QA Checklist

- [ ] COGs validados con `rio cogeo validate`
- [ ] Cache Redis con TTL
- [ ] Rescale + colormap explícitos
- [ ] Auth GCS via service account
- [ ] Cold-start <2 s primer tile
