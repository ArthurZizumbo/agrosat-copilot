# US-018.2 — Integrar bloque Sentinel-1 a la matriz de fusión sobre PASTIS-R

**Status**: Backlog
**Epic**: E3 (Feature Engineering)
**Estimación**: 1-2 días dev (CPU local + GEE en background)
**Origen**: Extensión de US-018 fase L2b. La generación de bloques `srtm + era5 + geom`
sobre las 2433 parcelas PASTIS-R completas se ejecutó en S4; el bloque Sentinel-1
quedó pendiente por costo computacional (estimado ~9 horas de GEE en vivo).

## Contexto

El notebook `notebooks/feature_engineering/03c_fe_alphaearth_pastis.ipynb`
construye la matriz de fusión sobre PASTIS-R completo. La integración inicial
agregó 117 columnas (64 AE + 24 ERA5 + 3 SRTM + 3 geom + 5 stats × 17 indices
parcial = 181 cols efectivas dependiendo de qué bloques se incluyen), pero
descartó el bloque Sentinel-1 (10 cols `s1_vv_*` + `s1_vh_*`) por su costo:

- Sentinel-1 GRD revisita Francia ~cada 6 días → ~130 imágenes/parcela/año
  (65 ascending + 70 descending según `metadata.geojson` PASTIS)
- Despeckle Lee 7x7 default (decisión histórica US-016) es una convolución
  espacial cara antes de cada `reduceRegions`
- Costo medido: **~13 segundos por parcela** sobre PASTIS-FR 2019
- Total proyectado sobre 2433 parcelas: **~9 horas de GEE en vivo**

## Lo que ya está hecho

- `scripts/generate_fusion_blocks_pastis.py` ya tiene la lógica de muestreo
  S1 cableada: basta con quitar `--skip-s1` y dejar correr
- `sample_s1_roi_for_parcels` en `ml/ingest/gee_sampler.py:774` opera sobre
  cualquier GeoDataFrame con `parcel_id + geometry` (no hay nada hardcoded
  a Italia)
- `build_fused_features` en `ml/features/fusion.py` ya acepta `s1_frame`
  inyectado vía parámetro
- 03c ya tiene el frame parcel-level y los folds oficiales PASTIS-R listos
  para consumir el bloque S1 cuando exista

## Plan de implementación

1. **Lanzar generación S1 en background** (~9h overnight o background day):
   ```bash
   poetry run python scripts/generate_fusion_blocks_pastis.py \
       --year 2019 \
       --cache-key pastis_fr_full \
       --skip-srtm --skip-era5
   ```
   Output esperado: `data/cache/gee/s1_pastis_fr_full_2019_both_lee_7x7_dB.parquet`
   (~150 KB, 2433 × 12 cols)

2. **Editar `scripts/build_fusion_notebook.py`** para cargar el bloque S1
   junto a SRTM/ERA5/geom existentes:
   ```python
   s1_path = REPO_ROOT / 'data/cache/gee/s1_pastis_fr_full_2019_both_lee_7x7_dB.parquet'
   s1_frame = pl.read_parquet(s1_path) if s1_path.exists() else None
   fused = build_fused_features(
       parcels_gdf, year=2019,
       ae_frame=ae_frame,
       srtm_frame=srtm_frame,
       era5_frame=era5_frame,
       s1_frame=s1_frame,  # <-- agregar
       blocks=('alphaearth', 'sentinel1', 'srtm', 'era5_monthly', 'geometry'),
   )
   ```

3. **Re-ejecutar 03c**: `make feature-fusion-notebook`

4. **Actualizar conclusiones dinámicas**: el bloque S1 agrega 10 features
   de backscatter VV+VH × 5 stats. La hipótesis agronómica es que Sentinel-1
   complementa a Sentinel-2 (AE+índices) bajo nubes y captura humedad de
   suelo, lo que debería mejorar F1 sobre clases de cultivo dependientes
   de irrigación (`Soft winter wheat`, `Corn`).

## Alternativas más rápidas (si 9h es bloqueante)

| Opción | Speedup | Trade-off |
|---|---|---|
| Disable despeckle (`despeckle="none"`) | 3-5x (~2-3h) | Stats más ruidosos, desviación del default histórico |
| Solo ascending (`orbit_pass="ascending"`) | 2x (~4h) | Pierde mitad de visitas, sesgo iluminación |
| Reducir a 6 meses (mayo-octubre, época vegetativa) | 2x (~4h) | Pierde dinámica invernal |
| Submuestrear a 1 imagen/mes | 5-10x (~1-1.5h) | Modificar sampler (param `temporal_stride`) |

## Definition of Done

- [ ] Cache `s1_pastis_fr_full_2019_both_lee_7x7_dB.parquet` generado y commiteado/DVC
- [ ] 03c actualizado para consumir el bloque S1 (cell de carga + comparativa de 4 vistas)
- [ ] Comparativa AE-only vs spectral-temporal vs fusion sin S1 vs fusion con S1 con números empíricos
- [ ] Limitación "S1 no integrado" eliminada del bloque limitaciones del 03c
- [ ] `make check` limpio + papermill exit 0

## Decisión registrada

S4 2026-05-17: el equipo elige diferir S1 para no bloquear el cierre del
Avance 2. Las 181 columnas (sin S1) ya proveen comparativa simétrica
benchmark-grade entre AlphaEarth y fusion parcial sobre PASTIS-R completo.
