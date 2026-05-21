---
name: alphaearth-patch-vs-parcel-aggregation
description: AlphaEarth sampling en PASTIS-R — trade-off centroide del patch vs reduceRegions sobre polígono completo, con caveat de heterogeneidad intra-patch
metadata:
  type: project
---

En US-018 (fusion AlphaEarth + spectral-temporal sobre PASTIS-R) el cache AE actual usa centroide del patch (`scripts/generate_alphaearth_pastis_full.py` → `sample_alphaearth_at_coords` con `ee.Geometry.Point` + `Reducer.first()`). Esto muestrea 1 píxel AE de 10 m sobre un patch de 128×128 px (~1.28 km²) → ~0.006 % del área.

**Why**: PASTIS-R no es monoparcela — cada patch de 128×128 contiene N parcelas heterogéneas con clases distintas. El centroide puede caer en linde, camino o parcela vecina, sesgando el embedding respecto a la label dominante.

**How to apply**: 
- Para la próxima iteración (US-018b o cuando se regenere el cache AE) preferir `reduceRegions` sobre el polígono **de cada parcela individual** (no del bbox del patch). Patrón canónico ya está en `ml/ingest/gee_sampler.py:915-985` (`sample_srtm_terrain`) — replicable ~40 LOC.
- Si se hace por patch (no parcela), considerar `Reducer.mode()` ponderado por máscara semántica para evitar mezclar clases.
- El `fusion_manifest.json` actual ya documenta la limitación explícitamente — válido para baseline, pero registrar como deuda técnica.
- Coste GEE por polígono ~igual que por punto: el reductor `mean` es server-side barato a 10 m sobre 1.28 km².

Relacionado: [[us-018-fusion-baseline]] (si se crea).
