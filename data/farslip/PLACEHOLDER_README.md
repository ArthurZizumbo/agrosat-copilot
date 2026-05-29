# Datos PLACEHOLDER en data/farslip/ — NO usar como dato real

Renombrados a `*_PLACEHOLDER.parquet` el 2026-05-29 tras la auditoría
([docs/audit/us-023-preview-v2-audit.md](../../docs/audit/us-023-preview-v2-audit.md)).
Estos archivos **NO contienen embeddings reales** y deben **regenerarse** antes de
usarse en cualquier evaluación o entregable.

| Archivo | Por qué es placeholder | Evidencia |
|---------|------------------------|-----------|
| `remoteclip_embeddings_pastis_PLACEHOLDER.parquet` | RemoteCLIP nunca corrió sobre imágenes reales. 48641 filas pero solo **2 patrones únicos** (varianza-cero). | `np.unique(emb, axis=0)` = 2/48641. El extractor `ml/ingest/remoteclip_extractor.py` existe pero no se ejecutó en producción. |
| `embeddings_italy_PLACEHOLDER.parquet` | FarSLIP extraído con `mode="placeholder"` (default): `randn` determinista seeded por parcela, **no derivado de crops reales**. | `ml/farslip/extract_embeddings.py` default `mode='placeholder'`; cierre US-022-c documenta "placeholder determinista seeded, crops reales diferidos a US-025". |
| `embeddings_italy_v1_PLACEHOLDER.parquet` | Ídem (versión v1). | Ídem. |
| `embeddings_italy_v2_PLACEHOLDER.parquet` | Ídem (versión v2). | Ídem. |

## Para regenerar dato real

- **RemoteCLIP**: ejecutar `ml/ingest/remoteclip_extractor.py` con `chendelong/RemoteCLIP-ViT-B-32`
  sobre crops PASTIS-R reales (verificar que NO cae al fallback `openai/clip-vit-base-patch32`).
- **FarSLIP italy**: ejecutar `ml/farslip/extract_embeddings.py` con `--mode real`
  (`_project_parcels_to_embeddings_real`) sobre los crops italianos reales (US-025 pendiente).

Una vez regenerados con dato real, quitar el sufijo `_PLACEHOLDER` (con `dvc move` para los
tracked por DVC) y actualizar los consumidores listados en el commit de renombrado.
