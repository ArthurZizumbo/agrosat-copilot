# notebooks/best_model — Avance 5: mejora del modelo final

Cuadernos del **Avance 5** (entrega dom 7-jun-2026): partir del mejor modelo del
Avance 4 (**TSViT-pheno**) y mejorarlo hacia el umbral de producción atacando el
desbalance de clases. Se separa de `notebooks/segmentation/` (que aloja los 6
modelos individuales del Avance 4) para no mezclar entregables.

## Contenido

| Archivo | Qué es | Cómo se genera |
|---------|--------|----------------|
| `Avance5.Equipo17.ipynb` | Integrador del modelo final: recap Avance 4, run mejorado (class-weights + augmentation), comparativa baseline vs mejorado, error analysis per-clase, figuras, compuerta de producción | `poetry run python scripts/build_avance5_notebook.py` |

> El `.ipynb` es un **esqueleto reproducible**: se genera con el builder, se
> ejecuta end-to-end en **Colab con runtime GPU** (no en tu laptop) y se commitea
> con outputs poblados (regla CLAUDE.md 12). Pon `RUN_TRAINING=True` y ajusta
> `PASTIS_ROOT` en la celda de configuración.

## Cómo correr en Colab

1. Runtime con GPU (L4/A100 en Colab Pro; T4 en free).
2. Ejecutar las celdas en orden: la celda de bootstrap **monta Drive, clona el
   repo (`github.com/ArthurZizumbo/agrosat-copilot`, branch
   `user/abocanegra/semana-5`, pide token si es privado) e instala
   dependencias** automáticamente — el notebook es standalone, igual que los
   `04*_segmentation`. Cuando el branch se mergee a `main`, ajustar `_branch` en
   la celda de bootstrap.
3. Datos `PASTIS-R` y artefactos (métricas, figuras, checkpoints, MLflow) viven
   en el **Drive compartido** (`MyDrive/Integrador/`), no en el repo.
4. Poner `RUN_TRAINING=True` para entrenar (el checkpoint se guarda en Drive y la
   corrida es reanudable).

## Contexto y decisiones

- Plan completo, lifts ajustados y árbol de decisión de producción:
  [`docs/us-planning/avance5-mejora-modelo-final.md`](../../docs/us-planning/avance5-mejora-modelo-final.md).
- Palancas implementadas en `ml/train/train_segmentation.py`
  (`--class-balance`, `--augment`) y `ml/data/pastis_seg_dataset.py`.
- Los **4 ensambles obligatorios** (EPIC 6) son el otro componente del Avance 5;
  irán en un cuaderno hermano cuando `ml/ensemble/` esté implementado.
