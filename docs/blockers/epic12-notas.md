# Blockers EPIC 12 (Transferencia Multi-Region) — validacion 2026-06-25

> Pasada de validacion autonoma US-074..077. Las 4 US tienen notebook ejecutado
> con datos reales y resultados verificados en disco. Los blockers son deuda de
> versionado/lineage, ninguno de resultado.

## B-E12-1 — US-075: MLflow lineage en fallback file, no en server :5010, severidad BAJA

**Que**: el finetune FR->Catalonia (Sen4AgriNet) registro su lineage en
`file:./mlruns` en vez del server Docker MLflow :5010.

**Causa**: el server MLflow :5010 estaba caido en la VM H100 durante la corrida
(AC-6 quedo parcial). El resultado SI se materializo:
`reports/segmentation/sen4agrinet_transfer_result.json` (zero-shot mIoU 0.0 ->
few-shot mIoU **0.2468**, f1_macro 0.3005, pixel-acc 0.918, 40 epocas, 10 train /
20 val parches).

**Impacto**: ninguno en el resultado (es real y reproducible). Solo el lineage no
quedo en el server central.

**Accion recomendada**: re-registrar el run en :5010 cuando el server este arriba,
o aceptar el fallback file como evidencia (el JSON tiene las metricas completas).

## B-E12-2 — US-075: checkpoint y subset solo en la VM, severidad BAJA

**Que**: el checkpoint finetuneado (`best.pt` de Catalonia) y el subset `.nc`
(`data/sen4agrinet/`, 943 MB) solo existen en la VM `F:/projects/agrosat-copilot/`,
no replicados a local. El `data/sen4agrinet.dvc` (88 files, 942.96 MB) esta
versionado pero no se hizo `dvc pull` a local.

**Impacto**: 4 tests del adapter Sen4AgriNet quedan skipped en local (sin los .nc).
El resultado del transfer no se ve afectado (esta en el JSON).

**Accion recomendada**: `dvc pull data/sen4agrinet.dvc` a local o dejar el dato en
la VM (es voluminoso). El checkpoint best.pt conviene `dvc add` + push desde la VM.

## B-E12-3 — RESUELTO: US-076 DVC del subset crudo

**Que**: el manual-test de US-076 (2026-06-20) marcaba como PENDIENTE el `dvc add`
del subset crudo EuroCropsML.

**Estado**: YA RESUELTO en esta rama FINAL. `data/transfer/eurocropsml.dvc`
commiteado (commit `4714106`, md5 `d066fc1a...dir`, 706.736 files, 4.5 GB). AC-10
pasa de PARCIAL a OK. Sin accion pendiente.

## B-E12-4 — US-077: validacion metrica F1 queda FUTURE (por diseno), severidad INFORMATIVA

**Que**: la demo Mexico (aguacate/guayaba) es **zero-shot CUALITATIVO**, NO una
metrica F1/accuracy.

**Causa**: por diseno — no hay ground truth de campo mexicano. El notebook tiene
un caveat prominente (celda 0) y un meta-test AST confirma 0 usos de
`f1_score`/`accuracy_score`/`classification_report` (honestidad verificada).

**Impacto**: ninguno — es el alcance correcto y documentado de la US.

**Accion recomendada**: ninguna. La validacion metrica formal queda FUTURE
(requeriria etiquetas de campo de Mexico).
