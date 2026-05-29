# ADR-008 — Rediseño de calendario por desfase de Avances y movimiento de Presentación a 27-jun

**Status**: Aceptada · 2026-05-29
**Fecha**: 2026-05-29
**Decisores**: Arthur Zizumbo (MLOps Lead) · equipo Equipo 17
**Reemplaza el calendario de**: `CLAUDE.md` / `AGENTS.md` (sección "Calendario Inamovible") y `context/RefinamientoPlaneacionAgroSatCopilot_v6.md` §10.3 (líneas 730-739)
**Relacionado**: [`docs/audit/us-023-preview-v2-audit.md`](../audit/us-023-preview-v2-audit.md) (auditoría 28-may que documenta el desfase)

---

## Contexto

A 29-may-2026 el proyecto está **~1 semana desfasado** respecto al plan v6:

- El plan v6 §10.3 fijaba **Avance 4 (modelos) el dom 24-may** y **Avance 5 (final + ensambles) el dom 31-may**.
- En la práctica, **Avance 4 se entrega esta semana** (la semana S6 que el plan reservaba para Avance 5). La auditoría del 28-may confirmó que el Avance 4 (24-may) no se había entregado.
- El sistema del curso ahora muestra la **Presentación Final el dom 27-jun-2026** (antes 21-jun), lo que **otorga +1 semana de holgura** que absorbe exactamente el desfase actual.

**Decisión de esta ronda (scope limitado)**: solo se **replanean fechas**. El alcance técnico de modelos (cuántas de las 6 arquitecturas de segmentación, ensambles, VLM) se decide por separado con el equipo y, si aplica, con Dr. Camacho (ver pregunta abierta en la auditoría). Este ADR **no recorta alcance**; solo recorre el calendario.

## Decisión

Adoptar el calendario rediseñado siguiente como nuevo calendario oficial. Cada Avance pendiente se corre **+1 semana**; los Avances ya entregados (0-3) quedan fijos como histórico.

### Calendario rediseñado (oficial desde 2026-05-29)

| Avance | Fecha original (v6) | Fecha nueva | Estado | Entregable |
|--------|---------------------|-------------|--------|------------|
| Avance 0 | dom 26-abr | dom 26-abr (fija) | ENTREGADO | Propuesta PDF |
| Avance 1 (EDA) | dom 3-may | dom 3-may (fija) | ENTREGADO | Notebooks EDA |
| Avance 2 (FE) | dom 17-may | dom 17-may (fija) | ENTREGADO | Notebooks Feature Engineering |
| Avance 3 (Baseline) | mié 20-may | mié 20-may (fija) | ENTREGADO | Baseline tabular |
| **Avance 4 (Modelos)** | dom 24-may | **dom 31-may** | **ESTA SEMANA** | Modelos alternativos (alcance a confirmar con equipo) |
| **Avance 5 (Final + Ensambles)** | dom 31-may | **dom 7-jun** | Pendiente | Modelo final + ensambles |
| **Avance 6 (Conclusiones)** | dom 7-jun | **dom 14-jun** | Pendiente | PDF conclusiones |
| **Avance 7 (Resumen)** | dom 14-jun | **dom 21-jun** | Pendiente | PDF resumen |
| **Presentación Final** | dom 21-jun | **dom 27-jun** | Pendiente | Demo + presentación |
| Buffer + Paper Track | 22-jun → 3-jul | **28-jun → 3-jul** | Opcional | Paper Track (post-presentación) |

### Secuenciación semanal rediseñada (reemplaza §10.3)

```
S1  (20-26 abr): E0 Setup + Avance 0 PDF                          → Avance 0 dom 26-abr   [HECHO]
S2  (27-abr a 3-may): E1 Ingesta + E2 EDA univariado             → Avance 1 dom 3-may    [HECHO]
S3  (4-10 may):  E2 completo + arrancar E3 FE                                            [HECHO]
S4  (11-17 may): E3 FE + arrancar E4 Baseline                    → Avance 2 dom 17-may   [HECHO]
S5  (18-24 may): E4 Baseline                                     → Avance 3 mié 20-may   [HECHO]
S6  (25-31 may): US-023-preview correcciones + E5 modelos        → Avance 4 dom 31-may   [EN CURSO]
S7  (1-7 jun):   E5 cierre + E6 ensambles + modelo final         → Avance 5 dom 7-jun
S8  (8-14 jun):  E6 VLM/agente + E8 backend + conclusiones        → Avance 6 dom 14-jun
S9  (15-21 jun): E9 frontend + E10 observabilidad + resumen       → Avance 7 dom 21-jun
S10 (22-27 jun): Pulido final + dry-runs + grabar demo            → Presentación dom 27-jun
S11 (28-jun a 3-jul): Buffer + Paper Track opcional
```

### Ventanas de cómputo H100/L4 — corridas +1 semana

| Ventana | Original | Nueva | Uso |
|---------|----------|-------|-----|
| V1 | 18-20 may | (consumida/omitida — ver auditoría: 0h H100 reales) | Baselines |
| V2 | 25-27 may | **1-3 jun** | U-TAE + TSViT + Swin-UNETR (si se mantienen) |
| V3 | 28-30 may | **4-6 jun** | Gemma 4 26B-MoE LoRA (si se mantiene) |
| V4 | 1-3 jun | **8-10 jun** | Qwen3-VL LoRA + ensambles |
| V5 | 5-7 jun | **12-14 jun** | Qwen3.5-35B-A3B vLLM |
| V6 | 18-20 jun | **24-26 jun** | Warm vLLM demo |

> Nota: la auditoría 28-may documenta que la VM H100 Azure está **parked** y 0h consumidas. Estas ventanas son válidas solo si se reactiva H100; en caso contrario el cómputo recae en L4 GCP y el alcance debe ajustarse en la decisión de scope (separada de este ADR de fechas).

## Consecuencias

- **Positiva**: el +1 semana de la presentación (27-jun) absorbe íntegro el desfase de 1 semana sin comprimir ningún Avance.
- **Positiva**: el espaciado dominical del curso se preserva (todos los Avances caen en domingo, salvo el histórico A3 del miércoles).
- **Riesgo abierto (NO resuelto por este ADR)**: el alcance técnico de Avances 4-5 (6 segmentadores, ensambles, VLM) sigue sobredimensionado para la capacidad del equipo según la auditoría. Este ADR mueve fechas; **la decisión de recorte de alcance queda pendiente** y debe tomarse antes de cerrar Avance 4.
- **Acción de sincronización**: actualizar la línea "Calendario Inamovible" en `CLAUDE.md` y su espejo `AGENTS.md`, y §10.3 del plan v6. (Ambos espejos deben quedar idénticos — ver `.claude/CLAUDE.md`.)
