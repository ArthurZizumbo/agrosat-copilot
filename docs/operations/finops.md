# FinOps — Presupuesto y control de costos AgroSatCopilot

**Corte:** 7-jun-2026 · **Mantenedor:** Arthur Zizumbo (MLOps lead)

> Contenido movido desde `CLAUDE.md` raíz (que ahora es guía operacional, sin plan/presupuesto). El plan completo vive en [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md).

## Presupuesto objetivo

| Concepto | Monto | Nota |
|----------|-------|------|
| Operativo mensual | **~$115 USD/mes** | Con scale-to-zero (Cloud Run `min_instances=0`) |
| Training único (cuando aplicaba spot) | $262 (spot) — $602 (on-demand) | Histórico; la H100 del sponsor ahora es 24/7 sin costo para el equipo |
| GCP acumulado a la fecha | ~$0.30-0.49 USD | Holgado |
| Gemini API (descripciones FarSLIP + chat) | centavos | ~$0.0001/descripción; cabe en el operativo |

**Nota créditos:** el "Trial credit for GenAI App Builder" (~$17,178) es de Vertex AI Search/Agent Builder, **NO** cubre la Gemini API de generación de texto (SKU distinta). No se necesita: las descripciones cuestan centavos.

## Cómputo

- **H100 NVL 96GB**: prestada por el sponsor, 24/7 (no apagar). VM `gjcamacho-gpuh1`. Acceso: [`docs/infra/acceso-vm-h100-tunnel.md`](../infra/acceso-vm-h100-tunnel.md).
- **GCP L4 24GB** (`agrosat-farslip-trainer-dev`): spot con daemon de auto-shutdown por idle. Pararlo antes de runs manuales largos.

## Controles FinOps activos

- **Cloud SQL dev** con `activation_policy=NEVER` (apagada para ahorrar; var Terraform `db_activation_policy` evita drift).
- **Disco `farslip-data`** reducido 250→125 GB (GCP no encoge in-place: snapshot→disco nuevo→rsync→import TF).
- **Cloud Run** scale-to-zero verificado con `make scale-to-zero-check`.
- **Auditoría** mensual con `make cost-audit`.

## Comandos

```bash
make cost-audit            # reporte de costos GCP + Azure
make scale-to-zero-check   # verifica min_instances=0 en Cloud Run
make azure-h100-start/stop # control de la VM H100 (módulo de referencia)
```
