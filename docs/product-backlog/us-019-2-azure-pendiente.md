# Backlog · US-019.2 — Azure pendiente: resolver tfvars + opcional aplicar modulo H100

**Origen**: el antiguo `us-019-1-terraform-tfstate-import-azure.md` agrupaba
DOS bloques distintos:
- **Bloque A**: importar `agrosat-tfstate` al state de Terraform.
- **Bloque B**: resolver placeholders Azure (`REPLACE-WITH-AZURE-SUBSCRIPTION-ID`)
  + opcional aplicar modulo H100.

El bloque A se consolido en [`us-023-cierre-fenologico-farslip-dvc.md`](us-023-cierre-fenologico-farslip-dvc.md)
§P4.3 (housekeeping infra). Este archivo conserva **solo el bloque B**, que
el usuario decidio (2026-05-23) **dejar pendiente a un lado** porque US-022b
y descendientes corren en GPU local + L4 GCP (no requieren H100 Azure).

**Status**: parked · no se retoma hasta decidir ventana H100
**Epic**: E5 (infra avanzada)
**SP**: 2
**Prioridad**: baja — solo se vuelve relevante si se decide ejecutar una
ventana H100 antes del 21-jun (presentacion final). Mientras tanto, el plan
v6 §"Presupuesto Computo" V1-V6 H100 esta en pausa.

---

## Alcance

### Resolver placeholders Azure en tfvars

`infrastructure/terraform/environments/dev/terraform.tfvars` contiene:

```hcl
azure_subscription_id = "REPLACE-WITH-AZURE-SUBSCRIPTION-ID"
admin_ssh_public_key  = "ssh-ed25519 AAAA...REPLACE-ME agrosat-mlops"
allowed_ssh_cidrs     = ["203.0.113.10/32", ...]  # IPs placeholder
```

Pasos cuando se retome:

1. `az login` + `az account set` (Arthur).
2. `az account show --query id -o tsv` → obtener subscription real.
3. Generar/colocar clave SSH publica del equipo (3 devs).
4. Resolver IPs SSH reales o decidir alternativa via Bastion / IAP.
5. **NO commitear** valores reales — `terraform.tfvars` esta gitignored;
   actualizar `terraform.tfvars.example` con instrucciones de bootstrap.
6. `terraform plan` sobre modulo Azure debe ser limpio.

### Opcional — Aplicar modulo H100 (ventana V1-V6)

Solo si se decide ejecutar entrenamiento H100 antes del 21-jun:
- Aplicar modulo Azure: H100 NVL 96 GB spot + NIC + NSG + auto-shutdown.
- Documentar costo real observado vs estimado (~$8/h spot).
- Decidir backend del state: mismo GCS (`gs://agrosat-tfstate/dev/`) o Azure
  Blob dedicado.

---

## Criterios de aceptacion

| AC | Criterio |
|----|----------|
| AC-1 | `terraform.tfvars` sin placeholders `REPLACE-*` (`grep REPLACE` vacio) |
| AC-2 | `terraform plan` sobre `environments/dev/` reporta 0 cambios pendientes en modulo Azure |
| AC-3 | `terraform.tfvars.example` actualizado con instrucciones de bootstrap |
| AC-4 | (opcional) modulo H100 aplicado con costo registrado en `docs/h100_log.md` |

---

## Por que parked

- US-022b y descendientes (US-023) corren en GPU local + L4 GCP — no requieren H100.
- Los entrenamientos pesados del plan v6 (Gemma 4 26B-MoE LoRA, Qwen3-VL LoRA,
  vLLM serving Qwen3.5-35B) son del EPIC 7+ (post-presentacion).
- El bloque B requiere ~3h de Arthur + decisiones de equipo (claves SSH, IPs)
  que no son urgentes ahora.

---

## Referencias

- US-019 handoff: [`docs/us-handoff/us-019.md`](../us-handoff/us-019.md)
- Manual-test US-019 §"Pendientes para un humano":
  [`docs/manual-test/us-019.md`](../manual-test/us-019.md)
- Bloque A consolidado en: [`us-023-cierre-fenologico-farslip-dvc.md`](us-023-cierre-fenologico-farslip-dvc.md) §P4.3
- Skills: `agrosat-terraform` + `agrosat-azure-h100`
