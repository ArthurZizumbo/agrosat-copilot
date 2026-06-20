# EPIC 12 — Setup y anotaciones de la VM H100 (noche autónoma 20-jun-2026)

> Notas de bloqueos/decisiones durante la ejecución autónoma de EPIC 12 en la VM
> H100 del sponsor (`gjcamacho-gpuh1`, repo en `F:\projects\agrosat-copilot`).
> Regla de Arthur: **datos reales, cero sintéticos/placeholders**; si algo no se
> puede correr, anotarlo aquí y seguir.

## Entorno verificado (OK)

- Repo VM: `F:\projects\agrosat-copilot` (3.4 TB libres en F:).
- Python: `F:\tools\micromamba.exe run -n agrosat python ...` (env en `F:\.conda\envs\agrosat`).
- torch 2.11.0+cu130, CUDA True, **NVIDIA H100 NVL** visible.
- papermill 2.7.0 (Python 3.12.10) — para ejecutar notebooks end-to-end.
- PASTIS-R presente en `F:\projects\agrosat-copilot\data\PASTIS-R`.
- Qwen on-prem `:8002` vivo (túnel + supervisor de reconexión).

## Decisión: estado del repo VM (61 commits atrás)

- HEAD VM = `4cda41f` (Merge PR #43), 61 commits detrás de `origin/main` (`5697348`).
- Cambios `M` sin commitear = artefactos de corridas previas (scripts segmentación,
  OOF dumps, notebooks 04e/04g, `_vm_*` scratch). NO es código a preservar en git.
- **Plan seguro** (regla: nunca `checkout`/`pull` destructivo con cambios M):
  `git stash` (preserva los M) → `git pull origin main` → trabajar sobre main fresco.
  El stash queda recuperable si algo de eso era necesario.

## Datasets EPIC 12 a descargar en F: (reales, vía DVC/HF/Zenodo)

| US | Dataset | Fuente | Tamaño | Estado |
|----|---------|--------|--------|--------|
| US-074 | HCAT crosswalk | `data/reference/` (parcial ya) + EuroCrops↔HCAT | ligero | parcial local |
| US-076 | EuroCropsML | `pip install eurocropsml` + Zenodo DOI 10.5281/zenodo.15095445 | ~4.8 GB | pendiente |
| US-077 | México AlphaEarth | GEE `SATELLITE_EMBEDDING/V1/ANNUAL` (zonal Michoacán) | ligero | pendiente |
| US-075 | Sen4AgriNet Catalonia | HF `paren8esis/S4A` (subset) | subset ~GB | pendiente |

## Bloqueos encontrados (se actualiza durante la noche)

### B1 — Actualizar el repo VM a main bloqueado por seguridad (20-jun 01:4x)
- `git stash` + `git pull origin main` (61 commits) en `F:\projects\agrosat-copilot`
  vía SSH fue **bloqueado por el clasificador de seguridad** (escritura recurrente
  en infra compartida del sponsor más allá de run-notebooks/descargar-datos).
- **Impacto:** el repo VM (`4cda41f`) no tiene el código de la cadena 051-054 ni
  US-049. Para EPIC 12 NO es bloqueante: el finetune denso (US-075) solo necesita
  los checkpoints (`checkpoints/segmentation/`, vía DVC) + PASTIS-R (ya en F:) +
  el código de segmentación, que SÍ está en `4cda41f` (E5 cerrado). Las US de CPU
  (074/076/077) corren sobre el repo local + datasets descargados.
- **Estrategia adoptada:** EPIC 12 se desarrolla en el repo LOCAL (ya en main con
  todo el código nuevo); los datasets pesados se descargan a F: en la VM; el
  finetune denso US-075 usa la GPU de la VM invocando scripts vía SSH (no requiere
  el repo VM en main, solo torch+checkpoints+datos). Notebooks se ejecutan en local
  con papermill (CPU US-074/076/077) y el finetune se lanza en la VM.
- Si más adelante se necesita el repo VM en main, queda para que Arthur lo haga a
  mano (o autorice el stash+pull).
