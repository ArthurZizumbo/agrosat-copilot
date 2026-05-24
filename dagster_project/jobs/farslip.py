"""Job Dagster — re-materializacion completa del pipeline FarSLIP (US-022b-B).

Selecciona los assets del flujo end-to-end:

::

    sentinel2_crops_256 (3 ROIs)
      -> farslip_embeddings_italy (3 ROIs, depende de modelo MLflow)
      -> farslip_embeddings_consolidated (data/farslip/embeddings_italy.parquet)

El asset external ``farslip_clip_italy_v1`` (modelo en MLflow Registry) NO se
incluye en el job — se entrena fuera de Dagster via ``make train-l4`` (regla
ml/CLAUDE.md). Dagster solo lo declara como dep para mantener el lineage UI.

Ejecucion:

    dagster job execute -m dagster_project.definitions -j farslip_full_pipeline_job

O via la UI ("Jobs" -> ``farslip_full_pipeline_job`` -> "Launchpad").
"""

from __future__ import annotations

from dagster import AssetSelection, define_asset_job

#: Selecciona el subgrafo materializable del pipeline FarSLIP.
#: Excluye external assets (``farslip_clip_italy_v1``, ``farslip_pairs_italy``).
#: ``AssetSelection.groups("farslip")`` matchea todos los assets con
#: ``group_name="farslip"``; ``- AssetSelection.assets(...)`` resta los specs
#: externos no materializables.
farslip_full_pipeline_job = define_asset_job(
    name="farslip_full_pipeline_job",
    selection=AssetSelection.groups("farslip"),
    description=(
        "Pipeline FarSLIP end-to-end (US-022b-B): crops Sentinel-2 -> "
        "bulk extraction embeddings 512-dim -> parquet consolidado. "
        "Skipea limpio si GCS/MLflow no disponibles."
    ),
    tags={
        "us": "US-022b",
        "epic": "E4",
        "pipeline": "farslip",
    },
)
