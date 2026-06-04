"""Dagster job — full re-materialization of the FarSLIP pipeline (US-022b-B).

Selects the assets of the end-to-end flow:

::

    sentinel2_crops_256 (3 ROIs)
      -> farslip_embeddings_italy (3 ROIs, depends on MLflow model)
      -> farslip_embeddings_consolidated (data/farslip/embeddings_italy.parquet)

The external asset ``farslip_clip_italy_v1`` (model in the MLflow Registry) is
NOT included in the job — it is trained outside Dagster via ``make train-l4``
(ml/CLAUDE.md rule). Dagster only declares it as a dep to keep the lineage UI.

Execution:

    dagster job execute -m dagster_project.definitions -j farslip_full_pipeline_job

Or via the UI ("Jobs" -> ``farslip_full_pipeline_job`` -> "Launchpad").
"""

from __future__ import annotations

from dagster import AssetSelection, define_asset_job

#: Selects the materializable subgraph of the FarSLIP pipeline.
#: Excludes external assets (``farslip_clip_italy_v1``, ``farslip_pairs_italy``).
#: ``AssetSelection.groups("farslip")`` matches all assets with
#: ``group_name="farslip"``; ``- AssetSelection.assets(...)`` subtracts the
#: non-materializable external specs.
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
