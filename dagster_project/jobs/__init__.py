"""Jobs Dagster — selecciones de assets ejecutables como una unidad.

Cada job es una ``AssetSelection`` materializable via ``make`` o desde la UI.
"""

from dagster_project.jobs.farslip import farslip_full_pipeline_job

__all__ = ["farslip_full_pipeline_job"]
