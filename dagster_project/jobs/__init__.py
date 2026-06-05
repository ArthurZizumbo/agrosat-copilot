"""Dagster jobs — selections of assets executable as a unit.

Each job is an ``AssetSelection`` materializable via ``make`` or from the UI.
"""

from dagster_project.jobs.farslip import farslip_full_pipeline_job

__all__ = ["farslip_full_pipeline_job"]
