"""Punto de entrada Dagster — agrega assets, resources, jobs y schedules.

Arranque: ``poetry run dagster dev -m dagster_project.definitions``.

Conforme avanzan las US se irán agregando assets reales (alphaearth, sentinel,
dinov3, features, models, drift, stac) y sus resources/schedules.
"""

from dagster import Definitions, load_assets_from_modules

from dagster_project.assets import farslip, features, health, sentinel2_crops

all_assets = load_assets_from_modules([health, features, sentinel2_crops, farslip])

defs = Definitions(
    assets=all_assets,
    resources={},
)
