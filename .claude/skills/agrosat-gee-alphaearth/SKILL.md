---
name: agrosat-gee-alphaearth
description: Access Google Earth Engine and download AlphaEarth Foundations v2.1 (Satellite Embedding V1 Annual) 64-dim embeddings for AgroSatCopilot. Use when ingesting AlphaEarth data, configuring service account authentication, exporting to GCS as COG, or querying GEE collections.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot GEE + AlphaEarth Skill

## Rules — NON-NEGOTIABLE

- Service account con rol Earth Engine Resource Writer
- Credenciales JSON solo en Secret Manager / `.env.local`, jamás en Git
- Colección oficial: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Exports vía `ee.batch.Export.image.toCloudStorage` con formato COG
- Compresión DEFLATE + nodata declarado
- 3 ROI italianos + control PASTIS-R francés
- Atribución obligatoria a Google DeepMind en `docs/licenses/DATA_LICENSE.md`

## Setup

```python
import ee
import json
from pathlib import Path

def authenticate_gee(service_account_json_path: str, project_id: str):
    """Autentica Earth Engine con service account."""
    sa_email = json.loads(Path(service_account_json_path).read_text())["client_email"]
    credentials = ee.ServiceAccountCredentials(sa_email, service_account_json_path)
    ee.Initialize(credentials, project=project_id)
```

## Download AlphaEarth para ROI

```python
def download_alphaearth(roi_geom: dict, roi_name: str, year: int, bucket: str = "agrosat-data"):
    """
    Lanza export a GCS:
    gs://agrosat-data/alphaearth/{roi_name}/{year}.tif
    """
    region = ee.Geometry(roi_geom)
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    image = collection.filterDate(f"{year}-01-01", f"{year+1}-01-01").first().clip(region)

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"alphaearth_{roi_name}_{year}",
        bucket=bucket,
        fileNamePrefix=f"alphaearth/{roi_name}/{year}",
        region=region,
        scale=10,
        crs="EPSG:32633",
        fileFormat="GeoTIFF",
        formatOptions={
            "cloudOptimized": True,
            "compression": "DEFLATE",
            "noData": -9999,
        },
        maxPixels=1e10,
    )
    task.start()
    return task.id
```

## Config ROIs

```yaml
# config/rois.yaml
rois:
  - name: pianura_padana
    region: Italy
    bbox: [9.0, 44.5, 12.0, 46.0]
    crs: EPSG:4326
    preferred_crs_projection: EPSG:32633
    crops: [arroz, maíz, soja, trigo]
    geometry_path: data/reference/rois/pianura_padana.geojson
  - name: toscana_centrale
    bbox: [10.5, 43.0, 12.0, 44.0]
    crops: [olivo, vid, girasol, trigo_duro]
  - name: apulia
    bbox: [15.0, 40.5, 17.5, 42.0]
    crops: [olivo, vid, hortalizas]
  - name: pastis_r_control
    region: France
    bbox: [-0.5, 43.0, 5.0, 46.0]
```

## CLI Script

```python
# scripts/download_alphaearth.py
import typer
import yaml
from pathlib import Path

app = typer.Typer()

@app.command()
def download(roi: str = "all", year_from: int = 2017, year_to: int = 2025):
    cfg = yaml.safe_load(Path("config/rois.yaml").read_text())
    authenticate_gee(".env/gee-service-account.json", "agrosat-prod")
    rois = cfg["rois"] if roi == "all" else [r for r in cfg["rois"] if r["name"] == roi]
    for r in rois:
        geom = json.loads(Path(r["geometry_path"]).read_text())
        for year in range(year_from, year_to + 1):
            task_id = download_alphaearth(geom, r["name"], year)
            print(f"Launched task {task_id}: {r['name']} {year}")

if __name__ == "__main__":
    app()
```

## Dagster Asset

```python
# dagster_project/assets/alphaearth.py
from dagster import asset, RetryPolicy, MaterializeResult, MetadataValue

@asset(
    retry_policy=RetryPolicy(max_retries=3, delay=60.0),
    deps=[],
)
def alphaearth_annual(context, gee: GEEResource):
    """Descarga AlphaEarth v2.1 annual para los 3 ROI italianos + PASTIS."""
    config = load_rois_config()
    results = []
    for roi in config["rois"]:
        for year in range(2017, 2026):
            task_id = gee.export_alphaearth(roi, year)
            results.append({"roi": roi["name"], "year": year, "task_id": task_id})
    return MaterializeResult(
        metadata={
            "num_exports": MetadataValue.int(len(results)),
            "rois": MetadataValue.json(results),
        }
    )
```

## Atribución Legal

```markdown
# docs/licenses/DATA_LICENSE.md (sección AlphaEarth)
AlphaEarth Foundations v2.1 — Google DeepMind
Source: Google Earth Engine Data Catalog `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
License: GEE Terms of Service (https://earthengine.google.com/terms/)
Permitted: research + commercial with attribution
Citation: Brown et al. (2024). AlphaEarth Foundations. Google DeepMind.
```

## QA Checklist GEE

- [ ] Service account con permisos mínimos
- [ ] Credenciales en Secret Manager
- [ ] COG validado con `rio cogeo validate`
- [ ] Tabla `alphaearth_tiles` actualizada en PostGIS
- [ ] Atribución en DATA_LICENSE.md
- [ ] Asset Dagster con retry + lineage
- [ ] Tests con mock `ee.batch`
