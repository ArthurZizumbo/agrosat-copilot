# Deploy a Streamlit Community Cloud

Esta carpeta contiene **todo lo específico** para desplegar el dashboard EDA
del Avance 1 (`app/eda_dashboard.py`) en
[share.streamlit.io](https://share.streamlit.io). El resto del repo no se
contamina con configuración de despliegue.

## Archivos

| Archivo | Para qué sirve |
|---------|----------------|
| `streamlit_app.py` | Entry point. Streamlit Cloud lo ejecuta; este script ajusta `sys.path` a la raíz del repo y delega en `app/eda_dashboard.py` |
| `requirements.txt` | Dependencias Python. Streamlit Cloud no usa Poetry — necesita un `requirements.txt` plano |
| `runtime.txt` | Fija Python 3.12 (necesario porque el dashboard usa sintaxis 3.12+) |
| `packages.txt` | Dependencias de sistema Linux (GDAL, GEOS, PROJ) para `geopandas` y `shapely` |

## Configuración en Streamlit Cloud

Al hacer clic en **New app** desde el dashboard:

| Campo | Valor |
|-------|-------|
| Repository | `ArthurZizumbo/agrosat-copilot` |
| Branch | `us-013` (o `main` después del merge) |
| Main file path | `deploy/streamlit/streamlit_app.py` |
| App URL (opcional) | `agrosat-eda-avance1` o el que prefieras |
| Python version (advanced) | 3.12 |

## Cómo funciona el entry point

`streamlit_app.py` está dos niveles por debajo de la raíz del repo:

```
agrosat-copilot/
├── app/
│   └── eda_dashboard.py        ← dashboard real
├── ml/
│   └── report/                  ← módulos que el dashboard importa
└── deploy/
    └── streamlit/
        ├── streamlit_app.py     ← este entry point
        ├── requirements.txt
        ├── runtime.txt
        ├── packages.txt
        └── README.md            ← este archivo
```

El entry point hace:

1. Calcula `repo_root = Path(__file__).resolve().parents[2]`.
2. Agrega `repo_root` a `sys.path` para que `from ml.report.notebook_content import …`
   funcione desde el sandbox de Streamlit Cloud.
3. Ejecuta `app/eda_dashboard.py` con `runpy.run_path` para que Streamlit lo
   interprete como script principal.

## Hardening de seguridad

El `.streamlit/config.toml` del repo root declara explícitamente:

- `enableXsrfProtection = true` — bloquea POSTs cross-origin (Streamlit Cloud
  expone el dashboard publicamente, así que CSRF es vector real).
- `enableCORS = false` — el dashboard no necesita servir requests desde
  otros origenes; bloquear CORS reduce superficie.

Estos dos flags aplican tanto en local como en Streamlit Community Cloud y son
los unicos del bloque `[server]` que Streamlit Cloud no ignora.

## Limitaciones conocidas del deploy gratuito

1. **Mapa folium sin tiles PASTIS** — `data/PASTIS-R/metadata.geojson` está en
   `.gitignore` (pesa 19 MB y vive en DVC). El dashboard degrada graceful y
   muestra solo las 3 ROIs italianas.
2. **PDF no se puede generar** — WeasyPrint requiere GTK3 runtime que
   Streamlit Cloud no instala. El PDF se sigue generando localmente con
   `make eda-pdf`.
3. **Memoria** — el plan gratuito ofrece 1 GB. `geopandas` + `polars` +
   `folium` consumen ~400 MB; queda margen suficiente.
4. **Cold start** — la primera carga tarda 30–60 s mientras instala
   dependencias del `requirements.txt`.

## Si querés probar el deploy local antes de subirlo

```powershell
# Desde la raíz del repo:
poetry run streamlit run deploy/streamlit/streamlit_app.py
```

Debería abrir el mismo dashboard que `make eda-dashboard`.

## Cómo actualizar el deploy

Streamlit Cloud hace re-deploy automático cuando hacés `git push` a la rama
configurada. No hay paso manual.

Si tenés que cambiar las dependencias:

1. Editá `deploy/streamlit/requirements.txt` o `packages.txt`.
2. `git add` + `git commit` + `git push`.
3. Streamlit Cloud detecta el cambio y reinstala automáticamente.
