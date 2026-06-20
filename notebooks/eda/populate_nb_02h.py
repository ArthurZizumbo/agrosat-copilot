"""Build ``notebooks/eda/02h_mexico_demo.ipynb`` from source cells (US-077).

Idempotent generator: writes the notebook skeleton (no outputs); papermill then
executes it end-to-end so it is committed WITH outputs. Run from repo root:

    poetry run python notebooks/eda/populate_nb_02h.py
    poetry run papermill notebooks/eda/02h_mexico_demo.ipynb \
        notebooks/eda/02h_mexico_demo.ipynb

Demo CUALITATIVA zero-shot Mexico aguacate/guayaba (datos REALES de GEE, cero
sinteticos):
- AlphaEarth zonal 64-dim REAL por AOI (firma satelital del cultivo).
- Curva NDVI temporal REAL 2023 de Sentinel-2 (perenne arboreo = estable).
- Descripcion fenologica REAL via Gemini Flash (alineacion fenologia-texto).
- Encuadre HCAT cualitativo (PERMANENT_WOODY/orchard, US-074) por ANALOGIA.
- CAVEAT explicito: SIN F1/accuracy para Mexico (no hay ground-truth curado).
- Modo `degraded=true`: si GEE falla, placeholder explicito SIN curva inventada.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = Path(__file__).resolve().parent / "02h_mexico_demo.ipynb"


def _md(text: str) -> dict:
    return new_markdown_cell(text)


def _code(src: str) -> dict:
    return new_code_cell(src)


def _params(src: str) -> dict:
    cell = new_code_cell(src)
    cell.metadata["tags"] = ["parameters"]
    return cell


CELLS: list[dict] = [
    _md(
        "# 02h - Demo Mexico aguacate/guayaba (zero-shot CUALITATIVO, US-077)\n"
        "\n"
        "Aplicamos el **mismo pipeline** del baseline europeo (embedding "
        "AlphaEarth zonal + curva NDVI fenologica de Sentinel-2 + descripcion-"
        "texto con Gemini) a dos zonas productoras reales de Mexico, usando "
        "AlphaEarth `V1/ANNUAL`, que es **global e incluye Mexico** (CC-BY-4.0).\n"
        "\n"
        "> **METODOLOGIA ZERO-SHOT CUALITATIVA -- SIN CLAIM DE EXACTITUD.** El "
        "objetivo es mostrar que la **metodologia es replicable** a otras zonas, "
        "no medir su exactitud. **NO reportamos F1 ni accuracy para Mexico**: no "
        "existe un conjunto de ground-truth curado de aguacate/guayaba que "
        "permita validarlo. Lo que mostramos es REAL (la curva NDVI de la "
        "parcela, su embedding AlphaEarth y la descripcion fenologica generada); "
        "lo que dejamos como **trabajo futuro** es la validacion metrica (un "
        "F1>=0.80 mexicano requiere muestras etiquetadas curadas, fuera de "
        "alcance de esta presentacion).\n"
        "\n"
        "**Lo que SI mostramos** (todo REAL, de GEE):\n"
        "- El embedding AlphaEarth 64-dim zonal de cada AOI (la firma satelital).\n"
        "- La curva NDVI temporal 2023 de Sentinel-2 (la fenologia observada).\n"
        "- La descripcion fenologica generada por Gemini (la alineacion "
        "fenologia-texto).\n"
        "\n"
        "**Lo que NO mostramos** (y por que):\n"
        "- Ningun F1/accuracy/clasificador sobre Mexico: no hay ground-truth "
        "curado de aguacate/guayaba. Cualquier numero de clasificacion seria "
        "una sobre-afirmacion. El encuadre HCAT (orchard) es **analogia "
        "cualitativa**, NO una prediccion.\n"
        "\n"
        "Aguacate y guayaba son cultivos **perennes arboreos** (no estacionales): "
        "su firma NDVI esperada es alta y relativamente **estable todo el anio** "
        "(sin el ciclo siembra-cosecha de un cereal anual). Eso es justo lo que "
        "la curva REAL debe revelar."
    ),
    _params(
        "# Parametros papermill (defaults para entregable; degraded para el smoke de CI).\n"
        "year = 2023\n"
        "cloud_pct_max = 40       # mas permisivo que el 30 europeo (nubosidad tropical)\n"
        "buffer_m = 1500          # ~7 km2 por AOI (promedia la huerta)\n"
        "gemini_model = 'gemini-3.5-flash'\n"
        "degraded = False         # True = sin GEE; muestra el pipeline + placeholder\n"
    ),
    _code(
        "from __future__ import annotations\n"
        "\n"
        "from dataclasses import replace\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "from IPython.display import Markdown, display\n"
        "\n"
        "from ml.utils.notebook_setup import find_repo_root, configure_ee_from_env\n"
        "from ml.ingest.gee_sampler import init_ee\n"
        "from ml.transfer import mexico_demo as mx\n"
        "\n"
        "repo_root = find_repo_root()\n"
        "# AOIs reales (coords EPSG:4326 verificadas en GEE); aplicamos el buffer del parametro.\n"
        "aois = tuple(replace(a, buffer_m=buffer_m) for a in mx.DEFAULT_AOIS)\n"
        "\n"
        "# Bootstrap GEE: service account .env/gee-service-account.json (agrosat-copilot).\n"
        "if not degraded:\n"
        "    try:\n"
        "        proj, sa = configure_ee_from_env(repo_root)\n"
        "        default_sa = repo_root / '.env' / 'gee-service-account.json'\n"
        "        sa_path = sa if sa is not None else default_sa\n"
        "        sa_path = sa_path if sa_path.is_file() else None\n"
        "        init_ee(service_account_json=sa_path, project=proj)\n"
        "        auth_msg = (\n"
        "            f'Earth Engine inicializado (project=`{proj}`, '\n"
        "            f'service_account={\"si\" if sa_path else \"ADC\"}).'\n"
        "        )\n"
        "    except Exception as exc:  # noqa: BLE001 -- degradar sin romper el notebook\n"
        "        degraded = True\n"
        "        auth_msg = (\n"
        "            f'GEE no disponible ({exc}); modo `degraded` -> '\n"
        "            'placeholder, sin curva inventada.'\n"
        "        )\n"
        "else:\n"
        "    auth_msg = (\n"
        "        'Modo `degraded` forzado por parametro: pipeline mostrado, '\n"
        "        'sin llamadas GEE.'\n"
        "    )\n"
        "display(Markdown(f'**Auth GEE**: {auth_msg}'))\n"
    ),
    _md(
        "## 1. Las AOIs mexicanas (coordenadas REALES)\n"
        "\n"
        "Dos zonas productoras reales y documentadas. Geometria = "
        "`ee.Geometry.Point(lon,lat).buffer(buffer_m)`. Centroides elegidos sobre "
        "vegetacion arborea conocida, lejos de mancha urbana."
    ),
    _code(
        "aoi_table = pl.DataFrame(\n"
        "    {\n"
        "        'aoi': [a.name for a in aois],\n"
        "        'cultivo': [a.crop for a in aois],\n"
        "        'lon': [a.lon for a in aois],\n"
        "        'lat': [a.lat for a in aois],\n"
        "        'buffer_m': [a.buffer_m for a in aois],\n"
        "        'fenologia_esperada': [a.expected_phenology for a in aois],\n"
        "    }\n"
        ")\n"
        "display(aoi_table)\n"
        "display(Markdown(\n"
        "    '- **AOI-1 aguacate**: Uruapan / faldas del Tancitaro, Michoacan '\n"
        "    '(cinturon aguacatero).\\n'\n"
        "    '- **AOI-2 guayaba**: Calvillo, Aguascalientes (capital nacional de la guayaba).'\n"
        "))\n"
    ),
    _md(
        "## 2. Embedding AlphaEarth 64-dim zonal REAL (la firma satelital)\n"
        "\n"
        "`reduceRegion(mean)` del mosaico AlphaEarth del anio sobre el buffer de "
        "cada AOI -> un vector de 64 dimensiones REAL por cultivo. Es la "
        "representacion aprendida del foundation model EO, exactamente la misma "
        "feature que alimenta el baseline europeo."
    ),
    _code(
        "alphaearth_rows = []\n"
        "embeddings = {}\n"
        "if not degraded:\n"
        "    for a in aois:\n"
        "        vec = mx.extract_alphaearth_zonal(a, year)\n"
        "        embeddings[a.name] = vec\n"
        "        if vec.size == mx.ALPHAEARTH_N_DIMS:\n"
        "            alphaearth_rows.append(\n"
        "                {\n"
        "                    'aoi': a.name,\n"
        "                    'cultivo': a.crop,\n"
        "                    'n_dims': int(vec.size),\n"
        "                    'norm': float(np.linalg.norm(vec)),\n"
        "                    'A00': float(vec[0]),\n"
        "                    'A01': float(vec[1]),\n"
        "                    'A02': float(vec[2]),\n"
        "                }\n"
        "            )\n"
        "        else:\n"
        "            display(Markdown(\n"
        "                f'> **AOI `{a.name}`**: AlphaEarth vacio (GEE no devolvio datos). '\n"
        "                'Sin vector inventado.'\n"
        "            ))\n"
        "    if alphaearth_rows:\n"
        "        display(pl.DataFrame(alphaearth_rows))\n"
        "else:\n"
        "    display(Markdown('> **Modo degradado**: no se consulta AlphaEarth (sin GEE).'))\n"
    ),
    _code(
        "# Figura 1: barra del embedding 64-dim por AOI (la firma satelital real).\n"
        "valid_emb = {k: v for k, v in embeddings.items() if v.size == mx.ALPHAEARTH_N_DIMS}\n"
        "if not degraded and valid_emb:\n"
        "    fig, ax = plt.subplots(figsize=(11, 4))\n"
        "    dims = np.arange(mx.ALPHAEARTH_N_DIMS)\n"
        "    width = 0.4\n"
        "    for i, (name, vec) in enumerate(valid_emb.items()):\n"
        "        ax.bar(dims + (i - 0.5) * width, vec, width=width, label=name)\n"
        "    ax.set_xlabel('Dimension del embedding AlphaEarth (A00..A63)')\n"
        "    ax.set_ylabel('Valor zonal medio (real)')\n"
        "    ax.set_title('Embedding AlphaEarth 64-dim por AOI (firma satelital del cultivo)')\n"
        "    ax.legend()\n"
        "    ax.grid(True, axis='y', alpha=0.3)\n"
        "    display(fig)\n"
        "    plt.close(fig)\n"
        "else:\n"
        "    display(Markdown('> **Modo degradado / sin datos**: figura del embedding omitida.'))\n"
    ),
    _md(
        "## 3. Curva NDVI temporal REAL 2023 (la fenologia observada)\n"
        "\n"
        "Coleccion Sentinel-2 `S2_SR_HARMONIZED` con mascara de nubes QA60, NDVI "
        "`normalizedDifference(['B8','B4'])` y `reduceRegion(mean)` por imagen. Se "
        "reporta el numero de imagenes usadas (auditable). La firma esperada del "
        "perenne arboreo es una curva **alta y estable**, sin el pico unico "
        "estacional de un cultivo anual."
    ),
    _code(
        "ndvi_series = {}\n"
        "if not degraded:\n"
        "    for a in aois:\n"
        "        ser = mx.extract_s2_ndvi_series(a, year, cloud_pct_max=cloud_pct_max)\n"
        "        ndvi_series[a.name] = ser\n"
        "        if ser.is_empty():\n"
        "            display(Markdown(\n"
        "                f'> **AOI `{a.name}`**: serie NDVI vacia (GEE no devolvio imagenes). '\n"
        "                'Sin curva inventada.'\n"
        "            ))\n"
        "        else:\n"
        "            display(Markdown(\n"
        "                f'- **{a.name}** ({a.crop}): {ser.height} imagenes S2 usadas, '\n"
        "                f'NDVI medio {ser[\"ndvi\"].mean():.3f}, '\n"
        "                f'rango [{ser[\"ndvi\"].min():.3f}, {ser[\"ndvi\"].max():.3f}].'\n"
        "            ))\n"
        "else:\n"
        "    display(Markdown('> **Modo degradado**: no se consulta Sentinel-2 (sin GEE).'))\n"
    ),
    _code(
        "# Figura 2: NDVI vs DOY 2023, una linea por AOI (la fenologia real).\n"
        "valid_ndvi = {k: v for k, v in ndvi_series.items() if not v.is_empty()}\n"
        "if not degraded and valid_ndvi:\n"
        "    fig, ax = plt.subplots(figsize=(10, 5))\n"
        "    for name, ser in valid_ndvi.items():\n"
        "        s = ser.sort('doy')\n"
        "        ax.plot(\n"
        "            s['doy'].to_numpy(), s['ndvi'].to_numpy(),\n"
        "            marker='o', ms=3, lw=1, label=name,\n"
        "        )\n"
        "    ax.set_xlabel('Dia del anio (DOY) 2023')\n"
        "    ax.set_ylabel('NDVI zonal medio (real, Sentinel-2)')\n"
        "    ax.set_title('Curva NDVI temporal 2023 por AOI (perenne arboreo)')\n"
        "    ax.set_ylim(-0.1, 1.0)\n"
        "    ax.legend()\n"
        "    ax.grid(True, alpha=0.3)\n"
        "    display(fig)\n"
        "    plt.close(fig)\n"
        "else:\n"
        "    display(Markdown(\n"
        "        '> **Modo degradado / sin datos**: curva NDVI omitida (sin numeros).'\n"
        "    ))\n"
    ),
    _md(
        "## 4. Descripcion fenologica REAL (alineacion fenologia-texto, Gemini)\n"
        "\n"
        "Pasamos la curva NDVI REAL al generador de descripciones fenologicas "
        "(`ml/features/phenology_description.py`, prompt 3-bloques de Wen et al., "
        "Gemini 3.5 Flash, `temperature=0.0`). Es exactamente el modulo del "
        "proyecto europeo, reusado sin tocar. La descripcion captura "
        "cualitativamente la firma del perenne arboreo."
    ),
    _code(
        "descriptions = {}\n"
        "if not degraded and valid_ndvi:\n"
        "    for name, ser in valid_ndvi.items():\n"
        "        aoi = next(a for a in aois if a.name == name)\n"
        "        text = mx.describe_phenology(ser, aoi, model=gemini_model)\n"
        "        descriptions[name] = text\n"
        "        display(Markdown(f'**{name}** ({aoi.crop}): {text}'))\n"
        "else:\n"
        "    display(Markdown(\n"
        "        '> **Modo degradado / sin datos**: no se genera descripcion '\n"
        "        '(requiere la curva NDVI real + `GEMINI_API_KEY`). Sin texto inventado.'\n"
        "    ))\n"
    ),
    _md(
        "## 5. Encuadre HCAT cualitativo (PERMANENT_WOODY / orchard, US-074)\n"
        "\n"
        "El cultivo perenne arboreo (aguacate/guayaba) se **encuadra por analogia "
        "fenologica** en la familia `PERMANENT_WOODY` del label-space `hcat-macro` "
        "de US-074 (clases `orchard` / `vineyard`). Es la familia arborea/perenne "
        "mas cercana del espacio europeo.\n"
        "\n"
        "> Esto es un **encuadre por analogia fenologica, NO una prediccion ni un "
        "F1**. No se llama a ningun clasificador ni se asigna probabilidad: solo "
        "se nombra la familia analoga del espacio de cultivos comun."
    ),
    _code(
        "framing = mx.hcat_perennial_framing()\n"
        "display(pl.DataFrame(\n"
        "    {'clase_hcat_macro': list(framing.keys()), 'familia_macro': list(framing.values())}\n"
        "))\n"
        "display(Markdown(\n"
        "    'La firma NDVI alta y estable de aguacate/guayaba es **analoga** a la de '\n"
        "    'un huerto (`orchard`, familia `PERMANENT_WOODY`) del espacio HCAT comun. '\n"
        "    'Es una analogia cualitativa, **no** una clasificacion.'\n"
        "))\n"
    ),
    _md(
        "## 6. Export de los artefactos REALES (ligeros, al Git)\n"
        "\n"
        "Serie NDVI por AOI + embedding AlphaEarth por AOI. Son KB, datos REALES "
        "de GEE; el cache crudo (`data/cache/gee/*.parquet`) NO se commitea "
        "(se regenera determinista)."
    ),
    _code(
        "ndvi_out = repo_root / 'data' / 'transfer' / 'mexico_demo_ndvi.parquet'\n"
        "emb_out = repo_root / 'data' / 'transfer' / 'mexico_demo_alphaearth.parquet'\n"
        "if not degraded and valid_ndvi and valid_emb:\n"
        "    ndvi_out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    ndvi_frames = [\n"
        "        ser.with_columns(pl.lit(name).alias('aoi')) for name, ser in valid_ndvi.items()\n"
        "    ]\n"
        "    ndvi_all = pl.concat(ndvi_frames).select('aoi', 'date', 'doy', 'ndvi')\n"
        "    ndvi_all.write_parquet(ndvi_out)\n"
        "    band_names = [f'A{i:02d}' for i in range(mx.ALPHAEARTH_N_DIMS)]\n"
        "    emb_rows = []\n"
        "    for name, vec in valid_emb.items():\n"
        "        row = {'aoi': name, 'cultivo': next(a.crop for a in aois if a.name == name)}\n"
        "        row.update({b: float(vec[i]) for i, b in enumerate(band_names)})\n"
        "        emb_rows.append(row)\n"
        "    emb_all = pl.DataFrame(emb_rows)\n"
        "    emb_all.write_parquet(emb_out)\n"
        "    rb_ndvi = pl.read_parquet(ndvi_out)\n"
        "    rb_emb = pl.read_parquet(emb_out)\n"
        "    assert rb_ndvi.height == ndvi_all.height, 'readback NDVI mismatch'\n"
        "    assert rb_emb.height == emb_all.height, 'readback embedding mismatch'\n"
        "    display(Markdown(\n"
        "        f'NDVI -> `{ndvi_out}` ({rb_ndvi.height} filas reales); '\n"
        "        f'embedding -> `{emb_out}` ({rb_emb.height} AOIs).'\n"
        "    ))\n"
        "else:\n"
        "    display(Markdown(\n"
        "        '> **Modo degradado / sin datos**: no se escribe parquet (sin datos reales).'\n"
        "    ))\n"
    ),
    _md(
        "## Conclusiones\n"
        "\n"
        "**Que mostramos**: el **mismo** pipeline (AlphaEarth zonal + curva NDVI "
        "fenologica + descripcion-texto con Gemini) aplicado a dos huertas reales "
        "de Mexico -- aguacate en Uruapan/Tancitaro (Michoacan) y guayaba en "
        "Calvillo (Aguascalientes). La metodologia se **replica** a otra zona sin "
        "ningun cambio: AlphaEarth es global e incluye Mexico.\n"
        "\n"
        "**Lo honesto del ejercicio**:\n"
        "- Todo lo mostrado es REAL y sale de GEE: el embedding AlphaEarth de 64 "
        "dimensiones, la curva NDVI 2023 de Sentinel-2 (con el numero de imagenes "
        "reportado) y la descripcion fenologica generada por Gemini.\n"
        "- **NO reportamos F1 ni accuracy para Mexico.** No existe un conjunto de "
        "ground-truth curado de aguacate/guayaba que permita validar una "
        "clasificacion. El encuadre HCAT (orchard) es una **analogia cualitativa** "
        "de la firma fenologica, no una prediccion.\n"
        "- La curva alta y estable es coherente con un cultivo **perenne arboreo** "
        "(sin el pico estacional unico de un cereal anual).\n"
        "\n"
        "**Lo que sigue**: una validacion metrica (un F1>=0.80 mexicano) requiere "
        "muestras etiquetadas curadas de aguacate/guayaba, que quedan como trabajo "
        "futuro. Hoy la evidencia es la **replicabilidad metodologica** zero-shot, "
        "con la honestidad de no afirmar una exactitud que no medimos."
    ),
]


def main() -> None:
    """Write the notebook skeleton (no outputs) to :data:`NB_PATH`."""
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
    print(f"Wrote {NB_PATH} with {len(CELLS)} cells.")


if __name__ == "__main__":
    main()
