"""Build ``notebooks/eda/02g_eurocropsml_fewshot.ipynb`` from source cells (US-076).

Idempotent generator: writes the notebook skeleton (no outputs); papermill then
executes it end-to-end so it is committed WITH outputs. Run from repo root:

    poetry run python notebooks/eda/populate_nb_02g.py
    poetry run papermill notebooks/eda/02g_eurocropsml_fewshot.ipynb \
        notebooks/eda/02g_eurocropsml_fewshot.ipynb

Curva k-shot REAL (datos de EuroCropsML, cero sinteticos):
- Contexto del domain gap transnacional (LV+PT->EE, LV->EE; Francia NO esta).
- Conteos reales por pais + distribucion macro HCAT (long-tail).
- Curva F1-macro vs k (escala log) con barras de error (seeds) y la referencia
  del paper @500-shot anotada (0.66 pre-train-LV / 0.57 sin-pretrain).
- Tabla por escenario + export del parquet de resultados.
- Modo `degraded=true`: si el subset no esta descargado, muestra el pipeline y
  un placeholder explicito SIN numeros fabricados (regla Arthur).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = Path(__file__).resolve().parent / "02g_eurocropsml_fewshot.ipynb"


def _md(text: str) -> dict:
    return new_markdown_cell(text)


def _code(src: str) -> dict:
    cell = new_code_cell(src)
    return cell


def _params(src: str) -> dict:
    cell = new_code_cell(src)
    cell.metadata["tags"] = ["parameters"]
    return cell


CELLS: list[dict] = [
    _md(
        "# 02g - EuroCropsML few-shot transnacional (US-076)\n"
        "\n"
        "Cuantificacion del **domain gap** entre regiones europeas: cuantas "
        "muestras locales del pais destino hacen falta para cerrar la brecha de "
        "transferencia. Reusamos el **recipe** del baseline tabular (XGBoost "
        "`multi:softprob`, `ml/train/baseline.py`) sobre los splits k-shot "
        "**pre-codificados** de EuroCropsML.\n"
        "\n"
        "**Aclaraciones honestas** (ver `docs/us-planning/us-076.md`):\n"
        "- **Francia NO esta en EuroCropsML.** Los paises son Estonia (EE), "
        "Latvia (LV) y Portugal (PT). El protocolo del paper (arXiv 2407.17458, "
        "Tabla II) es **LV+PT -> EE** y **LV -> EE**, no `Francia -> Estonia`.\n"
        "- **EuroCropsML no trae embeddings AlphaEarth.** Cada parcela es una "
        "serie temporal anual de medianas Sentinel-2 L1C (13 bandas). Reusamos el "
        "*recipe* XGB, pero el vector se deriva de esa serie S2 (no de AlphaEarth); "
        "la variante AlphaEarth-via-GEE queda FUTURE.\n"
        "- Las etiquetas `EC_hcat_c` se alinean al label-space **`hcat-macro`** de "
        "US-074 (`data/reference/hcat_crosswalk.parquet`).\n"
        "\n"
        "Es **evidencia del gap medido**, NO un claim de exactitud zero-shot."
    ),
    _params(
        "# Parametros papermill (defaults reducidos para el smoke de CI).\n"
        "k_shots = [1, 5, 10, 20, 100, 200, 500]\n"
        "source_lv = ['latvia']\n"
        "source_lvpt = ['latvia', 'portugal']\n"
        "target = 'estonia'\n"
        "n_seeds = 3\n"
        "data_root = 'data/transfer/eurocropsml'\n"
        "max_parcels = None  # cap por region; None = todas las parcelas reales\n"
        "degraded = False    # True = no hay datos descargados; muestra placeholder\n"
    ),
    _code(
        "from __future__ import annotations\n"
        "\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "from IPython.display import Markdown, display\n"
        "\n"
        "from ml.utils.notebook_setup import find_repo_root\n"
        "from ml.transfer import eurocropsml_fewshot as fs\n"
        "from ml.transfer.eurocropsml_fewshot import EuroCropsMLDataMissing\n"
        "from ml.transfer.label_align import align_codes_to_hcat_macro\n"
        "\n"
        "repo_root = find_repo_root()\n"
        "root = repo_root / data_root\n"
        "seeds = tuple(range(n_seeds))\n"
        "results_path = repo_root / 'data' / 'transfer' / 'eurocropsml_fewshot_results.parquet'\n"
        "\n"
        "# Deteccion robusta del subset real: si no hay .npz, pasamos a modo degradado.\n"
        "data_present = root.exists() and any(root.rglob('*.npz'))\n"
        "if not data_present:\n"
        "    degraded = True\n"
        "display(Markdown(\n"
        "    f\"**Subset EuroCropsML**: `{root}` -- \"\n"
        "    + ('detectado (datos reales).' if data_present else "
        "'NO descargado -> modo `degraded`.')\n"
        "))\n"
    ),
    _md(
        "## 1. Conteos reales por pais y distribucion macro HCAT\n"
        "\n"
        "Cada parcela trae su codigo `EC_hcat_c` (HCAT v3, 10 digitos). Lo "
        "colapsamos a los macro-grupos de US-074. El long-tail (grassland/pasture) "
        "domina, igual que en PASTIS-R."
    ),
    _code(
        "if not degraded:\n"
        "    region_counts = {}\n"
        "    macro_rows = []\n"
        "    for reg in [target, *source_lvpt]:\n"
        "        samples = fs.load_region_samples(root, reg, max_parcels=max_parcels)\n"
        "        region_counts[reg] = len(samples)\n"
        "        macro = align_codes_to_hcat_macro([s.hcat_code for s in samples])\n"
        "        for label in macro:\n"
        "            macro_rows.append({'region': reg, 'macro_hcat_group': label})\n"
        "    counts_df = pl.DataFrame({'region': list(region_counts), "
        "'n_parcels': list(region_counts.values())})\n"
        "    display(counts_df)\n"
        "    macro_df = (\n"
        "        pl.DataFrame(macro_rows)\n"
        "        .group_by('region', 'macro_hcat_group')\n"
        "        .agg(pl.len().alias('n'))\n"
        "        .sort('region', 'n', descending=[False, True])\n"
        "    )\n"
        "    display(macro_df)\n"
        "else:\n"
        "    display(Markdown(\n"
        "        '> **Modo degradado**: subset no descargado. Ejecuta '\n"
        "        '`poetry run python -c \"from pathlib import Path; '\n"
        "        'from ml.transfer.eurocropsml_fewshot import download_eurocropsml; '\n"
        "        'download_eurocropsml(Path(\\'data/transfer/eurocropsml\\'))\"` '\n"
        "        'y re-ejecuta. Sin numeros fabricados (ver '\n"
        "        '`docs/blockers/epic12-vm-setup.md`).'\n"
        "    ))\n"
    ),
    _md(
        "## 2. Curva k-shot REAL (F1-macro vs k)\n"
        "\n"
        "Para cada `k` y cada `seed` entrenamos el recipe XGB sobre `k` muestras "
        "por clase del pais destino (EE) y medimos F1-macro en el query set. Dos "
        "escenarios con pre-train (LV->EE, LV+PT->EE) y la referencia sin "
        "pre-train (solo las k muestras de EE)."
    ),
    _code(
        "if not degraded:\n"
        "    frames = []\n"
        "    # Escenario A: pre-train LV -> EE (el 0.66 del paper).\n"
        "    frames.append(fs.run_fewshot_curve(\n"
        "        root, source=source_lv, target=target, k_shots=k_shots,\n"
        "        seeds=seeds, use_pretrain=True, max_parcels=max_parcels))\n"
        "    # Escenario B: pre-train LV+PT -> EE.\n"
        "    frames.append(fs.run_fewshot_curve(\n"
        "        root, source=source_lvpt, target=target, k_shots=k_shots,\n"
        "        seeds=seeds, use_pretrain=True, max_parcels=max_parcels))\n"
        "    # Referencia: sin pre-train (solo k muestras EE).\n"
        "    frames.append(fs.run_fewshot_curve(\n"
        "        root, source=source_lv, target=target, k_shots=k_shots,\n"
        "        seeds=seeds, use_pretrain=False, max_parcels=max_parcels))\n"
        "    curve = pl.concat(frames)\n"
        "    # Distinguimos la referencia sin pre-train en la etiqueta de escenario.\n"
        "    curve = curve.with_columns(\n"
        "        pl.when(pl.col('use_pretrain'))\n"
        "        .then(pl.col('source') + pl.lit('->') + pl.col('target'))\n"
        "        .otherwise(pl.lit('sin-pretrain->') + pl.col('target'))\n"
        "        .alias('scenario')\n"
        "    )\n"
        "    display(curve.head(8))\n"
        "else:\n"
        "    curve = None\n"
        "    display(Markdown('> **Modo degradado**: no se computa la curva (sin datos).'))\n"
    ),
    _md("## 3. Tabla por escenario (media +/- std sobre seeds)"),
    _code(
        "if not degraded and curve is not None:\n"
        "    summary = (\n"
        "        curve.group_by('scenario', 'k')\n"
        "        .agg(\n"
        "            pl.col('f1_macro').mean().alias('f1_mean'),\n"
        "            pl.col('f1_macro').std(ddof=0).fill_null(0.0).alias('f1_std'),\n"
        "            pl.col('n_classes').max().alias('n_classes'),\n"
        "        )\n"
        "        .sort('scenario', 'k')\n"
        "    )\n"
        "    display(summary)\n"
        "else:\n"
        "    summary = None\n"
    ),
    _md(
        "## 4. Figura: F1-macro vs k (escala log)\n"
        "\n"
        "Eje x logaritmico (1..500). Una linea por escenario con banda de error "
        "(std sobre seeds). Las lineas punteadas marcan la referencia del paper "
        "@500-shot (accuracy de su red temporal: 0.66 pre-train-LV, 0.57 "
        "sin-pretrain) -- comparacion de **forma**, no de valor exacto (recipe "
        "tabular XGB vs red del paper; input S2 reducido)."
    ),
    _code(
        "if not degraded and summary is not None:\n"
        "    fig, ax = plt.subplots(figsize=(8, 5))\n"
        "    for scenario in summary['scenario'].unique().sort().to_list():\n"
        "        sub = summary.filter(pl.col('scenario') == scenario).sort('k')\n"
        "        ks = sub['k'].to_numpy()\n"
        "        mean = sub['f1_mean'].to_numpy()\n"
        "        std = sub['f1_std'].to_numpy()\n"
        "        ax.plot(ks, mean, marker='o', label=scenario)\n"
        "        ax.fill_between(ks, mean - std, mean + std, alpha=0.15)\n"
        "    ax.axhline(0.66, ls='--', color='grey', lw=1)\n"
        "    ax.axhline(0.57, ls=':', color='grey', lw=1)\n"
        "    ax.text(1.1, 0.665, 'paper @500 pre-train-LV (0.66)', fontsize=8, color='grey')\n"
        "    ax.text(1.1, 0.575, 'paper @500 sin-pretrain (0.57)', fontsize=8, color='grey')\n"
        "    ax.set_xscale('log')\n"
        "    ax.set_xlabel('k (muestras por clase del pais destino, escala log)')\n"
        "    ax.set_ylabel('F1-macro (query set EE)')\n"
        "    ax.set_title('EuroCropsML few-shot: cierre del domain gap LV[+PT]->EE')\n"
        "    ax.legend(loc='lower right', fontsize=9)\n"
        "    ax.grid(True, which='both', alpha=0.3)\n"
        "    display(fig)\n"
        "    plt.close(fig)\n"
        "else:\n"
        "    display(Markdown(\n"
        "        '> **Modo degradado**: figura omitida (placeholder, sin numeros). '\n"
        "        'La curva se genera al descargar el subset real.'\n"
        "    ))\n"
    ),
    _md("## 5. Export del parquet de resultados (datos reales, al Git)"),
    _code(
        "if not degraded and curve is not None:\n"
        "    results_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    curve.write_parquet(results_path)\n"
        "    readback = pl.read_parquet(results_path)\n"
        "    assert readback.height == curve.height, 'readback mismatch'\n"
        "    display(Markdown(f'Curva escrita en `{results_path}` "
        "({readback.height} filas reales).'))\n"
        "else:\n"
        "    display(Markdown('> **Modo degradado**: no se escribe parquet (sin datos reales).'))\n"
    ),
    _md(
        "## 6. (FUTURE) AlphaEarth-via-GEE en los centroides\n"
        "\n"
        "EuroCropsML trae el centroide de cada parcela. La variante de muestrear "
        "AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` en esos centroides via GEE "
        "(`ml/ingest/gee_sampler.py::sample_alphaearth_at_coords`) daria el "
        "'AlphaEarth real' del enunciado, pero requiere jobs GEE de horas sobre "
        "EE+LV+PT y un service account con cuota. Queda **FUTURE**; no se ejecuta "
        "aqui y no bloquea la curva (que ya es real sobre la serie S2 del dataset)."
    ),
    _md(
        "## Conclusiones\n"
        "\n"
        "**Que mide esto**: cuantas parcelas locales de Estonia hacen falta para "
        "que un clasificador entrenado en Latvia (y opcionalmente Portugal) "
        "acierte el cultivo en Estonia. Es transferencia **few-shot**, no "
        "zero-shot: con muy pocas muestras locales el modelo va perdido, y la "
        "exactitud sube conforme agregamos mas (cierre del *domain gap*).\n"
        "\n"
        "**Lo honesto del experimento**:\n"
        "- La curva sale de los datos REALES de EuroCropsML (serie Sentinel-2 por "
        "parcela), nunca de numeros inventados.\n"
        "- Usamos el mismo *recipe* (XGBoost) que el baseline del proyecto, pero el "
        "input es la serie satelital del propio dataset (no AlphaEarth, que "
        "EuroCropsML no incluye). Por eso comparamos la **forma** de la curva con "
        "el paper, no el valor exacto.\n"
        "- El espacio de cultivos es el mismo `hcat-macro` que el resto de la "
        "transferencia multi-region.\n"
        "\n"
        "**Lo que sigue**: muestrear AlphaEarth en los centroides via GEE para una "
        "curva 'AlphaEarth real' directamente comparable con el baseline frances, "
        "y sumar Sen4AgriNet como segunda region de transferencia."
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
