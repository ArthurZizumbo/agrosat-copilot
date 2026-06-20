"""Build ``notebooks/eda/02f_crosswalk_hcat.ipynb`` from source cells (US-074).

Idempotent generator: writes the notebook skeleton (no outputs); papermill then
executes it end-to-end so it is committed WITH outputs. Run from repo root:

    poetry run python notebooks/eda/populate_nb_02f.py
    poetry run papermill notebooks/eda/02f_crosswalk_hcat.ipynb \
        notebooks/eda/02f_crosswalk_hcat.ipynb

Figuras y tablas (datos REALES del crosswalk csv/parquet, cero sinteticos):
- F1 Barplot horizontal n_parcels por clase PASTIS (long-tail Meadow domina).
- F2 Comparativo 18 clases vs 11 macro HCAT vs 6 familias legadas.
- F3 Treemap (matplotlib puro) de la jerarquia HCAT (arable/pasture/permanent).
- T1 Tabla crosswalk 18 filas PASTIS->HCAT-leaf (display HTML).
- T2 Tabla colapso 11 macro-grupos con n_parcels acumulado (group_by Polars).
- T3 Demo viva del seam: get_label_space('hcat-macro').class_names (display).
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = Path(__file__).resolve().parent / "02f_crosswalk_hcat.ipynb"


def _md(text: str) -> dict:
    return new_markdown_cell(text)


def _code(src: str) -> dict:
    return new_code_cell(src)


CELLS: list[dict] = [
    _md(
        "# 02f - Crosswalk taxonomico PASTIS-18 -> HCAT v3 (US-074)\n"
        "\n"
        "Espacio de etiquetas unificado para la transferencia multi-region "
        "(EPIC 12). Mapea las 18 clases PASTIS-R al estandar **HCAT v3** "
        "(Harmonized Crop and Agricultural Types, codigos de 10 digitos, prefijo "
        "`33` = `crop_type`) y las colapsa a macro-clases via el nivel de grupo "
        "de la jerarquia.\n"
        "\n"
        "**Insumos reales del repo** (regla Arthur: cero sinteticos):\n"
        "- `data/reference/eurocrops_hcat3.csv` - diccionario canonico HCAT v3 "
        "(codigos verificados).\n"
        "- `data/reference/eurocrops_fr_2018_rpg_to_hcat3.csv` - crosswalk RPG "
        "frances -> HCAT3 (referencia cruzada).\n"
        "- `data/reference/pastis_class_mapping.json` - nombres PASTIS + conteos "
        "reales de parcelas.\n"
        "\n"
        "El crosswalk se materializa en `data/reference/hcat_crosswalk.parquet` "
        "y el label-space `hcat-macro` se registra en `ml/eval/class_remap.py` "
        "SIN tocar el clasificador."
    ),
    _code(
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.patches as mpatches\n"
        "import polars as pl\n"
        "from IPython.display import display\n"
        "\n"
        "REPO = Path.cwd()\n"
        "while not (REPO / 'pyproject.toml').exists() and REPO != REPO.parent:\n"
        "    REPO = REPO.parent\n"
        "REF = REPO / 'data' / 'reference'\n"
        "\n"
        "hcat = pl.read_csv(REF / 'eurocrops_hcat3.csv',\n"
        "                   schema_overrides={'HCAT3_code': pl.Utf8})\n"
        "rpg = pl.read_csv(REF / 'eurocrops_fr_2018_rpg_to_hcat3.csv',\n"
        "                  schema_overrides={'HCAT3_code': pl.Utf8,\n"
        "                                    'HCAT2_code': pl.Utf8})\n"
        "mapping = json.loads((REF / 'pastis_class_mapping.json')\n"
        "                     .read_text(encoding='utf-8'))\n"
        "print(f'eurocrops_hcat3:  {hcat.shape}  (nodos HCAT v3)')\n"
        "print(f'rpg_fr_to_hcat3:  {rpg.shape}  (RPG-FR -> HCAT3)')\n"
        "print(f'pastis classes:   {len(mapping[\"classes\"])} (0..19)')"
    ),
    _md(
        "## 1. Construccion del crosswalk 18 filas (tabla PASTIS->HCAT-leaf)\n"
        "\n"
        "Se reutiliza `ml.data.hcat_crosswalk.build_crosswalk`, que resuelve cada "
        "clase PASTIS contra el CSV real por nombre agronomico y **valida** que "
        "cada `hcat_leaf_code` exista en `eurocrops_hcat3.csv` (no se inventan "
        "codigos). Tres clases sin hoja 1:1 quedan marcadas `approx`."
    ),
    _code(
        "from ml.data.hcat_crosswalk import build_crosswalk\n"
        "\n"
        "cw = build_crosswalk()\n"
        "approx = cw.filter(pl.col('match_quality') == 'approx')['pastis_name'].to_list()\n"
        "print('filas:', cw.height, '| approx (sin hoja 1:1):', approx)\n"
        "crosswalk_tbl = cw.select(['semantic18_id', 'pastis_name', 'hcat_leaf_name',\n"
        "                           'hcat_leaf_code', 'hcat_group_code',\n"
        "                           'macro_hcat_group', 'macro_hcat_l1_6',\n"
        "                           'n_parcels', 'match_quality'])\n"
        "# Tabla crosswalk 18 filas PASTIS-18 -> HCAT codes -> macro-clases (HTML).\n"
        "display(crosswalk_tbl)"
    ),
    _md(
        "## 2. Long-tail PASTIS: Meadow domina (~45% de la masa)\n"
        "\n"
        "El conteo real de parcelas (del JSON) muestra el desbalance que hunde el "
        "F1-macro a 18 clases: Meadow (31292) y Corn (13123) concentran la masa. "
        "Barplot horizontal ordenado por `n_parcels`."
    ),
    _code(
        "total = cw['n_parcels'].sum()\n"
        "meadow = cw.filter(pl.col('pastis_name') == 'Meadow')['n_parcels'][0]\n"
        "print(f'total parcelas (18 clases): {total} | Meadow: {meadow} '\n"
        "      f'({100 * meadow / total:.1f}%)')\n"
        "ordered = cw.sort('n_parcels')\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "ax.barh(ordered['pastis_name'].to_list(),\n"
        "        ordered['n_parcels'].to_list(), color='#4c72b0')\n"
        "ax.set_xlabel('n parcelas (PASTIS-R real)')\n"
        "ax.set_title('Long-tail de las 18 clases PASTIS-R (Meadow domina)')\n"
        "for i, v in enumerate(ordered['n_parcels'].to_list()):\n"
        "    ax.text(v + 200, i, str(v), va='center', fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "plt.close(fig)"
    ),
    _md(
        "## 3. Colapso a 11 macro-clases HCAT (tabla con n_parcels acumulado)\n"
        "\n"
        "El nodo de grupo L2 (6 digitos significativos; L1 para pasture/permanent) "
        "produce 10 macro-grupos de cultivo. Con la macro-clase `void` "
        "(partial-label, background/fuera de nomenclatura) el vocabulario completo "
        "es **11**. Meadow queda **aislado** en `grassland`, mitigando el long-tail."
    ),
    _code(
        "from ml.data.hcat_crosswalk import MACRO_HCAT_GROUPS\n"
        "\n"
        "# Tabla colapso macro-grupos con n_parcels acumulado (group_by Polars).\n"
        "macro = (cw.group_by('macro_hcat_group', 'hcat_group_code')\n"
        "           .agg(pl.col('n_parcels').sum().alias('n_parcels'),\n"
        "                pl.len().alias('n_clases_pastis'),\n"
        "                pl.col('pastis_name').sort().str.join(', ').alias('clases_pastis'))\n"
        "           .sort('n_parcels', descending=True))\n"
        "print('macro-grupos de cultivo:', cw['macro_hcat_group'].n_unique(),\n"
        "      '| + void =', len(MACRO_HCAT_GROUPS), 'macro-clases canonicas')\n"
        "print('familias legadas (l1_6):', cw['macro_hcat_l1_6'].n_unique())\n"
        "display(macro)"
    ),
    _md(
        "## 4. Comparativa de compresion: 18 vs 11 macro vs 6 familias\n"
        "\n"
        "El grupo HCAT comprime el long-tail. Las 6 familias legadas (`hcat_l1_6`, "
        "XGB 0.6535 F1-macro) son aun mas gruesas; las 11 macro-clases HCAT son el "
        "punto canonico de E12 (mas finas para multi-region)."
    ),
    _code(
        "l16 = (cw.group_by('macro_hcat_l1_6')\n"
        "         .agg(pl.col('n_parcels').sum().alias('n'))\n"
        "         .sort('n', descending=True))\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "axes[0].bar(range(cw.height), cw.sort('n_parcels', descending=True)\n"
        "            ['n_parcels'].to_list(), color='#c44e52')\n"
        "axes[0].set_title(f'18 clases PASTIS (n={cw.height})')\n"
        "axes[0].set_xlabel('clase')\n"
        "axes[0].set_ylabel('n parcelas')\n"
        "axes[1].bar(macro['macro_hcat_group'].to_list(),\n"
        "            macro['n_parcels'].to_list(), color='#55a868')\n"
        "axes[1].set_title(f'macro HCAT cultivo (n={macro.height}) + void = 11')\n"
        "axes[1].tick_params(axis='x', rotation=90)\n"
        "axes[2].bar(l16['macro_hcat_l1_6'].to_list(),\n"
        "            l16['n'].to_list(), color='#8172b3')\n"
        "axes[2].set_title(f'6 familias legadas (n={l16.height})')\n"
        "axes[2].tick_params(axis='x', rotation=90)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "plt.close(fig)"
    ),
    _md(
        "## 5. Jerarquia HCAT (4 niveles) tocada por PASTIS-18 -- treemap\n"
        "\n"
        "Treemap por rama HCAT L1 (`arable 3301` / `pasture 3302` / `permanent "
        "3303`), area proporcional a `n_parcels`, agrupado por macro-grupo (L2). "
        "Solo se dibujan los nodos que PASTIS realmente alcanza. Implementado con "
        "rectangulos de matplotlib (sin dependencias extra)."
    ),
    _code(
        "def _slice_layout(items, x, y, w, h):\n"
        "    # Slice-and-dice simple: corta el lado mayor proporcional al peso.\n"
        "    total = sum(v for _, v in items)\n"
        "    out, off = [], 0.0\n"
        "    for label, val in items:\n"
        "        frac = val / total if total else 0.0\n"
        "        if w >= h:\n"
        "            out.append((label, val, x + off * w, y, frac * w, h))\n"
        "        else:\n"
        "            out.append((label, val, x, y + off * h, w, frac * h))\n"
        "        off += frac\n"
        "    return out\n"
        "\n"
        "branch = cw.with_columns(\n"
        "    pl.col('hcat_leaf_code').str.slice(0, 4).alias('hcat_l1_code'))\n"
        "L1_NAME = {'3301': 'arable (3301)', '3302': 'pasture (3302)',\n"
        "           '3303': 'permanent (3303)'}\n"
        "l1_tot = (branch.group_by('hcat_l1_code')\n"
        "                .agg(pl.col('n_parcels').sum().alias('n'))\n"
        "                .sort('hcat_l1_code'))\n"
        "l1_items = [(c, n) for c, n in zip(l1_tot['hcat_l1_code'].to_list(),\n"
        "                                   l1_tot['n'].to_list())]\n"
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        "cmap = plt.get_cmap('tab20')\n"
        "macro_order = sorted(cw['macro_hcat_group'].unique().to_list())\n"
        "color = {m: cmap(i % 20) for i, m in enumerate(macro_order)}\n"
        "for (l1, _, lx, ly, lw, lh) in _slice_layout(l1_items, 0, 0, 1, 1):\n"
        "    ax.add_patch(mpatches.Rectangle((lx, ly), lw, lh, fill=False,\n"
        "                                    edgecolor='black', lw=2.5))\n"
        "    ax.text(lx + lw / 2, ly + lh + 0.012, L1_NAME[l1], ha='center',\n"
        "            va='bottom', fontsize=10, fontweight='bold')\n"
        "    sub = (branch.filter(pl.col('hcat_l1_code') == l1)\n"
        "                 .group_by('macro_hcat_group')\n"
        "                 .agg(pl.col('n_parcels').sum().alias('n'))\n"
        "                 .sort('n', descending=True))\n"
        "    sub_items = [(m, n) for m, n in zip(sub['macro_hcat_group'].to_list(),\n"
        "                                        sub['n'].to_list())]\n"
        "    for (m, val, mx, my, mw, mh) in _slice_layout(sub_items, lx, ly, lw, lh):\n"
        "        ax.add_patch(mpatches.Rectangle((mx, my), mw, mh, facecolor=color[m],\n"
        "                                        edgecolor='white', lw=1.0))\n"
        "        if mw * mh > 0.012:\n"
        "            ax.text(mx + mw / 2, my + mh / 2, f'{m}\\n{val}', ha='center',\n"
        "                    va='center', fontsize=8)\n"
        "ax.set_xlim(0, 1)\n"
        "ax.set_ylim(0, 1.06)\n"
        "ax.axis('off')\n"
        "ax.set_title('Jerarquia HCAT v3 (area ~ n_parcels) -- ramas L1 -> macro L2')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "plt.close(fig)"
    ),
    _md(
        "## 6. Materializacion del parquet + roundtrip\n"
        "\n"
        "Se escribe `data/reference/hcat_crosswalk.parquet` (Polars, codigos como "
        "`Utf8` para preservar ceros) y se valida la lectura."
    ),
    _code(
        "from ml.data.hcat_crosswalk import (CROSSWALK_PARQUET, load_crosswalk,\n"
        "                                    write_crosswalk)\n"
        "\n"
        "p = write_crosswalk()\n"
        "rt = load_crosswalk()\n"
        "assert rt.height == 18\n"
        "assert rt.schema['hcat_leaf_code'] == pl.Utf8\n"
        "assert rt.equals(cw)\n"
        "size_kb = CROSSWALK_PARQUET.stat().st_size / 1024\n"
        "print(f'parquet: {p}  ({size_kb:.1f} KB, {rt.height} filas)')\n"
        "display(rt.head(3))"
    ),
    _md(
        "## 7. Demo viva del seam: label-space `hcat-macro` registrado (US-053)\n"
        "\n"
        "El registry US-053 queda AMPLIADO sin tocar `classify.py`. `hcat-macro` "
        "mantiene los 18 ids y EXPONE el mapeo macro en `class_names`. Se muestra "
        "`get_label_space('hcat-macro').class_names` y se verifica que `france-9` "
        "sigue intacto."
    ),
    _code(
        "from ml.eval.class_remap import get_label_space, list_label_spaces\n"
        "\n"
        "print('label-spaces:', list_label_spaces())\n"
        "hm = get_label_space('hcat-macro')\n"
        "print('kept ids:', len(hm.kept_class_ids), '| dropped:', hm.dropped_class_ids)\n"
        "print('france-9 intacto:', len(get_label_space('france-9').kept_class_ids),\n"
        "      'kept')\n"
        "seam = pl.DataFrame({'sem18_id': list(hm.class_names),\n"
        "                     'macro_label': list(hm.class_names.values())})\n"
        "# Demo viva: get_label_space('hcat-macro').class_names\n"
        "display(seam)"
    ),
    _md(
        "## 8. Convencion void/background + partial-label (UniSeg)\n"
        "\n"
        "| Dataset | Background | Void / out-of-scope | Unificado |\n"
        "|---------|-----------|---------------------|-----------|\n"
        "| PASTIS-R | id `0` ('stuff') | id `19` (fuera de nomenclatura / <50% overlap) | `ignore_index=255` |\n"
        "| Sen4AgriNet | clase `0` del `linear_encoder` | sin codigo FAO-ICC -> HCAT | `0`->ignore; sin-mapeo->null-class |\n"
        "| EuroCropsML | sin pixel background (tabular) | clase HCAT ausente en el split | ausencia->null-class (no falso negativo) |\n"
        "\n"
        "**Regla unica**: `background/void -> ignore`; `ausencia cross-dataset -> "
        "null-class` (partial-label, no negativo duro). `ignore_index=255` reutiliza "
        "`HARNESS_IGNORE_INDEX` de US-030. La 11a macro-clase `void` absorbe el "
        "background/fuera-de-nomenclatura como partial-label.\n"
        "\n"
        "**Recomendacion de loss (US-075/US-076, no implementada aqui)**: BCE "
        "class-independent (un sigmoide por clase HCAT, no softmax exclusivo) + "
        "cross-dataset relation loss que solo penaliza clases presentes en cada "
        "dataset, evitando el conflicto de gradiente entre regiones. US-074 entrega "
        "el espacio comun y la convencion; el loss lo consume la US siguiente."
    ),
    _code(
        "from ml.eval.class_remap import HARNESS_IGNORE_INDEX\n"
        "\n"
        "void = mapping['classes']['19']\n"
        "bg = mapping['classes']['0']\n"
        "ix = HARNESS_IGNORE_INDEX\n"
        "print(f\"PASTIS id 0 (background): {bg['name']} -> ignore_index={ix}\")\n"
        "print(f\"PASTIS id 19 (void): {void['name']} \"\n"
        "      f\"(n_parcels={void['n_parcels']}) -> ignore_index={ix}\")\n"
        "print('void_convention en crosswalk:',\n"
        "      cw['void_convention'].unique().to_list())\n"
        "print('macro-clase partial-label que absorbe background/void: void')"
    ),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, NB_PATH)
    print(json.dumps({"written": str(NB_PATH), "cells": len(CELLS)}))


if __name__ == "__main__":
    main()
