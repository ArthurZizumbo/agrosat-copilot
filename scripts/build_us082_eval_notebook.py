"""Builder of the US-082 notebook (complete Italian dataset + honest TL re-eval).

Generates ``notebooks/transfer/us082_dataset_completo_eval.ipynb`` reproducibly
(same idempotent ``nbformat`` + ``typer`` pattern as the sibling builders
``scripts/build_us079_copilot_notebook.py`` / ``scripts/build_us078_eda_notebook.py``).
The notebook closes US-082: it explains the 1 %-pilot root cause of the F1 0.13
(US-079), runs the formal EDA of the complete 1,438-patch dataset, and re-evaluates
the transfer along the THREE label-space vias Arthur decided, comparing them and
issuing an honest per-class verdict.

The three vias (the spine of the notebook):

- VIA A -- the native 39 HCAT leaves scored as-is ("no reagrupar clases").
- VIA B -- the Italian leaves mapped to the champion's input label space (the
  conserved crosswalk to PASTIS-18 / france-12 the French members already know).
- VIA C -- the full procedure replicated end-to-end on the new 1,438-patch dataset
  (AlphaEarth extraction -> per-member training -> fold-5 OOF -> Voting-3 -> eval).

What the notebook shows (8 sections, Avance5 style):

1. Framing + the 1 %-pilot root cause (884 parcels vs 107,493 real).
2. Formal EDA of the complete dataset (``ml.transfer.eda_italia``): volume,
   per-class support, temporal ceiling (dates vs PASTIS), co-occurrence.
3. Separability post-extraction (``ml.transfer.separability_italia``): JM /
   Bhattacharyya per class -- reliable now (hundreds of parcels/class) vs the
   spurious JM=2.0 of the pilot.
4. The three vias compared (``ml.transfer.eval_three_ways``): macro-F1 + n classes
   >= 0.6 / >= 0.8 per via, side by side.
5. A/B pilot (884) vs complete (107k): the table that proves the sampling was the
   cause, not the dataset.
6. F1 stratified by the patch date count (the temporal ceiling, honest).
7. Verdict: how many classes really rescue at each F1 gate (KPI-2).
8. Conclusions + handoff (US-083 UDA if the ceiling is domain/temporal).

HARD RULE -- REAL VALUES ONLY. The metric cells read the real artefacts produced
by the H100 extraction + training run (the full features parquet, the OOF, the
report JSONs). If an artefact is absent, the cell prints an HONEST pending state;
it NEVER emits a placeholder number. When re-run with the artefacts present (via
papermill after the VM run), the cells populate with the real metrics.

Visible prose (markdown, captions, prints) is Spanish with accents; code,
identifiers, comments and docstrings stay in English ASCII (project convention).
No emojis.

Usage::

    poetry run python scripts/build_us082_eval_notebook.py \\
        --out notebooks/transfer/us082_dataset_completo_eval.ipynb \\
        --data-dir data/pastis_italia_2018 \\
        --features data/features/alphaearth_italia_2018_full1438.parquet \\
        --report-dir reports/transfer_italia

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/transfer/us082_dataset_completo_eval.ipynb")
_DEFAULT_DATA = Path("data/pastis_italia_2018")
_DEFAULT_FEATURES = Path("data/features/alphaearth_italia_2018_full1438.parquet")
_DEFAULT_PILOT = Path("data/features/alphaearth_italia_2018.parquet")
_DEFAULT_REPORT = Path("reports/transfer_italia")

#: PASTIS-France references (already-measured EPIC 6 constants, not run outputs):
#: the dataset the Italian transfer is contrasted against.
_PASTIS_DATES: int = 43
_PASTIS_PATCHES: int = 2468
_PASTIS_PARCELS: int = 124000


def _build_cells(data_dir: str, features: str, pilot: str, report_dir: str) -> list:
    """Build the markdown + code cells of the US-082 notebook.

    Args:
        data_dir: Repo-relative path to the complete Italian dataset (US-078).
        features: Path to the complete-extraction features parquet (``_full1438``).
        pilot: Path to the 1 %-pilot features parquet (for the A/B contrast).
        report_dir: Directory holding the runner report JSONs (real metrics).

    Returns:
        The ordered list of ``nbformat`` cells.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    # ---------------------------------------------------------------- Cover ---
    cells.append(
        md(
            "# US-082 - Dataset Italia COMPLETO + re-evaluacion honesta del TL\n\n"
            "### El F1 0.13 NO era el dataset: era el 1 % del dataset\n\n"
            "**Equipo 17** - AgroSatCopilot - Transfer learning mediterraneo (EPIC 12)\n\n"
            "---\n\n"
            "El F1-macro **0.13** del TL Italia (US-079) no se debio a un dataset pobre ni a un "
            "bug: el modelo entreno y evaluo sobre el **PILOTO del 1 %** (20 patches / 884 "
            "parcelas) en vez del dataset completo (**1,438 patches / 107,493 parcelas**, ya en "
            "disco). Esta US re-extrae AlphaEarth sobre el dataset completo y re-evalua el "
            "transfer por **tres vias** que Arthur decidio comparar:\n\n"
            "- **Via A** - las **39 hojas HCAT nativas** tal cual (sin reagrupar clases).\n"
            "- **Via B** - las hojas italianas **mapeadas al espacio de entrada del campeon** (el "
            "crosswalk conservado a PASTIS-18 / france-12 que los miembros franceses ya conocen).\n"
            "- **Via C** - el **procedimiento completo replicado** de extremo a extremo sobre el "
            "dataset nuevo (extraccion AlphaEarth -> entreno xgb/TSViT/U-TAE -> OOF fold-5 -> "
            "Voting-3 -> eval), igual que se hizo con PASTIS-Francia.\n\n"
            "> **Solo valores reales.** Toda metrica se lee de los artefactos REALES que produce "
            "la corrida de extraccion + entrenamiento en la H100 (el parquet de features completo, "
            "el OOF, los report JSON). Si un artefacto no existe aun, la celda lo dice y muestra "
            "el estado **pendiente**, nunca un numero inventado. NO se reagrupan clases."
        )
    )

    # --------------------------------------------- parameters (papermill) ---
    cells.append(
        code(
            "# Parametros (papermill). Sobreescribe con `papermill -p <name> <value>`.\n"
            f'data_dir = "{data_dir}"   # dataset Italia completo (US-078, 1438 patches)\n'
            f'features_full = "{features}"   # features AlphaEarth del dataset completo (_full1438)\n'
            f'features_pilot = "{pilot}"   # features del piloto del 1 % (para el A/B)\n'
            f'report_dir = "{report_dir}"   # report JSON del runner (metricas reales del TL)\n'
            "year = 2018   # campaña del embedding anual (Italia 2018)\n"
            f"pastis_dates = {_PASTIS_DATES}   # fechas medias PASTIS-Francia (referencia)\n"
            f"pastis_patches = {_PASTIS_PATCHES}   # patches PASTIS-Francia (referencia)\n"
            f"pastis_parcels = {_PASTIS_PARCELS}   # parcelas PASTIS-Francia (referencia)"
        )
    )
    cells[-1].metadata = {"tags": ["parameters"]}

    # ------------------------------------------------------------------ Setup ---
    cells.append(
        md(
            "## Preparacion del entorno\n\n"
            "Resolvemos la raiz del repositorio y forzamos UTF-8 en la salida (la consola de "
            "Windows usa cp1252 y la prosa/logs llevan acentos). Todo lo que sigue lee de rutas "
            "del repo; nada se descarga ni se fabrica."
        )
    )
    cells.append(
        code(
            "import sys, io, json\n"
            "from pathlib import Path\n\n"
            "if hasattr(sys.stdout, 'reconfigure'):\n"
            "    sys.stdout.reconfigure(encoding='utf-8')\n\n"
            "def _find_repo_root(start: Path) -> Path:\n"
            "    cur = start.resolve()\n"
            "    for parent in [cur, *cur.parents]:\n"
            "        if (parent / 'pyproject.toml').is_file():\n"
            "            return parent\n"
            "    return cur\n\n"
            "REPO = _find_repo_root(Path.cwd())\n"
            "if str(REPO) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO))\n"
            "print('repo root:', REPO)"
        )
    )

    # ------------------------------------------- 1. Root cause (pilot 1%) ---
    cells.append(
        md(
            "## 1. Causa raiz: el piloto del 1 %\n\n"
            "El runner de features uso 20 patches (884 parcelas) en vez de los 1,438 patches "
            "(107,493 parcelas) reales. Verificamos el conteo real del dataset en disco -- es la "
            "evidencia que motiva toda la US."
        )
    )
    cells.append(
        code(
            "import polars as pl\n"
            "meta = pl.read_parquet(Path(REPO) / data_dir / 'metadata.parquet')\n"
            "n_patches = meta.height\n"
            "n_parcels = int(meta['n_parcelas'].sum()) if 'n_parcelas' in meta.columns else None\n"
            "print(f'Dataset completo en disco: {n_patches} patches, {n_parcels} parcelas')\n"
            "print(f'PASTIS-Francia (referencia): {pastis_patches} patches, {pastis_parcels} parcelas')\n"
            "print(f'Italia / PASTIS: {100*n_patches/pastis_patches:.0f}% patches, "
            "{100*n_parcels/pastis_parcels:.0f}% parcelas')\n"
            "# Piloto del 1 %: 20 patches / 884 parcelas (lo que uso US-079).\n"
            "pilot_path = Path(REPO) / features_pilot\n"
            "if pilot_path.is_file():\n"
            "    pilot_df = pl.read_parquet(pilot_path)\n"
            "    print(f'Piloto del 1 %% (US-079): {pilot_df.height} filas de parcela')\n"
            "else:\n"
            "    print('PENDIENTE: piloto no presente en disco (no bloquea el dataset completo)')"
        )
    )

    # ------------------------------------------------------- 2. Formal EDA ---
    cells.append(
        md(
            "## 2. EDA formal del dataset completo\n\n"
            "`ml.transfer.eda_italia.compute_italia_eda` resume volumen, soporte por clase, el "
            "techo temporal (fechas vs PASTIS) y la co-ocurrencia inter-clase, todo desde el "
            "`metadata.parquet` real. Las clases con >= 200 patches son candidatas a F1 alto."
        )
    )
    cells.append(
        code(
            "from ml.transfer.eda_italia import compute_italia_eda\n"
            "eda = compute_italia_eda(Path(REPO) / data_dir)\n"
            "print(json.dumps(eda.summary(), indent=2))\n"
            'print(f\'\\nTecho temporal: media {eda.date_stats["mean"]:.1f} fechas '
            "(PASTIS {pastis_dates}); {eda.n_patches_weak_phenology} patches con < 16 fechas')\n"
            "# Tabla de soporte por clase (las candidatas con >= 200 patches primero).\n"
            "pl.DataFrame(eda.per_class).head(40)"
        )
    )

    # -------------------------------------------------- 3. Separability ---
    cells.append(
        md(
            "## 3. Separabilidad post-extraccion (JM / Bhattacharyya)\n\n"
            "Con el dataset completo (cientos de parcelas/clase) la covarianza de 64 dims ya es "
            "estimable, asi que la distancia Jeffries-Matusita es confiable -- a diferencia del "
            "JM=2.0 espurio del piloto (covarianza degenerada con 1-5 parcelas/clase). Requiere "
            "el parquet de features completo (`_full1438`); si no existe aun, la celda lo dice."
        )
    )
    cells.append(
        code(
            "feat_path = Path(REPO) / features_full\n"
            "if feat_path.is_file():\n"
            "    from ml.transfer.separability_italia import compute_separability\n"
            "    sep = compute_separability(features_path=feat_path, italia_root=Path(REPO) / data_dir)\n"
            "    print(json.dumps(sep.summary() if hasattr(sep, 'summary') else {}, indent=2, default=str))\n"
            "    display(pl.DataFrame(sep.per_class) if hasattr(sep, 'per_class') else sep)\n"
            "else:\n"
            "    print(f'PENDIENTE: {features_full} no existe. Corre la extraccion AlphaEarth full '\n"
            "          '1438 en la H100 y vuelve a ejecutar este notebook con papermill.')"
        )
    )

    # --------------------------------------------- 4. The three vias ---
    cells.append(
        md(
            "## 4. Las tres vias comparadas\n\n"
            "`ml.transfer.eval_three_ways.compare_three_ways` puntua las predicciones densas del "
            "Voting-3 Italia en las tres vias: **A** (39 hojas nativas), **B** (mapeo al crosswalk "
            "PASTIS) y **C** (procedimiento completo). La tabla muestra macro-F1, nº de clases >= "
            "0.6 y >= 0.8 por via. Requiere las predicciones densas del OOF fold-5 (report del "
            "runner); si no existen, estado pendiente honesto."
        )
    )
    cells.append(
        code(
            "rep = Path(REPO) / report_dir / 'three_ways_comparison.json'\n"
            "if rep.is_file():\n"
            "    table = json.loads(rep.read_text(encoding='utf-8'))\n"
            "    display(pl.DataFrame(table['table'] if isinstance(table, dict) else table))\n"
            "else:\n"
            "    print(f'PENDIENTE: {rep} no existe. La via C (entreno + OOF + voto) corre en la '\n"
            "          'H100; este notebook puebla la tabla al re-ejecutarse con el report presente.')"
        )
    )

    # ----------------------------------------- 5. A/B pilot vs complete ---
    cells.append(
        md(
            "## 5. A/B: piloto (884) vs completo (107k)\n\n"
            "La tabla que prueba que el muestreo era la causa: el mismo pipeline sobre el piloto "
            "del 1 % vs el dataset completo. Si el F1 por-clase sube al pasar de 884 a 107k "
            "parcelas, la causa raiz queda demostrada (no el domain-shift)."
        )
    )
    cells.append(
        code(
            "ab = Path(REPO) / report_dir / 'ab_pilot_vs_full.json'\n"
            "if ab.is_file():\n"
            "    display(pl.DataFrame(json.loads(ab.read_text(encoding='utf-8'))))\n"
            "else:\n"
            "    print(f'PENDIENTE: {ab} no existe. Se genera tras re-entrenar sobre el dataset '\n"
            "          'completo (H100) y contrastar con el OOF del piloto.')"
        )
    )

    # ------------------------------------ 6. F1 stratified by dates ---
    cells.append(
        md(
            "## 6. F1 estratificado por nº de fechas (techo temporal)\n\n"
            "Italia tiene media 24.3 fechas (vs 43 PASTIS) y muy variable (9-40). Estratificar el "
            "F1 por nº de fechas del patch aisla el techo temporal honesto: los patches de pocas "
            "fechas padean fuerte a 32 timesteps y su fenologia es pobre."
        )
    )
    cells.append(
        code(
            "strat = Path(REPO) / report_dir / 'f1_by_date_count.json'\n"
            "if strat.is_file():\n"
            "    display(pl.DataFrame(json.loads(strat.read_text(encoding='utf-8'))))\n"
            "else:\n"
            "    print(f'PENDIENTE: {strat} no existe. Se genera en la eval del TL completo (H100).')"
        )
    )

    # --------------------------------------------------- 7. Verdict ---
    cells.append(
        md(
            "## 7. Veredicto: cuantas clases rescatan de verdad (KPI-2)\n\n"
            "El objetivo honesto NO es un solo F1-macro-39 global (cuyo techo lo fijan las ~23 "
            "clases de cola minoritarias reales), sino **cuantas clases superan F1 >= 0.6 y >= "
            "0.8** con el dataset completo. Target: >= 10 clases >= 0.6 y >= 5 clases >= 0.8."
        )
    )
    cells.append(
        code(
            "rep = Path(REPO) / report_dir / 'three_ways_comparison.json'\n"
            "if rep.is_file():\n"
            "    table = json.loads(rep.read_text(encoding='utf-8'))\n"
            "    rows = table['table'] if isinstance(table, dict) else table\n"
            "    for r in rows:\n"
            "        print(f\"Via {r['via']} ({r['label_space']}): macro-F1 {r['macro_f1']}, \"\n"
            "              f\"{r['n_classes_ge_0.6']} clases >= 0.6, {r['n_classes_ge_0.8']} >= 0.8\")\n"
            "    via_a = next((r for r in rows if r['via'] == 'A'), None)\n"
            "    if via_a:\n"
            "        ok6 = via_a['n_classes_ge_0.6'] >= 10\n"
            "        ok8 = via_a['n_classes_ge_0.8'] >= 5\n"
            '        print(f\'\\nKPI-2 (via A nativa): >=10 clases>=0.6 {"OK" if ok6 else "NO"}; \'\n'
            '              f\'>=5 clases>=0.8 {"OK" if ok8 else "NO"}\')\n'
            "else:\n"
            "    print('PENDIENTE: sin el report de las tres vias no hay veredicto. Corre la H100.')"
        )
    )

    # ----------------------------------------------- 8. Conclusions ---
    cells.append(
        md(
            "## 8. Conclusiones y handoff\n\n"
            "- **Si el F1 por-clase sube** con el dataset completo: la causa raiz era el muestreo "
            "del 1 %, confirmado por el A/B (seccion 5).\n"
            "- **Si NO sube significativamente** pese al dato completo: el techo es el "
            "domain-shift / temporal (24 vs 43 fechas), no el muestreo. Pivota a **US-083 (UDA "
            "fenologica: ClimID-UDA + class-aware MMD)**.\n"
            "- Las ~23 clases de cola minoritarias reales seguiran dificiles aun con re-extraccion "
            "(soporte estructuralmente bajo).\n\n"
            "> Provenance al cerrar: `US-082 @ <git_sha7> + mlflow:<run_id> + dvc:<rev>` "
            "(features `_full1438` versionadas con DVC, run del re-entreno en el server :5010)."
        )
    )

    return cells


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Output .ipynb path.")] = _DEFAULT_OUT,
    data_dir: Annotated[Path, typer.Option(help="Complete Italian dataset root.")] = _DEFAULT_DATA,
    features: Annotated[
        Path, typer.Option(help="Complete-extraction features parquet.")
    ] = _DEFAULT_FEATURES,
    pilot: Annotated[
        Path, typer.Option(help="1%-pilot features parquet (A/B contrast).")
    ] = _DEFAULT_PILOT,
    report_dir: Annotated[
        Path, typer.Option(help="Runner report JSON directory.")
    ] = _DEFAULT_REPORT,
) -> None:
    """Write the US-082 notebook to ``out`` (structure only; no executed outputs)."""
    nb = nbf.v4.new_notebook()
    nb.cells = _build_cells(str(data_dir), str(features), str(pilot), str(report_dir))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    typer.echo(f"Wrote {out} ({len(nb.cells)} cells, structure only).")


if __name__ == "__main__":
    app()
