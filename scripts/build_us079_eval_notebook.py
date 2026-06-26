"""Builder of the US-079 evaluation notebook (Italian transfer + Voting-3).

Generates ``notebooks/transfer/us079_transfer_italia_eval.ipynb`` programmatically
and reproducibly (same pattern as the sibling ``scripts/build_us078_eda_notebook.py``
and the other ``scripts/build_*_notebook.py`` builders). The notebook is step 4-5
of the US-079 plan: it EVALUATES the dense transfer the runner
``scripts/run_transfer_italia.py`` produced -- it does NOT re-train. It reads the
real ``report.json`` + ``voting_softmax.npz`` under
``checkpoints/transfer/voting-italia/<run>`` and the test masks under
``data/pastis_italia_2018``, with NO placeholders and NO fabricated numbers.

What the notebook shows:

1. Cover + framing (transfer Francia->Italia with the deployment-winner Voting-3).
2. The learned Voting-3 weights (AC2, interpretability).
3. The fine vs coarse dense metrics (mIoU + F1-macro) of the Voting-3 and each
   member (AC4, hierarchical eval).
4. The honest discard curve: F1-macro vs number of best classes retained, and the
   largest subset with F1 > 0.9 (AC3, the deployment label space).
5. The per-class F1 / IoU table (new Mediterranean classes flagged) + the dense
   confusion matrix.
6. The transfer delta (fine-tune vs zero-shot French champion, AC4).
7. The granularity demo (a parcel PASTIS would call a coarse bucket and the
   enriched model calls the fine Italian leaf, AC5).

If the runner has not produced a ``report.json`` yet (the H100 train is gated on
the full dataset), the notebook says so explicitly and shows the pending state --
it never invents results.

Visible prose (markdown, captions, prints) is Spanish with accents; code,
identifiers, comments and docstrings stay in English ASCII (project convention).
No emojis.

Usage::

    poetry run python scripts/build_us079_eval_notebook.py \\
        --out notebooks/transfer/us079_transfer_italia_eval.ipynb \\
        --report-dir checkpoints/transfer/voting-italia/us079 \\
        --data-dir data/pastis_italia_2018

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/transfer/us079_transfer_italia_eval.ipynb")
_DEFAULT_REPORT = Path("checkpoints/transfer/voting-italia/us079")
_DEFAULT_DATA = Path("data/pastis_italia_2018")


def _build_cells(report_dir: str, data_dir: str) -> list:
    """Build the markdown + code cells of the US-079 eval notebook.

    Args:
        report_dir: Repo-relative path to the runner output (``report.json`` +
            ``voting_softmax.npz``), injected into the parameters cell.
        data_dir: Repo-relative path to the homologue dataset (test masks).

    Returns:
        The ordered list of ``nbformat`` cells.
    """
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    cells.append(
        md(
            "# US-079 - Transfer Francia->Italia + Voting-3 (evaluacion)\n\n"
            "### Equipo 17 - AgroSatCopilot - Transfer learning mediterraneo (EPIC 12)\n\n"
            "---\n\n"
            "Este cuaderno **evalua** la extension del modelo campeon al homologo "
            "italiano de US-078. Los miembros densos (TSViT-pheno, U-TAE y el TSViT "
            "Full-M) se **afinaron de verdad** sobre los patches italianos partiendo del "
            "checkpoint PASTIS, con la **bandera de reciclaje**: las filas de la cabeza "
            "de las clases conservadas (las que mapean a PASTIS, p.ej. `vineyards`->"
            "`Grapevine`, `durum_hard_wheat`->`Winter durum wheat`) se warm-startean "
            "desde la cabeza francesa, y las clases nuevas mediterraneas (`olive`, "
            "bosque, ...) parten de cero.\n\n"
            "El combinador es el **Voting ponderado de 3 pesos** -- el ganador del "
            "despliegue en EPIC 6 (`france-10` 0.9069, `france-9` 0.92), no el "
            "Stacking. Aprende los pesos sobre las predicciones densas post-softmax "
            "italianas con validacion cruzada **por fold espacial** (anti-fuga, OOF).\n\n"
            "Todas las cifras se leen del `report.json` real que produjo "
            f"`scripts/run_transfer_italia.py` (bajo `{report_dir}`): no hay numeros "
            "inventados. Si el entrenamiento en la H100 aun no corrio (esta condicionado "
            "al dataset completo), el cuaderno lo dice explicitamente."
        )
    )

    cells.append(
        code(
            "# Parametros (papermill).\n"
            f'report_dir = "{report_dir}"\n'
            f'data_dir = "{data_dir}"\n'
            "f1_threshold = 0.9  # objetivo de calidad: F1-macro sobre las mejores clases\n"
            "n_demo = 2          # patches para la demo de granularidad\n"
        )
    )
    cells[-1].metadata = {"tags": ["parameters"]}

    cells.append(
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from matplotlib.colors import ListedColormap\n"
            "\n"
            "REPORT_DIR = Path(report_dir)\n"
            "DATA_ROOT = Path(data_dir)\n"
            "report_path = REPORT_DIR / 'report.json'\n"
            "HAS_REPORT = report_path.is_file()\n"
            "report = json.loads(report_path.read_text(encoding='utf-8')) if HAS_REPORT else None\n"
            "if HAS_REPORT:\n"
            "    print(f'Reporte US-079 encontrado: run={report[\"run\"]}, '\n"
            "          f'fold de test={report[\"test_fold\"]}, miembros={report[\"members\"]}')\n"
            "else:\n"
            "    print('AVISO: no hay report.json todavia. El entrenamiento real en la '\n"
            "          'H100 esta condicionado al dataset completo (~1226 patches). '\n"
            "          'Ejecuta scripts/run_transfer_italia.py para poblar este cuaderno.')\n"
        )
    )

    # ------------------------------------------------- learned weights (AC2) ---
    cells.append(
        md(
            "## 1. Pesos aprendidos del Voting-3 (AC2)\n\n"
            "El Voting ponderado aprende **un peso convexo por miembro** (suman 1) que "
            "maximiza el F1-macro denso en validacion OOF. Tres pesos -- frente a los "
            "54 del meta-LogReg del Stacking -- es lo que da al Voting su mejor "
            "generalizacion en transfer. Aqui los reportamos: la magnitud de cada peso "
            "dice cuanto confia el ensamble en cada miembro sobre el dominio italiano."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT:\n"
            "    weights = report['voting_weights']\n"
            "    wdf = pl.DataFrame({'miembro': list(weights.keys()),\n"
            "                        'peso': [round(v, 4) for v in weights.values()]}).sort('peso', descending=True)\n"
            "    display(wdf)\n"
            "    fig, ax = plt.subplots(figsize=(7, 3.2))\n"
            "    ax.barh(wdf['miembro'].to_list()[::-1], wdf['peso'].to_list()[::-1], color='#6a1b9a')\n"
            "    ax.set_xlabel('peso convexo (suma = 1)')\n"
            "    ax.set_title('Pesos aprendidos del Voting-3 sobre Italia')\n"
            "    ax.grid(axis='x', alpha=0.3); plt.tight_layout(); plt.show()\n"
            "    print(f\"F1-macro OOF (spatial-CV) del Voting-3: {report['voting_oof_f1_macro']}\")\n"
            "else:\n"
            "    print('Pendiente: pesos del Voting-3 (se reportan al correr el runner).')\n"
        )
    )

    # --------------------------------------------- fine vs coarse (AC4) ---
    cells.append(
        md(
            "## 2. Evaluacion jerarquica: fino vs coarse (AC4)\n\n"
            "La evaluacion se hace a **dos granularidades**. La **fina** usa el espacio "
            "de etiquetas italiano completo (las clases mediterraneas incluidas). La "
            "**coarse** colapsa cada clase fina a un bucket comun con PASTIS (p.ej. "
            "`apples`/`peach`/`plums` -> `Orchard`), de modo que un modelo que solo "
            "conoce la taxonomia gruesa de PASTIS es comparable con el modelo "
            "enriquecido. Reportamos mIoU + F1-macro por pixel para el Voting-3 y para "
            "cada miembro individual."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT:\n"
            "    rows = []\n"
            "    rows.append({'modelo': 'voting-3', **report['voting_eval']})\n"
            "    for name, ev in report['member_eval'].items():\n"
            "        rows.append({'modelo': name, **ev})\n"
            "    evdf = pl.DataFrame(rows).select(\n"
            "        ['modelo', 'fine_f1_macro', 'fine_miou', 'coarse_f1_macro', 'coarse_miou', 'n_pixels']\n"
            "    ).sort('fine_f1_macro', descending=True)\n"
            "    display(evdf)\n"
            "    fig, ax = plt.subplots(figsize=(9, 4))\n"
            "    x = np.arange(evdf.height)\n"
            "    ax.bar(x - 0.2, evdf['fine_f1_macro'].to_list(), 0.4, label='F1 fino', color='#1565c0')\n"
            "    ax.bar(x + 0.2, evdf['coarse_f1_macro'].to_list(), 0.4, label='F1 coarse', color='#2e7d32')\n"
            "    ax.set_xticks(x); ax.set_xticklabels(evdf['modelo'].to_list(), rotation=20, ha='right')\n"
            "    ax.set_ylabel('F1-macro (pixel)'); ax.set_title('Fino vs coarse por modelo')\n"
            "    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Pendiente: metricas fino/coarse (se reportan al correr el runner).')\n"
        )
    )

    # --------------------------------------------- discard curve (AC3) ---
    cells.append(
        md(
            "## 3. Curva de descarte honesto y subconjunto F1 > 0.9 (AC3)\n\n"
            "El objetivo de calidad de US-079 es **F1-macro > 0.9 sobre las ~10 clases "
            "mejor resueltas** (espejo del `france-10` 0.9069 del Voting-3 en PASTIS). "
            "Para localizar ese subconjunto sin trampa, ordenamos las clases por su F1 "
            "por clase (descendente) y reportamos el F1-macro de cada prefijo de `n` "
            "clases. Ninguna clase se descarta en silencio: la curva completa hace "
            "explicito donde cae el F1 por debajo del umbral."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT:\n"
            "    curve = pl.DataFrame(report['discard_curve']).select(['n_classes', 'macro_f1'])\n"
            "    best = report['best_subset_f1_over_0.9']\n"
            "    fig, ax = plt.subplots(figsize=(9, 4))\n"
            "    ax.plot(curve['n_classes'].to_list(), curve['macro_f1'].to_list(), marker='o', color='#c62828')\n"
            "    ax.axhline(f1_threshold, color='grey', linestyle='--', label=f'umbral {f1_threshold}')\n"
            "    ax.axvline(best['n_classes'], color='#2e7d32', linestyle=':',\n"
            "               label=f\"mejor subconjunto: {best['n_classes']} clases (F1 {best['macro_f1']})\")\n"
            "    ax.set_xlabel('n clases retenidas (mejores primero)'); ax.set_ylabel('F1-macro')\n"
            "    ax.set_title('Curva de descarte honesto del Voting-3 sobre Italia')\n"
            "    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"
            "    print(f\"Mayor subconjunto con F1-macro >= {f1_threshold}: \"\n"
            "          f\"{best['n_classes']} clases, F1 {best['macro_f1']}\")\n"
            "    print('Clases:', ', '.join(best['classes']))\n"
            "else:\n"
            "    print('Pendiente: curva de descarte (se reporta al correr el runner).')\n"
        )
    )

    # --------------------------------------------- per-class + confusion ---
    cells.append(
        md(
            "## 4. F1 por clase + matriz de confusion (AC5)\n\n"
            "El detalle por clase muestra que clases mediterraneas **nuevas** (sin "
            "warm-start desde PASTIS, marcadas `es_nueva`) aprende el backbone frances, "
            "y con que soporte. La matriz de confusion densa (normalizada por fila = "
            "recall por clase) localiza las confusiones residuales tras el transfer."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT:\n"
            "    pc = pl.DataFrame(report['voting_per_class']).select(\n"
            "        ['leaf', 'is_new', 'f1', 'iou', 'support']\n"
            "    ).rename({'leaf': 'clase', 'is_new': 'es_nueva'}).sort('f1', descending=True)\n"
            "    with pl.Config(tbl_rows=40):\n"
            "        display(pc)\n"
            "    n_new_good = pc.filter((pl.col('es_nueva')) & (pl.col('f1') >= 0.5)).height\n"
            "    print(f'Clases nuevas mediterraneas con F1 >= 0.5: {n_new_good}')\n"
            "else:\n"
            "    print('Pendiente: F1 por clase (se reporta al correr el runner).')\n"
        )
    )
    cells.append(
        code(
            "if HAS_REPORT and (REPORT_DIR / 'voting_softmax.npz').is_file():\n"
            "    from ml.transfer.italia_label_space import build_italia_label_space\n"
            "    from ml.eval.transfer_italia_eval import probs_to_class_map\n"
            "    from ml.eval.dense_metrics import dense_confusion_figure\n"
            "    ls = build_italia_label_space(italia_root=DATA_ROOT)\n"
            "    with np.load(REPORT_DIR / 'voting_softmax.npz') as data:\n"
            "        vote_probs = {int(k): data[k] for k in data.files}\n"
            "    vote_preds = probs_to_class_map(vote_probs)\n"
            "    ann = DATA_ROOT / 'ANNOTATIONS'\n"
            "    preds = np.concatenate([vote_preds[p].reshape(-1) for p in sorted(vote_preds)])\n"
            "    target = np.concatenate([np.load(ann / f'TARGET_{p}.npy').reshape(-1) for p in sorted(vote_preds)])\n"
            "    id_to_leaf = ls.id_to_leaf()\n"
            "    fig = dense_confusion_figure(preds, target, class_names=id_to_leaf, ignore_index=0, normalize=True)\n"
            "    fig.set_size_inches(11, 9); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Pendiente: matriz de confusion (necesita voting_softmax.npz del runner).')\n"
        )
    )

    # ------------------------------------------------------ transfer delta ---
    cells.append(
        md(
            "## 5. Delta del transfer: fine-tune vs zero-shot (AC4)\n\n"
            "La cota inferior es el **campeon frances zero-shot**: el checkpoint PASTIS "
            "aplicado tal cual a Italia, mapeando sus predicciones a las clases "
            "conservadas (las nuevas mediterraneas, que nunca vio, caen a fondo). El "
            "delta = (fine-tune) - (zero-shot) cuantifica cuanto aporta afinar de "
            "verdad. Un delta positivo confirma que el transfer adapta el backbone al "
            "vocabulario nuevo en vez de forzar todo por la taxonomia francesa."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT and report.get('transfer_delta'):\n"
            "    d = report['transfer_delta']\n"
            "    ddf = pl.DataFrame({'metrica': list(d.keys()), 'delta': list(d.values())})\n"
            "    display(ddf)\n"
            "    fig, ax = plt.subplots(figsize=(7, 3.2))\n"
            "    colors = ['#2e7d32' if v >= 0 else '#c62828' for v in d.values()]\n"
            "    ax.barh(list(d.keys())[::-1], list(d.values())[::-1], color=colors[::-1])\n"
            "    ax.axvline(0, color='black', linewidth=0.8)\n"
            "    ax.set_title('Delta del transfer (fine-tune - zero-shot)')\n"
            "    ax.grid(axis='x', alpha=0.3); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Pendiente: delta del transfer (se reporta al correr el runner con --no-zero-shot off).')\n"
        )
    )

    # ------------------------------------------------------ granularity demo ---
    cells.append(
        md(
            "## 6. Demo de granularidad (papaya/fruits, AC5)\n\n"
            "La hipotesis de taxonomia enriquecida, hecha visible: mostramos parcelas "
            "donde el modelo extendido dice la **clase fina italiana** (p.ej. `olive`, "
            "que PASTIS no tiene) frente al bucket coarse que un modelo sin granularidad "
            "usaria. Cada ejemplo: RGB del patch, prediccion fina del Voting-3 y la "
            "verdad densa, sobre un patch del fold de test."
        )
    )
    cells.append(
        code(
            "if HAS_REPORT and (REPORT_DIR / 'voting_softmax.npz').is_file():\n"
            "    from ml.transfer.italia_label_space import build_italia_label_space\n"
            "    from ml.eval.transfer_italia_eval import probs_to_class_map\n"
            "    ls = build_italia_label_space(italia_root=DATA_ROOT)\n"
            "    id_to_leaf = ls.id_to_leaf()\n"
            "    with np.load(REPORT_DIR / 'voting_softmax.npz') as data:\n"
            "        vote_probs = {int(k): data[k] for k in data.files}\n"
            "    vote_preds = probs_to_class_map(vote_probs)\n"
            "    s2d = DATA_ROOT / 'DATA_S2'; ann = DATA_ROOT / 'ANNOTATIONS'\n"
            "    n_cls = ls.num_classes\n"
            "    cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(n_cls, 2))))\n"
            "    sel = sorted(vote_preds)[:n_demo]\n"
            "    fig, axes = plt.subplots(len(sel), 3, figsize=(12, 4 * len(sel)))\n"
            "    axes = np.atleast_2d(axes)\n"
            "    for r, pid in enumerate(sel):\n"
            "        stack = np.load(s2d / f'S2_{pid}.npy'); mask = np.load(ann / f'TARGET_{pid}.npy')\n"
            "        t_mid = stack.shape[0] // 2\n"
            "        rgb = np.transpose(stack[t_mid, [2, 1, 0]].astype('float32'), (1, 2, 0))\n"
            "        p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)\n"
            "        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)\n"
            "        axes[r, 0].imshow(rgb); axes[r, 0].set_title(f'Patch {pid} - RGB'); axes[r, 0].axis('off')\n"
            "        axes[r, 1].imshow(vote_preds[pid], cmap=cmap, vmin=0, vmax=n_cls - 1, interpolation='nearest')\n"
            "        axes[r, 1].set_title(f'Patch {pid} - prediccion fina (Voting-3)'); axes[r, 1].axis('off')\n"
            "        axes[r, 2].imshow(mask, cmap=cmap, vmin=0, vmax=n_cls - 1, interpolation='nearest')\n"
            "        axes[r, 2].set_title(f'Patch {pid} - verdad densa'); axes[r, 2].axis('off')\n"
            "        present = sorted({int(c) for c in np.unique(mask) if int(c) != 0})\n"
            "        print(f'Patch {pid} clases presentes:', [id_to_leaf.get(c, c) for c in present])\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Pendiente: demo de granularidad (necesita voting_softmax.npz del runner).')\n"
        )
    )

    # ------------------------------------------------------------ closing ---
    cells.append(
        md(
            "## 7. Conclusiones\n\n"
            "Cuando el runner corre sobre el dataset completo, este cuaderno responde a "
            "las preguntas de US-079: (1) los miembros densos se afinaron al espacio "
            "italiano con warm-start verificado de las clases conservadas; (2) el "
            "Voting-3 aprendio sus pesos sobre Italia (reportados arriba); (3) el "
            "objetivo de F1-macro > 0.9 sobre las mejores ~10 clases se mide con la "
            "curva de descarte honesto; (4) la evaluacion jerarquica fino/coarse y el "
            "delta del transfer cuantifican lo ganado frente al zero-shot; (5) la demo "
            "de granularidad hace visible la taxonomia enriquecida. El run de MLflow "
            "(`us079-transfer-italia`) lleva los tags `data_version` + `code_version`."
        )
    )
    return cells


@app.command()
def build(
    out: Annotated[Path, typer.Option(help="Ruta de salida del notebook.")] = _DEFAULT_OUT,
    report_dir: Annotated[
        Path, typer.Option(help="Ruta de la salida del runner (report.json).")
    ] = _DEFAULT_REPORT,
    data_dir: Annotated[
        Path, typer.Option(help="Ruta del dataset homologo (mascaras de test).")
    ] = _DEFAULT_DATA,
) -> None:
    """Write the US-079 eval notebook (unexecuted; papermill populates outputs).

    Args:
        out: Output ``.ipynb`` path.
        report_dir: Repo-relative path to the runner output the notebook reads.
        data_dir: Repo-relative path to the homologue dataset.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = _build_cells(
        str(report_dir).replace("\\", "/"), str(data_dir).replace("\\", "/")
    )
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    typer.echo(f"Notebook escrito en {out} ({len(nb.cells)} celdas).")


if __name__ == "__main__":
    app()
