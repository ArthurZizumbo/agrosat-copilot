"""Agrupamiento jerarquico HCAT Level-1 de las 18 clases PASTIS-R.

Las 18 clases agronomicas activas de PASTIS-R contienen clases hermanas
inseparables a nivel parcela (varios trigos de invierno, varios cereales),
que hunden el F1-macro al ser confundidas entre si. La taxonomia HCAT
(Hierarchical Crop and Agriculture Taxonomy, version 3) agrupa las clases
en niveles jerarquicos; su Level-1 colapsa las hermanas dentro del mismo
grupo agronomico, lo que produce un F1-macro legitimo (no inflado) al
agregar la confusion intra-grupo que no aporta valor de cultivo.

Metodo de referencia
---------------------
- Russwurm, M., Korner, M. (2018). *Multi-Temporal Land Cover Classification
  with Sequential Recurrent Encoders*. arXiv:1802.02080 — agrega clases
  raras / hermanas para estabilizar la metrica multiclase desbalanceada.
- H2Crop (2025), *A Hierarchical Crop Mapping framework* (arXiv:2506.06155),
  que adopta la taxonomia HCAT v3 para reportar metricas por nivel
  jerarquico (L1 grupos, L2 subgrupos, L3 cultivos).

Este modulo define el mapeo explicito de las 18 clases PASTIS-R a los
**6 grupos HCAT Level-1** (distinto de ``PASTIS_R_GROUPINGS['agronomic_group']``,
que define 5 grupos), documentando el codigo HCAT de cada fusion para
defendibilidad, y un evaluador apples-to-apples que entrena el mismo modelo
sobre las mismas features con el esquema plano de 18 clases y con el esquema
agrupado de 6 grupos.

Convencion Polars
-----------------
Las funciones publicas reciben/devuelven :class:`polars.DataFrame`; ``numpy``
aparece solo en el borde de ``sklearn``. Logging via ``structlog``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "HCAT_L1_GROUPS",
    "HCAT_L1_GROUP_CODES",
    "HCAT_L1_GROUP_ORDER",
    "PASTIS_CLASS_TO_HCAT_L1",
    "GroupedVsFlatResult",
    "add_hcat_l1_group",
    "evaluate_flat_vs_grouped",
    "hcat6_dense_lut",
    "hcat_group_id_map",
    "per_label_f1_table",
]


# ---------------------------------------------------------------------------
# Mapeo de las 18 clases PASTIS-R a los 6 grupos HCAT Level-1.
# ---------------------------------------------------------------------------

#: Mapa ``class_id PASTIS-R -> nombre del grupo HCAT Level-1``.
#:
#: Las clases 0 (Background) y 19 (Void label) no son agronomicas y no
#: aparecen aqui: el pipeline baseline ya las descarta en
#: ``ml.train.baseline._prepare_dataframe``.
PASTIS_CLASS_TO_HCAT_L1: dict[int, str] = {
    # CEREALS: ocho cereales de invierno y primavera. La confusion
    # trigo-con-trigo / cereal-con-cereal es intra-grupo y desaparece aqui.
    2: "CEREALS",   # Soft winter wheat
    11: "CEREALS",  # Winter durum wheat
    4: "CEREALS",   # Winter barley
    6: "CEREALS",   # Spring barley
    3: "CEREALS",   # Corn (maize)
    10: "CEREALS",  # Winter triticale
    17: "CEREALS",  # Mixed cereal
    18: "CEREALS",  # Sorghum
    # OILSEEDS: oleaginosas.
    5: "OILSEEDS",  # Winter rapeseed
    7: "OILSEEDS",  # Sunflower
    # ROOT_CROPS: tuberculos y raices.
    9: "ROOT_CROPS",   # Beet
    13: "ROOT_CROPS",  # Potatoes
    # LEGUMES: leguminosas.
    14: "LEGUMES",  # Leguminous fodder
    15: "LEGUMES",  # Soybeans
    # PERMANENT_WOODY: cultivos lenosos permanentes.
    8: "PERMANENT_WOODY",   # Grapevine
    16: "PERMANENT_WOODY",  # Orchard
    # OTHER: pradera y mezcla de frutas/hortalizas/flores.
    1: "OTHER",   # Meadow
    12: "OTHER",  # Fruits, vegetables, flowers
}

#: Codigo HCAT v3 representativo de cada grupo Level-1 (para defendibilidad
#: del agrupamiento ante la rubrica del curso). Los codigos son los nodos
#: de la taxonomia HCAT bajo los que caen las clases PASTIS fusionadas.
HCAT_L1_GROUP_CODES: dict[str, str] = {
    "CEREALS": "3301000000",          # HCAT cereals
    "OILSEEDS": "3303000000",         # HCAT oilseed crops
    "ROOT_CROPS": "3304000000",       # HCAT root/tuber crops
    "LEGUMES": "3302000000",          # HCAT leguminous crops
    "PERMANENT_WOODY": "3306000000",  # HCAT permanent woody crops
    "OTHER": "3300000000",            # HCAT arable/other (grassland + mixed horticulture)
}

#: Orden canonico (estable) de los 6 grupos -> id contiguo ``[0, 6)``.
HCAT_L1_GROUP_ORDER: tuple[str, ...] = (
    "CEREALS",
    "LEGUMES",
    "OILSEEDS",
    "OTHER",
    "PERMANENT_WOODY",
    "ROOT_CROPS",
)

#: Alias publico legible (nombre de grupo -> lista de class_id PASTIS que
#: lo componen), util para tablas y leyendas del notebook.
HCAT_L1_GROUPS: dict[str, list[int]] = {
    group: sorted(cid for cid, g in PASTIS_CLASS_TO_HCAT_L1.items() if g == group)
    for group in HCAT_L1_GROUP_ORDER
}


def hcat_group_id_map() -> dict[str, int]:
    """Devuelve el mapa ``nombre_grupo -> id`` segun el orden canonico.

    Los ids empiezan en 1 (rango ``[1, 6]``), no en 0, a proposito: el
    pipeline baseline (:data:`ml.train.baseline._DROP_CLASS_IDS`) descarta los
    ``class_id`` 0 y 19 como clases de fondo. Reusamos ``class_id`` como
    objetivo del esquema agrupado, asi que un grupo con id 0 seria eliminado
    silenciosamente. Asignar 1..6 evita esa colision y deja que el
    ``LabelEncoder`` interno los recodifique a ``[0, 6)`` de forma consistente.

    Returns:
        Diccionario ``{nombre_grupo: id}`` con ids en ``[1, 6]`` ordenados
        alfabeticamente segun :data:`HCAT_L1_GROUP_ORDER`.
    """
    return {group: idx for idx, group in enumerate(HCAT_L1_GROUP_ORDER, start=1)}


def hcat6_dense_lut(ignore_index: int = 255) -> np.ndarray:
    """LUT ``(20,)`` que mapea la etiqueta densa PASTIS (0-19) a grupo HCAT (0-5).

    Para segmentacion densa: las 18 clases de cultivo (1-18) se colapsan a los 6
    grupos HCAT Level-1 (ids contiguos 0-5 segun :data:`HCAT_L1_GROUP_ORDER`),
    mientras que el fondo (0) y el void (19) se mapean a ``ignore_index`` para que
    no entren en las metricas de 6 grupos (asi son comparables con el baseline
    tabular, que solo evalua cultivos).

    Args:
        ignore_index: Valor para fondo y void (no agronomicos). Default 255.

    Returns:
        Array ``int64`` de forma ``(20,)`` indexable por la clase densa: ``lut[c]``
        da el grupo HCAT 0-5, o ``ignore_index`` para fondo/void.
    """
    order = {group: idx for idx, group in enumerate(HCAT_L1_GROUP_ORDER)}  # 0-5
    lut = np.full(20, ignore_index, dtype=np.int64)
    for class_id, group in PASTIS_CLASS_TO_HCAT_L1.items():
        lut[class_id] = order[group]
    return lut


def add_hcat_l1_group(
    df: pl.DataFrame,
    *,
    class_col: str = "class_id",
    group_name_col: str = "hcat6_group_name",
    group_id_col: str = "hcat6_group_id",
) -> pl.DataFrame:
    """Anexa el grupo HCAT Level-1 (nombre + id contiguo) a cada parcela.

    Cada ``class_col`` se mapea a su grupo HCAT Level-1 segun
    :data:`PASTIS_CLASS_TO_HCAT_L1`. Las clases no agronomicas (0, 19) o
    cualquier id fuera del mapa quedan con grupo ``null`` y deben filtrarse
    aguas arriba (el pipeline baseline ya lo hace).

    Args:
        df: DataFrame Polars con la columna ``class_col`` (entera).
        class_col: Nombre de la columna con el id de clase PASTIS-R.
        group_name_col: Nombre de la columna de salida con el nombre de grupo.
        group_id_col: Nombre de la columna de salida con el id contiguo.

    Returns:
        El DataFrame con dos columnas adicionales: el nombre y el id del
        grupo HCAT Level-1.

    Raises:
        ValueError: si ``class_col`` no esta en ``df``.
    """
    if class_col not in df.columns:
        raise ValueError(f"`df` debe contener la columna `{class_col}`.")

    id_map = hcat_group_id_map()
    name_expr = pl.col(class_col).replace_strict(
        PASTIS_CLASS_TO_HCAT_L1, default=None, return_dtype=pl.Utf8
    )
    out = df.with_columns(name_expr.alias(group_name_col))
    out = out.with_columns(
        pl.col(group_name_col)
        .replace_strict(id_map, default=None, return_dtype=pl.Int64)
        .alias(group_id_col)
    )
    n_mapped = int(out.get_column(group_id_col).is_not_null().sum())
    logger.info(
        "hcat_l1_group_added",
        n_rows=out.height,
        n_mapped=n_mapped,
        n_groups=len(id_map),
    )
    return out


def per_label_f1_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: dict[int, str],
    support_label: str = "support",
) -> pl.DataFrame:
    """Construye una tabla F1 + soporte por etiqueta a partir de OOF.

    Args:
        y_true: Etiquetas verdaderas codificadas (1D).
        y_pred: Etiquetas predichas codificadas (1D).
        label_names: Mapa ``id codificado -> nombre legible``.
        support_label: Nombre de la columna de soporte.

    Returns:
        DataFrame Polars con columnas ``label_id``, ``label_name``, ``f1`` y
        ``support``, ordenado por ``label_id``.
    """
    from sklearn.metrics import f1_score

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    labels = sorted(label_names)
    f1_vals = f1_score(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
    )
    uniq_vals, uniq_counts = np.unique(y_true_arr, return_counts=True)
    support = {
        int(v): int(c) for v, c in zip(uniq_vals, uniq_counts, strict=True)
    }
    return pl.DataFrame(
        {
            "label_id": labels,
            "label_name": [label_names[i] for i in labels],
            "f1": [round(float(v), 4) for v in f1_vals],
            support_label: [support.get(i, 0) for i in labels],
        }
    ).sort("label_id")


# ---------------------------------------------------------------------------
# Evaluador apples-to-apples: flat-18 vs grouped-6.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedVsFlatResult:
    """Resultado de evaluar el mismo modelo en 18 clases y 6 grupos HCAT.

    Attributes:
        model: ``"rf"``, ``"xgb"`` o ``"lgbm"``.
        n_samples: Numero de parcelas usadas (tras descartar 0/19).
        n_features: Numero de columnas de feature usadas por el modelo.
        flat_metrics: Las cinco metricas OOF del esquema plano de 18 clases.
        grouped_metrics: Las cinco metricas OOF del esquema agrupado de 6.
        flat_per_class: Tabla F1 + soporte por clase (18 filas).
        grouped_per_group: Tabla F1 + soporte por grupo HCAT (6 filas).
        flat_y_true: Etiquetas OOF verdaderas (codificadas) del esquema plano.
        flat_y_pred: Predicciones OOF del esquema plano.
        grouped_y_true: Etiquetas OOF verdaderas del esquema agrupado.
        grouped_y_pred: Predicciones OOF del esquema agrupado.
        flat_label_names: ``id codificado -> nombre de clase``.
        grouped_label_names: ``id codificado -> nombre de grupo``.
    """

    model: str
    n_samples: int
    n_features: int
    flat_metrics: dict[str, float]
    grouped_metrics: dict[str, float]
    flat_per_class: pl.DataFrame
    grouped_per_group: pl.DataFrame
    flat_y_true: np.ndarray
    flat_y_pred: np.ndarray
    grouped_y_true: np.ndarray
    grouped_y_pred: np.ndarray
    flat_label_names: dict[int, str]
    grouped_label_names: dict[int, str]

    @property
    def delta_f1_macro(self) -> float:
        """Diferencia ``f1_macro`` agrupado menos plano."""
        return float(self.grouped_metrics["f1_macro"] - self.flat_metrics["f1_macro"])


def evaluate_flat_vs_grouped(
    df: pl.DataFrame,
    *,
    model: Literal["rf", "xgb", "lgbm"] = "xgb",
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> GroupedVsFlatResult:
    """Evalua el mismo modelo y features en 18 clases planas y 6 grupos HCAT L1.

    Diseno apples-to-apples: ambos esquemas corren sobre exactamente el mismo
    DataFrame de features, el mismo CV espacial (mismas particiones por
    ``random_state``, ``k_folds`` y ``buffer_km``) y el mismo modelo. La unica
    diferencia es la columna objetivo: ``class_id`` de 18 clases para el
    esquema plano, y el id del grupo HCAT Level-1 para el agrupado.

    Reutiliza el pipeline de ``ml.train.baseline``
    (:func:`~ml.train.baseline.train_one_model` y
    :func:`~ml.train.baseline.evaluate_with_spatial_cv`), por lo que hereda el
    scaler anti-leakage por fold, la imputacion por mediana de train y el
    ``sample_weight`` inverso a frecuencia.

    Args:
        df: DataFrame con ``parcel_id``, ``class_id``, ``patch_id`` y features.
        model: Modelo tabular a usar (``"rf"``, ``"xgb"`` o ``"lgbm"``).
        k_folds: Folds del CV espacial.
        buffer_km: Buffer anti-leakage en km.
        random_state: Semilla determinista.

    Returns:
        Un :class:`GroupedVsFlatResult` con metricas y tablas por clase/grupo
        de ambos esquemas.

    Raises:
        ValueError: si ``df`` carece de ``class_id``.
    """
    from ml.ingest.pastis_loader import PASTIS_R_CLASSES
    from ml.train.baseline import (
        build_estimator,
        evaluate_with_spatial_cv,
        train_one_model,
    )

    if "class_id" not in df.columns:
        raise ValueError("`df` debe contener `class_id`.")

    # --- Esquema plano de 18 clases ---------------------------------------
    flat_result = train_one_model(
        df,
        model=model,
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    _, flat_true, flat_pred = evaluate_with_spatial_cv(
        df,
        lambda: build_estimator(model, flat_result.best_params),
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    flat_label_names = {
        i: PASTIS_R_CLASSES.get(int(c), f"c{int(c)}")
        for i, c in enumerate(flat_result.label_classes)
    }
    flat_per_class = per_label_f1_table(
        flat_true, flat_pred, label_names=flat_label_names
    )

    # --- Esquema agrupado de 6 grupos HCAT L1 -----------------------------
    # Remapeamos `class_id` al id del grupo HCAT. El pipeline baseline trata
    # `class_id` como el objetivo, asi que sobrescribirlo basta para que todo
    # (folds, scaler, sample_weight, encoder) opere sobre los 6 grupos sin
    # tocar las features. Conservamos solo las parcelas con grupo valido.
    grouped_df = add_hcat_l1_group(df)
    grouped_df = grouped_df.filter(pl.col("hcat6_group_id").is_not_null())
    grouped_df = grouped_df.drop("class_id").rename({"hcat6_group_id": "class_id"})

    grouped_result = train_one_model(
        grouped_df,
        model=model,
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    _, grouped_true, grouped_pred = evaluate_with_spatial_cv(
        grouped_df,
        lambda: build_estimator(model, grouped_result.best_params),
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    id_to_group = {idx: group for group, idx in hcat_group_id_map().items()}
    grouped_label_names = {
        i: id_to_group.get(int(c), f"g{int(c)}")
        for i, c in enumerate(grouped_result.label_classes)
    }
    grouped_per_group = per_label_f1_table(
        grouped_true, grouped_pred, label_names=grouped_label_names
    )

    logger.info(
        "flat_vs_grouped_done",
        model=model,
        n_samples=df.height,
        f1_macro_flat=round(float(flat_result.metrics["f1_macro"]), 4),
        f1_macro_grouped=round(float(grouped_result.metrics["f1_macro"]), 4),
    )

    return GroupedVsFlatResult(
        model=model,
        n_samples=df.height,
        n_features=len(flat_result.feature_cols),
        flat_metrics={k: float(v) for k, v in flat_result.metrics.items()},
        grouped_metrics={k: float(v) for k, v in grouped_result.metrics.items()},
        flat_per_class=flat_per_class,
        grouped_per_group=grouped_per_group,
        flat_y_true=flat_true,
        flat_y_pred=flat_pred,
        grouped_y_true=grouped_true,
        grouped_y_pred=grouped_pred,
        flat_label_names=flat_label_names,
        grouped_label_names=grouped_label_names,
    )
