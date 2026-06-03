"""Descriptores compactos de firma espectral por parcela (US-023-preview P5).

Genera ``spectral_signature_*`` como bloque opcional para ``fusion.py`` con
descriptores agronomicamente justificados de la curva espectral muestreada
por parcela:

- ``rep`` (default, **Frampton et al. 2013**, DOI 10.1016/j.isprsjprs.2013.04.007):
  Red Edge Position estacional. Posicion (en nm) del punto de inflexion de
  la curva de reflectancia entre el rojo y el infrarrojo cercano. La REP
  varia con el contenido de clorofila y la fenologia del cultivo;
  bibliografia de teledeteccion agronomica la documenta como uno de los
  descriptores compactos mas fiables del estado del cultivo.
- ``sam`` (Spectral Angle Mapper): coseno del angulo espectral entre la
  firma de la parcela y el centroide de la clase mayoritaria observada en
  fit. Es ``1.0`` cuando la parcela "se parece" a la firma media de la
  clase mayoritaria, ``-1.0`` cuando es ortogonal. Util como base learner
  contrastivo.
- ``redge_moments``: momentos estadisticos (mean, var, skew) de la
  reflectancia red-edge agregada por parcela. Captura la forma de la
  curva red-edge en 3 numeros compactos.

Decisiones canonicas (US-023-preview plan §11 D-3):

- Default ``rep`` por defecto: bien establecido en literatura, computable
  desde S2 ya muestreado, no requiere nueva ingesta GEE.
- Sklearn-compatible (``BaseEstimator`` + ``TransformerMixin``) para
  encajar en Pipelines y pasar el contrato de los tests de US-022b.
- Polars in / Polars out: el DataFrame de entrada ya esta limpio y
  filtrado; el caller (notebook 05 / ``fusion.py``) hace los joins.
- **No consume cuota GEE**: las bandas y stats vienen del parquet de
  features fusionadas (``data/features/*``). El modulo solo combina cols
  ya muestreadas.

Layout de salida (orden estable, downstream depende):

::

    parcel_id (i64) | year (i16) |
    spectral_signature_000 .. spectral_signature_{K-1} (K)

Donde ``K`` depende del descriptor:
- ``rep``: ``K = len(phenology_anchors)`` (default 3 — SOG/peak/senescence
  -> ``spectral_signature_000, 001, 002``).
- ``sam``: ``K = 1`` (un solo angulo escalar).
- ``redge_moments``: ``K = 3 * len(phenology_anchors)`` (mean, var, skew
  por ancla; default 9 cols).

El bloque entra a ``fusion.py`` via ``LEFT JOIN`` sobre ``(parcel_id, year)``,
mismo patron que FarSLIP y phenology_text.

Referencias agronomicas
-----------------------
- Frampton, W.J. et al. (2013), *Evaluating the capabilities of Sentinel-2
  for quantitative estimation of biophysical variables in vegetation*,
  ISPRS J. 82, 83-92. DOI 10.1016/j.isprsjprs.2013.04.007.
- Kruse, F.A. et al. (1993), *The Spectral Image Processing System (SIPS)*,
  Remote Sens. Environ. 44, 145-163 (SAM).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import polars as pl
import structlog
from sklearn.base import BaseEstimator, TransformerMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_PHENOLOGY_ANCHORS",
    "DEFAULT_REDGE_BANDS",
    "SpectralSignatureFeatures",
    "compute_rep",
]


#: Phenology anchors over which the spectral signature is computed. Each
#: anchor maps to a set of columns of the input DataFrame (see
#: :meth:`SpectralSignatureFeatures._extract_anchor_bands`).
DEFAULT_PHENOLOGY_ANCHORS: Final[tuple[str, ...]] = ("sog", "peak", "senescence")

#: Sentinel-2 red-edge bands that dominate the REP computation. The
#: central wavelengths (nm) come from the official
#: ESA Sentinel-2 MSI documentation: B05=703.9, B06=740.2, B07=782.5, B08=835.1.
DEFAULT_REDGE_BANDS: Final[tuple[str, ...]] = ("b05", "b06", "b07", "b08")

#: Central wavelengths in nanometers (ESA S2 MSI specs).
_BAND_WAVELENGTHS_NM: Final[dict[str, float]] = {
    "b04": 664.6,
    "b05": 703.9,
    "b06": 740.2,
    "b07": 782.5,
    "b08": 835.1,
    "b8a": 864.7,
}

#: Supported descriptor types.
Descriptor = Literal["rep", "sam", "redge_moments"]


def compute_rep(
    reflectance_b04: np.ndarray,
    reflectance_b05: np.ndarray,
    reflectance_b06: np.ndarray,
    reflectance_b07: np.ndarray,
) -> np.ndarray:
    """Calcula Red Edge Position (REP) linear-4-bands (Frampton et al. 2013).

    Implementa la formula linealizada de la version "Red Edge Position
    Linear 4-bands" del paper de Frampton (eq. 1):

    .. math::

        REP = 705 + 35 \\times
        \\frac{(R_{B04} + R_{B07}) / 2 - R_{B05}}{R_{B06} - R_{B05}}

    El resultado esta en nm y suele oscilar entre 700 y 740 nm para
    vegetacion sana. Cultivos estresados o suelos desnudos producen valores
    fuera de ese rango — la formula los tolera (no se acota artificialmente).

    Args:
        reflectance_b04: Reflectancia B04 (red, ~665 nm) shape ``(N,)``.
        reflectance_b05: Reflectancia B05 (red-edge 1, ~704 nm) shape ``(N,)``.
        reflectance_b06: Reflectancia B06 (red-edge 2, ~740 nm) shape ``(N,)``.
        reflectance_b07: Reflectancia B07 (red-edge 3, ~783 nm) shape ``(N,)``.

    Returns:
        Vector REP en nm, shape ``(N,)``, dtype ``float64``. Valores
        ``NaN`` cuando la formula degenera (denominador ~0 o entradas NaN).

    Raises:
        ValueError: si los 4 arrays no tienen el mismo shape.
    """
    arrays = (reflectance_b04, reflectance_b05, reflectance_b06, reflectance_b07)
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(
            f"Las 4 bandas deben tener el mismo shape; recibido {shapes!r}."
        )
    b04 = reflectance_b04.astype(np.float64, copy=False)
    b05 = reflectance_b05.astype(np.float64, copy=False)
    b06 = reflectance_b06.astype(np.float64, copy=False)
    b07 = reflectance_b07.astype(np.float64, copy=False)

    denom = b06 - b05
    # Avoid division by zero by producing explicit NaN values.
    safe_denom = np.where(np.abs(denom) > 1e-12, denom, np.nan)
    numerator = (b04 + b07) / 2.0 - b05
    rep = 705.0 + 35.0 * (numerator / safe_denom)
    return rep


class SpectralSignatureFeatures(BaseEstimator, TransformerMixin):
    """Genera features compactas derivadas de la firma espectral por parcela.

    Sklearn-compatible: encaja en ``sklearn.pipeline.Pipeline`` con
    ``StandardScaler``, ``XGBRegressor``, etc. El metodo ``fit`` aprende
    (cuando aplica) el centroide de la clase mayoritaria para el descriptor
    ``sam``; ``transform`` siempre devuelve un :class:`polars.DataFrame`
    con columnas ``parcel_id, year, spectral_signature_NNN``.

    Args:
        descriptor: Tipo de descriptor. Uno de ``"rep"`` (default, Frampton
            et al. 2013 Red Edge Position), ``"sam"`` (Spectral Angle Mapper
            vs centroide de la clase mayoritaria) o ``"redge_moments"``
            (mean/var/skew de la reflectancia red-edge en cada ancla).
        phenology_anchors: Anclajes temporales sobre los que se calcula
            cada descriptor. Default ``("sog", "peak", "senescence")``.
            Para cada ancla se buscan columnas ``{ancla}_{banda}`` (e.g.
            ``sog_b05``); si no existen, el ancla se rellena con NaN.
        bands: Bandas red-edge requeridas (default
            ``("b05", "b06", "b07", "b08")``). Para ``rep`` se usan B04..B07.
        parcel_id_col: Nombre de la columna identificadora (default
            ``"parcel_id"``).
        year_col: Nombre de la columna de anio (default ``"year"``).
        class_col: Columna de clase usada por ``sam`` para calcular el
            centroide en ``fit``. Si es ``None``, ``sam`` calcula contra
            un vector de unos (mero fallback) y avisa via warning
            estructurado.
    """

    def __init__(
        self,
        descriptor: Descriptor = "rep",
        phenology_anchors: tuple[str, ...] = DEFAULT_PHENOLOGY_ANCHORS,
        bands: tuple[str, ...] = DEFAULT_REDGE_BANDS,
        parcel_id_col: str = "parcel_id",
        year_col: str = "year",
        class_col: str | None = "class_id",
    ) -> None:
        self.descriptor = descriptor
        self.phenology_anchors = phenology_anchors
        self.bands = bands
        self.parcel_id_col = parcel_id_col
        self.year_col = year_col
        self.class_col = class_col

    # ------------------------------------------------------------------
    # Sklearn API.
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pl.DataFrame,
        y: object | None = None,
    ) -> SpectralSignatureFeatures:
        """Aprende el centroide de la clase mayoritaria (solo ``sam``).

        Args:
            X: DataFrame Polars con al menos ``parcel_id``, ``year`` y las
                columnas espectrales requeridas por el descriptor.
            y: Ignorado (sklearn signature).

        Returns:
            La instancia ``self`` para encadenar.

        Raises:
            ValueError: si ``descriptor`` no es uno de los soportados.
        """
        if self.descriptor not in ("rep", "sam", "redge_moments"):
            raise ValueError(
                f"`descriptor` debe ser 'rep', 'sam' o 'redge_moments'; "
                f"recibido {self.descriptor!r}."
            )

        self.centroid_: np.ndarray | None = None
        if self.descriptor == "sam":
            self.centroid_ = self._fit_centroid(X)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Produce el DataFrame ``parcel_id, year, spectral_signature_NNN``.

        Args:
            X: DataFrame Polars con las columnas espectrales requeridas.

        Returns:
            DataFrame Polars con shape ``(N, 2 + K)`` donde ``K`` depende
            del descriptor (3 para ``rep`` default, 1 para ``sam``, 9 para
            ``redge_moments`` default).

        Raises:
            ValueError: si ``parcel_id`` o ``year`` no estan en ``X``.
        """
        self._validate_input(X)

        n = X.height
        if self.descriptor == "rep":
            feats = self._transform_rep(X)
        elif self.descriptor == "sam":
            feats = self._transform_sam(X)
        else:  # redge_moments
            feats = self._transform_redge_moments(X)

        k = feats.shape[1]
        feat_cols = [f"spectral_signature_{i:03d}" for i in range(k)]

        out_dict: dict[str, list[object]] = {
            self.parcel_id_col: X.get_column(self.parcel_id_col).to_list(),
            self.year_col: X.get_column(self.year_col).to_list(),
        }
        for j, name in enumerate(feat_cols):
            out_dict[name] = feats[:, j].tolist()

        schema: dict[str, pl.DataType] = {
            self.parcel_id_col: X.schema[self.parcel_id_col],
            self.year_col: X.schema[self.year_col],
        }
        for name in feat_cols:
            schema[name] = pl.Float32()

        logger.info(
            "spectral_signature_transformed",
            descriptor=self.descriptor,
            n_rows=n,
            n_features=k,
        )
        return pl.DataFrame(out_dict, schema=schema)

    def fit_transform(  # type: ignore[override]
        self,
        X: pl.DataFrame,
        y: object | None = None,
        **fit_params: object,
    ) -> pl.DataFrame:
        """Sklearn fit_transform: ``self.fit(X, y).transform(X)``."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Helpers privados.
    # ------------------------------------------------------------------

    def _validate_input(self, X: pl.DataFrame) -> None:
        """Valida que el DataFrame trae las columnas minimas."""
        if not isinstance(X, pl.DataFrame):
            raise TypeError(
                f"`X` debe ser un polars.DataFrame; recibido {type(X)!r}."
            )
        missing = [c for c in (self.parcel_id_col, self.year_col) if c not in X.columns]
        if missing:
            raise ValueError(
                f"`X` no contiene columnas requeridas: {missing}. "
                f"Esperadas al menos: ['{self.parcel_id_col}', '{self.year_col}']."
            )

    def _extract_anchor_bands(
        self,
        X: pl.DataFrame,
        anchor: str,
        bands: Sequence[str],
    ) -> np.ndarray:
        """Devuelve matriz ``(N, len(bands))`` de reflectancias.

        Busca columnas con prefijo ``{anchor}_{band}``. Si no existen,
        intenta el fallback de columnas estilo subset US-018:
        ``{band}_mean`` (ignora el ancla). Si tampoco existen, rellena
        con NaN para mantener el contrato de shape.
        """
        n = X.height
        out = np.full((n, len(bands)), np.nan, dtype=np.float64)
        for j, band in enumerate(bands):
            candidate_cols = [
                f"{anchor}_{band}",
                f"{band}_{anchor}",
                f"{band}_mean",
                band,
            ]
            for col in candidate_cols:
                if col in X.columns and X.schema[col].is_numeric():
                    out[:, j] = X.get_column(col).cast(pl.Float64).to_numpy()
                    break
        return out

    def _transform_rep(self, X: pl.DataFrame) -> np.ndarray:
        """Calcula REP en cada ancla fenologica.

        Requiere las 4 bandas B04/B05/B06/B07 por ancla. Si alguna falta,
        la columna resultante queda en NaN para esa ancla.
        """
        n = X.height
        out = np.full((n, len(self.phenology_anchors)), np.nan, dtype=np.float64)
        required_bands = ("b04", "b05", "b06", "b07")
        for j, anchor in enumerate(self.phenology_anchors):
            bands_matrix = self._extract_anchor_bands(X, anchor, required_bands)
            out[:, j] = compute_rep(
                bands_matrix[:, 0],
                bands_matrix[:, 1],
                bands_matrix[:, 2],
                bands_matrix[:, 3],
            )
        return out

    def _transform_sam(self, X: pl.DataFrame) -> np.ndarray:
        """Calcula Spectral Angle Mapper vs centroide aprendido en fit.

        Devuelve un escalar por parcela: el coseno del angulo entre la
        firma media (concatenacion de bandas red-edge en las anclas) de la
        parcela y el centroide. Sin centroide aprendido (fit no llamado o
        sin ``class_col``), produce coseno vs un vector de unos.
        """
        signatures = self._stack_signatures(X)
        if self.centroid_ is None:
            centroid = np.ones(signatures.shape[1], dtype=np.float64)
            logger.warning(
                "spectral_signature_sam_no_centroid",
                hint="llamar fit() con class_col valido para SAM significativo",
            )
        else:
            centroid = self.centroid_

        # Cosine similarity row-wise, robust to NaN (replaces them with 0).
        sig = np.where(np.isfinite(signatures), signatures, 0.0)
        cen = np.where(np.isfinite(centroid), centroid, 0.0)
        num = sig @ cen
        denom = np.linalg.norm(sig, axis=1) * (np.linalg.norm(cen) + 1e-12)
        safe_denom = np.where(denom > 1e-12, denom, np.nan)
        return (num / safe_denom).reshape(-1, 1)

    def _transform_redge_moments(self, X: pl.DataFrame) -> np.ndarray:
        """Calcula mean/var/skew de las bandas red-edge por ancla.

        Devuelve ``K = 3 * len(phenology_anchors)`` columnas: por cada
        ancla, los 3 momentos estadisticos sobre la curva red-edge.
        """
        n = X.height
        out = np.full((n, 3 * len(self.phenology_anchors)), np.nan, dtype=np.float64)
        for j, anchor in enumerate(self.phenology_anchors):
            bands_matrix = self._extract_anchor_bands(X, anchor, self.bands)
            # Impute NaN within each row with the row's own mean.
            row_means = np.nanmean(bands_matrix, axis=1)
            imputed = bands_matrix.copy()
            for i in range(n):
                if np.isnan(imputed[i]).any():
                    fill_value = row_means[i] if np.isfinite(row_means[i]) else 0.0
                    imputed[i] = np.where(np.isnan(imputed[i]), fill_value, imputed[i])
            mean_row = imputed.mean(axis=1)
            var_row = imputed.var(axis=1)
            # Classic skewness: m3 / m2^(3/2). Tolerates degenerate distributions.
            centred = imputed - mean_row[:, None]
            m2 = (centred**2).mean(axis=1)
            m3 = (centred**3).mean(axis=1)
            safe_m2 = np.where(m2 > 1e-12, m2, np.nan)
            skew_row = m3 / np.power(safe_m2, 1.5)
            out[:, j * 3] = mean_row
            out[:, j * 3 + 1] = var_row
            out[:, j * 3 + 2] = skew_row
        return out

    def _stack_signatures(self, X: pl.DataFrame) -> np.ndarray:
        """Construye la firma concatenada (anchors x bands) por parcela.

        Returns:
            Matriz ``(N, len(phenology_anchors) * len(bands))`` con todas
            las reflectancias en el orden ``anchor0_band0, anchor0_band1,
            ..., anchorK_bandJ``.
        """
        n = X.height
        out = np.full(
            (n, len(self.phenology_anchors) * len(self.bands)),
            np.nan,
            dtype=np.float64,
        )
        for a_idx, anchor in enumerate(self.phenology_anchors):
            bands_matrix = self._extract_anchor_bands(X, anchor, self.bands)
            start = a_idx * len(self.bands)
            out[:, start : start + len(self.bands)] = bands_matrix
        return out

    def _fit_centroid(self, X: pl.DataFrame) -> np.ndarray:
        """Calcula el centroide de la clase mayoritaria para SAM."""
        if self.class_col is None or self.class_col not in X.columns:
            logger.warning(
                "spectral_signature_no_class_col",
                class_col=self.class_col,
                hint="SAM degrada a coseno vs vector de unos",
            )
            return np.ones(
                len(self.phenology_anchors) * len(self.bands), dtype=np.float64
            )

        class_counts = (
            X.group_by(self.class_col)
            .len()
            .sort("len", descending=True)
        )
        majority_class = class_counts.row(0, named=True)[self.class_col]
        subset = X.filter(pl.col(self.class_col) == majority_class)
        signatures = self._stack_signatures(subset)
        # Centroid = column-by-column mean, ignoring NaN.
        centroid = np.nanmean(signatures, axis=0)
        # If the whole column was NaN, fill it with 0 (contributes no angle).
        centroid = np.where(np.isfinite(centroid), centroid, 0.0)
        logger.info(
            "spectral_signature_centroid_fitted",
            majority_class=majority_class,
            n_parcels_class=subset.height,
            n_signature_dims=centroid.size,
        )
        return centroid
