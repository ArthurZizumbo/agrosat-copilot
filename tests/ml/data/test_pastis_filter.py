"""Tests para ``ml.data.pastis_filter.PastisFilter`` (US-032).

Cubren la rama nueva ``dominance_ratio`` (regla 3:1 Meadow per-patch) con
**golden values** sobre TARGET sinteticos de conteos exactos, los edge cases
definidos en el plan, la exclusion de Background(0)/Void(19) independiente de
``ignore_index``, la parametrizacion por ``target_classes`` / ``meadow_class``,
y un test de **regresion** que verifica que el modo ``coverage`` legacy queda
intacto. Ninguno de estos tests depende del dataset PASTIS-R real.

Dos tests adicionales (``@pytest.mark.skipif``) corren sobre PASTIS-R real si
esta descargado: ``TARGET_10102`` -> DROP y ``TARGET_10004`` -> KEEP, las
referencias documentadas en el plan.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ml.data.pastis_filter import PastisFilter

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_REAL_ANN_DIR = _REAL_PASTIS_ROOT / "ANNOTATIONS"

_pastis_present = _REAL_ANN_DIR.exists() and any(_REAL_ANN_DIR.glob("TARGET_*.npy"))

# PASTIS-R patches son nativamente 128x128 = 16384 px.
_H = _W = 128
_N_PX = _H * _W


# ---------------------------------------------------------------------------
# Helpers: TARGET sintetico con conteos exactos
# ---------------------------------------------------------------------------


def _target_from_counts(counts: dict[int, int]) -> np.ndarray:
    """Construye un TARGET ``(3, 128, 128)`` uint8 con conteos exactos por clase.

    El canal 0 (semantica) se rellena en orden de aparicion del dict; los px
    restantes quedan en Background (0). Los canales 1 y 2 (instancia / heatmap
    en PASTIS-R) van a cero. ``PastisFilter`` solo lee el canal 0.

    Args:
        counts: mapa ``{class_id: n_pixels}``. La suma debe ser ``<= 16384``.

    Returns:
        Array ``(3, 128, 128)`` uint8 con el canal 0 conteniendo exactamente
        ``counts[c]`` pixeles de cada clase ``c`` (resto = Background).
    """
    total = sum(counts.values())
    if total > _N_PX:
        raise ValueError(f"counts sum {total} exceeds {_N_PX} pixels")
    flat = np.zeros(_N_PX, dtype=np.uint8)
    cursor = 0
    for class_id, n_px in counts.items():
        flat[cursor : cursor + n_px] = class_id
        cursor += n_px
    semantic = flat.reshape(_H, _W)
    zeros = np.zeros_like(semantic)
    return np.stack([semantic, zeros, zeros], axis=0)


def _write_metadata(root: Path, fold_by_pid: dict[int, int]) -> None:
    """Escribe un ``metadata.geojson`` minimo con ``ID_PATCH`` + ``Fold``.

    Args:
        root: raiz del dataset sintetico.
        fold_by_pid: mapa ``{patch_id: fold}``.
    """
    features = [
        {
            "id": str(pid),
            "type": "Feature",
            "properties": {"ID_PATCH": int(pid), "Fold": int(fold), "TILE": "T30UXV"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }
        for pid, fold in fold_by_pid.items()
    ]
    (root / "metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def _make_root(tmp_path: Path, patches: dict[int, dict[int, int]], fold: int = 1) -> Path:
    """Construye una raiz PASTIS-R sintetica con ``patches`` (id -> counts).

    Args:
        tmp_path: directorio temporal del test.
        patches: mapa ``{patch_id: {class_id: n_px}}``.
        fold: fold comun para todos los patches.

    Returns:
        Path a la raiz con ``ANNOTATIONS/`` + ``metadata.geojson``.
    """
    root = tmp_path / "PASTIS-R"
    ann = root / "ANNOTATIONS"
    ann.mkdir(parents=True, exist_ok=True)
    for pid, counts in patches.items():
        np.save(ann / f"TARGET_{pid}.npy", _target_from_counts(counts))
    _write_metadata(root, {pid: fold for pid in patches})
    return root


def _mask_from_counts(counts: dict[int, int]) -> np.ndarray:
    """Devuelve el canal semantico ``(H, W)`` int32 listo para ``_passes``."""
    return _target_from_counts(counts)[0].astype(np.int32)


def _filter(root: Path, **kwargs: object) -> PastisFilter:
    """Construye un ``PastisFilter`` apuntando a ``root`` con kwargs extra."""
    return PastisFilter(pastis_root=root, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 6.1 Filtro 3:1 (dominance_ratio) — golden values
# ---------------------------------------------------------------------------


def test_dominance_keep_exact_3to1(tmp_path: Path) -> None:
    """Meadow=12288, class2=4096 -> 12288 <= 3*4096=12288 -> KEEP (`<=` inclusivo)."""
    root = _make_root(tmp_path, {1: {1: 12288, 2: 4096}})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0, meadow_class=1)
    passes, ratio_obs = f._passes(_mask_from_counts({1: 12288, 2: 4096}))
    assert passes is True
    assert ratio_obs == pytest.approx(3.0)


def test_dominance_drop_above_3to1(tmp_path: Path) -> None:
    """Meadow=12289, class2=4095 -> 12289 > 3*4095=12285 -> DROP."""
    root = _make_root(tmp_path, {1: {1: 12289, 2: 4095}})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts({1: 12289, 2: 4095}))
    assert passes is False
    assert ratio_obs > 3.0


def test_dominance_drop_just_above_3to1(tmp_path: Path) -> None:
    """Meadow ~= 3.0001*second -> DROP (limite superior estricto, no inclusivo)."""
    second = 4000
    meadow = math.ceil(3.0001 * second)  # 12001 -> 12001 > 3*4000=12000
    assert meadow + second <= _N_PX
    root = _make_root(tmp_path, {1: {1: meadow, 2: second}})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts({1: meadow, 2: second}))
    assert passes is False
    assert ratio_obs > 3.0


def test_dominance_keep_below_3to1(tmp_path: Path) -> None:
    """Meadow=8192, class2=8192 -> 8192 <= 24576 -> KEEP holgado."""
    root = _make_root(tmp_path, {1: {1: 8192, 2: 8192}})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts({1: 8192, 2: 8192}))
    assert passes is True
    assert ratio_obs == pytest.approx(1.0)


def test_dominance_keep_no_meadow(tmp_path: Path) -> None:
    """Meadow=0, class2=4096 -> KEEP (no dominado), metric 0.0."""
    root = _make_root(tmp_path, {1: {2: 4096}})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts({2: 4096}))
    assert passes is True
    assert ratio_obs == 0.0


def test_dominance_drop_no_competitor(tmp_path: Path) -> None:
    """Meadow>0 sin otra clase objetivo (resto Background) -> DROP, metric inf."""
    root = _make_root(tmp_path, {1: {1: 12288}})  # resto = Background(0)
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts({1: 12288}))
    assert passes is False
    assert math.isinf(ratio_obs)


def test_dominance_all_background(tmp_path: Path) -> None:
    """Mascara sin Meadow ni clases objetivo -> KEEP (meadow==0)."""
    root = _make_root(tmp_path, {1: {0: _N_PX}})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts({0: _N_PX}))
    assert passes is True
    assert ratio_obs == 0.0


# ---------------------------------------------------------------------------
# 6.2 Exclusion Background/Void independiente de ignore_index
# ---------------------------------------------------------------------------


def test_bg_void_excluded_from_second(tmp_path: Path) -> None:
    """Background y Void mayores que la clase objetivo NO se eligen como 2da.

    Background=8000, Void=2000, Meadow=2000, class2=4384. La 2da clase debe ser
    class2 (4384), no Background ni Void -> 2000 <= 3*4384 -> KEEP.
    """
    counts = {0: 8000, 19: 2000, 1: 2000, 2: 4384}
    root = _make_root(tmp_path, {1: counts})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts(counts))
    assert passes is True
    assert ratio_obs == pytest.approx(2000 / 4384)


def test_decision_invariant_to_ignore_index(tmp_path: Path) -> None:
    """Misma mascara con ignore_index 255 vs 19 -> decision dominance identica."""
    counts = {0: 8000, 19: 2000, 1: 2000, 2: 4384}
    root = _make_root(tmp_path, {1: counts})
    mask = _mask_from_counts(counts)
    f255 = _filter(root, target_classes=[2], mode="dominance_ratio", ignore_index=255)
    f19 = _filter(root, target_classes=[2], mode="dominance_ratio", ignore_index=19)
    assert f255._passes(mask) == f19._passes(mask)


def test_void_not_picked_as_second(tmp_path: Path) -> None:
    """Void (19) presente y enorme nunca actua como 2da clase competidora.

    Meadow=12288, Void=4095, sin otra clase objetivo -> second_px=0 -> DROP
    (Void no cuenta como competidor pese a su conteo alto).
    """
    counts = {1: 12288, 19: 4095}
    root = _make_root(tmp_path, {1: counts})
    f = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    passes, ratio_obs = f._passes(_mask_from_counts(counts))
    assert passes is False
    assert math.isinf(ratio_obs)


# ---------------------------------------------------------------------------
# 6.3 Parametrizacion por n_classes / target_classes
# ---------------------------------------------------------------------------


def test_second_class_restricted_to_target(tmp_path: Path) -> None:
    """target_classes restringe la 2da clase: [2,3] -> KEEP, [3] -> DROP."""
    counts = {1: 6000, 2: 2500, 3: 1800}
    root = _make_root(tmp_path, {1: counts})
    mask = _mask_from_counts(counts)

    f_both = _filter(root, target_classes=[2, 3], mode="dominance_ratio", ratio=3.0)
    passes_both, _ = f_both._passes(mask)
    assert passes_both is True  # second=2500 -> 6000 <= 7500

    f_only3 = _filter(root, target_classes=[3], mode="dominance_ratio", ratio=3.0)
    passes_only3, _ = f_only3._passes(mask)
    assert passes_only3 is False  # second=1800 -> 6000 > 5400


def test_meadow_class_configurable(tmp_path: Path) -> None:
    """meadow_class cambia que clase se acota -> decision OPUESTA sobre el patch.

    counts: class1=1000, class2=10000, class3=3000 (suma 14000 <= 16384).
    - meadow_class=2: clase acotada = 2 (10000), 2da entre {1,3} = 3000 ->
      10000 > 3*3000=9000 -> DROP (ratio 10000/3000 ~ 3.33).
    - meadow_class=1: clase acotada = 1 (1000), 2da entre {2,3} = 10000 ->
      1000 <= 3*10000=30000 -> KEEP (ratio 1000/10000 = 0.1).
    """
    counts = {1: 1000, 2: 10000, 3: 3000}
    root = _make_root(tmp_path, {1: counts})
    mask = _mask_from_counts(counts)

    f_m2 = _filter(root, target_classes=[1, 3], mode="dominance_ratio", ratio=3.0, meadow_class=2)
    passes_m2, ratio_m2 = f_m2._passes(mask)
    assert passes_m2 is False
    assert ratio_m2 == pytest.approx(10000 / 3000)

    f_m1 = _filter(root, target_classes=[2, 3], mode="dominance_ratio", ratio=3.0, meadow_class=1)
    passes_m1, ratio_m1 = f_m1._passes(mask)
    assert passes_m1 is True
    assert ratio_m1 == pytest.approx(1000 / 10000)
    assert passes_m1 != passes_m2


def test_target_classes_is_set_order_independent(tmp_path: Path) -> None:
    """target_classes=[3,2] da el mismo resultado que [2,3] (ordena por conteo)."""
    counts = {1: 6000, 2: 2500, 3: 1800}
    root = _make_root(tmp_path, {1: counts})
    mask = _mask_from_counts(counts)
    f_ab = _filter(root, target_classes=[2, 3], mode="dominance_ratio", ratio=3.0)
    f_ba = _filter(root, target_classes=[3, 2], mode="dominance_ratio", ratio=3.0)
    assert f_ab._passes(mask) == f_ba._passes(mask)


# ---------------------------------------------------------------------------
# 6.4 Regresion del modo coverage legacy (NO romper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("min_coverage", "expected"),
    [(0.20, True), (0.25, True), (0.30, False)],
)
def test_coverage_legacy_unchanged(tmp_path: Path, min_coverage: float, expected: bool) -> None:
    """Modo coverage default: patch coverage=0.25 -> KEEP/KEEP/DROP (`>=`).

    class2=4096, resto Background valido (ignore_index=255 no excluye nada) ->
    coverage = 4096/16384 = 0.25.
    """
    counts = {2: 4096, 0: _N_PX - 4096}
    root = _make_root(tmp_path, {1: counts})
    f = _filter(root, target_classes=[2], min_coverage=min_coverage)  # mode default
    passes, cov = f._passes(_mask_from_counts(counts))
    assert cov == pytest.approx(0.25)
    assert passes is expected


def test_coverage_legacy_is_default(tmp_path: Path) -> None:
    """Sin pasar ``mode`` el filtro usa ``coverage`` (default preservado)."""
    f = _filter(_make_root(tmp_path, {1: {2: 4096}}), target_classes=[2])
    assert f.mode == "coverage"
    assert f.ratio == 3.0
    assert f.meadow_class == 1


def test_coverage_all_ignore_returns_false_zero(tmp_path: Path) -> None:
    """Mascara toda ignore_index -> (False, 0.0) (comportamiento heredado)."""
    counts = {19: _N_PX}
    root = _make_root(tmp_path, {1: counts})
    f = _filter(root, target_classes=[2], ignore_index=19)  # coverage mode
    passes, cov = f._passes(_mask_from_counts(counts))
    assert passes is False
    assert cov == 0.0


def test_passes_returns_2tuple_both_modes(tmp_path: Path) -> None:
    """``_passes`` devuelve ``tuple[bool, float]`` en ambos modos."""
    counts = {1: 8192, 2: 8192}
    root = _make_root(tmp_path, {1: counts})
    mask = _mask_from_counts(counts)

    f_cov = _filter(root, target_classes=[2])
    res_cov = f_cov._passes(mask)
    assert isinstance(res_cov, tuple) and len(res_cov) == 2
    assert isinstance(res_cov[0], (bool, np.bool_)) and isinstance(res_cov[1], float)

    f_dom = _filter(root, target_classes=[2], mode="dominance_ratio")
    res_dom = f_dom._passes(mask)
    assert isinstance(res_dom, tuple) and len(res_dom) == 2
    assert isinstance(res_dom[0], bool) and isinstance(res_dom[1], float)


def test_filter_folds_runs_both_modes(tmp_path: Path) -> None:
    """``filter_folds([1])`` corre sin TypeError en coverage y dominance_ratio."""
    patches = {
        100: {1: 12288, 2: 4096},  # dominance KEEP (3:1 exacto)
        101: {1: 12289, 2: 4095},  # dominance DROP
        102: {2: 8192, 0: 8192},  # dominance KEEP (meadow=0); coverage=0.5
    }
    root = _make_root(tmp_path, patches)

    f_cov = _filter(root, target_classes=[2], min_coverage=0.50)
    kept_cov = f_cov.filter_folds([1])
    assert set(kept_cov) == {102}  # solo 102 alcanza coverage 0.50

    f_dom = _filter(root, target_classes=[2], mode="dominance_ratio", ratio=3.0)
    kept_dom = f_dom.filter_folds([1])
    assert set(kept_dom) == {100, 102}


def test_coverage_stats_runs_both_modes(tmp_path: Path) -> None:
    """``coverage_stats([1])`` no lanza y reporta n_patches en ambos modos."""
    patches = {100: {1: 12288, 2: 4096}, 101: {1: 8192, 2: 8192}}
    root = _make_root(tmp_path, patches)

    stats_cov = _filter(root, target_classes=[2]).coverage_stats([1])
    assert stats_cov["n_patches"] == 2

    stats_dom = _filter(
        root, target_classes=[2], mode="dominance_ratio"
    ).coverage_stats([1])
    assert stats_dom["n_patches"] == 2


# ---------------------------------------------------------------------------
# 6.X Referencia REAL sobre PASTIS-R (opt-in si esta descargado)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _pastis_present, reason="PASTIS-R no descargado en data/PASTIS-R/ANNOTATIONS/."
)
def test_real_target_10102_drops() -> None:
    """``TARGET_10102`` (Meadow >> 3*Corn) -> DROP con la regla 3:1 real."""
    f = PastisFilter(
        pastis_root=_REAL_PASTIS_ROOT,
        target_classes=[2, 3, 8],
        mode="dominance_ratio",
        ratio=3.0,
        meadow_class=1,
    )
    mask = f._load_mask(10102)
    passes, ratio_obs = f._passes(mask)
    assert passes is False
    assert ratio_obs > 3.0


@pytest.mark.skipif(
    not _pastis_present, reason="PASTIS-R no descargado en data/PASTIS-R/ANNOTATIONS/."
)
def test_real_target_10004_keeps() -> None:
    """``TARGET_10004`` (Meadow=0) -> KEEP (no dominado por Meadow)."""
    f = PastisFilter(
        pastis_root=_REAL_PASTIS_ROOT,
        target_classes=[2, 3, 8],
        mode="dominance_ratio",
        ratio=3.0,
        meadow_class=1,
    )
    mask = f._load_mask(10004)
    passes, ratio_obs = f._passes(mask)
    assert passes is True
    assert ratio_obs == 0.0
