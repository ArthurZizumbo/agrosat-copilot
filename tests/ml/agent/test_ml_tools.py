"""ML / perceiver tool tests (US-045 AC-4, AC-7).

Covers ``classify_new_parcel``, ``explain_prediction`` and ``compare_models``.

- ``classify`` is driven by a real posterior row taken from the
  ``xgb-alphaearth`` OOF parquet (real probabilities, never random); the DB
  embedding fetch is mocked. The no-embedding path is also exercised.
- ``explain`` mocks ``session_scoped_conn`` (DB) and the phenology descriptor
  (no Gemini / litellm network), asserting the structured phenology text and the
  vigor mapping from the real scalar landmarks.
- ``compare`` reads the *real* per-parcel OOF parquets shipped in
  ``ml/eval/oof/`` as deterministic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.agent.tools.classify as classify_mod
import ml.agent.tools.compare as compare_mod
import ml.agent.tools.explain as explain_mod
from ml.agent.schemas import (
    ClassificationResult,
    ClassifyParcelInput,
    CompareModelsInput,
    ExplainPredictionInput,
    Explanation,
    GeoJSONGeometry,
    ModelComparison,
)
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

from .conftest import (
    SESSION_A,
    FakeConn,
    FakeRecord,
    fake_session_scoped_conn,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OOF_DIR = _REPO_ROOT / "ml" / "eval" / "oof"
# A parcel present in the real OOF parquets (composite canonical id).
_REAL_PARCEL = "10003_1103071"
_POLYGON = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}


def _real_xgb_posterior() -> np.ndarray:
    """Return a real 18-class posterior row from the xgb-alphaearth OOF parquet.

    Sourcing the vector from the actual model output keeps the classify test
    grounded in real probabilities (per the US-045 "no random" rule).
    """
    path = _OOF_DIR / f"oof_parcel_xgb-alphaearth_fold{5}.parquet"
    if not path.exists():
        pytest.skip(f"OOF fixture not found: {path} (run `dvc pull ml/eval/oof`).")
    frame = canonical_parcel_id(pl.read_parquet(path), col="canonical_parcel_id")
    row = frame.filter(pl.col("canonical_parcel_id") == _REAL_PARCEL)
    if row.height == 0:
        pytest.skip(f"parcel {_REAL_PARCEL} absent from xgb OOF fixture.")
    return row.select(PROB_COLUMNS).to_numpy().astype(np.float64)[0]


class _FakeClassifier:
    """``_XgbAlphaEarthClassifier`` stand-in returning a fixed real posterior."""

    def __init__(self, proba: np.ndarray, class_names: dict[int, str]) -> None:
        self._proba = proba
        self.class_names = class_names

    def predict_proba_18(self, embedding: np.ndarray) -> np.ndarray:
        return self._proba


# ---------------------------------------------------------------------------
# classify_new_parcel
# ---------------------------------------------------------------------------
async def test_classify_restrict_default_nine_classes(monkeypatch, make_ctx) -> None:
    """US-053: the default (restrict ON) returns the france-9 resolved posterior.

    The default ``restrict_to_resolved_classes=True`` masks the 18-class posterior
    down to the nine ``france-9`` classes and renormalizes; the surfaced classes
    are the real semantic18 names of those nine and they sum to ~1.
    """
    from ml.eval.class_remap import get_label_space, restrict_posterior

    proba = _real_xgb_posterior()
    class_names = {i: f"class_{i}" for i in range(18)}

    async def _fake_fetch_embedding(ctx, year, aoi):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    space = get_label_space("france-9")
    assert isinstance(out, ClassificationResult)
    # Nine resolved classes, renormalized to ~1.
    assert len(out.class_probabilities) == 9
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # The surfaced labels are the real france-9 crop names, not class_xx.
    assert set(out.class_probabilities) == set(space.class_names.values())
    # The reported argmax matches the renormalized restricted distribution.
    expected = restrict_posterior(proba, space)
    top_cid = max(expected, key=lambda c: expected[c])
    assert out.crop_class == space.class_names[top_cid]
    assert out.confidence == pytest.approx(float(expected[top_cid]))


async def test_classify_restrict_off_full_eighteen(monkeypatch, make_ctx) -> None:
    """With ``restrict_to_resolved_classes=False`` the full 18-class posterior shows."""
    proba = _real_xgb_posterior()
    expected_idx = int(np.argmax(proba))
    class_names = {i: f"class_{i}" for i in range(18)}

    async def _fake_fetch_embedding(ctx, year, aoi):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            restrict_to_resolved_classes=False,
        ),
        make_ctx(),
    )

    assert isinstance(out, ClassificationResult)
    assert out.crop_class == f"class_{expected_idx}"
    assert out.confidence == pytest.approx(float(proba[expected_idx]))
    # Full 18-class posterior surfaced and summing to ~1 (post-softmax).
    assert len(out.class_probabilities) == 18
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


async def test_classify_use_stacking_with_oof(monkeypatch, make_ctx) -> None:
    """``use_stacking=True`` serves the Stacking-5 posterior when one is available.

    ``_stacking_posterior`` is stubbed to return a real OOF-shaped 18-vector (so
    no PASTIS-R / OOF I/O happens); the restricted result must reflect THAT
    posterior, not the xgb fallback. The classifier is patched to fail if invoked
    so the test proves the stacking branch was taken.
    """
    from ml.eval.class_remap import get_label_space, restrict_posterior

    stack_proba = _real_xgb_posterior()  # real probabilities, never random

    async def _fake_fetch_embedding(ctx, year, aoi):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    async def _fake_stacking(ctx, inp):
        return stack_proba

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(classify_mod, "_stacking_posterior", _fake_stacking)
    # The xgb fallback must NOT predict when stacking succeeds.
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: _FakeClassifier(np.full(18, np.nan), {i: f"class_{i}" for i in range(18)}),
    )

    out = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019, use_stacking=True),
        make_ctx(),
    )

    space = get_label_space("france-9")
    expected = restrict_posterior(stack_proba, space)
    top_cid = max(expected, key=lambda c: expected[c])
    assert out.crop_class == space.class_names[top_cid]
    assert out.confidence == pytest.approx(float(expected[top_cid]))


async def test_classify_use_stacking_degrades_without_oof(monkeypatch, make_ctx) -> None:
    """``use_stacking=True`` with no OOF degrades to xgb-alphaearth, no crash (AC-8).

    ``_resolve_canonical_parcel_id`` returns a parcel id but ``_load_stacking_five``
    raises ``FileNotFoundError`` (DVC not pulled); ``run`` must fall back to the xgb
    posterior and emit the structured ``classify_stacking_unavailable`` warning
    rather than propagating the error.
    """
    proba = _real_xgb_posterior()
    class_names = {i: f"class_{i}" for i in range(18)}

    async def _fake_fetch_embedding(ctx, year, aoi):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    async def _fake_resolve(ctx, aoi):
        return "10003_1103071"

    def _raise_no_oof():
        raise FileNotFoundError("Stacking-5 OOF parquet missing (dvc not pulled).")

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(classify_mod, "_resolve_canonical_parcel_id", _fake_resolve)
    monkeypatch.setattr(classify_mod, "_load_stacking_five", _raise_no_oof)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(
            session_id=SESSION_A,
            aoi=_POLYGON,
            year=2019,
            use_stacking=True,
            restrict_to_resolved_classes=False,
        ),
        make_ctx(),
    )

    # Fell back to xgb (full 18-class posterior since restrict is off), no crash.
    assert isinstance(out, ClassificationResult)
    assert len(out.class_probabilities) == 18
    expected_idx = int(np.argmax(proba))
    assert out.crop_class == f"class_{expected_idx}"


async def test_classify_needs_gee_when_no_embedding(monkeypatch, make_ctx) -> None:
    """With no persisted embedding the tool returns the controlled sentinel."""

    async def _no_embedding(ctx, year, aoi):
        return None

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _no_embedding)
    # The classifier must NOT be loaded down this path.
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: pytest.fail("classifier should not load without an embedding"),
    )

    out = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert out.crop_class == "needs_gee_sampling"
    assert 0.0 < out.confidence < 1.0  # uniform prior, not a fabricated certainty


async def test_fetch_embedding_resolves_by_aoi_intersection(monkeypatch, make_ctx) -> None:
    """``_fetch_parcel_embedding`` resolves the parcel by AOI intersection (B-2).

    Regression for US-045 B-2: the embedding must be chosen by ``ST_Intersects``
    against the drawn AOI, not by "session's most recently updated parcel". We
    drive the real ``_fetch_parcel_embedding`` over a :class:`FakeConn` and assert
    the SQL contains the spatial predicate and that the AOI GeoJSON is bound as a
    parameter (``$3``) -- proving the AOI geometry actually reaches the query.
    """
    import ml.agent.db as agent_db

    embedding = list(np.linspace(0.0, 1.0, 64, dtype=np.float64))
    conn = FakeConn(fetchrow_row=FakeRecord(alphaearth_embedding=embedding))
    monkeypatch.setattr(agent_db, "session_scoped_conn", fake_session_scoped_conn(conn))

    aoi = GeoJSONGeometry(**_POLYGON)
    out = await classify_mod._fetch_parcel_embedding(make_ctx(), 2019, aoi)

    assert out is not None
    assert out.shape == (64,)
    # The spatial query (not the RLS set_config) carries the intersection clause.
    spatial_calls = [c for c in conn.calls if "ST_Intersects" in c[0]]
    assert len(spatial_calls) == 1, "expected exactly one ST_Intersects query"
    sql, args = spatial_calls[0]
    assert "ST_SetSRID(ST_GeomFromGeoJSON($3), 4326)" in sql
    assert "p.session_id = $1" in sql  # multi-tenant filter preserved
    assert "fp.year = $2" in sql
    # session_id, year and the AOI GeoJSON are bound positionally as $1, $2, $3.
    assert args[0] == SESSION_A
    assert args[1] == 2019
    bound_aoi = json.loads(args[2])
    assert bound_aoi["type"] == "Polygon"
    assert bound_aoi["coordinates"] == _POLYGON["coordinates"]


async def test_fetch_embedding_none_when_aoi_misses(monkeypatch, make_ctx) -> None:
    """No persisted parcel intersects the AOI -> ``None`` (caller routes to GEE).

    Regression for US-045 B-2: when ``ST_Intersects`` matches nothing the fetch
    returns ``None`` so ``run`` emits the controlled ``needs_gee_sampling`` result
    instead of borrowing an unrelated parcel's embedding.
    """
    import ml.agent.db as agent_db

    conn = FakeConn(fetchrow_row=None)  # no parcel intersects the AOI
    monkeypatch.setattr(agent_db, "session_scoped_conn", fake_session_scoped_conn(conn))

    aoi = GeoJSONGeometry(**_POLYGON)
    out = await classify_mod._fetch_parcel_embedding(make_ctx(), 2019, aoi)

    assert out is None


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------
async def test_explain_with_mocked_descriptor(monkeypatch, make_ctx) -> None:
    """``explain_prediction`` builds phenology text + uses the mocked descriptor.

    The FFT harmonics drive the reconstructed curve; the descriptor is mocked so
    no Gemini/litellm call happens. We assert the structured phenology text,
    vigor, and that the mocked description is surfaced.
    """
    # ``_reconstruct_curve`` requires the full amp/phase set for k=0..3
    # (DEFAULT_FFT_HARMONICS=3); supply all eight so the real inverse-FFT yields
    # a finite non-zero NDVI curve and the descriptor path runs.
    phenology_json = {
        "NDVI_fft_amp_0": 0.55,
        "NDVI_fft_amp_1": 0.20,
        "NDVI_fft_amp_2": 0.05,
        "NDVI_fft_amp_3": 0.02,
        "NDVI_fft_phase_0": 0.0,
        "NDVI_fft_phase_1": 0.10,
        "NDVI_fft_phase_2": 0.30,
        "NDVI_fft_phase_3": 0.50,
    }
    record = {
        "crop_class": "wheat",
        "confidence": 0.88,
        "phenology": json.dumps(phenology_json),
        "sog_doy": 95,
        "peak_doy": 185,
        "peak_value": 0.82,
        "senescence_doy": 265,
        "ndvi_auc": 120.5,
        "maturity_duration_days": 170,
    }

    async def _fake_fetch(ctx, parcel_id):
        return dict(record)

    # ``explain`` imports ``session_scoped_conn`` lazily inside ``_fetch_parcel``;
    # patch the fetch helper itself (the DB boundary) instead.
    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch)

    captured = {}

    def _fake_descriptor(curve, *, parcel_id=None, crop_type_hint=None, temperature=0.0):
        captured["called"] = True
        captured["len"] = int(np.asarray(curve).size)
        captured["hint"] = crop_type_hint
        return "Descripcion fenologica generada por el descriptor (mock)."

    # Patch where the tool imports it (lazy import inside run()).
    monkeypatch.setattr(
        "ml.features.phenology_description.generate_phenology_description",
        _fake_descriptor,
    )

    out = await explain_mod.run(
        ExplainPredictionInput(session_id=SESSION_A, parcel_id=11), make_ctx()
    )

    assert isinstance(out, Explanation)
    assert out.parcel_id == 11
    assert out.crop_class == "wheat"
    assert out.confidence == pytest.approx(0.88)
    assert out.vigor == "high"  # peak_value 0.82 >= 0.7
    assert out.phenology_text  # non-empty structured block
    assert "dia 95" in out.phenology_text  # real SOG landmark surfaced
    assert captured.get("called") is True
    assert out.description == "Descripcion fenologica generada por el descriptor (mock)."


async def test_explain_falls_back_without_fft(monkeypatch, make_ctx) -> None:
    """Without FFT harmonics the description falls back to the structured text.

    No LLM call must happen in this path; the description equals the phenology
    text block built from the real scalar landmarks.
    """
    record = {
        "crop_class": "maize",
        "confidence": 0.5,
        "phenology": None,  # no harmonics
        "sog_doy": 100,
        "peak_doy": 190,
        "peak_value": 0.35,  # low vigor
        "senescence_doy": 260,
        "ndvi_auc": 80.0,
        "maturity_duration_days": None,
    }

    async def _fake_fetch(ctx, parcel_id):
        return dict(record)

    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch)
    monkeypatch.setattr(
        "ml.features.phenology_description.generate_phenology_description",
        lambda *a, **k: pytest.fail("descriptor must not be called without a curve"),
    )

    out = await explain_mod.run(
        ExplainPredictionInput(session_id=SESSION_A, parcel_id=12), make_ctx()
    )

    assert out.vigor == "low"  # 0.35 < 0.4
    assert out.description == out.phenology_text
    assert out.phenology_text.startswith("Fenologia:")


# ---------------------------------------------------------------------------
# compare_models (deferred) -- real OOF parquet fixtures
# ---------------------------------------------------------------------------
def _oof_present(*models: str) -> bool:
    """True if every model's per-parcel OOF parquet exists locally."""
    return all((_OOF_DIR / f"oof_parcel_{m}_fold5.parquet").exists() for m in models)


def test_predict_for_parcel_reads_real_oof() -> None:
    """``_predict_for_parcel`` returns the real argmax for a known parcel.

    utae and xgb-alphaearth both classify parcel ``10003_1103071`` as class 1 in
    the shipped OOF parquets -- a deterministic, real-data assertion.
    """
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")
    assert compare_mod._predict_for_parcel("utae", _REAL_PARCEL) == 1
    assert compare_mod._predict_for_parcel("xgb-alphaearth", _REAL_PARCEL) == 1


def test_predict_for_parcel_missing_model_returns_none() -> None:
    """A model with no OOF parquet is reported as ``None`` (never fabricated)."""
    assert compare_mod._predict_for_parcel("does-not-exist", _REAL_PARCEL) is None


def _owns_parcel(monkeypatch) -> None:
    """Patch the multi-tenant gate so the parcel is treated as session-owned.

    ``compare.run`` resolves the parcel against ``parcels`` (session-scoped)
    before any OOF read. These OOF-focused tests are not about the DB gate, so we
    stub ownership to ``True`` -- the dedicated gate behaviour is covered by
    ``test_compare_models_rejects_foreign_parcel`` /
    ``test_compare_models_accepts_owned_parcel`` below.
    """

    async def _belongs(ctx, parcel_id):
        return True

    # These OOF tests exercise the legacy numeric-cast bridge, so the parcel
    # carries no stored canonical id (US-079): force the DB lookup to ``None`` (no
    # real DB) so ``_compute_comparison`` falls back to the integer-id cast.
    async def _no_canonical(ctx, parcel_id):
        return None

    monkeypatch.setattr(compare_mod, "_parcel_belongs_to_session", _belongs)
    monkeypatch.setattr(classify_mod, "fetch_canonical_parcel_id", _no_canonical)


async def test_compare_models_full_agreement(monkeypatch, make_ctx) -> None:
    """Two agreeing models over the real parcel yield ``agreement == 1.0``.

    The DB integer parcel id is bridged to the real composite canonical id so the
    tool reads the genuine OOF rows (no mock of the probabilities themselves).
    """
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    _owns_parcel(monkeypatch)
    # The DB int parcel id has no composite OOF row, so redirect the per-model
    # lookup to the real composite parcel id. ``_predict_for_parcel`` still reads
    # the genuine OOF parquet and takes the real argmax -- no faked probabilities.
    real_predict = compare_mod._predict_for_parcel
    monkeypatch.setattr(
        compare_mod,
        "_predict_for_parcel",
        lambda model, canonical_id: real_predict(model, _REAL_PARCEL),
    )

    out = await compare_mod.run(
        CompareModelsInput(
            session_id=SESSION_A, parcel_id=10003, models=["utae", "xgb-alphaearth"]
        ),
        make_ctx(),
    )

    assert isinstance(out, ModelComparison)
    assert set(out.predictions) == {"utae", "xgb-alphaearth"}
    # Both predict the same class -> identical labels, perfect agreement.
    assert len(set(out.predictions.values())) == 1
    assert out.agreement == pytest.approx(1.0)


async def test_compare_models_empty_when_no_match(monkeypatch, make_ctx) -> None:
    """A plain int parcel id matches no composite OOF row -> empty, no crash."""
    if not _oof_present("utae", "tsvit-pheno"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    _owns_parcel(monkeypatch)
    out = await compare_mod.run(
        CompareModelsInput(session_id=SESSION_A, parcel_id=999_999, models=["utae", "tsvit-pheno"]),
        make_ctx(),
    )

    assert out.predictions == {}
    assert out.agreement == 0.0


async def test_compare_models_enqueues_when_defer_wired(monkeypatch, make_ctx) -> None:
    """When a defer hook is wired the job is enqueued and a typed result returned."""
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    _owns_parcel(monkeypatch)
    enqueued = {}

    async def _defer(job_name, payload):
        enqueued["job"] = job_name
        enqueued["payload"] = payload
        return "handle-123"

    out = await compare_mod.run(
        CompareModelsInput(
            session_id=SESSION_A, parcel_id=999_999, models=["utae", "xgb-alphaearth"]
        ),
        make_ctx(defer=_defer),
    )

    assert enqueued["job"] == "compare_models"
    assert enqueued["payload"]["parcel_id"] == 999_999
    assert isinstance(out, ModelComparison)  # inline result still returned


async def test_fetch_canonical_parcel_id_reads_column(monkeypatch, make_ctx) -> None:
    """``fetch_canonical_parcel_id`` returns the stored id (US-079), else ``None``."""
    import ml.agent.db as agent_db

    conn = FakeConn(fetchval_value="10003_1103071")
    monkeypatch.setattr(agent_db, "session_scoped_conn", fake_session_scoped_conn(conn))
    assert await classify_mod.fetch_canonical_parcel_id(make_ctx(), 52) == "10003_1103071"

    conn_none = FakeConn(fetchval_value=None)
    monkeypatch.setattr(agent_db, "session_scoped_conn", fake_session_scoped_conn(conn_none))
    assert await classify_mod.fetch_canonical_parcel_id(make_ctx(), 52) is None


async def test_compare_models_uses_stored_canonical_id(monkeypatch, make_ctx) -> None:
    """A stored canonical id resolves the real OOF rows for an int parcel id (US-079).

    ``parcel_id=52`` never matches a composite OOF key by the numeric cast; the
    stored ``canonical_parcel_id`` is what bridges it to the genuine fold-5 rows.
    """
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    async def _belongs(ctx, parcel_id):
        return True

    async def _canonical(ctx, parcel_id):
        return _REAL_PARCEL

    monkeypatch.setattr(compare_mod, "_parcel_belongs_to_session", _belongs)
    monkeypatch.setattr(classify_mod, "fetch_canonical_parcel_id", _canonical)

    out = await compare_mod.run(
        CompareModelsInput(session_id=SESSION_A, parcel_id=52, models=["utae", "xgb-alphaearth"]),
        make_ctx(),
    )
    # Real OOF rows resolved via the stored canonical id (not the numeric "52").
    assert set(out.predictions) == {"utae", "xgb-alphaearth"}


# -- multi-tenant gate (B-1 regression): parcel ownership before any OOF read --
async def test_compare_models_rejects_foreign_parcel(monkeypatch, make_ctx) -> None:
    """A parcel not owned by the session yields the controlled empty comparison.

    Regression for B-1: ``compare_models`` declared ``session_id`` but never used
    it, reading global OOF artifacts by ``parcel_id`` alone (cross-tenant leak).
    The gate now resolves the id against ``parcels WHERE id=$1 AND session_id=$2``
    inside ``session_scoped_conn``; a miss returns ``predictions={}`` and never
    touches the OOF parquets or the defer hook.
    """
    # ``fetchrow`` returns ``None`` -> parcel not visible to the session.
    fake_conn = FakeConn(fetchrow_row=None)
    monkeypatch.setattr("ml.agent.db.session_scoped_conn", fake_session_scoped_conn(fake_conn))
    monkeypatch.setattr(
        compare_mod,
        "_predict_for_parcel",
        lambda *a, **k: pytest.fail("OOF must not be read for a foreign parcel"),
    )

    async def _defer(job_name, payload):
        pytest.fail("defer must not be called for a foreign parcel")

    out = await compare_mod.run(
        CompareModelsInput(session_id=SESSION_A, parcel_id=42, models=["utae", "xgb-alphaearth"]),
        make_ctx(defer=_defer),
    )

    assert isinstance(out, ModelComparison)
    assert out.parcel_id == 42
    assert out.predictions == {}
    assert out.agreement == 0.0
    # The ownership query ran session-scoped with the id + session bound ($1,$2).
    ownership = [c for c in fake_conn.calls if "FROM parcels" in c[0]]
    assert len(ownership) == 1
    assert ownership[0][1] == (42, SESSION_A)


async def test_compare_models_accepts_owned_parcel(monkeypatch, make_ctx) -> None:
    """An owned parcel passes the gate and the OOF comparison runs as before."""
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    # ``fetchrow`` returns a row -> parcel belongs to the session.
    fake_conn = FakeConn(fetchrow_row=FakeRecord({"?column?": 1}))
    monkeypatch.setattr("ml.agent.db.session_scoped_conn", fake_session_scoped_conn(fake_conn))
    # Bridge the int id to the real composite OOF row (real argmax, no fakes).
    real_predict = compare_mod._predict_for_parcel
    monkeypatch.setattr(
        compare_mod,
        "_predict_for_parcel",
        lambda model, canonical_id: real_predict(model, _REAL_PARCEL),
    )

    out = await compare_mod.run(
        CompareModelsInput(
            session_id=SESSION_A, parcel_id=10003, models=["utae", "xgb-alphaearth"]
        ),
        make_ctx(),
    )

    assert isinstance(out, ModelComparison)
    assert set(out.predictions) == {"utae", "xgb-alphaearth"}
    assert out.agreement == pytest.approx(1.0)
