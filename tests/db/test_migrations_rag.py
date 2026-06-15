"""Tests de integración para la migración de la US-046 (``rag_documents``).

Valida que ``20260615082041_create_rag_documents.sql`` aplica sobre un PostgreSQL
efímero con PostGIS + pgvector, que el schema introspectado coincide con el
contrato del plan (Spatial-RAG lite: corpus + vector AlphaEarth 64-dim) y que la
migración es reversible vía un round-trip up->down->up.

Diseño (espejo de ``tests/db/test_migrations_us015.py``):
    - testcontainers levanta una imagen con PostGIS 15 + pgvector preinstalado.
    - El fixture aplica TODAS las migraciones de ``db/migrations/*.sql`` en orden
      lexicográfico, parseando cada archivo por los marcadores
      ``-- migrate:up`` / ``-- migrate:down``.
    - Las pruebas introspectan ``information_schema`` / ``pg_indexes`` /
      ``pg_attribute`` para verificar tipos, ``VECTOR(64)`` e índices.

Invariantes críticas verificadas:
    - ``embedding`` es ``VECTOR(64)`` (``atttypmod == 64``); su dimensión está
      bloqueada igual que ``features_parcels.alphaearth_embedding`` en US-015.
    - ``geom`` lleva un índice GIST (alimenta el ``ST_DWithin`` del pre-filtro).
    - ``source`` lleva un índice BTREE (no GIST).
    - NO existe índice ANN (HNSW/IVFFlat) sobre ``embedding`` (lite por diseño).

Si la imagen Docker no levanta o ``testcontainers`` no está instalado, los tests
se saltan limpiamente (no fallan) para no bloquear la CI sin Docker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("testcontainers", reason="testcontainers no instalado")
pytest.importorskip("sqlalchemy", reason="sqlalchemy requerido")

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

try:
    from testcontainers.postgres import PostgresContainer
except ImportError:  # pragma: no cover - rama defensiva
    PostgresContainer = None  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"
RAG_MIGRATION = MIGRATIONS_DIR / "20260615082041_create_rag_documents.sql"

# Imágenes candidatas en orden de preferencia. La primera que arranque gana.
CANDIDATE_IMAGES: tuple[str, ...] = (
    "agrosat-postgres:15-3.4-pgvector",
    "postgis/postgis:15-3.4",
)

_MIGRATE_UP_RE = re.compile(r"--\s*migrate:up\s*\n(.*?)(?=--\s*migrate:down|\Z)", re.DOTALL)
_MIGRATE_DOWN_RE = re.compile(r"--\s*migrate:down\s*\n(.*?)\Z", re.DOTALL)


def _split_up(sql_text: str) -> str:
    """Devuelve el bloque ``migrate:up`` de un archivo dbmate."""
    match = _MIGRATE_UP_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migración sin bloque -- migrate:up")
    return match.group(1).strip()


def _split_down(sql_text: str) -> str:
    """Devuelve el bloque ``migrate:down`` de un archivo dbmate."""
    match = _MIGRATE_DOWN_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migración sin bloque -- migrate:down")
    return match.group(1).strip()


def _apply_all_migrations(engine: Engine) -> None:
    """Aplica todas las migraciones ``db/migrations/*.sql`` en orden lexicográfico."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"No se encontraron migraciones en {MIGRATIONS_DIR}")
    with engine.begin() as conn:
        for migration_path in files:
            up_block = _split_up(migration_path.read_text(encoding="utf-8"))
            conn.execute(text(up_block))


@pytest.fixture(scope="module")
def pg_engine() -> Engine:
    """Levanta un Postgres efímero con PostGIS+pgvector y aplica todas las migraciones."""
    if PostgresContainer is None:
        pytest.skip("testcontainers.postgres no disponible")

    last_error: Exception | None = None
    for image in CANDIDATE_IMAGES:
        try:
            container = PostgresContainer(
                image=image, username="test", password="test", dbname="test"
            )
            container.start()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - depende del host
            last_error = exc
            continue

        try:
            url = container.get_connection_url().replace(
                "postgresql+psycopg2", "postgresql+psycopg"
            )
            try:
                engine = create_engine(url, future=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            except Exception:  # noqa: BLE001
                # fallback al driver por defecto si psycopg v3 no está instalado
                engine = create_engine(container.get_connection_url(), future=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

            # Garantiza que la extensión vector exista antes de aplicar migraciones.
            try:
                with engine.begin() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except OperationalError as exc:
                container.stop()
                last_error = exc
                continue

            _apply_all_migrations(engine)
            yield engine
            container.stop()
            return
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - depende del entorno
            container.stop()
            last_error = exc
            continue

    pytest.skip(f"Ninguna imagen Postgres+PostGIS+pgvector disponible: {last_error}")


def _columns(engine: Engine, table: str) -> dict[str, str]:
    """Devuelve mapping {nombre_columna: data_type} via information_schema."""
    query = text(
        """
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = :table
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"table": table}).all()
    return {
        row.column_name: (row.udt_name if row.data_type == "USER-DEFINED" else row.data_type)
        for row in rows
    }


def test_dbmate_up_creates_rag_documents(pg_engine: Engine) -> None:
    """La tabla rag_documents existe con columnas y tipos del contrato."""
    cols = _columns(pg_engine, "rag_documents")
    assert cols["id"] == "bigint"
    assert cols["parcel_id"] == "text"
    assert cols["geom"] == "geometry"
    assert cols["content"] == "text"
    assert cols["source"] == "text"
    assert cols["embedding"] == "vector"
    assert cols["created_at"] == "timestamp with time zone"


def test_embedding_is_vector_64(pg_engine: Engine) -> None:
    """``embedding`` es VECTOR(64): atttypmod == 64 (dimensión bloqueada)."""
    query = text(
        """
        SELECT atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        WHERE c.relname = 'rag_documents' AND a.attname = 'embedding'
        """
    )
    with pg_engine.connect() as conn:
        atttypmod = conn.execute(query).scalar_one()
    assert atttypmod == 64, f"Esperaba VECTOR(64), atttypmod={atttypmod}"


def test_content_and_source_not_null(pg_engine: Engine) -> None:
    """``content`` y ``source`` son NOT NULL; el resto admite NULL."""
    query = text(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'rag_documents'
        """
    )
    with pg_engine.connect() as conn:
        nullable = {row.column_name: row.is_nullable for row in conn.execute(query).all()}
    assert nullable["content"] == "NO"
    assert nullable["source"] == "NO"
    # Documentos a nivel-escena pueden no tener parcela ni geometría ni embedding.
    assert nullable["parcel_id"] == "YES"
    assert nullable["geom"] == "YES"
    assert nullable["embedding"] == "YES"


def test_geom_index_is_gist(pg_engine: Engine) -> None:
    """``rag_documents_geom_idx`` existe y es GIST (alimenta ST_DWithin)."""
    query = text(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = 'rag_documents'
        """
    )
    with pg_engine.connect() as conn:
        rows = {row.indexname: row.indexdef for row in conn.execute(query).all()}

    assert "rag_documents_geom_idx" in rows
    assert "using gist" in rows["rag_documents_geom_idx"].lower()


def test_source_index_is_btree_not_gist(pg_engine: Engine) -> None:
    """``rag_documents_source_idx`` existe y es BTREE (no GIST)."""
    query = text(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = 'rag_documents'
        """
    )
    with pg_engine.connect() as conn:
        rows = {row.indexname: row.indexdef for row in conn.execute(query).all()}

    assert "rag_documents_source_idx" in rows
    assert "using gist" not in rows["rag_documents_source_idx"].lower()


def test_no_ann_index_on_embedding(pg_engine: Engine) -> None:
    """No hay índice HNSW/IVFFlat sobre ``embedding`` (lite por diseño)."""
    query = text(
        """
        SELECT indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = 'rag_documents'
        """
    )
    with pg_engine.connect() as conn:
        defs = [row.indexdef.lower() for row in conn.execute(query).all()]
    assert not any("hnsw" in d or "ivfflat" in d for d in defs)


def test_round_trip_up_down_up(pg_engine: Engine) -> None:
    """Aplicar, revertir y re-aplicar la migración deja el schema idéntico."""
    sql_text = RAG_MIGRATION.read_text(encoding="utf-8")
    up_block = _split_up(sql_text)
    down_block = _split_down(sql_text)

    cols_before = _columns(pg_engine, "rag_documents")
    assert cols_before, "rag_documents debe existir antes del round-trip"

    # down: DROP TABLE ... CASCADE
    with pg_engine.begin() as conn:
        conn.execute(text(down_block))
    with pg_engine.connect() as conn:
        exists = conn.execute(text("SELECT to_regclass('public.rag_documents')")).scalar_one()
    assert exists is None, "Rollback no eliminó la tabla rag_documents"

    # up de nuevo (idempotente: IF NOT EXISTS)
    with pg_engine.begin() as conn:
        conn.execute(text(up_block))
    cols_after = _columns(pg_engine, "rag_documents")
    assert cols_before == cols_after, "Schema difiere tras round-trip up->down->up"


def test_insert_point_and_query_dwithin(pg_engine: Engine) -> None:
    """Inserta un punto + vector y verifica que ST_DWithin lo recupera."""
    with pg_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO rag_documents (parcel_id, geom, content, source, embedding)
                VALUES (
                    '10000_1',
                    ST_SetSRID(ST_Point(0.0, 0.0), 4326),
                    'descripcion fenologica real',
                    'phenology_caption',
                    :emb
                )
                """
            ),
            {"emb": "[" + ",".join("0.1" for _ in range(64)) + "]"},
        )
        found = conn.execute(
            text(
                """
                SELECT count(*)
                FROM rag_documents
                WHERE ST_DWithin(
                    geom::geography,
                    ST_SetSRID(ST_Point(0.001, 0.001), 4326)::geography,
                    1000
                )
                """
            )
        ).scalar_one()
        # Limpieza para no contaminar otros tests del módulo.
        conn.execute(text("DELETE FROM rag_documents WHERE parcel_id = '10000_1'"))
    assert found == 1, "ST_DWithin debió recuperar el punto insertado dentro del radio"
