"""Importa runs MLflow desde un file store local al servidor MLflow (Postgres).

Caso de uso (US-025): los runs reales de TSViT se entrenaron en la VM L4 y su
tracking quedo en un file store ``./mlruns/<exp>/`` (formato de archivos),
mientras el servidor MLflow local del proyecto es un Docker con backend
Postgres (``http://localhost:5010``). Ambos almacenes son distintos: el servidor
no lee la carpeta ``./mlruns/``. Este modulo reconstruye cada run del file store
en el experimento del servidor via :class:`mlflow.tracking.MlflowClient`,
preservando params, tags (incluidos ``code_version`` y ``data_version``,
exigidos por la regla 10 de ``CLAUDE.md``), las series de metricas por epoch y
los timestamps originales.

Operativo permanente (no es un script ``scripts/_*.py`` ad-hoc): la seleccion de
runs se hace por *allowlist* de ``run_id`` para no arrastrar smokes ni intentos
abandonados que comparten ``run_name`` con los reales. La importacion es
idempotente: re-ejecutar detecta el ``run_name`` ya presente y lo omite.

Uso CLI::

    python -m ml.utils.import_runs_from_filestore \\
        --src-experiment-dir mlruns/965679031955557780 \\
        --run-ids 3955879d26e4498a860517c10867d672,63aacbec1ffb45d493d15ceb63d73210 \\
        --dest-experiment-id 7
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)

#: Map from file store status (integer) to the MLflow status string.
_STATUS_MAP = {"1": "RUNNING", "2": "SCHEDULED", "3": "FINISHED", "4": "FAILED", "5": "KILLED"}

#: Tags that are NOT copied: ``mlflow.runName`` is injected by ``create_run`` via
#: ``run_name=`` (copying it again would duplicate it).
_SKIP_TAGS = frozenset({"mlflow.runName"})


@contextlib.contextmanager
def _silence_mlflow_url() -> Iterator[None]:
    """Suprime el ``print`` de URL de MLflow (contiene un emoji).

    ``MlflowClient.set_terminated``/``create_run`` escriben en ``stdout`` una
    linea decorada con un emoji (``\\U0001f3c3``); en consolas Windows con
    codificacion cp1252 eso lanza ``UnicodeEncodeError``. Redirige ``stdout`` a
    un buffer durante la llamada para que el log estructurado (structlog, en
    ``stderr``) siga visible sin romper la importacion.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _read_run_dir(run_dir: Path) -> dict:
    """Parsea un directorio de run del file store a un dict estructurado.

    Args:
        run_dir: Ruta al directorio del run (``<exp>/<run_id>/``).

    Returns:
        Dict con ``meta`` (start_time, end_time, status, run_name, user_id),
        ``params`` (dict), ``tags`` (dict) y ``metrics`` (dict
        ``nombre -> list[(value, timestamp_ms, step)]`` con las series por epoch).

    Raises:
        FileNotFoundError: si falta ``meta.yaml``.
    """
    meta_path = run_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no existe meta.yaml en {run_dir}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    params: dict[str, str] = {}
    params_dir = run_dir / "params"
    if params_dir.is_dir():
        for p in params_dir.iterdir():
            if p.is_file():
                params[p.name] = p.read_text(encoding="utf-8").strip()

    tags: dict[str, str] = {}
    tags_dir = run_dir / "tags"
    if tags_dir.is_dir():
        for t in tags_dir.iterdir():
            if t.is_file():
                tags[t.name] = t.read_text(encoding="utf-8").strip()

    metrics: dict[str, list[tuple[float, int, int]]] = {}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.is_dir():
        for m in metrics_dir.iterdir():
            if not m.is_file():
                continue
            points: list[tuple[float, int, int]] = []
            for line in m.read_text(encoding="utf-8").splitlines():
                if not line.strip():  # guards against empty lines / CRLF on Windows
                    continue
                ts, val, step = line.split()
                points.append((float(val), int(ts), int(step)))
            metrics[m.name] = points

    return {"meta": meta, "params": params, "tags": tags, "metrics": metrics}


def _run_exists(client: MlflowClient, experiment_id: str, run_name: str) -> str | None:
    """Devuelve el run_id existente con ese ``run_name`` en el experimento, o None.

    El match es por ``run_name`` (no por ``run_id``) porque ``create_run`` genera
    un id nuevo en Postgres: el id del file store no se preserva.

    Args:
        client: Cliente MLflow apuntando al servidor destino.
        experiment_id: Id del experimento destino.
        run_name: Nombre del run a buscar.

    Returns:
        El ``run_id`` del primer match, o ``None`` si no existe.
    """
    existing = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=1,
    )
    return str(existing[0].info.run_id) if existing else None


def import_run(
    client: MlflowClient,
    run_dir: Path,
    dest_experiment_id: str,
    *,
    recreate: bool = False,
    upload_artifacts: bool = False,
) -> str | None:
    """Reconstruye un run del file store en el experimento destino del servidor.

    Idempotente: si ya existe un run con el mismo ``run_name`` en el destino, lo
    omite (o lo borra y recrea si ``recreate=True``). La creacion se envuelve en
    try/except que borra el run si algo falla antes de ``set_terminated``, para
    no dejar un run ``RUNNING`` incompleto que bloquearia futuras corridas.

    Args:
        client: Cliente MLflow apuntando al servidor destino.
        run_dir: Directorio del run en el file store.
        dest_experiment_id: Id del experimento destino.
        recreate: Si ``True`` y el run ya existe, lo borra y vuelve a crear.
        upload_artifacts: Si ``True`` sube ``best.pt`` via ``log_artifact``
            (requiere proxied-artifacts habilitado en el servidor).

    Returns:
        El ``run_id`` creado, o ``None`` si se omitio (ya existia y no recreate).
    """
    from mlflow.entities import Metric, Param

    parsed = _read_run_dir(run_dir)
    meta = parsed["meta"]
    run_name = meta["run_name"]

    existing_id = _run_exists(client, dest_experiment_id, run_name)
    if existing_id is not None:
        if not recreate:
            logger.info(
                "import_run_skip", run_name=run_name, reason="ya existe", existing=existing_id[:12]
            )
            return None
        client.delete_run(existing_id)
        logger.info("import_run_deleted_for_recreate", run_name=run_name, deleted=existing_id[:12])

    # Tags to copy (excludes mlflow.runName, which create_run sets via run_name=).
    run_tags = {k: v for k, v in parsed["tags"].items() if k not in _SKIP_TAGS}

    with _silence_mlflow_url():
        run = client.create_run(
            experiment_id=dest_experiment_id,
            start_time=int(meta["start_time"]),
            run_name=run_name,
            tags=run_tags,
        )
    run_id = str(run.info.run_id)
    try:
        # Params in a single batch.
        params = [Param(k, str(v)) for k, v in parsed["params"].items()]
        # Metrics: all the per-epoch series, preserving timestamp and step.
        metric_entities: list[Metric] = []
        for name, points in parsed["metrics"].items():
            for value, ts, step in points:
                metric_entities.append(Metric(name, value, ts, step))
        # log_batch accepts <1000 metrics / <100 params per call.
        client.log_batch(run_id, metrics=metric_entities, params=params, tags=[])

        if upload_artifacts:
            best_pt = run_dir / "artifacts" / "checkpoint" / "best.pt"
            if best_pt.is_file():
                with _silence_mlflow_url():
                    client.log_artifact(run_id, str(best_pt), artifact_path="checkpoint")
                logger.info("import_run_artifact_uploaded", run_name=run_name, file="best.pt")

        status = _STATUS_MAP.get(str(meta.get("status")), "FINISHED")
        with _silence_mlflow_url():
            client.set_terminated(run_id, status=status, end_time=int(meta["end_time"]))
    except Exception:
        # Do not leave an incomplete RUNNING run: delete it to keep idempotency.
        client.delete_run(run_id)
        logger.error("import_run_failed_rolled_back", run_name=run_name, run_id=run_id[:12])
        raise

    logger.info(
        "import_run_done",
        run_name=run_name,
        run_id=run_id[:12],
        n_metrics=len(metric_entities),
        n_params=len(params),
        artifacts=upload_artifacts,
    )
    return run_id


def import_runs_from_filestore(
    src_experiment_dir: Path | str,
    run_ids: list[str],
    dest_experiment_id: str,
    *,
    tracking_uri: str | None = None,
    recreate: bool = False,
    upload_artifacts: bool = False,
) -> dict[str, str | None]:
    """Importa una allowlist de runs de un file store al servidor MLflow.

    Args:
        src_experiment_dir: Directorio del experimento en el file store
            (``mlruns/<exp_id>/``).
        run_ids: Allowlist de ``run_id`` a importar (los demas se ignoran).
        dest_experiment_id: Id del experimento destino en el servidor.
        tracking_uri: URI del servidor; si ``None`` usa
            :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.
        recreate: Recrear runs ya existentes en vez de omitirlos.
        upload_artifacts: Subir ``best.pt`` (requiere proxied-artifacts).

    Returns:
        Dict ``run_id_origen -> run_id_destino`` (``None`` si se omitio).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from ml.utils.mlflow_utils import resolve_tracking_uri

    uri = tracking_uri or resolve_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    src = Path(src_experiment_dir)
    result: dict[str, str | None] = {}
    for rid in run_ids:
        run_dir = src / rid
        if not run_dir.is_dir():
            logger.warning("import_run_missing_dir", run_id=rid, path=str(run_dir))
            result[rid] = None
            continue
        result[rid] = import_run(
            client,
            run_dir,
            dest_experiment_id,
            recreate=recreate,
            upload_artifacts=upload_artifacts,
        )
    logger.info("import_runs_summary", uri=uri, dest_exp=dest_experiment_id, imported=result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description="Importa runs MLflow de un file store local al servidor Postgres."
    )
    parser.add_argument("--src-experiment-dir", required=True)
    parser.add_argument(
        "--run-ids", required=True, help="Lista separada por comas de run_id (allowlist)."
    )
    parser.add_argument("--dest-experiment-id", required=True)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--upload-artifacts", action="store_true")
    args = parser.parse_args(argv)

    import_runs_from_filestore(
        args.src_experiment_dir,
        [r.strip() for r in args.run_ids.split(",") if r.strip()],
        args.dest_experiment_id,
        tracking_uri=args.tracking_uri,
        recreate=args.recreate,
        upload_artifacts=args.upload_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
