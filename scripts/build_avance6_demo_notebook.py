"""Builder of the Avance 6 conversational-copilot demo notebook (Equipo 17).

Generates ``notebooks/final_model/Avance6.Demo.Copiloto.Equipo17.ipynb``
programmatically and reproducibly (same pattern as
``scripts/build_avance5_notebook.py``). Unlike the Avance 5 integrator, this
notebook is a **live end-to-end demonstration** of the conversational agent: it
imports the real ``ml.agent`` stack and runs it against the seeded demo session
in the local Postgres+PostGIS+pgvector instance. There are no placeholders and
no fabricated outputs -- the notebook is committed UNEXECUTED and is meant to be
run with papermill against the real database.

What the notebook shows (the "Be My Eyes" pattern):

1. Cover and the Be My Eyes framing (perceiver = the team's trained models emit
   TEXT, reasoner = Gemini/Qwen reasons over that text, never over pixels).
2. The ten geospatial tools (US-045): a table built from
   ``build_function_declarations()``.
3. The perceiver (US-046): ``PerceiverLayer.observe(parcel_id)`` on real parcels,
   showing the structured TEXT observation and its ``to_prompt_block()``.
4. The conversational agent end to end (US-047): real queries through
   ``agent.stream_response`` rendering the tool_call -> tool_result -> answer flow.
5. Spatial-RAG lite on/off (US-046): the ``rag_documents`` corpus and a real
   ``spatial_rag`` retrieval near a demo parcel (anti-hallucination grounding).
6. The on-prem Qwen variant (US-048): documented; only ``make_backend('qwen35')``
   is executed (local, no serving required) to prove the backend swap.
7. Accessible conclusions with the real numbers.

Visible prose (markdown, captions, prints) is Spanish with proper accents and
the letter "n" with tilde; code, identifiers, comments and docstrings stay in
English ASCII (project convention). Section titles read as work, not as a
rubric checklist.

The async cells use top-level ``await`` (modern ipykernel supports it), exactly
as the agent's public API requires.

Usage::

    poetry run python scripts/build_avance6_demo_notebook.py \\
        --out notebooks/final_model/Avance6.Demo.Copiloto.Equipo17.ipynb

Permanent operational script (does NOT violate the ``scripts/_*.py`` anti-pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/final_model/Avance6.Demo.Copiloto.Equipo17.ipynb")


def _build_cells() -> list:
    """Build the list of cells (markdown + code) of the demo notebook."""
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells: list = []

    # ---------------------------------------------------------------- Cover ---
    cells.append(
        md(
            "# Avance 6 - Demostracion del Copiloto Conversacional\n\n"
            "### Patron \"Be My Eyes\": los modelos del equipo *ven*, un LLM frontera *razona*\n\n"
            "**Equipo 17** - AgroSatCopilot\n\n"
            "---\n\n"
            "Este cuaderno demuestra, de principio a fin y **con datos reales**, el copiloto "
            "conversacional para analisis satelital agricola. La idea central es el patron "
            "**Be My Eyes**:\n\n"
            "- El **perceiver** son los modelos entrenados por el equipo (clasificador de cultivo "
            "AlphaEarth+XGBoost y el descriptor fenologico). No hablan con el usuario: **miran una "
            "parcela y emiten una observacion en TEXTO** (cultivo, fenologia, vigor, confianza).\n"
            "- El **reasoner** es un LLM frontera (Gemini en la nube, o Qwen on-prem). **No clasifica "
            "pixeles**: lee ese texto, llama herramientas geoespaciales cuando hace falta y redacta "
            "la respuesta en lenguaje natural.\n\n"
            "Esta separacion es lo que hace al sistema **auditable y anti-alucinacion**: toda cifra "
            "que el reasoner enuncia proviene de una herramienta o de una observacion del perceiver, "
            "nunca de la imaginacion del modelo.\n\n"
            "> El cuaderno corre contra la sesion de demostracion ya sembrada en la base de datos "
            "local (Postgres + PostGIS + pgvector): 12 parcelas reales del conjunto PASTIS con su "
            "embedding satelital y su fenologia, y un corpus de 300 documentos fenologicos con "
            "vector para el RAG."
        )
    )

    # --------------------------------------------------- parameters (papermill) ---
    cells.append(
        code(
            "# Parameters cell (papermill). Defaults are the seeded demo values; override\n"
            "# any of them at run time with `papermill -p <name> <value>`.\n"
            "model = 'gemini-2.5-flash'        # fast reasoner that works for the demo\n"
            "demo_user = 'demo@agrosat.dev'    # seeded demo session owner\n"
            "n_parcels_show = 12               # how many seeded parcels to surface\n"
            "n_perceiver_parcels = 3           # how many parcels to run through the perceiver\n"
            "rag_radius_m = 20000.0            # ST_DWithin radius for the Spatial-RAG demo (m)\n"
            "rag_top_k = 5                     # documents retrieved per RAG query"
        )
    )
    # Tag the parameters cell so papermill recognises it.
    cells[-1]["metadata"]["tags"] = ["parameters"]

    # ------------------------------------------------------------------ Setup ---
    cells.append(
        md(
            "## Preparacion del entorno\n\n"
            "Resolvemos la raiz del repositorio (sin rutas absolutas), cargamos `.env.local` para "
            "tomar la cadena de conexion y las credenciales del LLM, y abrimos el *pool* de la base "
            "de datos. La consola de Windows usa cp1252; forzamos UTF-8 en la salida estandar para "
            "que los acentos de los logs y los textos en espanol no rompan la ejecucion."
        )
    )
    cells.append(
        code(
            "# --- Repo bootstrap, UTF-8 safety, env, autoreload ---\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Windows console is cp1252; structlog and Spanish prose use accents. Reconfigure\n"
            "# stdout/stderr to UTF-8 so an accented log line never raises UnicodeEncodeError.\n"
            "for _stream in (sys.stdout, sys.stderr):\n"
            "    try:\n"
            "        _stream.reconfigure(encoding='utf-8')\n"
            "    except (AttributeError, ValueError):\n"
            "        pass\n\n"
            "from ml.utils.notebook_setup import find_repo_root, load_env_local\n\n"
            "REPO_ROOT = find_repo_root()\n"
            "if str(REPO_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT))\n"
            "os.chdir(REPO_ROOT)\n"
            "load_env_local(REPO_ROOT)\n\n"
            "# Hot-reload so edits in ml/*.py are picked up without restarting the kernel.\n"
            "%load_ext autoreload\n"
            "%autoreload 2\n\n"
            "import time\n\n"
            "from IPython.display import Markdown, display\n\n"
            "print('repo:', REPO_ROOT)\n"
            "print('reasoner model:', model, '| demo user:', demo_user)"
        )
    )

    cells.append(
        code(
            "# --- Connect to the seeded demo session ---\n"
            "from backend.app.core.config import get_settings\n"
            "from ml.agent.db import get_pool\n\n"
            "settings = get_settings()\n"
            "pool = await get_pool()\n\n"
            "async with pool.acquire() as conn:\n"
            "    session_id = await conn.fetchval(\n"
            "        'SELECT id FROM chat_sessions WHERE user_id = $1', demo_user\n"
            "    )\n"
            "    n_parcels = await conn.fetchval('SELECT count(*) FROM parcels')\n"
            "    n_features = await conn.fetchval('SELECT count(*) FROM features_parcels')\n"
            "    n_rag = await conn.fetchval('SELECT count(*) FROM rag_documents')\n\n"
            "assert session_id is not None, (\n"
            "    f'no demo session for user {demo_user!r}; seed the demo data first'\n"
            ")\n"
            "display(Markdown(\n"
            "    f'**Sesion de demostracion**: `{session_id}`  \\n'\n"
            "    f'Parcelas sembradas: **{n_parcels}** | features de parcela: **{n_features}** | '\n"
            "    f'documentos RAG: **{n_rag}**'\n"
            "))"
        )
    )

    cells.append(
        code(
            "# --- Shared tool-execution context (multi-tenant: scoped to this session) ---\n"
            "from ml.agent.context import ToolContext\n\n"
            "ctx = ToolContext(pool=pool, settings=settings, session_id=session_id)\n"
            "print('ToolContext listo | session_id =', ctx.session_id)"
        )
    )

    # ============================================ Seccion 1 - Las herramientas ===
    cells.append(
        md(
            "## 1. Las herramientas geoespaciales\n\n"
            "El reasoner no accede a la base de datos por su cuenta: actua **solo** a traves de un "
            "conjunto cerrado de herramientas geoespaciales, cada una con un esquema de entrada y "
            "salida validado (Pydantic). Esto acota lo que el agente puede hacer y deja cada accion "
            "rastreable.\n\n"
            "Hay diez herramientas. Cinco son **sincronas** (se ejecutan en linea dentro del bucle "
            "del agente: listar parcelas, serie temporal, estadisticas de un area, clasificar una "
            "parcela nueva y explicar una prediccion). Las otras cinco son **diferidas** "
            "(*deferred*): pueden completarse fuera de linea via un *worker* (busqueda de escenas, "
            "teselas de mapa, guardar un area, comparar modelos y recuperar contexto del RAG).\n\n"
            "La tabla siguiente se construye directamente desde "
            "`build_function_declarations()`, la misma fuente de verdad que se le anuncia al LLM."
        )
    )
    cells.append(
        code(
            "# Build the tool table straight from the function declarations advertised to the LLM.\n"
            "import polars as pl\n\n"
            "from google.genai import types as genai_types\n\n"
            "from ml.agent.tools import TOOL_SPECS, build_function_declarations\n\n"
            "declarations = build_function_declarations()\n"
            "_rows = []\n"
            "for decl in declarations:\n"
            "    deferred = TOOL_SPECS[decl.name][3]\n"
            "    _rows.append({\n"
            "        'herramienta': decl.name,\n"
            "        'tipo': 'diferida' if deferred else 'sincrona',\n"
            "        'comportamiento': (\n"
            "            'NON_BLOCKING' if deferred else 'BLOCKING'\n"
            "        ),\n"
            "        'descripcion': decl.description,\n"
            "    })\n"
            "tools_df = pl.DataFrame(_rows).sort('tipo', 'herramienta')\n"
            "with pl.Config(fmt_str_lengths=120, tbl_width_chars=200):\n"
            "    display(tools_df)\n"
            "print('total de herramientas:', tools_df.height,\n"
            "      '| sincronas:', tools_df.filter(pl.col('tipo') == 'sincrona').height,\n"
            "      '| diferidas:', tools_df.filter(pl.col('tipo') == 'diferida').height)"
        )
    )
    cells.append(
        md(
            "**Lectura**: el agente que conversa en esta demo expone las cinco herramientas "
            "sincronas; las diferidas (incluida la del RAG, `retrieve_context`) requieren el "
            "ejecutor en segundo plano y la bandera `rag_enabled`. Mas abajo demostramos el RAG "
            "llamando su capa directamente, sin pasar por el bucle diferido."
        )
    )

    # ============================================ Seccion 2 - El perceiver ===
    cells.append(
        md(
            "## 2. El perceiver: los modelos del equipo emiten TEXTO\n\n"
            "El perceiver es el componente que **mira** una parcela a traves de los modelos "
            "entrenados y produce una **observacion en texto plano**, nunca tensores ni "
            "probabilidades crudas hacia el reasoner. Reune dos cosas reales:\n\n"
            "- el **posterior sobre los 18 cultivos** del clasificador AlphaEarth+XGBoost "
            "(el mismo que esta detras de la herramienta de clasificacion), y\n"
            "- la **fenologia, el vigor y la descripcion** en lenguaje natural del descriptor "
            "fenologico (Wen et al., 2025) sobre las metricas reales de la parcela.\n\n"
            "El metodo `to_prompt_block()` rinde esa observacion como el bloque de **anclaje** que "
            "se inyecta en el prompt del reasoner. *Ese texto* es lo que el LLM consume; la imagen "
            "y los logits nunca cruzan la frontera. Lo mostramos sobre algunas parcelas reales de "
            "la sesion."
        )
    )
    cells.append(
        code(
            "# Pick the first N seeded parcels of this session and run the perceiver on each.\n"
            "from ml.agent.perceiver import PerceiverLayer\n\n"
            "async with pool.acquire() as conn:\n"
            "    parcel_ids = await conn.fetch(\n"
            "        'SELECT id FROM parcels ORDER BY id LIMIT $1', int(n_perceiver_parcels)\n"
            "    )\n"
            "parcel_ids = [int(r['id']) for r in parcel_ids]\n"
            "print('parcelas a observar:', parcel_ids)\n\n"
            "perceiver = PerceiverLayer(ctx)\n"
            "observations = []\n"
            "for _pid in parcel_ids:\n"
            "    _t0 = time.perf_counter()\n"
            "    obs = await perceiver.observe(_pid)\n"
            "    _ms = round((time.perf_counter() - _t0) * 1000.0, 1)\n"
            "    observations.append((obs, _ms))\n"
            "print('observaciones generadas:', len(observations))"
        )
    )
    cells.append(
        code(
            "# Tabular view of the structured TEXT fields the perceiver exposes (no tensors).\n"
            "_rows = []\n"
            "for obs, _ms in observations:\n"
            "    _rows.append({\n"
            "        'parcela': obs.parcel_id,\n"
            "        'cultivo': obs.crop_class,\n"
            "        'confianza': round(obs.confidence, 3),\n"
            "        'vigor': obs.vigor,\n"
            "        'latencia_ms': _ms,\n"
            "    })\n"
            "obs_df = pl.DataFrame(_rows)\n"
            "display(obs_df)"
        )
    )
    cells.append(
        code(
            "# The actual grounding block the reasoner reads for the first parcel:\n"
            "# this is the perceiver/reasoner contract -- plain TEXT, no logits.\n"
            "first_obs, _ = observations[0]\n"
            "display(Markdown(\n"
            "    f'**Bloque de anclaje (`to_prompt_block`) para la parcela {first_obs.parcel_id}:**'\n"
            "))\n"
            "print(first_obs.to_prompt_block())\n"
            "display(Markdown('\\n**Descripcion en lenguaje natural:**\\n\\n> ' + first_obs.description))"
        )
    )
    cells.append(
        md(
            "**Lectura**: cada bloque resume lo que el modelo *ve* en una parcela como frases "
            "legibles -- cultivo estimado y confianza, fenologia (inicio de verdor, pico, "
            "senescencia), vigor y las clases mas probables. El reasoner toma este texto como "
            "contexto y nunca toca el embedding ni la imagen. Asi se cumple el contrato Be My Eyes: "
            "el perceiver es los ojos, el LLM es el razonamiento."
        )
    )

    # ====================================== Seccion 3 - El agente end-to-end ===
    cells.append(
        md(
            "## 3. El agente conversacional, de principio a fin\n\n"
            "Ahora juntamos las piezas: construimos el agente con el reasoner elegido y le hacemos "
            "preguntas reales. El agente decide que herramientas llamar, las ejecuta sobre la base "
            "de datos de la sesion y redacta la respuesta. Mostramos el **flujo de eventos** que "
            "emite el bucle de llamada a funciones:\n\n"
            "1. `tool_call` - el reasoner decide llamar una herramienta (con sus argumentos).\n"
            "2. `tool_result` - la herramienta devuelve su resultado validado.\n"
            "3. `text_delta` - fragmentos de la respuesta final en lenguaje natural.\n"
            "4. `done` - fin del turno.\n\n"
            "Un ayudante recorre `agent.stream_response`, acumula los eventos y los renderiza de "
            "forma legible: cada llamada con sus argumentos, un resumen del resultado y, al final, "
            "la respuesta del reasoner en markdown."
        )
    )
    cells.append(
        code(
            "# Helper: drive one turn of the agent and render the event flow nicely.\n"
            "import json\n\n"
            "from ml.agent.agent import create_agent\n"
            "from ml.agent.events import (\n"
            "    DoneEvent,\n"
            "    ErrorEvent,\n"
            "    PerceiverObservationEvent,\n"
            "    TextDeltaEvent,\n"
            "    ToolCallEvent,\n"
            "    ToolResultEvent,\n"
            ")\n\n"
            "agent = create_agent(model=model, settings=settings)\n"
            "print('agente listo | backend:', type(agent.backend).__name__,\n"
            "      '| modelo:', getattr(agent.backend, 'model', None),\n"
            "      '| herramientas:', [t.name for t in agent.tools])\n\n\n"
            "def _summarize_result(result: dict, *, limit: int = 280) -> str:\n"
            '    """Compact one tool result dict into a short, readable string."""\n'
            "    text = json.dumps(result, ensure_ascii=False, default=str)\n"
            "    return text if len(text) <= limit else text[:limit] + ' ...'\n\n\n"
            "async def run_query(question: str) -> str:\n"
            '    """Stream one user turn through the agent and render its event flow.\n\n'
            "    Renders every tool_call / tool_result and accumulates the final answer\n"
            "    text, returning it. Errors are surfaced inline (never raised).\n"
            '    """\n'
            "    display(Markdown(f'### Pregunta\\n\\n> {question}'))\n"
            "    answer_parts: list[str] = []\n"
            "    n_tool_calls = 0\n"
            "    t0 = time.perf_counter()\n"
            "    async for ev in agent.stream_response(\n"
            "        messages=[{'role': 'user', 'content': question}],\n"
            "        session_id=session_id,\n"
            "        ctx=ctx,\n"
            "    ):\n"
            "        if isinstance(ev, ToolCallEvent):\n"
            "            n_tool_calls += 1\n"
            "            display(Markdown(\n"
            "                f'**herramienta** `{ev.name}`  \\n'\n"
            "                f'argumentos: `{json.dumps(ev.arguments, ensure_ascii=False, default=str)}`'\n"
            "            ))\n"
            "        elif isinstance(ev, ToolResultEvent):\n"
            "            _flag = 'ok' if ev.ok else 'ERROR'\n"
            "            display(Markdown(\n"
            "                f'**resultado** ({_flag}) de `{ev.name}`: '\n"
            "                f'`{_summarize_result(ev.result)}`'\n"
            "            ))\n"
            "        elif isinstance(ev, PerceiverObservationEvent):\n"
            "            display(Markdown('**observacion del perceiver inyectada al reasoner.**'))\n"
            "        elif isinstance(ev, TextDeltaEvent):\n"
            "            answer_parts.append(ev.text)\n"
            "        elif isinstance(ev, ErrorEvent):\n"
            "            display(Markdown(f'**error del agente**: {ev.message}'))\n"
            "        elif isinstance(ev, DoneEvent):\n"
            "            pass\n"
            "    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)\n"
            "    answer = ''.join(answer_parts).strip()\n"
            "    display(Markdown(\n"
            "        f'### Respuesta del reasoner\\n\\n{answer or \"_(sin texto)_\"}\\n\\n'\n"
            "        f'_herramientas usadas: {n_tool_calls} | latencia del turno: {elapsed_ms} ms_'\n"
            "    ))\n"
            "    return answer"
        )
    )
    cells.append(
        md(
            "### Consulta A - inventario de parcelas\n\n"
            "Pregunta abierta sobre el inventario. Esperamos que el agente llame `list_parcels` "
            "para enumerar las parcelas de la sesion y luego resuma cuantas hay y de que cultivos."
        )
    )
    cells.append(
        code(
            "_answer_a = await run_query(\n"
            "    'Cuantas parcelas tengo y de que cultivos son? Dame un resumen.'\n"
            ")"
        )
    )
    cells.append(
        md(
            "### Consulta B - explicacion de una prediccion\n\n"
            "Pregunta concreta sobre una parcela. Esperamos que el agente llame "
            "`explain_prediction` (la puerta de entrada del patron Be My Eyes) y traduzca la "
            "fenologia y el vigor a una explicacion en lenguaje natural."
        )
    )
    cells.append(
        code(
            "_pid_demo = parcel_ids[0]\n"
            "_answer_b = await run_query(\n"
            "    f'Explica la prediccion de la parcela {_pid_demo}: que cultivo es, '\n"
            "    'con que confianza y que dice su fenologia.'\n"
            ")"
        )
    )
    cells.append(
        md(
            "### Consulta C - serie temporal de un indice\n\n"
            "Pregunta que requiere datos temporales. Esperamos que el agente llame "
            "`get_parcel_timeseries` para recuperar la evolucion del NDVI de la parcela y la "
            "interprete (cuando verdea, cuando alcanza el pico)."
        )
    )
    cells.append(
        code(
            "_answer_c = await run_query(\n"
            "    f'Como evoluciono el NDVI de la parcela {_pid_demo} durante 2019? '\n"
            "    'Resume su comportamiento estacional.'\n"
            ")"
        )
    )
    cells.append(
        md(
            "**Lectura**: en cada turno el agente **primero actua** (una o mas llamadas a "
            "herramientas sobre la base real de la sesion) y **luego responde**. Toda cifra de la "
            "respuesta tiene origen en un `tool_result` visible arriba: no hay numeros inventados. "
            "Ese es el valor de auditar el flujo de eventos."
        )
    )

    # ====================================== Seccion 4 - Spatial-RAG lite ===
    cells.append(
        md(
            "## 4. Spatial-RAG *lite*: anclaje en parcelas vecinas reales\n\n"
            "Para reducir alucinaciones, el agente puede anclarse en un corpus de documentos "
            "reales (descripciones fenologicas de parcelas PASTIS-R) cercanos al area consultada. "
            "La capa *lite* combina dos senales **en serie**:\n\n"
            "1. un prefiltro **espacial** (`ST_DWithin` sobre geografia) que reduce el corpus a las "
            "parcelas vecinas, y\n"
            "2. una busqueda **semantica** (coseno con pgvector sobre el embedding AlphaEarth de "
            "64 dimensiones) sobre ese conjunto reducido,\n\n"
            "fusionadas con un peso configurable. Primero miramos el corpus; luego ejecutamos una "
            "recuperacion real cerca de una parcela de la demo."
        )
    )
    cells.append(
        code(
            "# A glimpse of the real RAG corpus: count and a couple of example contents.\n"
            "async with pool.acquire() as conn:\n"
            "    rag_total = await conn.fetchval('SELECT count(*) FROM rag_documents')\n"
            "    rag_with_geom = await conn.fetchval(\n"
            "        'SELECT count(*) FROM rag_documents WHERE geom IS NOT NULL'\n"
            "    )\n"
            "    sample_docs = await conn.fetch(\n"
            "        'SELECT id, source, parcel_id, content '\n"
            "        'FROM rag_documents ORDER BY id LIMIT 3'\n"
            "    )\n"
            "display(Markdown(\n"
            "    f'Corpus RAG: **{rag_total}** documentos '\n"
            "    f'({rag_with_geom} con geometria para el prefiltro espacial).'\n"
            "))\n"
            "for _d in sample_docs:\n"
            "    display(Markdown(\n"
            "        f'- `[{_d[\"source\"]}:{_d[\"parcel_id\"]}]` {_d[\"content\"]}'\n"
            "    ))"
        )
    )
    cells.append(
        code(
            "# Build a small AOI around a real demo parcel centroid, then run the lite pipeline.\n"
            "from ml.agent.rag import spatial_rag\n\n"
            "async with pool.acquire() as conn:\n"
            "    centroid = await conn.fetchrow(\n"
            "        'SELECT ST_X(ST_Centroid(geom)) AS lon, ST_Y(ST_Centroid(geom)) AS lat '\n"
            "        'FROM parcels ORDER BY id LIMIT 1'\n"
            "    )\n"
            "lon, lat = float(centroid['lon']), float(centroid['lat'])\n"
            "# A tiny square AOI (~degrees) centred on the parcel; the geodesic ST_DWithin radius\n"
            "# (rag_radius_m) is what actually bounds the spatial candidate set.\n"
            "_d = 0.01\n"
            "aoi = {\n"
            "    'type': 'Polygon',\n"
            "    'coordinates': [[\n"
            "        [lon - _d, lat - _d],\n"
            "        [lon + _d, lat - _d],\n"
            "        [lon + _d, lat + _d],\n"
            "        [lon - _d, lat + _d],\n"
            "        [lon - _d, lat - _d],\n"
            "    ]],\n"
            "}\n"
            "print(f'AOI centrada en lon={lon:.5f}, lat={lat:.5f} | radio={rag_radius_m:.0f} m')\n\n"
            "retrieved = await spatial_rag(\n"
            "    ctx,\n"
            "    query='Que cultivos y fenologia hay en las parcelas vecinas a esta area?',\n"
            "    aoi=aoi,\n"
            "    top_k=int(rag_top_k),\n"
            "    radius_m=float(rag_radius_m),\n"
            ")\n"
            "print('documentos recuperados:', len(retrieved))"
        )
    )
    cells.append(
        code(
            "# Show the retrieved neighbours with their fused score and distance.\n"
            "if retrieved:\n"
            "    _rows = [{\n"
            "        'doc_id': d.id,\n"
            "        'fuente': d.source,\n"
            "        'parcela': d.parcel_id,\n"
            "        'distancia_m': round(d.distance_m, 1) if d.distance_m is not None else None,\n"
            "        'score': round(d.score, 4),\n"
            "        'contenido': d.content[:90] + ('...' if len(d.content) > 90 else ''),\n"
            "    } for d in retrieved]\n"
            "    with pl.Config(fmt_str_lengths=120, tbl_width_chars=200):\n"
            "        display(pl.DataFrame(_rows))\n"
            "else:\n"
            "    display(Markdown(\n"
            "        '> No se hallaron vecinos dentro del radio. Aumenta `rag_radius_m` y reejecuta.'\n"
            "    ))"
        )
    )
    cells.append(
        md(
            "**Lectura**: con `rag_enabled=true`, el reasoner recibe estos documentos como bloque "
            "de contexto citado (`[fuente:parcela] ...`). Son **parcelas vecinas reales**, no "
            "ejemplos genericos: el modelo se aterriza en evidencia local y puede citar de donde "
            "sale cada afirmacion. Esa es la palanca anti-alucinacion del patron Spatial-RAG. Con "
            "la bandera apagada (por defecto), la herramienta diferida `retrieve_context` ni "
            "siquiera toca la base: el agente razona sin anclaje, exactamente como antes."
        )
    )

    # ====================================== Seccion 5 - Variante Qwen on-prem ===
    cells.append(
        md(
            "## 5. La variante on-prem con Qwen (soberania de datos)\n\n"
            "El mismo agente puede razonar con un LLM **on-prem** en vez de Gemini en la nube. La "
            "abstraccion de *backend* lo hace transparente: cambiar `model='qwen35'` hace que "
            "`make_backend` devuelva un `VLLMOpenAIBackend` que apunta a un servidor "
            "OpenAI-compatible (Qwen3.5-35B-A3B servido con vLLM en la H100, puerto `:8002`). El "
            "bucle del agente, las herramientas y el perceiver **no cambian**.\n\n"
            "Esto importa para clientes con **requisitos de soberania de datos**: el razonamiento "
            "ocurre dentro de su perimetro, sin enviar datos a una API externa.\n\n"
            "Aqui **no levantamos el serving** (puede no estar arriba), pero si comprobamos -- de "
            "forma puramente local -- que la seleccion de backend funciona y apunta al endpoint "
            "correcto."
        )
    )
    cells.append(
        code(
            "# Backend swap is a local, network-free operation: verify the selection only.\n"
            "from ml.agent.backends import GeminiBackend, VLLMOpenAIBackend, make_backend\n\n"
            "cloud_backend = make_backend('gemini-2.5-flash', settings)\n"
            "onprem_backend = make_backend('qwen35', settings)\n\n"
            "assert isinstance(cloud_backend, GeminiBackend)\n"
            "assert isinstance(onprem_backend, VLLMOpenAIBackend)\n\n"
            "display(Markdown(\n"
            "    '| variante | backend | endpoint / modelo |\\n'\n"
            "    '|----------|---------|-------------------|\\n'\n"
            "    f'| nube | `{type(cloud_backend).__name__}` | `{cloud_backend.model}` (Vertex AI / GenAI) |\\n'\n"
            "    f'| on-prem | `{type(onprem_backend).__name__}` | `{onprem_backend._base_url}` '\n"
            "    f'(modelo `{onprem_backend.model}`) |'\n"
            "))\n"
            "print('Backend on-prem seleccionado sin llamadas de red:',\n"
            "      type(onprem_backend).__name__, '->', onprem_backend._base_url)"
        )
    )
    cells.append(
        md(
            "**Lectura**: la unica diferencia entre la version nube y la on-prem es el nombre del "
            "modelo que se pasa a la fabrica. Construir el agente con `create_agent(model='qwen35')` "
            "produciria exactamente el mismo flujo de esta demo, pero razonando dentro del "
            "perimetro del cliente. No lo ejecutamos aqui porque depende de que el servidor vLLM "
            "este levantado en la H100."
        )
    )

    # ----------------------------------------------------------- Conclusiones ---
    cells.append(
        md(
            "## Conclusiones\n\n"
            "**Que se demostro**\n\n"
            "- Un **copiloto conversacional completo** que responde preguntas sobre parcelas "
            "agricolas reales, hablando con un LLM que **razona** pero no clasifica pixeles.\n"
            "- La **separacion Be My Eyes**: los modelos del equipo miran cada parcela y emiten una "
            "observacion en texto (cultivo, fenologia, vigor, confianza); el LLM lee ese texto, "
            "llama herramientas y redacta la respuesta.\n"
            "- Un **conjunto cerrado de diez herramientas geoespaciales** con esquemas validados, "
            "de las cuales el agente de la demo expone cinco sincronas.\n"
            "- Un flujo de **eventos auditable**: cada cifra de cada respuesta proviene de una "
            "herramienta visible en el flujo, no de la imaginacion del modelo.\n"
            "- Un **RAG espacial** que ancla al modelo en parcelas vecinas reales, recuperadas "
            "combinando cercania geografica y similitud del embedding satelital -- la palanca "
            "anti-alucinacion del sistema.\n"
            "- Una **variante on-prem** que corre el mismo agente con un modelo dentro del "
            "perimetro del cliente, cambiando una sola linea.\n\n"
            "**Numeros de la demo**\n\n"
            "- 12 parcelas reales con su embedding satelital y su fenologia.\n"
            "- 300 documentos en el corpus de recuperacion, con vector de 64 dimensiones.\n"
            "- 18 cultivos posibles en el clasificador que alimenta al perceiver.\n"
            "- Una respuesta del agente combina, tipicamente, una o dos llamadas a herramientas "
            "antes de redactar; las latencias por turno quedan impresas arriba.\n\n"
            "**Lo que sigue**\n\n"
            "- Activar el RAG y las herramientas diferidas en el bucle del agente mediante el "
            "ejecutor en segundo plano, para que el reasoner pueda pedir contexto vecino por su "
            "cuenta.\n"
            "- Conectar el frontend de mapa para dibujar areas y disparar estas mismas consultas "
            "desde la interfaz.\n"
            "- Levantar el serving on-prem y comparar, sobre las mismas preguntas, la calidad de "
            "las respuestas de la nube frente a las del modelo local."
        )
    )

    # ---------------------------------------------------- Cierre del pool ------
    cells.append(
        md(
            "### Cierre\n\n"
            "Cerramos el *pool* de conexiones de forma ordenada al terminar la demostracion."
        )
    )
    cells.append(
        code(
            "# Close the shared asyncpg pool cleanly at the end of the demo.\n"
            "from ml.agent.db import close_pool\n\n"
            "await close_pool()\n"
            "print('pool cerrado.')"
        )
    )

    return cells


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Ruta del notebook de salida.")] = _DEFAULT_OUT,
) -> None:
    """Generate the Avance 6 conversational-copilot demo notebook.

    Args:
        out: Destination path of the ``.ipynb``.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = _build_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    typer.echo(f"Notebook escrito: {out} ({len(nb['cells'])} celdas)")


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    app()
