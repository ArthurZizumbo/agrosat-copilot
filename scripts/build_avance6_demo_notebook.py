"""Builder of the Avance 6 conversational-copilot demo notebook (Equipo 17).

Generates ``notebooks/final_model/Avance6.Demo.Copiloto.Equipo17.ipynb``
programmatically and reproducibly (same idempotent ``nbformat`` + ``typer``
pattern as the sibling ``scripts/build_*_notebook.py``). Unlike the Avance 5
integrator, this notebook is a **live end-to-end demonstration** of the
conversational agent: it imports the real ``ml.agent`` stack and runs it against
the seeded demo session in the local Postgres + PostGIS + pgvector instance. No
placeholders, no fabricated outputs -- it is committed UNEXECUTED and is meant to
be run with papermill against the real database and LLM credentials.

Scope: this is the **general** copilot demo over the seeded PASTIS parcels. The
Mediterranean transfer-learning story (US-079, original-vs-TL) lives in
``notebooks/transfer/`` (``us079_copilot_original_vs_tl`` for the copilot view and
``us079_transfer_italia_eval`` for the dense analysis), so it is deliberately NOT
duplicated here -- this notebook stays focused on the mechanism.

What the notebook shows (the "Be My Eyes" pattern):

1. Cover and the Be My Eyes framing (perceiver = the team's trained models emit
   TEXT, reasoner = a frontier/on-prem LLM reasons over that text, never pixels).
2. The geospatial tools (US-045): a table from ``demo.tool_inventory()``.
3. The perceiver (US-046): ``PerceiverLayer.observe`` on real parcels, surfacing
   the structured TEXT observation and its ``to_prompt_block()``.
4. The conversational agent end to end (US-047): three real queries that exercise
   the synchronous parcel tools, plus the AOI/deferred tools shown directly
   (``classify_new_parcel`` / ``get_aoi_stats`` / ``compare_models`` / ``add_aoi``).
5. Spatial-RAG lite (US-046): the corpus, a real retrieval, AND the reasoner
   answering over the retrieved neighbours (the anti-hallucination grounding IN USE).
6. The three reasoner backends (US-048/052): Gemini 3.5 Flash (cloud), Qwen3.6-VL
   (on-prem multimodal) and Qwen3.5-35B (on-prem text), with an honest availability
   probe and the SAME grounded question put to each -- which reasoner reasons best
   over the identical perceiver TEXT.
7. Accessible conclusions with the real numbers.

All the reusable driver/renderer logic lives in :mod:`ml.agent.demo` (unit-tested
in ``tests/ml/agent/test_demo.py``), so every cell here is a short call -- the
notebook is the narrative, not the implementation (notebooks/CLAUDE.md).

Visible prose (markdown, captions, prints) is Spanish with proper accents and the
letter "n" with tilde; code, identifiers, comments and docstrings stay English
ASCII (project convention). No emojis. The async cells use top-level ``await``
(modern ipykernel supports it), exactly as the agent's public API requires.

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
            '### Patron "Be My Eyes": los modelos del equipo *ven*, un LLM frontera *razona*\n\n'
            "**Equipo 17** - AgroSatCopilot\n\n"
            "---\n\n"
            "Este cuaderno demuestra, de principio a fin y **con datos reales**, el copiloto "
            "conversacional para analisis satelital agricola. La idea central es el patron "
            "**Be My Eyes**:\n\n"
            "- El **perceiver** son los modelos entrenados por el equipo (el ensamble campeon "
            "**Voting-3**: tsvit-pheno + utae + xgb-alphaearth, y el descriptor fenologico). No "
            "hablan con el usuario: "
            "**miran una parcela y emiten una observacion en TEXTO** (cultivo, fenologia, vigor, "
            "confianza).\n"
            "- El **reasoner** es un LLM frontera (Gemini en la nube) u on-prem (Qwen). **No "
            "clasifica pixeles**: lee ese texto, llama herramientas geoespaciales cuando hace falta "
            "y redacta la respuesta en lenguaje natural.\n\n"
            "Esta separacion es lo que hace al sistema **auditable y anti-alucinacion**: toda cifra "
            "que el reasoner enuncia proviene de una herramienta o de una observacion del perceiver, "
            "nunca de la imaginacion del modelo.\n\n"
            "> Esta es la demo **general** del copiloto sobre las parcelas PASTIS sembradas. El "
            "transfer learning mediterraneo (US-079, original vs Italia) se trata aparte, en "
            "`notebooks/transfer/` (`us079_copilot_original_vs_tl` y `us079_transfer_italia_eval`), "
            "para que este cuaderno se mantenga centrado en el mecanismo.\n\n"
            "> Corre contra la sesion de demostracion ya sembrada en la base local (Postgres + "
            "PostGIS + pgvector): parcelas reales de **PASTIS-R** (fold-5 retenido) que el campeon "
            "**Voting-3** puntua desde su OOF real, y un corpus de descripciones fenologicas "
            "(FarSLIP) con vector para el RAG."
        )
    )

    # --------------------------------------------- parameters (papermill) ---
    cells.append(
        code(
            "# Parameters cell (papermill). Defaults are the seeded demo values; override\n"
            "# any of them at run time with `papermill -p <name> <value>`.\n"
            "model = 'gemini-3.5-flash'        # default cloud reasoner (live if GEMINI_API_KEY set)\n"
            "demo_user = 'demo@agrosat.dev'    # seeded demo session owner\n"
            "n_perceiver_parcels = 3           # how many parcels to run through the perceiver\n"
            "classify_year = 2019              # AlphaEarth annual campaign for classify / aoi_stats\n"
            "classify_model = 'voting3'        # EPIC 12 deployment champion served by classify\n"
            "rag_radius_m = 20000.0            # ST_DWithin radius for the Spatial-RAG demo (m)\n"
            "rag_top_k = 5                     # documents retrieved per RAG query\n"
            "# The three reasoner backends contrasted in section 5 (make_backend resolves each).\n"
            "backend_models = ['gemini-3.5-flash', 'qwen3.6-vl', 'qwen35']"
        )
    )
    cells[-1]["metadata"]["tags"] = ["parameters"]

    # ------------------------------------------------------------------ Setup ---
    cells.append(
        md(
            "## Preparacion del entorno\n\n"
            "Resolvemos la raiz del repositorio (sin rutas absolutas), cargamos `.env.local` para "
            "tomar la cadena de conexion y las credenciales del LLM, y **silenciamos el ruido de "
            "logs** del agente (structlog emite una linea INFO por paso; en una demo en vivo eso "
            "tapa la respuesta). La consola de Windows usa cp1252; forzamos UTF-8 en la salida para "
            "que los acentos no rompan la ejecucion."
        )
    )
    cells.append(
        code(
            "# --- Repo bootstrap, UTF-8 safety, env, autoreload, quiet logging ---\n"
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
            "import polars as pl\n"
            "from IPython.display import Markdown, display\n\n"
            "# Reusable copilot-demo driver + renderers (unit-tested in tests/ml/agent/test_demo).\n"
            "# Keeping them in ml/agent/demo is what lets every cell below be a single short call.\n"
            "from ml.agent import demo\n\n"
            "# Silence the agent/perceiver/tool INFO-DEBUG structlog stream (warnings still show).\n"
            "demo.quiet_logging()\n\n"
            "print('repo:', REPO_ROOT)\n"
            "print('reasoner por defecto:', model, '| usuario demo:', demo_user)"
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
            "    # Pick the demo user's session that actually owns parcels (deterministic even\n"
            "    # if several demo sessions exist), so the perceiver/tools have data to work on.\n"
            "    session_id = await conn.fetchval(\n"
            "        'SELECT cs.id FROM chat_sessions cs '\n"
            "        'LEFT JOIN parcels p ON p.session_id = cs.id '\n"
            "        'WHERE cs.user_id = $1 '\n"
            "        'GROUP BY cs.id ORDER BY count(p.id) DESC, cs.id LIMIT 1',\n"
            "        demo_user,\n"
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
            "Cinco son **sincronas** (se ejecutan en linea dentro del bucle del agente: listar "
            "parcelas, serie temporal, estadisticas de un area, clasificar y explicar). Las otras "
            "cinco son **diferidas** (*deferred*): se completan fuera de linea via un *worker* "
            "(buscar escenas, teselas de mapa, guardar un area, comparar modelos y recuperar "
            "contexto del RAG). La tabla se construye desde `build_function_declarations()` -- la "
            "misma fuente de verdad que se le anuncia al LLM."
        )
    )
    cells.append(
        code(
            "# The tool table, straight from the declarations advertised to the LLM. The whole\n"
            "# rendering lives in ml/agent/demo so this cell is one line.\n"
            "_tools = demo.tool_inventory()"
        )
    )

    # ============================================ Seccion 2 - El perceiver ===
    cells.append(
        md(
            "## 2. El perceiver: los modelos del equipo emiten TEXTO\n\n"
            "El perceiver **mira** una parcela a traves de los modelos entrenados y produce una "
            "**observacion en texto plano**, nunca tensores ni probabilidades crudas hacia el "
            "reasoner. Reune el **posterior del campeon Voting-3** -- el voto ponderado de "
            "`tsvit-pheno` + `utae` + `xgb-alphaearth` (ganador de despliegue, F1-macro 0.9069 sobre "
            "`france-10`) -- resuelto sobre el fold-5 real de PASTIS via `canonical_parcel_id`, mas "
            "la **fenologia / vigor / descripcion** del descriptor (Wen et al., 2025) cuando la "
            "parcela tiene metricas fenologicas.\n\n"
            "El metodo `to_prompt_block()` rinde esa observacion como el bloque de **anclaje** que "
            "se inyecta en el prompt del reasoner. *Ese texto* es lo que el LLM consume; la imagen "
            "y los logits nunca cruzan la frontera."
        )
    )
    cells.append(
        code(
            "# Run the perceiver over the first N seeded parcels and tabulate the TEXT fields it\n"
            "# exposes (no tensors). observe_parcels + perceiver_table live in ml/agent/demo.\n"
            "from ml.agent.perceiver import PerceiverLayer\n\n"
            "async with pool.acquire() as conn:\n"
            "    # Session-scoped (multi-tenant): only this session's parcels.\n"
            "    _rows = await conn.fetch(\n"
            "        'SELECT id FROM parcels WHERE session_id = $1 ORDER BY id LIMIT $2',\n"
            "        session_id, int(n_perceiver_parcels),\n"
            "    )\n"
            "parcel_ids = [int(r['id']) for r in _rows]\n\n"
            "perceiver = PerceiverLayer(ctx)\n"
            "observations = await demo.observe_parcels(perceiver, parcel_ids)\n"
            "demo.perceiver_table(observations)"
        )
    )
    cells.append(
        code(
            "# The actual grounding block the reasoner reads for the first parcel: plain TEXT,\n"
            "# no logits -- the perceiver/reasoner contract.\n"
            "first_obs = observations[0][0]\n"
            "display(Markdown(\n"
            "    f'**Bloque de anclaje (`to_prompt_block`) -- parcela {first_obs.parcel_id}:**'\n"
            "))\n"
            "print(first_obs.to_prompt_block())\n"
            "display(Markdown('\\n**Descripcion en lenguaje natural:**\\n\\n> ' + first_obs.description))"
        )
    )
    cells.append(
        md(
            "**Lectura**: cada bloque resume lo que el modelo *ve* como frases legibles -- cultivo "
            "estimado y confianza, fenologia (inicio de verdor, pico, senescencia), vigor y las "
            "clases mas probables. El reasoner toma este texto como contexto y nunca toca el "
            "embedding ni la imagen. Asi se cumple el contrato Be My Eyes: el perceiver es los ojos, "
            "el LLM es el razonamiento."
        )
    )

    # ====================================== Seccion 3 - El agente end-to-end ===
    cells.append(
        md(
            "## 3. El agente conversacional, de principio a fin\n\n"
            "Construimos el agente con el reasoner por defecto y le hacemos **preguntas reales**. "
            "El agente decide que herramientas llamar, las ejecuta sobre la base de la sesion y "
            "redacta la respuesta. El ayudante `demo.run_agent_turn` recorre `stream_response` y "
            "renderiza el flujo `tool_call -> tool_result -> respuesta`, de modo que cada cifra de "
            "la respuesta tiene un origen visible.\n\n"
            "Estas tres consultas ejercitan las herramientas **sincronas** que dependen de una "
            "parcela: `list_parcels`, `explain_prediction` y `get_parcel_timeseries`."
        )
    )
    cells.append(
        code(
            "# Build the agent on the default reasoner and put three real questions to it. The\n"
            "# driver (ml/agent/demo) renders the tool_call -> tool_result -> answer flow.\n"
            "from ml.agent.agent import create_agent\n\n"
            "agent = create_agent(model=model, settings=settings)\n"
            "print('agente listo | backend:', type(agent.backend).__name__,\n"
            "      '| modelo:', getattr(agent.backend, 'model', None),\n"
            "      '| herramientas:', [t.name for t in agent.tools])\n\n"
            "_pid = parcel_ids[0]\n"
            "_live_queries = [\n"
            "    ('inventario -> list_parcels',\n"
            "     'Cuantas parcelas tengo y de que cultivos son? Dame un resumen.'),\n"
            "    ('explicacion -> explain_prediction',\n"
            "     f'Explica la prediccion de la parcela {_pid}: que cultivo es, con que confianza '\n"
            "     'y que dice su fenologia.'),\n"
            "    ('serie temporal -> get_parcel_timeseries',\n"
            "     f'Como evoluciono el NDVI de la parcela {_pid} durante 2019? Resume su '\n"
            "     'comportamiento estacional.'),\n"
            "]\n"
            "for _label, _q in _live_queries:\n"
            "    await demo.run_agent_turn(\n"
            "        agent, _q, ctx=ctx, session_id=session_id, title=f'### {_label}'\n"
            "    )"
        )
    )
    cells.append(
        md(
            "### Las herramientas de area y diferidas, en crudo\n\n"
            "El bucle conversacional de Gemini no puede invocar las herramientas **diferidas** "
            "(su `behavior=NON_BLOCKING` lo rechaza la API estandar de generacion). Las "
            "demostramos **directamente** -- el mismo `tool_result` que consumiria el agente -- "
            "sobre un AOI pequeno alrededor de la primera parcela: `classify_new_parcel`, "
            "`get_aoi_stats`, `compare_models` y `add_aoi`. El ayudante `demo.run_tool` inyecta el "
            "`session_id` y **degrada de forma honesta**.\n\n"
            "`compare_models` contrasta los **tres miembros del Voting-3 + FarSLIP** (cada uno con "
            "su OOF fold-5 real), resuelto por el `canonical_parcel_id` de la parcela: cuando los "
            "miembros coinciden el acuerdo es 1.0, y cuando discrepan se ve el valor real del "
            "ensamble. `classify_new_parcel` sobre un AOI nuevo devuelve `needs_gee_sampling` (la "
            "parcela aun no tiene embedding AlphaEarth materializado: el agente dispararia el "
            "muestreo GEE) -- es la ruta honesta para un poligono recien dibujado."
        )
    )
    cells.append(
        code(
            "# Demonstrate the AOI-based + deferred tools DIRECTLY (run_tool injects the session id\n"
            "# and degrades honestly). A tiny AOI around the first parcel centroid.\n"
            "async with pool.acquire() as conn:\n"
            "    _c = await conn.fetchrow(\n"
            "        'SELECT ST_X(ST_Centroid(geom)) AS lon, ST_Y(ST_Centroid(geom)) AS lat '\n"
            "        'FROM parcels WHERE id = $1', _pid\n"
            "    )\n"
            "_lon, _lat = float(_c['lon']), float(_c['lat'])\n"
            "_d = 0.002\n"
            "_aoi = {'type': 'Polygon', 'coordinates': [[\n"
            "    [_lon - _d, _lat - _d], [_lon + _d, _lat - _d], [_lon + _d, _lat + _d],\n"
            "    [_lon - _d, _lat + _d], [_lon - _d, _lat - _d]]]}\n\n"
            "await demo.run_tool('classify_new_parcel',\n"
            "                    {'aoi': _aoi, 'year': classify_year, 'model': classify_model}, ctx)\n"
            "await demo.run_tool('get_aoi_stats', {'aoi': _aoi, 'year': classify_year}, ctx)\n"
            "# compare_models contrasts the three Voting-3 members + FarSLIP (its own real\n"
            "# fold-5 OOF) for the parcel, resolved by its stored canonical PASTIS-R id.\n"
            "await demo.run_tool('compare_models',\n"
            "                    {'parcel_id': _pid,\n"
            "                     'models': ['tsvit-pheno', 'utae', 'xgb-alphaearth',\n"
            "                                'farslip-ft18']}, ctx)\n"
            "await demo.run_tool('add_aoi', {'aoi': _aoi, 'name': f'Demo AOI parcela {_pid}'}, ctx)"
        )
    )
    cells.append(
        md(
            "**Lectura**: en cada turno el agente **primero actua** (una o mas llamadas a "
            "herramientas sobre la base real) y **luego responde**. Las herramientas de area y "
            "diferidas devuelven el mismo texto estructurado que el reasoner consume. Las dos "
            "restantes (`search_stac` y `get_tiles`) dependen de servicios externos (catalogo STAC "
            "/ CDSE y TiTiler) y corren via el *worker*; no se invocan aqui para no exigir esos "
            "servicios en la demo. Toda cifra tiene origen en un `tool_result` visible: no hay "
            "numeros inventados."
        )
    )

    # ====================================== Seccion 4 - Spatial-RAG lite ===
    cells.append(
        md(
            "## 4. Spatial-RAG *lite*: anclaje en parcelas vecinas reales\n\n"
            "Para reducir alucinaciones, el agente puede anclarse en un corpus de documentos reales "
            "(descripciones fenologicas de parcelas PASTIS-R) cercanos al area consultada. La capa "
            "*lite* combina **en serie** un prefiltro **espacial** (`ST_DWithin` sobre geografia) y "
            "una busqueda **semantica** (coseno con pgvector sobre el embedding AlphaEarth de 64 "
            "dimensiones), fusionados con un peso configurable. Primero recuperamos; luego el "
            "**reasoner razona sobre esos vecinos** (el RAG en uso)."
        )
    )
    cells.append(
        code(
            "# Real RAG corpus glimpse + a real retrieval near the first parcel. spatial_rag runs\n"
            "# the lite pipeline; rag_table (ml/agent/demo) renders the fused score + distance.\n"
            "from ml.agent.rag import spatial_rag\n\n"
            "async with pool.acquire() as conn:\n"
            "    _rag_total = await conn.fetchval('SELECT count(*) FROM rag_documents')\n"
            "    _rag_geom = await conn.fetchval(\n"
            "        'SELECT count(*) FROM rag_documents WHERE geom IS NOT NULL'\n"
            "    )\n"
            "display(Markdown(\n"
            "    f'Corpus RAG: **{_rag_total}** documentos ({_rag_geom} con geometria para el '\n"
            "    'prefiltro espacial).'\n"
            "))\n\n"
            "# The session's parcels are real PASTIS-R (same region as the corpus), so the AOI\n"
            "# around the first parcel centroid has real neighbours within the radius.\n"
            "aoi_rag = {'type': 'Polygon', 'coordinates': [[\n"
            "    [_lon - 0.05, _lat - 0.05], [_lon + 0.05, _lat - 0.05], [_lon + 0.05, _lat + 0.05],\n"
            "    [_lon - 0.05, _lat + 0.05], [_lon - 0.05, _lat - 0.05]]]}\n"
            "retrieved = await spatial_rag(\n"
            "    ctx,\n"
            "    query='Que cultivos y fenologia hay en las parcelas vecinas a esta area?',\n"
            "    aoi=aoi_rag, top_k=int(rag_top_k), radius_m=float(rag_radius_m),\n"
            ")\n"
            "demo.rag_table(retrieved)"
        )
    )
    cells.append(
        code(
            "# Spatial-RAG lite IN USE: feed the retrieved neighbours to the reasoner as a cited\n"
            "# grounding block and let it answer over THAT text. Honest: skipped cleanly if the\n"
            "# default reasoner is not reachable here (no fabricated answer).\n"
            "_avail_default, _ = demo.probe_availability([model], settings, display=False)\n"
            "if retrieved and _avail_default.get(model):\n"
            "    _ctx_block = '\\n'.join(f'[{d.source}:{d.parcel_id}] {d.content}' for d in retrieved)\n"
            "    _rag_q = (\n"
            "        'Contexto recuperado por Spatial-RAG (parcelas vecinas reales):\\n'\n"
            "        f'{_ctx_block}\\n\\nCon SOLO ese contexto, resume en dos frases que cultivos y '\n"
            "        'fenologia predominan en las parcelas vecinas y cita [fuente:parcela].'\n"
            "    )\n"
            "    await demo.run_backend_turn(\n"
            "        model, _rag_q, settings=settings, ctx=ctx, session_id=session_id,\n"
            "        availability=_avail_default,\n"
            "        title='### El reasoner razona sobre el contexto del RAG',\n"
            "    )\n"
            "elif not retrieved:\n"
            "    display(Markdown('> Sin vecinos en el radio: nada que anclar. Aumenta `rag_radius_m`.'))\n"
            "else:\n"
            "    display(Markdown(\n"
            "        f'> Reasoner `{model}` no disponible aqui; la celda se ejecuta con credenciales '\n"
            "        '/ endpoint vivos. El bloque de contexto de arriba es lo que recibiria el '\n"
            "        'reasoner.'\n"
            "    ))"
        )
    )
    cells.append(
        md(
            "**Lectura**: el reasoner recibe los documentos como bloque de contexto citado "
            "(`[fuente:parcela] ...`). Son **parcelas vecinas reales**, no ejemplos genericos: el "
            "modelo se aterriza en evidencia local y cita de donde sale cada afirmacion. Esa es la "
            "palanca anti-alucinacion del patron Spatial-RAG."
        )
    )

    # ====================================== Seccion 5 - Comparacion 3 backends ====
    cells.append(
        md(
            "## 5. Tres reasoners, un mismo perceiver: comparacion de backends\n\n"
            "La abstraccion `LLMBackend` / `make_backend` desacopla el bucle del agente del modelo "
            "concreto. Contrastamos **tres reasoners**:\n\n"
            "- **Gemini 3.5 Flash** -- nube (`GeminiBackend`, Vertex AI o la API GenAI).\n"
            "- **Qwen3.6-VL** -- on-prem **multimodal** servido por llama.cpp (`:8003`).\n"
            "- **Qwen3.5-35B** -- on-prem **texto** servido por vLLM (`:8002`). Soberania de datos: "
            "el razonamiento ocurre dentro del perimetro.\n\n"
            "Primero resolvemos cada nombre a su backend (sin red) y **sondeamos honestamente** "
            "cual esta vivo aqui (credenciales de Gemini, o endpoint OpenAI-compatible que "
            "responda). El perceiver es **el mismo** para los tres: la clase y la confianza no "
            "cambian con el LLM, porque el LLM no clasifica.\n\n"
            "> Los dos Qwen on-prem comparten la **misma H100**, asi que se evaluan **uno a la "
            "vez** (no caben los dos en VRAM a la vez). Cada corrida en vivo guarda su registro "
            "real bajo `reports/copilot_backends/` y la tabla final se rearma con todas las "
            "corridas: un backend que aun no se evaluo aparece como **no disponible**, sin inventar."
        )
    )
    cells.append(
        code(
            "# Resolve each reasoner to its backend (network-free) and probe availability HONESTLY.\n"
            "# Both renderings live in ml/agent/demo.\n"
            "demo.backend_overview(backend_models, settings)\n"
            "availability, _ = demo.probe_availability(backend_models, settings)"
        )
    )
    cells.append(
        code(
            "# Put the SAME grounded question (the perceiver's TEXT for the first parcel) to every\n"
            "# available backend. The dense observation is FIXED; only the reasoning over it varies.\n"
            "# The on-prem Qwen text and Qwen-VL share the single H100 GPU, so they are evaluated\n"
            "# ONE AT A TIME: each live pass saves its real record under reports/copilot_backends/\n"
            "# and the table is reassembled from every pass (a backend never run shows as such).\n"
            "_grounded_q = (\n"
            "    first_obs.to_prompt_block()\n"
            "    + '\\n\\nCon esa observacion del perceiver (no inventes cifras), di en una frase '\n"
            "    f'breve que cultivo es la parcela {first_obs.parcel_id}, su confianza y su vigor.'\n"
            ")\n"
            "_records_dir = REPO_ROOT / 'reports' / 'copilot_backends'\n"
            "_this_run = {}\n"
            "for _name in backend_models:\n"
            "    _rec = await demo.run_backend_turn(\n"
            "        _name, _grounded_q, settings=settings, ctx=ctx, session_id=session_id,\n"
            "        availability=availability,\n"
            "    )\n"
            "    _this_run[_name] = _rec\n"
            "    demo.save_backend_record(_rec, _records_dir)   # persists only a successful run\n"
            "# Prefer a persisted real record (possibly from a previous one-at-a-time pass).\n"
            "_persisted = demo.load_persisted_records(backend_models, _records_dir)\n"
            "backend_records = [_persisted.get(_n, _this_run[_n]) for _n in backend_models]\n"
            "demo.cross_backend_table(backend_records)"
        )
    )
    cells.append(
        md(
            "**Lectura**: con el perceiver fijo, lo que cambia entre backends es la **calidad del "
            "razonamiento sobre ese texto**, el uso de herramientas y la **latencia / coste**. "
            "Gemini y Qwen3.5 (texto) pueden ademas llamar herramientas; Qwen3.6-VL razona sobre el "
            "texto inyectado. Un backend no disponible aparece como tal -- sin cifras inventadas. "
            "Construir el agente con cualquiera de los tres produce el mismo flujo de esta demo, "
            "pero razonando dentro (on-prem) o fuera (nube) del perimetro del cliente."
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
            "observacion en texto; el LLM lee ese texto, llama herramientas y redacta la respuesta.\n"
            "- Un **conjunto cerrado de diez herramientas** con esquemas validados: las cinco "
            "sincronas y, en crudo, las de area y diferidas (`classify`, `get_aoi_stats`, "
            "`compare_models`, `add_aoi`).\n"
            "- Un **RAG espacial en uso**: el reasoner se ancla en parcelas vecinas reales, "
            "recuperadas combinando cercania geografica y similitud del embedding satelital, y cita "
            "su origen -- la palanca anti-alucinacion.\n"
            "- **Tres reasoners intercambiables** (Gemini nube, Qwen3.6-VL y Qwen3.5 on-prem) sobre "
            "el **mismo** perceiver, con sonda honesta de disponibilidad y la misma pregunta "
            "anclada para cada uno.\n\n"
            "**Lo que sigue**\n\n"
            "- Activar el RAG y las herramientas diferidas dentro del bucle del agente via el "
            "ejecutor en segundo plano, para que el reasoner pida contexto vecino por su cuenta.\n"
            "- Conectar el frontend de mapa para dibujar areas y disparar estas mismas consultas.\n"
            "- Para el transfer learning Francia -> Italia accedido por el copiloto, ver "
            "`notebooks/transfer/us079_copilot_original_vs_tl` (vista copiloto) y "
            "`us079_transfer_italia_eval` (analisis denso)."
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
