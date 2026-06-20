# Seguridad — Revision OWASP Top 10 (US-064)

> Cierre documental del MVP en su vertiente de seguridad (Presentacion Final
> 27-jun). Documenta el estado **real** de cada control en el codigo: que esta
> implementado (con ruta), que se difiere conscientemente a post-presentacion
> (Full), y que es deuda anotada (con enlace al blocker doc). No se reclama
> ningun control que no exista en el repo.

Epic E10 (Observabilidad y documentacion). Documento en espanol neutro, sin
codigo de aplicacion: solo describe controles ya presentes y los enlaza a su
evidencia. Las cifras y rutas citadas son verificables en el arbol del repo.

---

## 1. Modelo de amenazas del MVP

AgroSatCopilot es un SaaS conversacional **multi-tenant por `session_id`** para
analisis satelital agricola. La superficie de ataque del MVP:

| Componente | Tecnologia | Exposicion |
|---|---|---|
| API | FastAPI sobre Cloud Run (scale-to-zero) | Publica (HTTPS) |
| Frontend | Nuxt 4 SSR | Publico |
| Agente conversacional | Google ADK + reasoner Gemini cloud / Qwen on-prem | Interno (via service layer) |
| Datos | PostgreSQL 15 + PostGIS + pgvector | Privado (Cloud SQL) |
| Tiling | TiTiler / rio-tiler sobre COG en GCS | Publico de lectura (parametro `url`) |

Activos a proteger: aislamiento entre sesiones/tenants, secretos (claves de
Gemini/Vertex, credenciales de DB), integridad del catalogo geoespacial y la
disponibilidad del endpoint `/chat`. Vectores principales: acceso cruzado entre
sesiones, inyeccion (SQL / XSS via markdown del LLM), SSRF a traves del
parametro `url` del tiler, y abuso de cuota del reasoner.

---

## 2. Tabla OWASP Top 10 (2021)

Estados posibles: **Implementado** (control real, con ruta), **Full /
post-presentacion** (diferido conscientemente para el MVP demo) o **Deuda
anotada** (pendiente, enlazado al blocker doc).

| ID | Riesgo en AgroSatCopilot | Control / estado | Evidencia |
|----|--------------------------|------------------|-----------|
| **A01 Broken Access Control** | Acceso cruzado entre sesiones/tenants | Aislamiento por `session_id` en toda query (regla CLAUDE.md). RLS PostgreSQL por tenant **Deuda anotada** (pendiente US-051): a la fecha `aois` / `parcels` / `features_parcels` / `chat_sessions` sin policy RLS. | Regla multi-tenant CLAUDE.md; US-051; blocker [§ B18](blockers/epic10-notas.md) |
| **A02 Cryptographic Failures** | Trafico en claro | HTTPS. En el demo 27-jun se usa el TLS por defecto de Cloud Run. Cloud Load Balancer + certificado managed = **Full / post-presentacion**. | `infrastructure/` (TF); [ADR-009](decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) |
| **A03 Injection** | SQLi / XSS del markdown del LLM | **Implementado**: acceso a DB via SQLModel/SQLAlchemy parametrizado (sin SQL crudo concatenado). XSS del markdown renderizado del reasoner saneado en el frontend con `isomorphic-dompurify` (US-057). | `backend/app/services/`; US-057 (frontend) |
| **A04 Insecure Design** | Inferencia pesada bloqueando el request | **Implementado**: inferencia sincrona acotada en el MVP por diseno (ADR-012); el path asincrono via Pub/Sub queda para Full. | [ADR-012](decisions/ADR-012-inferencia-sincrona-mvp.md) |
| **A05 Security Misconfiguration** | CORS laxo, vars no declaradas, errores con stack | **Implementado**: CORS endurecido (`allow_headers` whitelisteado, no `*`; `allow_methods` explicito) en `create_app()`; `Settings` con `extra="forbid"` (rechaza vars no declaradas y defaults de dev fuera de `env=dev`); errores TiTiler/rio-tiler mapeados a respuestas HTTP tipadas (sin 500 desnudo). | `backend/app/main.py` (CORS L76-89, `add_exception_handlers` L116); `backend/app/core/config.py` |
| **A06 Vulnerable Components** | Dependencias con CVE | **Implementado / proceso**: deps pinneadas via Poetry (`pyproject.toml`) y `pnpm`; auditoria de CVE documentada en la skill `agrosat-security-audit` (check humano pre-deploy, no automatizado aun). | `pyproject.toml`; skill `agrosat-security-audit` |
| **A07 Identification & Authentication Failures** | Sesiones sin auth fuerte | En el MVP demo se usa `user_id` demo (hardcoded) para no bloquear la presentacion. Clerk OAuth2 + JWT con rotacion / refresh tokens = **Full / post-presentacion**. | [§ 4](#4-autenticacion-mvp-vs-full); skill `agrosat-security` |
| **A08 Software & Data Integrity Failures** | Datos/pesos no trazables | **Implementado**: rasters / COG / pesos versionados con DVC (nunca en Git); runs MLflow con tags `data_version` + `code_version`; migraciones solo rollforward via dbmate. | `*.dvc`; `ml/utils/mlflow_utils.py`; `db/migrations/` |
| **A09 Security Logging & Monitoring Failures** | Falta de trazas / alertas | **Implementado (scaffolding)**: logging estructurado `structlog` (US-065, eventos `chat_stream_*` / `chat_model_resolved` / `chat_turn_metrics`); `/metrics` Prometheus (US-059); `Settings.sentry_dsn` y `Settings.prometheus_pushgateway` declarados. Scrape GPU/alertas reales = **Deuda anotada** (sin Prometheus desplegado). | `backend/app/services/chat_service.py`; `backend/app/api/metrics.py`; blocker [§ B1-B4](blockers/epic10-notas.md) |
| **A10 Server-Side Request Forgery (SSRF)** | El parametro abierto `url` del tiler haria que el servidor consulte cualquier host | **Implementado**: `validate_cog_url()` rechaza todo `url` fuera del allowlist antes de que GDAL haga el fetch. Solo se permiten paths locales / `file://`, `gs://` del bucket configurado (`settings.gcs_data_bucket`), y `http(s)://` de hosts en `settings.tile_url_allowed_hosts`; cualquier otro esquema/host levanta `CogUrlNotAllowedError`. | `backend/app/services/cog_tiler.py` (`validate_cog_url` L56-87, invocado en `render_cog_tile` L219); US-055 |

---

## 3. Gestion de secretos

- **Dev**: secretos en `.env.local` (gitignored, nunca commiteado ni
  hardcodeado). `Settings` (`backend/app/core/config.py`) los lee via
  `get_settings()` con `@lru_cache`; toda var nueva debe declararse alli por el
  `extra="forbid"`.
- **Prod**: Secret Manager (GCP) / Key Vault (Azure).
- **Escaneo de secretos** — *correccion de la referencia muerta del plan v6*: el
  proyecto **NO usa `.pre-commit-config.yaml`** (regla irrevocable). No existe un
  hook `detect-secrets`. El secrets-scan vive en **`make secrets-scan`**
  (gitleaks) y en GitHub Actions (CI), no en un hook local de pre-commit.

```bash
make secrets-scan   # gitleaks sobre el arbol (parte de make check)
make check          # lint + secrets-scan + i18n-check (obligatorio antes de PR)
```

---

## 4. Autenticacion: MVP vs Full

| Aspecto | MVP (27-jun) | Full / post-presentacion |
|---|---|---|
| Identidad | `user_id` demo hardcoded | Clerk OAuth2 |
| Tokens | sesion simple por `session_id` | JWT con rotacion + refresh tokens |
| Transporte | TLS por defecto de Cloud Run | HTTPS via Cloud Load Balancer + cert managed |

El middleware de auth aun no existe: `backend/app/middleware/` solo tiene
`__init__.py` y `metrics.py` (US-059). La implementacion de Clerk + JWT refresh
se difiere conscientemente para no bloquear la demo; queda documentada aqui y en
la skill `agrosat-security`.

---

## 5. Penetration test manual basico (pre-presentacion)

Check **manual humano** pre-deploy (no automatizado en CI), alineado con la skill
`agrosat-security-audit` (cuya postura es documentada, ejecutada por humanos en
revisiones). Procedimiento contra el deploy de **staging** (sustituir
`STAGING_HOST` por el host real del Cloud Run de staging):

```bash
# 1. Escaneo de puertos (solo deberian responder 80/443 del Cloud Run frontend)
nmap -Pn -sT -p- --top-ports 1000 STAGING_HOST

# 2. Escaneo HTTP basico (cabeceras, metodos, banners, ficheros expuestos)
nikto -h https://STAGING_HOST

# 3. Verificacion manual del aislamiento por sesion (A01): dos session_id
#    distintos no deben leer datos del otro (curl con cabeceras X-Session-ID
#    diferentes contra /aois, /timeseries).
```

**Estado en esta sesion**: el pen-test no se ejecuto contra un endpoint live
(no hay deploy de staging activo). El procedimiento queda listo; la ejecucion es
un paso manual pre-deploy. Anotado en el blocker doc.

---

## 6. Enlaces

- Skills: `agrosat-security` (controles de auth/rate-limit/RLS),
  `agrosat-security-audit` (OWASP code-level + CIS GCP/Azure, postura
  documentada).
- ADRs: [ADR-009](decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md)
  (H100 + pivote FarSLIP + alcance v8),
  [ADR-012](decisions/ADR-012-inferencia-sincrona-mvp.md) (inferencia sincrona MVP).
- US relacionadas: US-051 (RLS por tenant), US-055 (SSRF allowlist del tiler),
  US-057 (saneado XSS del markdown), US-059 (Prometheus), US-065 (observabilidad
  de chat).
- Claims no verificables de esta sesion: [docs/blockers/epic10-notas.md](blockers/epic10-notas.md).
