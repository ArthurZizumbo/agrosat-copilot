# ADR-003 — Redis serverless via Upstash en lugar de GCP Memorystore

**Status**: Propuesta · pendiente visto bueno equipo
**Fecha**: 2026-05-12
**Decisores**: Arthur Zizumbo (MLOps Lead), Aaron Bocanegra, Isaac Avila
**US relacionada**: [US-001](../us-planning/us-001.md) (descubrimiento durante QA), pre-requisito para US-009
**Avance**: A0 (2026-04-26)

---

## Contexto

El plan v6 y la sub-CLAUDE.md de `infrastructure/` mencionan Redis como
servicio GCP Memorystore Basic 1 GB para tres casos de uso:

1. **TiTiler cache**: tiles COG con TTL 1h, MosaicJSON con TTL 24h
   (`agrosat-titiler-cog`).
2. **Rate limit** distribuido en `/chat` (10 req/min) y `/llm/switch`
   (5 req/min) via `slowapi`, compartido entre instancias Cloud Run.
3. **Session storage** del agente Google ADK (`agrosat-google-adk-agent`).

GCP Memorystore Basic 1 GB cuesta **~$35 USD/mes** y NO escala a cero (siempre
on). Esto representa el 30 % del presupuesto operativo objetivo ($115/mes).
Adicionalmente requiere un Serverless VPC Access Connector para que Cloud Run
pueda alcanzarlo, lo que suma ~$10/mes y complejidad de red.

El proyecto es academico, trafico bajo (3 devs + revisor + demo final), con
patron de uso bursty (sprints de avances) → Memorystore queda subutilizado.

## Decisión

1. Usar **Upstash Redis Pay-as-you-go** como backend Redis productivo en el
   entorno `dev`.
2. Mantener `redis:7-alpine` en `docker-compose.yml` para desarrollo local
   (offline-first; ningun dev necesita cuenta Upstash para correr `make dev`).
3. Inyectar dos secretos en `.env.local` (dev) y Secret Manager (cloud):
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`
4. Conexion desde backend FastAPI via `redis-py` con TLS (rediss://) o cliente
   REST oficial (`@upstash/redis` no aplica; en Python usamos `redis-py` con
   url estilo `rediss://default:<token>@<endpoint>:6379`).
5. NO provisionar `google_redis_instance` ni VPC connector en el modulo TF GCP.

## Consecuencias

### Positivas

- Costo: **~$0-10 USD/mes** en el patron actual (free tier 10k commands/dia +
  $0.20 por 100k requests adicionales). Vs $35-45/mes Memorystore + VPC.
- Scale-to-zero coherente con Cloud Run: solo pagamos por lo que usamos.
- Sin VPC connector → menos superficie de red, menos IAM, deploy mas rapido.
- REST API tambien disponible → util para Cloud Run Worker y para el
  frontend Nuxt si en el futuro queremos cache server-side accesible desde
  Nitro middleware.
- Misma interfaz Redis estandar (RESP) → el codigo backend no cambia si en
  staging/prod se decide migrar a Memorystore.

### Negativas

- **Latencia**: 30-50 ms p50 vs <1 ms en Memorystore intra-VPC. Aceptable
  para los 3 casos de uso (tile cache: cache miss compara con render TiTiler
  ~500 ms; rate limit: 50 ms es invisible para humanos; session storage:
  fuera del hot path SSE).
- **Cuenta externa**: requiere crear cuenta Upstash (1 dev gestiona, los
  secretos viven en Secret Manager).
- **Cuota**: si excedemos free tier hay que monitorear costos. Mitigacion:
  alerta de billing a $20/mes.

### Neutras

- Latencia mas alta podria sesgar resultados de benchmarks si midiesemos
  TiTiler. Para benchmarks reales (Paper Track) se debe medir contra
  Memorystore equivalente.

## Alternativas consideradas

| Alternativa | Razón rechazo |
|-------------|---------------|
| GCP Memorystore Basic 1 GB | $35/mes + VPC connector $10/mes; 30 % presupuesto operativo total para uso bursty academico |
| Redis self-host en Cloud Run | Cloud Run no garantiza persistencia entre cold starts → rate limit y session storage romperian |
| Redis en Compute Engine e2-micro | Aun mas operacion; defeats el objetivo de serverless |
| Cloud Memorystore Standard HA | $115/mes solo Redis; absurdo para academico |
| DragonflyDB self-host | Lo mismo que self-host Redis; misma complejidad |

## Implementacion

### En esta US (US-001)

- ADR documentado (este archivo).
- `.env.example` con placeholders `UPSTASH_REDIS_REST_URL=`, `UPSTASH_REDIS_REST_TOKEN=`
  + `REDIS_URL=` (local docker-compose).
- `infrastructure/CLAUDE.md` actualizado: tabla §Servicios GCP ya no incluye
  Memorystore; nota de Upstash externo.
- `infrastructure/terraform/modules/gcp/main.tf`: agregar dos `secret_id`
  a `local.secret_ids` (`agrosat-upstash-rest-url`, `agrosat-upstash-rest-token`).

### En US-009 (cuando se implemente backend rate limit)

- Crear cuenta Upstash gratis en `console.upstash.com`.
- Crear database `agrosat-dev` en region `eu-west-1` (proxima a europe-west1).
- Copiar REST URL + token a Secret Manager via `gcloud secrets versions add`.
- Backend: cliente `redis.asyncio.Redis.from_url(upstash_url, ssl=True)`.
- Test: rate limit funcional contra Upstash con `slowapi`.

## Referencias

- Upstash Redis pricing: https://upstash.com/pricing
- Plan v6 §10 FinOps — [`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v6.md)
- ADR-001 — [`ADR-001-no-cookiecutter-externo.md`](ADR-001-no-cookiecutter-externo.md)
- ADR-002 — [`ADR-002-single-env-dev.md`](ADR-002-single-env-dev.md)
- TiTiler cache spec — `agrosat-titiler-cog` skill
- Rate limit spec — `agrosat-security` skill
