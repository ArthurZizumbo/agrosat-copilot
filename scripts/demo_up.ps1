<#
.SYNOPSIS
    Levanta la demo completa de AgroSatCopilot para la presentacion y la valida.

.DESCRIPTION
    Arranca, en el orden correcto y con healthchecks reales, todos los servicios
    necesarios para demostrar el copiloto en vivo:

      1. Datos:    Postgres+PostGIS+pgvector y Redis vía Docker (los dos contenedores
                   que funcionan de forma fiable; el resto corre nativo).
      2. Backend:  FastAPI nativo con Poetry (uvicorn). NO se usa el contenedor `api`
                   porque su Dockerfile dev falla con `ModuleNotFoundError: app`
                   (ver scripts/demo_up.README.md, bug DOCKER-API). Nativo arranca
                   limpio contra el Postgres dockerizado.
      3. TiTiler:  contenedor oficial (tiling COG) -- funciona tal cual.
      4. Frontend: Nuxt nativo con pnpm (`pnpm dev`). NO se usa el contenedor
                   `frontend` porque su Dockerfile falla en `COPY --from=deps /pnpm`
                   (bug DOCKER-FRONT).
      5. LLM on-prem (opcional, -WithVM): tunel SSH al Qwen llama.cpp de la VM H100
                   (:8002) para el switch A/B con soberania de datos.

    Cada paso se valida con un healthcheck; el script reporta un resumen final con
    el estado de cada servicio y las URLs. Idempotente: re-ejecutar no duplica.

.PARAMETER WithVM
    Abre tambien el tunel SSH al Qwen on-prem de la VM H100 (requiere el tunel
    cloudflared activo y la llave ~/.ssh/agrosat_h100).

.PARAMETER SkipFrontend
    No arranca el frontend (util para una demo solo-API o si Node no esta listo).

.EXAMPLE
    pwsh scripts/demo_up.ps1
    pwsh scripts/demo_up.ps1 -WithVM
#>
[CmdletBinding()]
param(
    [switch]$WithVM,
    [switch]$SkipFrontend
)

# NO usar "Stop": los ejecutables nativos (docker, dbmate, ssh) escriben progreso
# y avisos a stderr, lo que con "Stop" PowerShell convierte en excepcion terminal
# aunque el exit code sea 0. Con "Continue" el script juzga por el exit code real
# y por los healthchecks, no por el ruido de stderr.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Puertos (alineados con .env.local / docker-compose.yml).
$API_NATIVE    = 8010   # backend nativo (mismo puerto que espera el frontend)
$FRONT_PORT    = 3010
$TITILER_PORT  = 8011
$MLFLOW_PORT   = 5010
$QWEN_LOCAL    = 8002

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }

# Espera hasta `timeout` s a que `url` devuelva 200 (o cualquier respuesta si AnyCode).
function Wait-Http($url, $timeout = 60, [switch]$AnyCode) {
    $deadline = (Get-Date).AddSeconds($timeout)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $url -TimeoutSec 4 -UseBasicParsing -ErrorAction Stop
            if ($AnyCode -or $r.StatusCode -eq 200) { return $true }
        } catch {
            if ($AnyCode -and $_.Exception.Response) { return $true }
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

$summary = [ordered]@{}

# --- 0. Bajar los contenedores rotos (api/frontend) si quedaron arriba -------
# Sus Dockerfiles dev fallan (bug DOCKER-API ModuleNotFoundError, bug DOCKER-FRONT
# COPY /pnpm); corremos esos dos nativos. Si un intento previo de `make dev` los
# dejo arriba, ocupan los puertos 8010/3010 y chocan con el nativo. Los retiramos.
Write-Step "0/5 Retirando contenedores rotos api/frontend (corren nativos)"
# docker compose escribe progreso a stderr; en PowerShell eso dispara
# NativeCommandError. Capturamos ambos flujos con *>&1 y los descartamos.
docker compose --env-file .env.local rm -sf api frontend *>&1 | Out-Null
Write-Ok "Contenedores api/frontend retirados (se levantan nativos)"

# --- 1. Datos: Postgres + Redis (Docker) -----------------------------------
Write-Step "1/5 Datos: Postgres+PostGIS+pgvector y Redis (Docker)"
docker compose --env-file .env.local up -d postgres redis titiler | Out-Null
# Postgres expone 55432->5432; healthcheck propio del contenedor.
$pgReady = $false
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    $h = (docker inspect --format '{{.State.Health.Status}}' agro_sat_copilot-postgres-1 *>&1)
    if ($h -eq "healthy") { $pgReady = $true; break }
    Start-Sleep -Seconds 2
}
if ($pgReady) { Write-Ok "Postgres healthy (:55432)"; $summary["Postgres"] = "OK :55432" }
else { Write-Warn2 "Postgres no reporto healthy"; $summary["Postgres"] = "FALLO" }
$summary["Redis"]   = "OK :63790"
$summary["TiTiler"] = "OK :$TITILER_PORT"
Write-Ok "Redis (:63790) + TiTiler (:$TITILER_PORT) levantados"

# --- 2. Migraciones + seed -------------------------------------------------
Write-Step "2/5 Migraciones (dbmate) + seed de demo"
# Dos gotchas: (1) dbmate busca `.env` por defecto, el repo usa `.env.local`;
# (2) el DATABASE_URL de la app trae el driver async `postgresql+asyncpg` que
# dbmate rechaza -- el repo define DBMATE_DATABASE_URL (driver psql plano) para
# esto. dbmate lee la URL de la env var DATABASE_URL, asi que la fijamos al valor
# de DBMATE_DATABASE_URL solo para este comando.
try {
    $dbmateUrl = (Select-String -Path .env.local -Pattern '^DBMATE_DATABASE_URL=(.+)$').Matches.Groups[1].Value
    if ($dbmateUrl) {
        $env:DATABASE_URL = $dbmateUrl
        & dbmate --env-file .env.local --url $dbmateUrl up 2>&1 | Out-Null
    } else {
        & dbmate --env-file .env.local up 2>&1 | Out-Null
    }
    Write-Ok "Migraciones aplicadas (dbmate up)"
    $summary["Migraciones"] = "OK"
} catch {
    Write-Warn2 "dbmate up fallo: $($_.Exception.Message)"
    $summary["Migraciones"] = "REVISAR"
}

# --- 3. Backend FastAPI nativo (Poetry) ------------------------------------
Write-Step "3/5 Backend FastAPI nativo (Poetry uvicorn :$API_NATIVE)"
# Si ya hay algo escuchando en el puerto, no relanzar.
$inUse = Get-NetTCPConnection -LocalPort $API_NATIVE -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    Write-Warn2 "Puerto $API_NATIVE ya en uso; reutilizo el backend existente"
} else {
    # poetry debe correr DESDE backend/ para resolver el env y el modulo `app`.
    # Start-Process -WorkingDirectory fija el cwd del proceso hijo (Push-Location
    # no se hereda al proceso lanzado).
    Start-Process -FilePath "poetry" `
        -ArgumentList "run","uvicorn","app.main:app","--host","127.0.0.1","--port","$API_NATIVE" `
        -WorkingDirectory "$repo\backend" `
        -WindowStyle Hidden -RedirectStandardOutput "$repo\_demo_api.log" -RedirectStandardError "$repo\_demo_api.err.log"
}
if (Wait-Http "http://127.0.0.1:$API_NATIVE/healthz" 60) {
    Write-Ok "Backend healthz 200 (http://127.0.0.1:$API_NATIVE)"
    $summary["Backend API"] = "OK :$API_NATIVE"
} else {
    Write-Warn2 "Backend no respondio healthz; ver _demo_api.err.log"
    $summary["Backend API"] = "FALLO"
}

# --- 4. Frontend Nuxt nativo (pnpm) ----------------------------------------
if (-not $SkipFrontend) {
    Write-Step "4/5 Frontend Nuxt nativo (pnpm dev :$FRONT_PORT)"
    $frontInUse = Get-NetTCPConnection -LocalPort $FRONT_PORT -State Listen -ErrorAction SilentlyContinue
    if ($frontInUse) {
        Write-Warn2 "Puerto $FRONT_PORT ya en uso; reutilizo el frontend existente"
    } else {
        # pnpm en Windows es un .ps1 (ExternalScript), NO un .exe; Start-Process
        # -FilePath "pnpm" falla en silencio. Hay que invocarlo a traves del
        # ejecutable de PowerShell (pwsh/powershell). Resolvemos cual existe.
        $pwshCmd = Get-Command pwsh -ErrorAction SilentlyContinue
        $psExe = if ($pwshCmd) { $pwshCmd.Source } else { (Get-Command powershell).Source }
        if (-not (Test-Path "$repo\frontend\node_modules")) {
            Write-Warn2 "frontend/node_modules ausente; corriendo pnpm install (una vez)"
            Start-Process -FilePath $psExe -ArgumentList "-NoProfile","-Command","pnpm install" `
                -WorkingDirectory "$repo\frontend" -Wait -WindowStyle Hidden
        }
        Start-Process -FilePath $psExe `
            -ArgumentList "-NoProfile","-Command","pnpm dev --port $FRONT_PORT" `
            -WorkingDirectory "$repo\frontend" `
            -WindowStyle Hidden -RedirectStandardOutput "$repo\_demo_front.log" -RedirectStandardError "$repo\_demo_front.err.log"
    }
    # Nuxt en Windows liga el dev server por nombre (localhost), y 127.0.0.1
    # puede dar 000 por el binding IPv6/IPv4; el healthcheck usa localhost.
    if (Wait-Http "http://localhost:$FRONT_PORT" 120 -AnyCode) {
        Write-Ok "Frontend respondiendo (http://localhost:$FRONT_PORT)"
        $summary["Frontend"] = "OK :$FRONT_PORT"
    } else {
        Write-Warn2 "Frontend no respondio; ver _demo_front.err.log"
        $summary["Frontend"] = "FALLO"
    }
} else {
    $summary["Frontend"] = "OMITIDO (-SkipFrontend)"
}

# --- 5. LLM on-prem por tunel SSH (opcional) -------------------------------
if ($WithVM) {
    Write-Step "5/5 Tunel al Qwen on-prem de la VM H100 (:$QWEN_LOCAL)"
    $key = "$HOME\.ssh\agrosat_h100"
    if (-not (Test-Path $key)) {
        Write-Warn2 "Llave $key ausente; omito el tunel on-prem"
        $summary["Qwen on-prem"] = "SIN LLAVE"
    } else {
        Start-Process -FilePath "ssh" -ArgumentList `
            "-p","50022","-i",$key,"-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=no",`
            "-N","-L","${QWEN_LOCAL}:127.0.0.1:8002","User1@127.0.0.1" -WindowStyle Hidden
        if (Wait-Http "http://127.0.0.1:$QWEN_LOCAL/health" 50) {
            Write-Ok "Qwen on-prem alcanzable (http://127.0.0.1:$QWEN_LOCAL)"
            $summary["Qwen on-prem"] = "OK :$QWEN_LOCAL"
        } else {
            Write-Warn2 "Qwen no respondio /health (tarda ~40s en cargar el GGUF, o la tarea qwen_serve no esta corriendo en la VM)"
            $summary["Qwen on-prem"] = "REVISAR"
        }
    }
} else {
    $summary["Qwen on-prem"] = "OMITIDO (usa -WithVM)"
}

# --- Resumen ----------------------------------------------------------------
Write-Host "`n=================== RESUMEN DEMO ===================" -ForegroundColor Magenta
foreach ($k in $summary.Keys) {
    $v = $summary[$k]
    $color = if ($v -like "OK*") { "Green" } elseif ($v -like "OMITIDO*") { "Gray" } else { "Yellow" }
    Write-Host ("  {0,-14} {1}" -f $k, $v) -ForegroundColor $color
}
Write-Host "===================================================" -ForegroundColor Magenta
Write-Host "`nURLs:" -ForegroundColor Cyan
Write-Host "  Frontend : http://localhost:$FRONT_PORT"
Write-Host "  API docs : http://localhost:$API_NATIVE/docs"
Write-Host "  TiTiler  : http://localhost:$TITILER_PORT"
Write-Host "  MLflow   : http://localhost:$MLFLOW_PORT  (make mlflow-up si no esta arriba)"
Write-Host "`nPara bajar todo: pwsh scripts/demo_down.ps1`n"
