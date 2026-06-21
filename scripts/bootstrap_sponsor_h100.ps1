<#
.SYNOPSIS
    Levanta el stack de AgroSatCopilot en la VM H100 del sponsor (Windows nativo).

.DESCRIPTION
    La VM del sponsor (gjcamacho-gpuh1) corre Windows nativo con el entorno
    micromamba `agrosat` en F: y NO puede usar Docker (nested virtualization
    desactivada). Por eso este arranque es NATIVO: backend FastAPI + MLflow con
    backend SQLite local + (opcional) serving Qwen3.5 vLLM en la H100. Postgres y
    el frontend Nuxt son opcionales y se omiten por defecto (la demo del sponsor
    se centra en el backend + modelos + serving on-prem).

    Componentes que arranca (en orden):
      1. MLflow tracking server nativo sobre SQLite  -> http://127.0.0.1:5010
      2. Backend FastAPI (uvicorn) con .env.local     -> http://127.0.0.1:8000
      3. (opcional -ServeQwen) vLLM Qwen3.5-35B-A3B    -> http://127.0.0.1:8002

    No usa docker compose. Para el arranque en la nube usar
    `scripts/bootstrap_cloud.sh`.

.PARAMETER RepoRoot
    Raiz del repo en la VM. Por defecto F:\projects\agrosat-copilot.

.PARAMETER Micromamba
    Ruta al ejecutable micromamba. Por defecto F:\tools\micromamba.exe.

.PARAMETER EnvName
    Nombre del entorno micromamba. Por defecto `agrosat`.

.PARAMETER ServeQwen
    Si se indica, lanza tambien el serving vLLM de Qwen3.5 en la H100.

.PARAMETER SkipMlflow
    Omite el arranque del servidor MLflow (usar cuando ya hay uno corriendo).

.EXAMPLE
    pwsh -File scripts/bootstrap_sponsor_h100.ps1
    pwsh -File scripts/bootstrap_sponsor_h100.ps1 -ServeQwen
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = 'F:\projects\agrosat-copilot',
    [string]$Micromamba = 'F:\tools\micromamba.exe',
    [string]$EnvName = 'agrosat',
    [switch]$ServeQwen,
    [switch]$SkipMlflow
)

$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Msg) Write-Host "[bootstrap-h100] $Msg" -ForegroundColor Cyan }
function Write-Warn2 { param([string]$Msg) Write-Host "[bootstrap-h100] WARN: $Msg" -ForegroundColor Yellow }

# --- Preconditions -----------------------------------------------------------
if (-not (Test-Path $RepoRoot)) { throw "RepoRoot no existe: $RepoRoot" }
if (-not (Test-Path $Micromamba)) { throw "micromamba no encontrado: $Micromamba" }

$run = @($Micromamba, 'run', '-n', $EnvName)
Set-Location $RepoRoot

Write-Step "Repo: $RepoRoot  |  env: $EnvName"

# .env.local es obligatorio (extra=forbid en backend Settings). No lo creamos
# nosotros: contiene secretos y debe existir ya en la VM.
if (-not (Test-Path (Join-Path $RepoRoot '.env.local'))) {
    Write-Warn2 ".env.local ausente. El backend FastAPI no arrancara sin el (extra=forbid)."
}

# Verifica que torch ve la H100 (sanity del entorno).
Write-Step "Verificando entorno (torch + CUDA)..."
& $run[0] $run[1..($run.Count - 1)] python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

$logDir = Join-Path $RepoRoot '.vm_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# --- 1. MLflow tracking server (SQLite nativo) -------------------------------
# Memoria del proyecto: en la VM el lineage va a file/SQLite local, NO al server
# Docker :5010 (no hay Docker). Arrancamos MLflow nativo sobre SQLite para que
# los runs de entrenamiento/eval tengan un tracking server real on-prem.
if (-not $SkipMlflow) {
    $mlflowUp = $false
    try {
        $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:5010/health' -TimeoutSec 3 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $mlflowUp = $true }
    } catch { $mlflowUp = $false }

    if ($mlflowUp) {
        Write-Step "MLflow ya esta arriba en :5010 (reuso)."
    } else {
        Write-Step "Arrancando MLflow (SQLite) en :5010..."
        $mlflowDb = Join-Path $RepoRoot 'mlflow.db'
        $artifacts = Join-Path $RepoRoot 'mlruns'
        New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
        $mlflowArgs = @(
            'run', '-n', $EnvName, 'mlflow', 'server',
            '--backend-store-uri', "sqlite:///$mlflowDb",
            '--default-artifact-root', $artifacts,
            '--host', '127.0.0.1', '--port', '5010'
        )
        Start-Process -FilePath $Micromamba -ArgumentList $mlflowArgs `
            -RedirectStandardOutput (Join-Path $logDir 'mlflow.out.log') `
            -RedirectStandardError (Join-Path $logDir 'mlflow.err.log') `
            -WindowStyle Hidden
        Write-Step "MLflow lanzado (log: $logDir\mlflow.*.log)."
    }
}

# --- 2. Backend FastAPI (uvicorn) --------------------------------------------
Write-Step "Arrancando backend FastAPI (uvicorn) en :8000..."
$apiArgs = @(
    'run', '-n', $EnvName, 'uvicorn', 'backend.app.main:app',
    '--host', '127.0.0.1', '--port', '8000'
)
Start-Process -FilePath $Micromamba -ArgumentList $apiArgs `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput (Join-Path $logDir 'api.out.log') `
    -RedirectStandardError (Join-Path $logDir 'api.err.log') `
    -WindowStyle Hidden

# Espera a que /healthz responda (hasta 60 s).
Write-Step "Esperando /healthz..."
$apiReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/healthz' -TimeoutSec 3 -UseBasicParsing
        if ($h.StatusCode -eq 200) { $apiReady = $true; break }
    } catch { }
}
if ($apiReady) { Write-Step "Backend OK -> http://127.0.0.1:8000 (docs en /docs)." }
else { Write-Warn2 "Backend no respondio /healthz en 60 s. Revisa $logDir\api.err.log" }

# --- 3. (opcional) Serving Qwen3.5 vLLM en la H100 ---------------------------
if ($ServeQwen) {
    Write-Step "Lanzando serving Qwen3.5-35B-A3B (vLLM) en :8002..."
    $serveScript = Join-Path $RepoRoot 'scripts\serve_qwen35.sh'
    if (Test-Path $serveScript) {
        # serve_qwen35.sh es bash; en la VM Windows se invoca via el bash de git
        # o WSL. Si no hay bash, se documenta el comando manual.
        $bash = (Get-Command bash -ErrorAction SilentlyContinue)
        if ($bash) {
            Start-Process -FilePath $bash.Source -ArgumentList @($serveScript) `
                -WorkingDirectory $RepoRoot `
                -RedirectStandardOutput (Join-Path $logDir 'qwen.out.log') `
                -RedirectStandardError (Join-Path $logDir 'qwen.err.log') `
                -WindowStyle Hidden
            Write-Step "Qwen serving lanzado (log: $logDir\qwen.*.log)."
        } else {
            Write-Warn2 "bash no disponible. Lanza el serving manual: $serveScript"
        }
    } else {
        Write-Warn2 "scripts/serve_qwen35.sh no existe; omito serving Qwen."
    }
}

# --- Resumen -----------------------------------------------------------------
Write-Host ""
Write-Step "Stack on-prem levantado. Endpoints:"
Write-Host "  - Backend API : http://127.0.0.1:8000  (/docs, /healthz, /metrics)"
if (-not $SkipMlflow) { Write-Host "  - MLflow      : http://127.0.0.1:5010" }
if ($ServeQwen) { Write-Host "  - Qwen vLLM   : http://127.0.0.1:8002 (OpenAI-compatible)" }
Write-Host ""
Write-Step "Logs en $logDir. Para detener: Get-Process python,mlflow | Stop-Process (con cuidado)."
Write-Step "Acceso remoto via tunel cloudflared: ver docs/infra/acceso-vm-h100-tunnel.md"
