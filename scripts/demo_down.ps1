<#
.SYNOPSIS
    Baja la demo de AgroSatCopilot levantada por scripts/demo_up.ps1.
.DESCRIPTION
    Detiene el backend nativo (uvicorn) y el frontend nativo (pnpm/nuxt) por puerto,
    cierra los tuneles SSH al Qwen on-prem, y baja los contenedores Docker.
    No apaga la VM ni el serving de LLMs en la VM (eso se gestiona alla).
#>
[CmdletBinding()]
param([switch]$KeepDocker)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Stop-Port($port, $label) {
    $procs = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $procs) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Write-Host "  detenido $label (pid $p, :$port)" -ForegroundColor Yellow
    }
}

Write-Host "==> Deteniendo servicios nativos" -ForegroundColor Cyan
Stop-Port 8010 "backend uvicorn"
Stop-Port 3010 "frontend nuxt"

Write-Host "==> Cerrando tuneles SSH al Qwen on-prem (:8002 texto + :8003 VL)" -ForegroundColor Cyan
# El tunel de demo_up reenvia 8002 (texto) y 8003 (VL) en un solo proceso ssh;
# emparejamos por cualquiera de los dos reenvios para cerrarlo.
Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match '8002:127.0.0.1:8002' -or $_.CommandLine -match '8003:127.0.0.1:8003' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host "  tunel cerrado (pid $($_.ProcessId))" -ForegroundColor Yellow }

if (-not $KeepDocker) {
    Write-Host "==> Bajando contenedores Docker (postgres/redis/titiler/mlflow)" -ForegroundColor Cyan
    docker compose --env-file .env.local stop postgres redis titiler mlflow | Out-Null
    Write-Host "  contenedores detenidos (datos persisten en el volumen)" -ForegroundColor Yellow
} else {
    Write-Host "==> Docker se mantiene arriba (-KeepDocker)" -ForegroundColor Gray
}

Write-Host "`nDemo detenida.`n" -ForegroundColor Green
