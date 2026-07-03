# Exporta la presentacion (docs/presentation) a PDF usando el modo print-pdf de
# Reveal.js y un navegador Chromium headless (Edge en Windows, Chrome como
# respaldo). Levanta un servidor HTTP efimero (fetch de content/*.json no
# funciona via file://), imprime y lo apaga.
#
# Uso:
#   pwsh scripts/presentation_pdf.ps1                # ES -> docs/presentation/AgroSatCopilot_presentacion_es.pdf
#   pwsh scripts/presentation_pdf.ps1 -Lang en       # EN
#   pwsh scripts/presentation_pdf.ps1 -Lang es,en    # ambos
param(
    [string[]]$Lang = @("es"),
    [int]$Port = 8123
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$deckDir = Join-Path $repoRoot "docs/presentation"

# `-Lang es,en` via `pwsh -File` llega como un solo string "es,en": separarlo.
$Lang = $Lang | ForEach-Object { $_ -split "," } | Where-Object { $_ }
$bad = $Lang | Where-Object { $_ -notin @("es", "en") }
if ($bad) { throw "Idiomas no soportados: $($bad -join ', ') (usar es|en)" }

# La impresion real la hace Playwright (page.pdf respeta el @page 1600x900 de
# Reveal; el print-to-pdf del CLI de Edge NO y pagina cada lamina en ~4 hojas).
$printer = Join-Path $PSScriptRoot "presentation_pdf.mjs"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Se requiere node (Playwright del frontend)." }

# Servidor HTTP efimero sobre el deck.
$server = Start-Process -FilePath "python" -ArgumentList "-m", "http.server", "$Port", "--directory", $deckDir `
    -PassThru -WindowStyle Hidden
try {
    # Esperar a que el servidor responda.
    $up = $false
    foreach ($i in 1..30) {
        try {
            Invoke-WebRequest -Uri "http://localhost:$Port/" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $up = $true; break
        } catch { Start-Sleep -Milliseconds 300 }
    }
    if (-not $up) { throw "El servidor local no respondio en el puerto $Port." }

    foreach ($l in $Lang) {
        $out = Join-Path $deckDir "AgroSatCopilot_presentacion_$l.pdf"
        # Pagina de impresion propia (una lamina = una hoja fija 1600x900), en
        # vez del pipeline print-pdf de reveal.js (fragil con temas custom).
        $url = "http://localhost:$Port/print.html?lang=$l"
        Write-Host "Imprimiendo $l -> $out"
        node $printer $url $out
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { throw "No se genero el PDF para '$l'." }
        $kb = [math]::Round((Get-Item $out).Length / 1kb)
        Write-Host "OK: $out ($kb KB)"
    }
}
finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
