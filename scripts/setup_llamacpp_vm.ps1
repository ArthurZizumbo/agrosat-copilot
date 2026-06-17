# US-048 (llama.cpp variant): install llama.cpp (CUDA, Windows x64) on the H100 VM.
#
# Downloads the official ggml-org/llama.cpp release binaries (CUDA 13.x build) plus
# the CUDA runtime DLLs into F:\tools\llamacpp, so llama-server.exe runs natively
# on the Windows VM against the H100 -- no WSL2/Docker (those are blocked).
#
# Pin Tag to a known release for reproducibility; defaults to b9656.
# CudaTag: use the build that matches the DRIVER's max CUDA version, not the
# newest. The VM driver 596.36 reports CUDA 13.2, but the cuda-13.3 build ships
# PTX the driver rejects ("unsupported toolchain"). cuda-12.4 is well below 13.2
# and works. Only bump to a newer build after confirming the driver supports it.
# Run on the VM (it has network). Idempotent: skips download if the exe exists.

param(
    [string]$Tag = "b9656",
    [string]$Dest = "F:\tools\llamacpp",
    [string]$CudaTag = "cuda-12.4"
)

$ErrorActionPreference = "Stop"
$repo = "ggml-org/llama.cpp"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$binZip   = "llama-$Tag-bin-win-$CudaTag-x64.zip"
$cudartZip = "cudart-llama-bin-win-$CudaTag-x64.zip"
$binUrl    = "https://github.com/$repo/releases/download/$Tag/$binZip"
$cudartUrl = "https://github.com/$repo/releases/download/$Tag/$cudartZip"

$serverExe = Join-Path $Dest "llama-server.exe"
if (Test-Path $serverExe) {
    Write-Output "[setup_llamacpp] llama-server.exe already present at $serverExe -- skipping."
} else {
    foreach ($pair in @(@($binUrl, $binZip), @($cudartUrl, $cudartZip))) {
        $url = $pair[0]; $name = $pair[1]
        $tmp = Join-Path $env:TEMP $name
        Write-Output "[setup_llamacpp] downloading $name ..."
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
        Write-Output "[setup_llamacpp] extracting $name -> $Dest"
        Expand-Archive -Path $tmp -DestinationPath $Dest -Force
        Remove-Item $tmp -Force
    }
}

# Verify the binary loads and reports CUDA devices.
Write-Output "[setup_llamacpp] verifying llama-server ..."
& $serverExe --version 2>&1 | Select-Object -First 5 | Write-Output
Write-Output "[setup_llamacpp] done. Binary at $serverExe"
