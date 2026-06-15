@echo off
REM US-048 (llama.cpp variant): serve Qwen on-prem on the Windows H100 VM NATIVELY.
REM
REM vLLM needs Linux (blocked on this Hyper-V guest, see docs/serving/qwen35.md).
REM llama.cpp's llama-server runs natively on Windows + CUDA and exposes an
REM OpenAI-compatible /v1/chat/completions endpoint, so the agent's
REM VLLMOpenAIBackend (US-047) works unchanged -- only the base URL differs.
REM
REM --parallel N gives N concurrent request slots (not just one user): plenty for
REM the demo. -ngl 99 offloads all layers to the H100. Single GGUF, ~18.6GB, fits.
REM
REM Override paths/params via environment before calling, or edit the defaults.
REM Launch from a scheduled task so it survives the SSH session (no -ExecutionPolicy
REM Bypass needed for a .bat). Stop by killing llama-server.exe.

setlocal

if "%LLAMA_SERVER%"=="" set "LLAMA_SERVER=F:\tools\llamacpp\llama-server.exe"
if "%QWEN_GGUF%"==""    set "QWEN_GGUF=F:\models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
if "%QWEN_PORT%"==""    set "QWEN_PORT=8002"
if "%QWEN_CTX%"==""     set "QWEN_CTX=32768"
if "%QWEN_PARALLEL%"=="" set "QWEN_PARALLEL=4"
if "%QWEN_NGL%"==""     set "QWEN_NGL=99"
if "%QWEN_ALIAS%"==""   set "QWEN_ALIAS=qwen35"

echo [serve_qwen_llamacpp] server=%LLAMA_SERVER%
echo [serve_qwen_llamacpp] model=%QWEN_GGUF%
echo [serve_qwen_llamacpp] port=%QWEN_PORT% ctx=%QWEN_CTX% parallel=%QWEN_PARALLEL% ngl=%QWEN_NGL%

if not exist "%LLAMA_SERVER%" (
  echo [serve_qwen_llamacpp] ERROR: llama-server not found at %LLAMA_SERVER% 1>&2
  echo [serve_qwen_llamacpp] Run scripts\setup_llamacpp_vm.ps1 first. 1>&2
  exit /b 2
)
if not exist "%QWEN_GGUF%" (
  echo [serve_qwen_llamacpp] ERROR: GGUF not found at %QWEN_GGUF% 1>&2
  echo [serve_qwen_llamacpp] Run: poetry run python scripts\download_qwen_gguf.py 1>&2
  exit /b 2
)

REM --jinja: use the model's chat template (Qwen3) for correct tool/role formatting.
REM --flash-attn: enable FlashAttention kernels on the H100.
"%LLAMA_SERVER%" ^
  -m "%QWEN_GGUF%" ^
  --host 127.0.0.1 ^
  --port %QWEN_PORT% ^
  -ngl %QWEN_NGL% ^
  --ctx-size %QWEN_CTX% ^
  --parallel %QWEN_PARALLEL% ^
  --flash-attn on ^
  --jinja ^
  --alias %QWEN_ALIAS%

endlocal
