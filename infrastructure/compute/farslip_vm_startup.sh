#!/bin/bash
# Startup script para la VM `agrosat-farslip-trainer-dev` (US-022-c P1 etapa 5 fix v4).
#
# Imagen base: Deep Learning VM `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`
# (Ubuntu 22.04 + Pytorch 2.9 + CUDA 12.9 + Python 3.12 + NVIDIA 580 preinstalados).
# Ejecutado por Compute Engine al primer boot y en cada start. Idempotente.
#
# Tareas (sin Docker — corremos Python directo, mas simple):
#   1) Verificar NVIDIA driver + Pytorch (ya preinstalados).
#   2) git clone repo agro_sat_copilot a /opt/agrosat (idempotente).
#   3) pip install deps adicionales (transformers, mlflow, dvc[gs], breizhcrops, etc.).
#   4) Descargar dataset farslip_pairs (~15 GB) desde GCS al boot SSD 100 GB.
#   5) Bajar daemon Python desde GCS + lanzar systemd service farslip-vm-daemon.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
exec > >(tee -a /var/log/farslip-vm-startup.log) 2>&1
echo "=== farslip-vm-startup v4 BEGIN: $(date -u +%FT%TZ) ==="

get_metadata() {
    curl -fsSL -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || echo ""
}

PROJECT_ID=$(get_metadata project-id)
SUBSCRIPTION_ID=$(get_metadata subscription-id)
DATA_BUCKET=$(get_metadata data-bucket)
DATA_PREFIX=$(get_metadata data-prefix)
DAEMON_URI=$(get_metadata daemon-uri)
IDLE_SHUTDOWN_SECONDS=$(get_metadata idle-shutdown-seconds)
MLFLOW_TRACKING_URI=$(get_metadata mlflow-tracking-uri)

echo "PROJECT_ID=$PROJECT_ID SUBSCRIPTION_ID=$SUBSCRIPTION_ID MLFLOW=$MLFLOW_TRACKING_URI"

# ----------------------------------------------------------------------------
# 1) Verificar NVIDIA + Pytorch (preinstalados en DLVM)
# ----------------------------------------------------------------------------
echo "[1/5] Verificando NVIDIA driver y Pytorch..."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "ERROR: nvidia-smi falla"; exit 1; }
/opt/conda/bin/python -c "import torch; print(f'Pytorch {torch.__version__} CUDA={torch.cuda.is_available()}')" || \
    python3 -c "import torch; print(f'Pytorch {torch.__version__} CUDA={torch.cuda.is_available()}')"

# DLVM trae conda env en /opt/conda. Usamos pip de ese conda para consistencia.
PIP_BIN="/opt/conda/bin/pip"
PYTHON_BIN="/opt/conda/bin/python"
[ -x "$PIP_BIN" ] || { PIP_BIN="pip3"; PYTHON_BIN="python3"; }

# ----------------------------------------------------------------------------
# 2) Clonar repo agro_sat_copilot (idempotente)
# ----------------------------------------------------------------------------
REPO_DIR="/opt/agrosat"
REPO_BRANCH="us-022-c-Farslip"
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[2/5] Clonando repo public agrosat-copilot rama $REPO_BRANCH..."
    git clone --branch "$REPO_BRANCH" https://github.com/ArthurZizumbo/agrosat-copilot.git "$REPO_DIR" || \
        git clone https://github.com/ArthurZizumbo/agrosat-copilot.git "$REPO_DIR"
else
    echo "[2/5] Repo ya existe en $REPO_DIR, git pull"
    cd "$REPO_DIR" && git fetch origin "$REPO_BRANCH" && git checkout "$REPO_BRANCH" && git pull origin "$REPO_BRANCH"
fi

# ----------------------------------------------------------------------------
# 3) Instalar deps adicionales (Pytorch ya esta, transformers/mlflow/dvc no)
# ----------------------------------------------------------------------------
echo "[3/5] Instalando deps FarSLIP..."
$PIP_BIN install --quiet --upgrade \
    transformers==5.8.0 \
    mlflow==3.12.0 \
    dvc[gs]==3.67.1 \
    breizhcrops==0.0.4.1 \
    sentence-transformers \
    polars \
    structlog \
    typer \
    google-cloud-pubsub \
    google-cloud-storage \
    pyyaml \
    rasterio

# ----------------------------------------------------------------------------
# 4) Descargar dataset farslip_pairs desde GCS al boot SSD
# ----------------------------------------------------------------------------
DATA_DIR="/opt/agrosat/data/farslip_pairs"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "[4/5] Descargando dataset gs://${DATA_BUCKET}/${DATA_PREFIX}/ (~15 GB)..."
    mkdir -p "$DATA_DIR"
    gcloud storage cp --recursive "gs://${DATA_BUCKET}/${DATA_PREFIX}/*" "$DATA_DIR/" 2>&1 | tail -10
    echo "[4/5] Dataset descargado: $(du -sh $DATA_DIR | cut -f1)"
else
    echo "[4/5] Dataset ya existe ($(du -sh $DATA_DIR | cut -f1)), skip"
fi

# ----------------------------------------------------------------------------
# 5) Daemon Pub/Sub + systemd service
# ----------------------------------------------------------------------------
echo "[5/5] Bajando daemon desde $DAEMON_URI ..."
mkdir -p /opt/farslip-daemon
gcloud storage cp "$DAEMON_URI" /opt/farslip-daemon/farslip_vm_daemon.py

cat > /etc/systemd/system/farslip-vm-daemon.service <<EOF
[Unit]
Description=AgroSatCopilot FarSLIP VM Daemon (Pub/Sub event-driven, US-022-c P1)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment="PROJECT_ID=${PROJECT_ID}"
Environment="SUBSCRIPTION_ID=${SUBSCRIPTION_ID}"
Environment="IDLE_SHUTDOWN_SECONDS=${IDLE_SHUTDOWN_SECONDS:-300}"
Environment="WORKDIR=${REPO_DIR}"
Environment="MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=${PYTHON_BIN} /opt/farslip-daemon/farslip_vm_daemon.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable farslip-vm-daemon
systemctl restart farslip-vm-daemon

echo "=== farslip-vm-startup v4 OK: $(date -u +%FT%TZ) ==="
echo "Monitor daemon: journalctl -u farslip-vm-daemon -f"
echo "Monitor startup: tail -f /var/log/farslip-vm-startup.log"
