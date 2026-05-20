#!/usr/bin/env bash
# Descarga local del dataset BreizhCrops (Russwurm et al., ISPRS Archives 2020)
# al layout que espera el paquete `breizhcrops`. Operativo permanente: se
# corre A MANO una sola vez por maquina. Los notebooks NUNCA auto-descargan.
#
# Uso:
#   bash scripts/download_breizhcrops.sh                # regiones default frh04 frh01
#   BC_REGIONS="frh04" bash scripts/download_breizhcrops.sh
#   BC_ROOT=/data/breizhcrops bash scripts/download_breizhcrops.sh
#
# Idempotente: omite cualquier archivo ya presente con tamano > 0.
# Las URLs S3 son estaticas y sin autenticacion (bucket publico eu-central-1).
#
# Layout resultante (root = data/breizhcrops/):
#   data/breizhcrops/classmapping.csv
#   data/breizhcrops/codes.csv
#   data/breizhcrops/2017/L2A/frh04.csv
#   data/breizhcrops/2017/L2A/frh04.h5      (descomprimido de frh04.h5.tar.gz)
#   data/breizhcrops/2017/L2A/frh01.csv
#   data/breizhcrops/2017/L2A/frh01.h5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BC_ROOT="${BC_ROOT:-${REPO_ROOT}/data/breizhcrops}"
BC_YEAR="${BC_YEAR:-2017}"
BC_LEVEL="${BC_LEVEL:-L2A}"
BC_REGIONS="${BC_REGIONS:-frh04 frh01}"

S3_BASE="https://breizhcrops.s3.eu-central-1.amazonaws.com"
LEVEL_DIR="${BC_ROOT}/${BC_YEAR}/${BC_LEVEL}"

mkdir -p "${LEVEL_DIR}"

# fetch URL DEST -- descarga solo si DEST no existe o esta vacio.
fetch() {
  local url="$1" dest="$2"
  if [[ -s "${dest}" ]]; then
    echo "skip (ya existe): ${dest}"
    return 0
  fi
  echo "descargando: ${url}"
  curl -fSL --retry 3 --retry-delay 5 -o "${dest}.part" "${url}"
  mv "${dest}.part" "${dest}"
  echo "ok: ${dest}"
}

# Tablas de referencia compartidas por todas las regiones.
fetch "${S3_BASE}/classmapping.csv" "${BC_ROOT}/classmapping.csv"
fetch "${S3_BASE}/codes.csv" "${BC_ROOT}/codes.csv"

for region in ${BC_REGIONS}; do
  echo ">>> region ${region}"
  idx_dest="${LEVEL_DIR}/${region}.csv"
  h5_dest="${LEVEL_DIR}/${region}.h5"
  targz_dest="${LEVEL_DIR}/${region}.h5.tar.gz"

  fetch "${S3_BASE}/${BC_YEAR}/${BC_LEVEL}/${region}.csv" "${idx_dest}"

  if [[ -s "${h5_dest}" ]]; then
    echo "skip (ya existe): ${h5_dest}"
  else
    # Reutiliza un .tar.gz preexistente (descargas previas en layout plano).
    flat_targz="${BC_ROOT}/${region}.h5.tar.gz"
    if [[ -s "${flat_targz}" && ! -s "${targz_dest}" ]]; then
      echo "moviendo .tar.gz preexistente al layout: ${flat_targz} -> ${targz_dest}"
      mv "${flat_targz}" "${targz_dest}"
    fi
    fetch "${S3_BASE}/${BC_YEAR}/${BC_LEVEL}/${region}.h5.tar.gz" "${targz_dest}"
    echo "descomprimiendo: ${targz_dest}"
    tar -xzf "${targz_dest}" -C "${LEVEL_DIR}"
    # El tar incluye el .h5 en la raiz; si quedo en subcarpeta lo reubicamos.
    if [[ ! -s "${h5_dest}" ]]; then
      found="$(find "${LEVEL_DIR}" -name "${region}.h5" -type f | head -n1 || true)"
      if [[ -n "${found}" && "${found}" != "${h5_dest}" ]]; then
        mv "${found}" "${h5_dest}"
      fi
    fi
    rm -f "${targz_dest}"
    echo "ok: ${h5_dest}"
  fi
done

echo ""
echo "BreizhCrops listo en: ${BC_ROOT}"
echo "Re-ejecuta el notebook con:"
echo "  poetry run papermill notebooks/eda/02d_eda_breizhcrops.ipynb \\"
echo "    notebooks/eda/02d_eda_breizhcrops.ipynb -p region frh04 -p year 2017"
