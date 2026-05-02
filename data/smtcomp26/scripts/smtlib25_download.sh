#!/usr/bin/env bash
# NOTE: John personal use on snork.
# Download, decompress, extract, and clean up one SMT-LIB 25 logic division from Zenodo.
# Usage: ./smtlib25_download.sh <LOGIC_NAME>
# Example: ./smtlib25_download.sh QF_NIA

set -e

ZENODO_RECORD="${ZENODO_RECORD:-15493090}"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="/home/z52lu/smtlib25"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <LOGIC_NAME>" >&2
  echo "Example: $0 QF_NIA" >&2
  exit 1
fi

LOGIC="$1"
if [[ ! "$LOGIC" =~ ^[A-Za-z0-9_+-]+$ ]]; then
  echo "Error: invalid logic name '$LOGIC'." >&2
  echo "Allowed characters: letters, digits, underscore, plus, hyphen." >&2
  exit 1
fi

if ! command -v wget &>/dev/null; then
  echo "Error: wget is required but was not found in PATH." >&2
  exit 1
fi
if ! command -v tar &>/dev/null; then
  echo "Error: tar is required but was not found in PATH." >&2
  exit 1
fi

ARCHIVE="${LOGIC}.tar.zst"
TARFILE="${LOGIC}.tar"
DOWNLOAD_URL="${BASE_URL}/${ARCHIVE}?download=1"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading ${ARCHIVE}..."
wget -O "$ARCHIVE" "$DOWNLOAD_URL"

echo "Decompressing..."
if command -v zstd &>/dev/null; then
  zstd -df "$ARCHIVE" -o "$TARFILE"
else
  # Fallback: use Python zstandard (pip install zstandard)
  PYTHON="${PYTHON:-python3}"
  if [[ -x "${SCRIPT_DIR}/../.venv/bin/python3" ]]; then
    PYTHON="${SCRIPT_DIR}/../.venv/bin/python3"
  fi
  if ! "$PYTHON" -c "
import zstandard as zstd
with open('$ARCHIVE', 'rb') as f_in, open('$TARFILE', 'wb') as f_out:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(f_in, f_out)
" 2>/dev/null; then
    echo "Error: zstd not found and Python zstandard not available." >&2
    echo "Install one of: (1) zstd binary, or (2) pip install zstandard" >&2
    exit 1
  fi
fi

echo "Extracting..."
tar -xf "$TARFILE" --no-same-owner

echo "Removing archives to free space..."
rm -f "$TARFILE" "$ARCHIVE"

echo "Done. Benchmarks are under ${TARGET_DIR}/"