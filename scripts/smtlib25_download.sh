#!/bin/bash

# Check if logic name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <logic_name> [download_path]"
    echo "Example: $0 QF_NRA"
    echo "Example: $0 QF_NRA /path/to/download"
    exit 1
fi

LOGIC_NAME=$1
DOWNLOAD_PATH=${2:-.}  # Use current directory if no path provided

# Create directory if it doesn't exist
mkdir -p "$DOWNLOAD_PATH"

URL="https://zenodo.org/records/15493090/files/${LOGIC_NAME}.tar.zst?download=1"
ARCHIVE="${DOWNLOAD_PATH}/${LOGIC_NAME}.tar.zst"

echo "Downloading ${LOGIC_NAME} to ${DOWNLOAD_PATH}..."
curl -L "$URL" -o "$ARCHIVE"

echo "Extracting ${ARCHIVE}..."
tar --use-compress-program=zstd -xf "$ARCHIVE" -C "$DOWNLOAD_PATH"

echo "Cleaning up..."
rm "$ARCHIVE"

echo "Done!"
