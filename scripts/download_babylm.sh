#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data/babylm"

declare -A DATASETS=(
    [10m]="https://osf.io/download/y7djq/"
    [100m]="https://osf.io/download/ywea7/"
    [dev]="https://osf.io/download/uyd7v/"
    [test]="https://osf.io/download/ftwu3"
)


DATASET="${1:-all}"

echo "**************************"
echo "Downloading BabyLM data"
echo "OSF Project: https://osf.io/ryjfm/"
echo "**************************"


mkdir -p "$DATA_DIR"
cd "$DATA_DIR"


download_data() {
    local name="$1"
    local url="$2"

    echo "Downloading: $name"
    if wget -q --show-progress -O "${name}.zip" "$url"; then
        unzip -q "${name}.zip"
        rm -f "${name}.zip"
        echo "$name downloaded and extracted"
    else
        echo "Failed to download $name"
        echo "Please visit https://osf.io/ryjfm/files/ and check the file IDs"
        exit 1
    fi
}


if [[ "$DATASET" == "all" ]]; then
    for key in "${!DATASETS[@]}"; do
        echo "-----------------------------"
        download_data "$key" "${DATASETS[$key]}"
    done
else
    if [[ -v DATASETS["$DATASET"] ]]; then
        download_data "$DATASET" "${DATASETS[$DATASET]}"
    else
        echo "wrong key: '$DATASET'"
        echo "Usage: $0 [10m|100m|dev|test|all]"
        exit 1
    fi
fi

echo "Download complete"
echo "Data location: ${DATA_DIR}"
echo ""
ls -lh
