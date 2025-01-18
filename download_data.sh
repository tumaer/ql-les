#!/bin/bash
# Download datasets from Zenodo
# Usage:
#   bash download_data.sh <dataset_name> <output_dir>
# Usage:
#   bash download_data.sh all datasets/

declare -A datasets
datasets["tgv_3d"]="3D_TGV_8000_10kevery100.zip"
datasets["tgv_2d"]="2D_TGV_2500_10kevery100.zip"

if [ $# -ne 2 ]; then
    echo "Usage: bash download_data.sh <dataset_name> <output_dir>"
    exit 1
fi

DATASET_NAME="$1"
OUTPUT_DIR="$2"
ZENODO_PREFIX="https://zenodo.org/records/10491868/files/"

# Check if there is a trailing slash in $OUTPUT_DIR and remove it
if [[ $OUTPUT_DIR == */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR%/}"
    echo "Output directory: ${OUTPUT_DIR}"
fi

# Create output directory if it doesn't exist
if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# Download the data
if [ "${DATASET_NAME}" == "all" ]; then
    echo "Downloading all datasets"
    for key in ${!datasets[@]}; do
        echo "Downloading ${key}"
        wget ${ZENODO_PREFIX}${datasets[${key}]} -P "${OUTPUT_DIR}/"
        ZIP_PATH=${OUTPUT_DIR}/${datasets[${key}]}
        python3 -c "import zipfile; zipfile.ZipFile('$ZIP_PATH', 'r').extractall('$OUTPUT_DIR')"
        rm ${ZIP_PATH}
    done
else
    echo "Downloading ${DATASET_NAME}"
    wget ${ZENODO_PREFIX}${datasets[${DATASET_NAME}]} -P "${OUTPUT_DIR}/"
    ZIP_PATH=${OUTPUT_DIR}/${datasets[${DATASET_NAME}]}
    python3 -c "import zipfile; zipfile.ZipFile('$ZIP_PATH', 'r').extractall('$OUTPUT_DIR')"
    rm ${ZIP_PATH}
fi
