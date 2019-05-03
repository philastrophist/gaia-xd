#!/bin/bash
set -e
source ~/.bashrc
ENVIRONMENT="gaia-xd"
conda activate "$ENVIRONMENT" || ( conda env create -f environment.yml && conda activate $ENVIRONMENT && pip install -r requirements.txt)
export MKL_THREADING_LAYER="GNU"

# go to directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

python run_ms.py async_20190310174529.vot 20 1e-5 5000 "$DIR"