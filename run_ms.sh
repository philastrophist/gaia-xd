#!/bin/bash
set -e
source ~/.bashrc
ENVIRONMENT="gaia-xd"
source activate "$ENVIRONMENT" || conda env create -f environment.yml
source activate "$ENVIRONMENT"
pip install -r requirements.txt
export MKL_THREADING_LAYER="GNU"

python run_ms.py async_20190310174529.vot 20 1e-5 5000 .