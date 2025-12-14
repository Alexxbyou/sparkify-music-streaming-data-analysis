#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlmain312
python Production/DataProcessing.py
python Production/Modelling.py
conda deactivate
