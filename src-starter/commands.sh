#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="out_baseline"

cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 \
  --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o "${OUT_DIR}"

cs_python run.py --name "${OUT_DIR}" --case baseline
