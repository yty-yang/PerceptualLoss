#!/bin/bash
# Run baseline + wd + wd-saliency on Test_50frame for comparison.
# All jobs run sequentially on the same GPU.
# Usage: bash run_compare.sh [GPU_ID]

GPU_ID=${1:-0}
VID=Test_50frame
LAMB=10.0
SCALE=s
LR_S1=2e-3
LR_S2=1e-4
GRAD_ACCUM=1
BATCH_SIZE=144

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    local name=$1
    shift
    echo "========================================"
    echo "Starting: ${name}"
    echo "========================================"
    bash "$@"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "FAILED: ${name} (exit code ${exit_code})"
        exit $exit_code
    fi
    echo "Done: ${name}"
}

run "baseline" \
    "${SCRIPT_DIR}/overfitting_uvg_nvrc.sh" \
    ${GPU_ID} ${VID} ${LAMB} ${SCALE} ${LR_S1} ${LR_S2} ${GRAD_ACCUM} ${BATCH_SIZE}

run "wd" \
    "${SCRIPT_DIR}/nvrc_loss.sh" \
    -gpu ${GPU_ID} -vid ${VID} -lamb ${LAMB} -scale ${SCALE} \
    -lr1 ${LR_S1} -lr2 ${LR_S2} -ga ${GRAD_ACCUM} -bs ${BATCH_SIZE} \
    -loss wd

run "wd-saliency" \
    "${SCRIPT_DIR}/nvrc_loss.sh" \
    -gpu ${GPU_ID} -vid ${VID} -lamb ${LAMB} -scale ${SCALE} \
    -lr1 ${LR_S1} -lr2 ${LR_S2} -ga ${GRAD_ACCUM} -bs ${BATCH_SIZE} \
    -loss wd-saliency
