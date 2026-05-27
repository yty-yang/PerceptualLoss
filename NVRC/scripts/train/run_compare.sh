#!/bin/bash
# Run baseline + wd + wd-saliency on Test_50frame for comparison.
# All jobs run sequentially on the same GPU.
# Usage: bash run_compare.sh

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
    "${SCRIPT_DIR}/nvrc_loss.sh" \
    -loss l1_ms-ssim -q

run "wd" \
    "${SCRIPT_DIR}/nvrc_loss.sh" \
    -loss wd -q

run "wd-saliency" \
    "${SCRIPT_DIR}/nvrc_loss.sh" \
    -loss wd-saliency -q -ga 2 -bs 72
