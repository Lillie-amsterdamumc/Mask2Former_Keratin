#!/usr/bin/env bash
# Run the pipeline steps inside an ALREADY ACTIVATED environment.
#
#   bash run_training_pipeline.sh [patch|train|all] MASK2FORMER_REPO DATA_ROOT OUTPUT_DIR
#
# The first argument is optional (default: all). Extra options for the scripts can be
# passed through environment variables:
#   PATCHER_ARGS='--label-map "151:1,225:1,255:0"'   (see adamantinoma_patcher.py --help)
#   TRAIN_ARGS='--max-iter 60000 --resume'           (see train_mask2former_all_patches.py --help)
#
# 'patch' needs only CPUs. 'train' needs a GPU. For a cluster use submit_pipeline.sh,
# which runs the two steps as separate jobs.
set -euo pipefail

step=all
if [[ ${1:-} == patch || ${1:-} == train || ${1:-} == all ]]; then
    step=$1
    shift
fi

if [[ $# -ne 3 ]]; then
    echo "Usage: bash run_training_pipeline.sh [patch|train|all] MASK2FORMER_REPO DATA_ROOT OUTPUT_DIR" >&2
    exit 2
fi

repo=$1
data_root=$2
output_dir=$3
config="$repo/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml"
if [[ ! -f "$config" ]]; then
    printf 'Mask2Former config not found: %s\n' "$config" >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patcher="$script_dir/adamantinoma_patcher.py"
trainer="$script_dir/train_mask2former_all_patches.py"
for script in "$patcher" "$trainer"; do
    if [[ ! -f "$script" ]]; then
        printf 'Required file missing: %s\n' "$script" >&2
        exit 1
    fi
done

if [[ $step == patch || $step == all ]]; then
    echo 'Step 1: Generate paired image and mask patches (CPU)'
    # shellcheck disable=SC2086
    eval "python -u \"\$patcher\" --data-root \"\$data_root\" ${PATCHER_ARGS:-}"
fi

if [[ $step == train || $step == all ]]; then
    python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "Training needs a GPU: run inside a GPU allocation.")'
    echo 'Step 2: Train Mask2Former on paired patches (GPU)'
    # shellcheck disable=SC2086
    eval "python -u \"\$trainer\" --repo \"\$repo\" --data-root \"\$data_root\" --output-dir \"\$output_dir\" ${TRAIN_ARGS:-}"
fi
