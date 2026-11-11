#!/usr/bin/env bash
# Submit the pipeline to Slurm as TWO jobs: patching on CPUs, then training on a GPU
# (the GPU job starts only if the patch job finished successfully).
#
# Required environment variables:
#   REPO        path to the Mask2Former repository
#   DATA_ROOT   folder containing images/ and masks/
#   OUTPUT_DIR  where checkpoints are written
#   ACTIVATE    file to source that activates the environment
#               (the activate_mask2former.sh created by install_mask2former.sbatch)
# Optional:
#   STEP        all (default) | patch | train
#   GPU_PARTITION, GPU_GRES (default gpu:1)    CPU_PARTITION
#   PATCH_MEM (default 128G: whole slides are loaded into RAM)   PATCH_CPUS (8)
#   TRAIN_MEM (48G)   PATCH_TIME / TRAIN_TIME (24:00:00)
#   PATCHER_ARGS, TRAIN_ARGS  extra options passed to the two scripts
#
# Example:
#   REPO=/appdata/users/$USER/codes/Mask2Former DATA_ROOT=/data/adamantinoma \
#   OUTPUT_DIR=/data/adamantinoma_out ACTIVATE=/appdata/users/$USER/activate_mask2former.sh \
#   GPU_PARTITION=gpu GPU_GRES=gpu:4g.47gb:1 \
#   PATCHER_ARGS='--label-map "151:1,225:1,255:0"' \
#   bash submit_pipeline.sh
#
# If training needs more than the time limit, run it again with
#   STEP=train TRAIN_ARGS='--resume' ... bash submit_pipeline.sh
set -euo pipefail

: "${REPO:?set REPO}" "${DATA_ROOT:?set DATA_ROOT}" "${OUTPUT_DIR:?set OUTPUT_DIR}" "${ACTIVATE:?set ACTIVATE}"
STEP="${STEP:-all}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/run_training_pipeline.sh"
[[ -f "$ACTIVATE" ]] || { echo "ACTIVATE file not found: $ACTIVATE" >&2; exit 1; }
[[ $STEP == all || $STEP == patch || $STEP == train ]] || { echo "STEP must be all, patch or train" >&2; exit 1; }

q() { printf '%q' "$1"; }

# Body of both jobs (they run under #!/bin/bash, so 'source' and conda work).
job_body() {  # $1 = patch|train
    printf '%s\n' \
        "source $(q "$ACTIVATE")" \
        "export PATCHER_ARGS=$(q "${PATCHER_ARGS:-}") TRAIN_ARGS=$(q "${TRAIN_ARGS:-}")" \
        "cd $(q "$script_dir")" \
        "bash $(q "$runner") $1 $(q "$REPO") $(q "$DATA_ROOT") $(q "$OUTPUT_DIR")"
}

submit() {  # $1 = body ; remaining args = sbatch options
    local body=$1; shift
    printf '#!/bin/bash\nset -eo pipefail\n%s\n' "$body" | sbatch --parsable "$@"
}

dep=()
if [[ $STEP == patch || $STEP == all ]]; then
    opts=(--job-name=m2f_patch --cpus-per-task="${PATCH_CPUS:-8}"
          --mem="${PATCH_MEM:-128G}" -t "${PATCH_TIME:-24:00:00}")
    [[ -n "${CPU_PARTITION:-}" ]] && opts+=(-p "$CPU_PARTITION")
    patch_id=$(submit "$(job_body patch)" "${opts[@]}")
    patch_id=${patch_id%%;*}
    echo "Patch job: $patch_id"
    dep=(--dependency="afterok:$patch_id")
fi

if [[ $STEP == train || $STEP == all ]]; then
    opts=(--job-name=m2f_train --cpus-per-task=8 --mem="${TRAIN_MEM:-48G}"
          -t "${TRAIN_TIME:-24:00:00}" --gres="${GPU_GRES:-gpu:1}")
    [[ -n "${GPU_PARTITION:-}" ]] && opts+=(-p "$GPU_PARTITION")
    train_id=$(submit "$(job_body train)" "${opts[@]}" ${dep[@]+"${dep[@]}"})
    echo "Train job: ${train_id%%;*}"
fi
echo "Follow with: squeue -u \$USER      logs: slurm-<jobid>.out in $(pwd)"
