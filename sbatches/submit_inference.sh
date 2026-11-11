#!/usr/bin/env bash
# Submit whole-slide inference as ONE GPU job.
#
# Required environment variables:
#   REPO        path to the Mask2Former repository
#   IMAGES_DIR  folder with the slides (.tif / .tiff) to predict
#   OUTPUT_DIR  where predictions/, overlays/ and inference_summary.csv are written
#   ACTIVATE    activate_mask2former.sh
#   WEIGHTS     checkpoint .pth   (or TRAIN_DIR = training output dir, then model_final.pth is used)
# Optional:
#   MASKS_DIR   ground-truth masks: IoU/Dice are computed
#   GPU_PARTITION, GPU_GRES (default gpu:1)   MEM (64G)   TIME (12:00:00)   CPUS (8)
#   INFER_ARGS  extra options, e.g. '--label-map "151:1,225:1,255:0" --infer-size 256'
#
# Example:
#   REPO=/appdata/users/$USER/codes/Mask2Former \
#   IMAGES_DIR=/net/beegfs/users/$USER/adamantinoma_keratin_test/Test/images \
#   MASKS_DIR=/net/beegfs/users/$USER/adamantinoma_keratin_test/Test/masks \
#   OUTPUT_DIR=/net/beegfs/users/$USER/m2f_predictions \
#   TRAIN_DIR=/net/beegfs/users/$USER/m2f_out \
#   ACTIVATE=/appdata/users/$USER/activate_mask2former.sh \
#   GPU_PARTITION=gpu GPU_GRES=gpu:4g.47gb:1 bash submit_inference.sh
set -euo pipefail

: "${REPO:?set REPO}" "${IMAGES_DIR:?set IMAGES_DIR}" "${OUTPUT_DIR:?set OUTPUT_DIR}" "${ACTIVATE:?set ACTIVATE}"
[[ -n "${WEIGHTS:-}" || -n "${TRAIN_DIR:-}" ]] || { echo "set WEIGHTS or TRAIN_DIR" >&2; exit 1; }
[[ -f "$ACTIVATE" ]] || { echo "ACTIVATE file not found: $ACTIVATE" >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

q() { printf '%q' "$1"; }
cmd="python -u $(q "$script_dir/infer_mask2former_wsi.py") --repo $(q "$REPO") --images-dir $(q "$IMAGES_DIR") --output-dir $(q "$OUTPUT_DIR")"
[[ -n "${WEIGHTS:-}" ]]   && cmd+=" --weights $(q "$WEIGHTS")"
[[ -n "${TRAIN_DIR:-}" ]] && cmd+=" --train-dir $(q "$TRAIN_DIR")"
[[ -n "${MASKS_DIR:-}" ]] && cmd+=" --masks-dir $(q "$MASKS_DIR")"
cmd+=" ${INFER_ARGS:-}"

opts=(--parsable --job-name=m2f_infer --cpus-per-task="${CPUS:-8}" --mem="${MEM:-64G}"
      -t "${TIME:-12:00:00}" --gres="${GPU_GRES:-gpu:1}")
[[ -n "${GPU_PARTITION:-}" ]] && opts+=(-p "$GPU_PARTITION")

job=$(printf '#!/bin/bash\nset -eo pipefail\nsource %s\n%s\n' "$(q "$ACTIVATE")" "$cmd" | sbatch "${opts[@]}")
echo "Inference job: ${job%%;*}"
echo "Follow with: squeue -u \$USER      log: slurm-${job%%;*}.out in $(pwd)"
