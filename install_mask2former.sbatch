#!/bin/bash
# =============================================================================
# Install the official Mask2Former (+ Detectron2) in a user-owned conda env.
# Works for any user on a Slurm cluster: no hard-coded usernames or paths.
#
# USAGE (choose YOUR cluster's GPU partition / GPU request):
#   sbatch -p <gpu_partition> --gres=gpu:1 install_mask2former.sbatch
#
# OPTIONAL overrides (environment variables, all have defaults):
#   BASE         where conda envs/packages live      (default: $HOME)
#   CODE_PARENT  where Mask2Former is cloned         (default: $BASE/codes)
#   ENV_NAME     conda environment name              (default: mask2former)
#   CONDA_MODULE name of a conda module to load, if your cluster has one
#                (e.g. CONDA_MODULE=anaconda3). If empty and conda is missing,
#                Miniforge is installed under $BASE/miniforge3.
# Example:
#   BASE=/scratch/$USER CONDA_MODULE=anaconda3 \
#     sbatch -p gpu --gres=gpu:1 install_mask2former.sbatch
#
# Log: slurm-<jobid>.out in the directory you submitted from.
# Success = last lines print "Mask2Former imports passed" and "DONE".
# =============================================================================
#SBATCH --job-name=install_mask2former
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00

set -euo pipefail

ENV_NAME="${ENV_NAME:-mask2former}"
BASE="${BASE:-$HOME}"
CODE_PARENT="${CODE_PARENT:-$BASE/codes}"
CONDA_MODULE="${CONDA_MODULE:-}"

echo "Job started on $(hostname) at $(date)"
echo "User: $USER | BASE=$BASE | CODE_PARENT=$CODE_PARENT | ENV_NAME=$ENV_NAME"

# ---- sanity checks (fail early with a clear message) -------------------------
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: need Linux x86_64, got $(uname -s) $(uname -m)" >&2; exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no GPU visible on this node. Submit with your cluster's GPU" >&2
  echo "       partition, e.g.: sbatch -p <gpu_partition> --gres=gpu:1 $0" >&2
  exit 1
fi
nvidia-smi
if ! curl -fsI -m 20 https://github.com >/dev/null 2>&1; then
  echo "ERROR: no internet from this node (github.com unreachable)." >&2
  echo "       Do the downloads on a login node, or ask your admins for a proxy." >&2
  exit 1
fi

mkdir -p "$BASE" "$CODE_PARENT"

# ---- conda ---------------------------------------------------------------------
if [[ -n "$CONDA_MODULE" ]]; then
  module load "$CONDA_MODULE"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found -> installing Miniforge under $BASE/miniforge3"
  curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o "$BASE/Miniforge3-Linux-x86_64.sh"
  bash "$BASE/Miniforge3-Linux-x86_64.sh" -b -p "$BASE/miniforge3"
  source "$BASE/miniforge3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda --version

export CONDA_ENVS_PATH="$BASE/.conda/envs"
export CONDA_PKGS_DIRS="$BASE/.conda/pkgs"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

# ---- env -------------------------------------------------------------------------
if ! conda env list | grep -q "/${ENV_NAME}\$"; then
  conda create -y -n "$ENV_NAME" --override-channels \
    -c nvidia/label/cuda-12.1.1 -c conda-forge \
    python=3.11 pip git gcc_linux-64=11 gxx_linux-64=11 \
    cuda-toolkit=12.1.1 ninja=1.11.1 'libstdcxx-ng>=13'
fi
set +u            # conda's activate scripts touch unset variables
conda activate "$ENV_NAME"
set -u

# ---- code --------------------------------------------------------------------------
cd "$CODE_PARENT"
if [[ ! -d Mask2Former ]]; then
  git clone https://github.com/facebookresearch/Mask2Former.git
fi
cd Mask2Former
echo "Mask2Former commit: $(git rev-parse HEAD)"

# ---- PyTorch -----------------------------------------------------------------------
python -m pip install 'setuptools==69.5.1' 'wheel==0.43.0' 'numpy==1.26.4'
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0 torchvision==0.16.0

export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS="${MAX_JOBS:-2}"

"$CC" --version
"$CXX" --version
"$CUDA_HOME/bin/nvcc" --version
python - <<'PY'
import torch, torchvision
print("PyTorch", torch.__version__, "torchvision", torchvision.__version__)
print("PyTorch CUDA", torch.version.cuda)
assert torch.version.cuda == "12.1"
assert torch.cuda.is_available(), "No GPU visible to PyTorch"
print("GPU", torch.cuda.get_device_name(0))
print("Compute capability", torch.cuda.get_device_capability(0))
print(torch.ones(1, device="cuda"))
PY

# ---- dependencies + extensions -------------------------------------------------------
cat > mask2former-install-constraints.txt <<'CONSTRAINTS'
torch==2.1.0+cu121
torchvision==0.16.0+cu121
numpy==1.26.4
Pillow==9.5.0
opencv-python-headless==4.8.1.78
CONSTRAINTS

python -m pip install -c mask2former-install-constraints.txt \
  opencv-python-headless -r requirements.txt

python -m pip install --no-build-isolation --no-cache-dir \
  -c mask2former-install-constraints.txt \
  'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@b599f139756bd3646a26a909caf86a1a159e53a7'

python -m pip install --no-build-isolation --no-cache-dir \
  --no-deps ./mask2former/modeling/pixel_decoder/ops

python -m pip check
python -c 'import torch; from detectron2 import _C; import MultiScaleDeformableAttention; import mask2former; print("Mask2Former imports passed")'

# ---- make the install usable from ANY directory ---------------------------------------
# mask2former is a folder inside the repo (no setup.py), so PYTHONPATH must point to it.
# A conda activate.d hook sets it automatically every time the env is activated.
REPO_DIR="$(pwd)"
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"

cat > "$ACT_DIR/mask2former.sh" <<'HOOK'
export M2F_OLD_PYTHONPATH="${PYTHONPATH:-}"
export M2F_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="__REPO_DIR__${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOOK
sed -i "s|__REPO_DIR__|$REPO_DIR|" "$ACT_DIR/mask2former.sh"

cat > "$DEACT_DIR/mask2former.sh" <<'HOOK'
export PYTHONPATH="${M2F_OLD_PYTHONPATH:-}"
export LD_LIBRARY_PATH="${M2F_OLD_LD_LIBRARY_PATH:-}"
unset M2F_OLD_PYTHONPATH M2F_OLD_LD_LIBRARY_PATH
HOOK

# One file the user sources in any new shell or inside any Slurm job.
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
ACTIVATE_FILE="$BASE/activate_mask2former.sh"
{
  echo "# Usage:  source $ACTIVATE_FILE"
  [[ -n "$CONDA_MODULE" ]] && echo "module load $CONDA_MODULE"
  echo "source \"$CONDA_SH\""
  echo "export CONDA_ENVS_PATH=\"$CONDA_ENVS_PATH\""
  echo "conda activate \"$ENV_NAME\""
} > "$ACTIVATE_FILE"

# Record exactly what was installed (reproducibility)
{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "mask2former_commit: $(git rev-parse HEAD)"
  echo "repo_dir: $REPO_DIR"
  python -m pip freeze
} > "$BASE/mask2former_install_info.txt"

# Final test: re-activate (so the hook runs) and import from a NEUTRAL directory
set +u
conda deactivate
source "$ACTIVATE_FILE"
set -u
cd /tmp
python -c 'import mask2former, detectron2, torch; from mask2former import add_maskformer2_config; print("FINAL TEST OK (works from any directory), CUDA:", torch.cuda.is_available())'

echo "DONE at $(date)"
echo "================================================================"
echo "To use in any new shell or Slurm job, just run:"
echo "    source $ACTIVATE_FILE"
echo "Mask2Former code: $REPO_DIR"
echo "Install record:   $BASE/mask2former_install_info.txt"
echo "================================================================"
