
# Mask2Former_Keratin

Semantic segmentation of whole-slide histopathology images (adamantinoma:
**Background / Stroma / Keratin**) with
[Mask2Former](https://github.com/facebookresearch/Mask2Former) on Detectron2,
built to run on a **Slurm GPU cluster**.

```
slides + masks ─patch─▶ 256 px patches ─train─▶ weights ─inference─▶ label masks ─▶ GeoJSON (QuPath)
```

This README is a complete walkthrough: install the environment, train, predict new
slides, and export annotations. Run the commands in the order shown.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Get the code](#2-get-the-code)
3. [Install the environment](#3-install-the-environment)
4. [Prepare the data](#4-prepare-the-data)
5. [Patch the slides and train](#5-patch-the-slides-and-train)
6. [Get the trained weights](#6-get-the-trained-weights)
7. [Predict new slides](#7-predict-new-slides)
8. [Export GeoJSON for QuPath](#8-export-geojson-for-qupath)
9. [Troubleshooting](#9-troubleshooting)


---

## 1. Repository layout

```
Install_env/  environment and dependency setup
src/          project source code (patching, training, inference, GeoJSON export)
sbatches/     Slurm batch and job-submission scripts
Checkpoints/  trained model checkpoints (large .pth files are tracked with Git LFS)
```

Keep the source files in `src/` and the related job scripts in `sbatches/` together; scripts may rely on their relative paths.

---

## 2. Get the code

Run this on the cluster (login node):

```bash
git clone https://github.com/Lillie-amsterdamumc/Mask2Former_Keratin.git
cd Mask2Former_Keratin
```

The project files are organised in `Install_env/`, `src/`, `sbatches/`, and `Checkpoints/` as shown above.

---

## 3. Install the environment

### 3.1 Find your GPU partition

```bash
sinfo -o "%P %G %D"
```

Look for the row whose `GRES` column contains `gpu`. The value in the `PARTITION`
column is what you pass to `-p` (leave out a trailing `*`). Your GPU request goes in
`--gres`, for example `gpu:1` or a MIG slice such as `gpu:4g.47gb:1`.

### 3.2 Submit the installer

Choose a location with **at least 25 GB free** (the conda environment alone is about
19 GB; do not use a small home directory):

```bash
export BASE=/path/with/plenty/of/space

BASE=$BASE sbatch -p <gpu_partition> --gres=gpu:1 Install_env/install_mask2former.sbatch
```

`sbatch` prints a job number. Follow the log:

```bash
tail -f slurm-<jobid>.out
```

The install is finished when the end of the log shows:

```
FINAL TEST OK (works from any directory), CUDA: True
DONE
```

(A `timm` `FutureWarning` is harmless.) Then check the result:

```bash
grep -c "FINAL TEST OK" slurm-<jobid>.out      # 1 = success
```

The installer creates a conda environment `mask2former` (Python 3.11, PyTorch 2.1.0 +
CUDA 12.1, Detectron2), clones the official Mask2Former into `$BASE/codes/Mask2Former`
and writes `$BASE/activate_mask2former.sh`. More options and troubleshooting:
[`Install_env/README_install.md`](Install_env/README_install.md).

### 3.3 Activate and test

In **every new shell**:

```bash
source $BASE/activate_mask2former.sh
python -c "import mask2former, detectron2; print('ok')"
```

On a login node `torch.cuda.is_available()` is `False`: that is normal, there is no GPU there.

The GeoJSON step also needs OpenCV; if `python -c "import cv2"` fails:

```bash
pip install opencv-python-headless
```

---

## 4. Prepare the data

```
DATA_ROOT/
├── images/<slide>.tif            (or .tiff)
└── masks/<slide>-mask.tif        (also accepted: -mask.tiff, _mask.tif, _mask.tiff, or the image's own name)
```

* Image and mask must have **exactly the same pixel size** (masks are never resized).
* A mask is a single-channel label image; the pixel value is the class id:
  `0 = Background, 1 = Stroma, 2 = Keratin`, `255 = ignore`.
* Slide names may contain spaces.

Check the values of one mask (the path **must be inside the quotes**):

```bash
source $BASE/activate_mask2former.sh
python - <<'EOF'
from PIL import Image; import numpy as np
Image.MAX_IMAGE_PIXELS = None
im = Image.open("/path/to/DATA_ROOT/masks/<one-slide>-mask.tif")
print("mode:", im.mode, "size:", im.size)
print("values:", np.unique(np.array(im)))
EOF
```

* `values: [0 1 2]` → nothing to do.
* Other values (for example `[151 225 255]`) → remap them in step 5 with
  `--label-map`, and decide which value is which class.

More detail: the data-format notes and input handling in `src/`.

---

## 5. Patch the slides and train

Set the paths once (no spaces around `=`):

```bash
export REPO=$BASE/codes/Mask2Former
export DATA_ROOT=/path/to/DATA_ROOT
export OUTPUT_DIR=/path/to/training_output
export ACTIVATE=$BASE/activate_mask2former.sh
export GPU_PARTITION=<gpu_partition>
export GPU_GRES=gpu:1
```

Submit patching and training:

```bash
cd sbatches
bash submit_pipeline.sh
```

This submits **two jobs**: patch extraction on CPUs (`m2f_patch`), then training on a
GPU (`m2f_train`), which starts only if patching succeeded. Two job numbers are printed.

If your masks need remapping, add (example):

```bash
PATCHER_ARGS='--label-map "151:1,225:1,255:0"' bash submit_pipeline.sh
```

### 5.1 Follow it

```bash
squeue -u $USER
tail -f slurm-<patch_jobid>.out
```

When patching is done, check the summary (every `status` should be `ok`):

```bash
cat $DATA_ROOT/patch_summary.csv
```

`no_mask`, `size_mismatch`, `read_error` or `bad_mask_values` mean that slide was skipped;
the others continue. Re-running skips slides that are already done (use `--overwrite`
through `PATCHER_ARGS` to redo them).

In the training log (`slurm-<train_jobid>.out`) you should see:

```
Loaded <N> image/mask pairs
Label check on 300 random masks, values present: [0, 1, 2]
   value   0: ..% of pixels
   ...
Training sees about ... patches = ... passes over the data
...
iter: ... loss: ...
```

If the label check lists values outside `0..2` or `255`, re-run patching with `--label-map`.

### 5.2 Options

| Variable | Meaning |
|---|---|
| `STEP=patch` / `STEP=train` | run only one of the two steps |
| `PATCHER_ARGS='...'` | options of `adamantinoma_patcher.py` (`--label-map`, `--patch-size`, `--stride`, `--overwrite`, `--strict`, ...) |
| `TRAIN_ARGS='--max-iter 60000'` | options of `train_mask2former_all_patches.py` (`--classes`, `--batch-size`, `--lr`, `--num-queries`, `--checkpoint-period`, `--weights`, ...) |
| `PATCH_MEM` (128G), `TRAIN_MEM` (48G), `PATCH_TIME`, `TRAIN_TIME` (24:00:00) | Slurm resources |

By default training runs 30 000 iterations with batch size 4. Check the line
*"passes over the data"* in the log: if it is below 1, consider a larger `--max-iter`.

### 5.3 Continue after the time limit

Checkpoints are written to `OUTPUT_DIR` every 2000 iterations. To continue:

```bash
STEP=train TRAIN_ARGS='--resume' bash submit_pipeline.sh
```

(`--max-iter` must stay the same as in the first run.)

---

## 6. Get the trained weights

Training writes `model_XXXXXXX.pth` files and `model_final.pth` to `OUTPUT_DIR`:

```bash
ls -lh $OUTPUT_DIR/*.pth
```

Each file should be several hundred MB. A file of only a few hundred **bytes** is
not a model (see the Git LFS entry in [Troubleshooting](#9-troubleshooting)).

To share weights, make a smaller weights-only file:

```bash
python - <<'EOF'
import torch
ck = torch.load("/path/to/training_output/model_final.pth", map_location="cpu")
torch.save({"model": ck["model"]}, "/path/to/model_weights_only.pth")
EOF
```

This repository tracks `.pth` checkpoints with Git LFS. The shared checkpoint is stored
at `Checkpoints/model_final.pth`. 

To check which architecture a weights file has (it must match `--classes` / `--num-queries`):

```bash
python - <<'EOF'
import torch
ck = torch.load("/path/to/weights.pth", map_location="cpu")
sd = ck["model"] if "model" in ck else ck
print("classes:", sd["sem_seg_head.predictor.class_embed.weight"].shape[0] - 1,
      " queries:", sd["sem_seg_head.predictor.query_feat.weight"].shape[0])
EOF
```

---

## 7. Predict new slides

Put the slides (`.tif` / `.tiff`) **directly** inside one folder. Then:

```bash
export IMAGES_DIR=/path/to/new/slides
export WEIGHTS=/path/to/model_final.pth          # or a weights-only file
export OUTPUT_DIR=/path/to/predictions

cd sbatches
bash submit_inference.sh
```

(`TRAIN_DIR=/path/to/training_output` can replace `WEIGHTS`: it then uses `model_final.pth`
from that folder.) Follow it with `squeue -u $USER` and `tail -f slurm-<jobid>.out`.

How it works: every slide is cut into overlapping 256 px patches (stride 128). The model is
run on **all** patches, each enlarged to 512 px as in training. Class probabilities are averaged
where patches overlap, and one full-size mask per slide is written.

Outputs in `OUTPUT_DIR`:

```
predictions/<slide>-pred.tif     label mask (value = class id), same size as the slide
overlays/<slide>-overlay.png     small colour preview: look at it before trusting the result
inference_summary.csv            predicted area per class for every slide (+ IoU / Dice with masks)
```

Options:

| Variable | Meaning |
|---|---|
| `MASKS_DIR=/path/to/masks` | ground-truth masks: IoU and Dice are computed |
| `INFER_ARGS='--overwrite'` | redo slides that already have a prediction |
| `INFER_ARGS='--infer-size 256'` | model input size (default 512); compare on a validation slide |
| `INFER_ARGS='--skip-blank'` | do not run the model on near-white patches (faster, but pale tissue can turn into empty squares) |
| `INFER_ARGS='--classes "A,B,C" --num-queries 100'` | must match the weights |
| `MEM`, `TIME`, `CPUS` | Slurm resources (large slides may need `MEM=128G`) |

Tip: try a folder with 2–3 slides first and inspect the overlays; then run everything.

---

## 8. Export GeoJSON for QuPath

On the cluster (CPU only, quick):

```bash
source $BASE/activate_mask2former.sh
python src/masks_to_geojson.py \
  --pred-dir $OUTPUT_DIR/predictions \
  --output-dir $OUTPUT_DIR/geojson
```

One `<slide>.geojson` per slide. Every connected region of a class becomes a polygon,
holes are kept; Stroma and Keratin are exported (Background is not). Coordinates are
slide pixels. In QuPath open the same slide and use
`File > Import objects from file...`.

Useful options:

| Option | Meaning |
|---|---|
| `--export Keratin` | export only this class |
| `--min-area 500` | drop regions smaller than 500 pixels (default 200) |
| `--min-hole-area 500` | fill holes smaller than this |
| `--smooth 5` | smooth edges and remove specks (morphological open and close) |
| `--simplify 2` | fewer polygon points, smaller files |

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `UnpicklingError: invalid load key, 'v'` when loading weights | The `.pth` is a **Git LFS pointer** (a tiny text file), not the model; see below. |

**Weights that are a Git LFS pointer:** if you downloaded a repository as a ZIP, or cloned it
without Git LFS, large files are replaced by pointer files of about 130 bytes that start with
`version https://git-lfs...`. Check with `ls -l` and `head -c 200 file.pth`.


