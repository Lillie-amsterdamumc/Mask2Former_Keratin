#!/usr/bin/env python3
"""Whole-slide inference with a trained Mask2Former (semantic segmentation).

For every slide in --images-dir the script
  1. cuts it into overlapping patches (same size/stride logic as the training patches),
  2. predicts each patch and adds up the class probabilities where patches overlap,
  3. writes one full-size label mask  <output-dir>/predictions/<slide>-pred.tif
     (pixel value = class id, same size as the slide),
  4. writes a small colour overlay    <output-dir>/overlays/<slide>-overlay.png,
  5. if --masks-dir is given: compares with the ground-truth masks and reports
     IoU / Dice per class (per slide and over all slides).

A summary is written to <output-dir>/inference_summary.csv.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # whole-slide TIFFs exceed Pillow's default limit

COLORS = np.array([[255, 255, 255], [0, 200, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]],
                  dtype=np.uint8)
MASK_SUFFIXES = ("-mask", "_mask", "")
TIFF_EXT = (".tif", ".tiff")


# ----------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------

def read_image_rgb(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def read_mask(path):
    """Single-channel label image as uint8 (palette images: indices, not colours)."""
    with Image.open(path) as img:
        if img.mode not in ("L", "P", "1", "I;16", "I;16B", "I;16L", "I"):
            raise ValueError(f"mask mode {img.mode!r} is not a single-channel label image")
        arr = np.array(img)
    if arr.max() > 255:
        raise ValueError(f"mask has values up to {int(arr.max())}; expected 8-bit labels")
    return arr.astype(np.uint8)


def find_mask(image_path, mask_dir):
    for suffix in MASK_SUFFIXES:
        for ext in TIFF_EXT:
            cand = mask_dir / f"{image_path.stem}{suffix}{ext}"
            if cand.exists() and cand.resolve() != image_path.resolve():
                return cand
    return None


def parse_label_map(text):
    lut = np.arange(256, dtype=np.uint8)
    for pair in filter(None, (text or "").split(",")):
        src, dst = (int(v) for v in pair.split(":"))
        lut[src] = dst
    return lut


# ----------------------------------------------------------------------------------
# tiling + stitching (no torch / detectron2 needed)
# ----------------------------------------------------------------------------------

def tile_positions(length, patch, stride):
    """Start positions covering the whole length; the last tile touches the border."""
    pos = list(range(0, length - patch + 1, stride))
    if not pos or pos[-1] + patch < length:
        pos.append(max(length - patch, 0))
    return pos


def is_blank(tile, white_level, blank_fraction):
    return (tile.min(axis=2) >= white_level).mean() >= blank_fraction


def predict_slide(image, predict_fn, patch, stride, batch_size, num_classes,
                  skip_blank=True, white_level=220, blank_fraction=0.95, log=print):
    """Return a (H, W) uint8 label map.

    predict_fn(batch) gets an array (B, patch, patch, 3) uint8 RGB and must return
    class probabilities (B, num_classes, patch, patch).
    """
    H, W = image.shape[:2]
    pad_h, pad_w = max(patch - H, 0), max(patch - W, 0)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
    Hp, Wp = image.shape[:2]

    acc = np.zeros((num_classes, Hp, Wp), dtype=np.float16)  # summed probabilities
    coords = [(y, x) for y in tile_positions(Hp, patch, stride)
              for x in tile_positions(Wp, patch, stride)]

    todo, n_blank = [], 0
    for y, x in coords:
        if skip_blank and is_blank(image[y:y + patch, x:x + patch], white_level, blank_fraction):
            acc[0, y:y + patch, x:x + patch] += 1.0   # empty glass counts as class 0
            n_blank += 1
        else:
            todo.append((y, x))
    log(f"   {len(coords)} tiles: {len(todo)} predicted, {n_blank} blank (set to class 0)")

    next_report = 0.1
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        batch = np.stack([image[y:y + patch, x:x + patch] for y, x in chunk])
        probs = np.asarray(predict_fn(batch), dtype=np.float32)
        for (y, x), p in zip(chunk, probs):
            acc[:, y:y + patch, x:x + patch] += p.astype(np.float16)
        if (i + len(chunk)) / len(todo) >= next_report:
            log(f"   {100 * (i + len(chunk)) // len(todo)}%")
            next_report += 0.1

    pred = acc.argmax(axis=0).astype(np.uint8)  # equal overlap counts per pixel: no division needed
    return pred[:H, :W]


# ----------------------------------------------------------------------------------
# metrics, overlay, saving
# ----------------------------------------------------------------------------------

def confusion(pred, gt, n):
    """Rows = ground truth, columns = prediction; pixels with gt >= n (e.g. 255) are ignored."""
    valid = gt < n
    idx = gt[valid].astype(np.int64) * n + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def scores(conf):
    tp = np.diag(conf).astype(float)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        dice = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return iou, dice


def make_overlay(image, pred, max_side, alpha=0.45):
    H, W = pred.shape
    scale = min(1.0, max_side / max(H, W))
    size = (max(1, int(W * scale)), max(1, int(H * scale)))
    img = np.array(Image.fromarray(image).resize(size, Image.BILINEAR)).astype(np.float32)
    lab = np.array(Image.fromarray(pred).resize(size, Image.NEAREST))
    out = img.copy()
    for c in range(1, len(COLORS)):          # class 0 (background) stays untouched
        m = lab == c
        out[m] = (1 - alpha) * img[m] + alpha * COLORS[c]
    return Image.fromarray(out.astype(np.uint8))


def save_mask(pred, path):
    try:
        Image.fromarray(pred).save(path, compression="tiff_adobe_deflate")
        return path
    except Exception as exc:  # e.g. libtiff without deflate
        fallback = path.with_suffix(".png")
        print(f"   (TIFF save failed: {exc}; writing PNG instead)")
        Image.fromarray(pred).save(fallback)
        return fallback


# ----------------------------------------------------------------------------------
# one slide
# ----------------------------------------------------------------------------------

def run_slide(image_path, predict_fn, args, class_names, lut, pred_dir, overlay_dir):
    n = len(class_names)
    slide = image_path.stem
    row = {"slide": slide, "status": "", "seconds": ""}
    conf = None
    out_path = pred_dir / f"{slide}-pred.tif"

    print("\n" + "=" * 70 + f"\nSlide: {slide}\n" + "=" * 70)
    if not args.overwrite and (out_path.exists() or out_path.with_suffix(".png").exists()) \
            and not args.masks_dir:
        print("   prediction exists (use --overwrite to redo). Skipping.")
        row["status"] = "already_done"
        return row, conf

    t0 = time.time()
    try:
        image = read_image_rgb(image_path)
    except Exception as exc:
        print(f"WARNING: cannot read {slide}: {exc}")
        row.update(status="read_error", detail=str(exc))
        return row, conf
    print(f"   size {image.shape[1]} x {image.shape[0]}")

    pred = predict_slide(image, predict_fn, args.patch_size, args.stride, args.batch_size, n,
                         skip_blank=not args.keep_blank, white_level=args.white_level,
                         blank_fraction=args.blank_fraction)
    saved = save_mask(pred, out_path)
    print(f"   saved {saved}")
    if not args.no_overlay:
        make_overlay(image, pred, args.overlay_max_side).save(overlay_dir / f"{slide}-overlay.png")

    frac = np.bincount(pred.reshape(-1), minlength=n)[:n] / pred.size
    for c, name in enumerate(class_names):
        row[f"frac_{name}"] = round(float(frac[c]), 5)
    print("   predicted area: " + "  ".join(f"{nm} {100 * f:.1f}%" for nm, f in zip(class_names, frac)))

    if args.masks_dir:
        mask_path = find_mask(image_path, args.masks_dir)
        if mask_path is None:
            print("   (no ground-truth mask found: no metrics for this slide)")
        else:
            try:
                gt = lut[read_mask(mask_path)]
                if gt.shape != pred.shape:
                    raise ValueError(f"mask size {gt.shape[::-1]} != image size {pred.shape[::-1]}")
                conf = confusion(pred, gt, n)
                iou, dice = scores(conf)
                for c, name in enumerate(class_names):
                    row[f"iou_{name}"] = "" if np.isnan(iou[c]) else round(float(iou[c]), 4)
                    row[f"dice_{name}"] = "" if np.isnan(dice[c]) else round(float(dice[c]), 4)
                print("   IoU:  " + "  ".join(f"{nm} {v:.3f}" for nm, v in zip(class_names, iou)))
                print("   Dice: " + "  ".join(f"{nm} {v:.3f}" for nm, v in zip(class_names, dice)))
            except Exception as exc:
                print(f"   WARNING: metrics skipped: {exc}")
                row["detail"] = f"metrics skipped: {exc}"
    row["status"] = "ok"
    row["seconds"] = round(time.time() - t0, 1)
    return row, conf


# ----------------------------------------------------------------------------------
# model (detectron2 + mask2former are imported only here)
# ----------------------------------------------------------------------------------

def resolve_weights(weights, train_dir):
    if weights:
        p = Path(weights)
        if not p.is_file():
            raise SystemExit(f"Weights file not found: {p}")
        return p
    final = train_dir / "model_final.pth"
    if final.is_file():
        return final
    ckpts = sorted(train_dir.glob("model_*.pth"))
    if ckpts:
        return ckpts[-1]
    raise SystemExit(f"No weights given and no model_*.pth in {train_dir}")


def build_predictor(args, config_file, class_names, weights):
    import torch
    import torch.nn.functional as F
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(class_names)
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = args.num_queries
    cfg.MODEL.WEIGHTS = ""          # everything comes from our checkpoint
    cfg.MODEL.DEVICE = "cuda"

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(str(weights))
    flip_to_bgr = cfg.INPUT.FORMAT == "BGR"
    P, S = args.patch_size, args.infer_size

    def predict(batch):
        inputs = []
        for tile in batch:
            t = torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1).float()  # RGB
            if flip_to_bgr:
                t = t.flip(0)
            if S != P:   # training saw the patches enlarged, so enlarge them here too
                t = F.interpolate(t[None], size=(S, S), mode="bilinear", align_corners=False)[0]
            inputs.append({"image": t, "height": P, "width": P})
        with torch.no_grad():
            outs = model(inputs)
        probs = torch.stack([o["sem_seg"] for o in outs]).float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probs.cpu().numpy()

    return predict


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Whole-slide inference with Mask2Former.")
    ap.add_argument("--repo", type=Path, required=True, help="Mask2Former repository")
    ap.add_argument("--images-dir", type=Path, required=True, help="folder with slide .tif/.tiff files")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--weights", default="", help="checkpoint .pth (default: model_final.pth of --train-dir)")
    ap.add_argument("--train-dir", type=Path, default=None,
                    help="training OUTPUT_DIR; used to find the weights when --weights is not given")
    ap.add_argument("--masks-dir", type=Path, default=None,
                    help="ground-truth masks (optional): enables IoU/Dice")
    ap.add_argument("--label-map", default="", help='remap ground-truth values, e.g. "151:1,225:1,255:0"')
    ap.add_argument("--classes", default="Background,Stroma,Keratin")
    ap.add_argument("--num-queries", type=int, default=100, help="must equal the value used in training")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--infer-size", type=int, default=512,
                    help="patches are enlarged to this size before the model (training used enlarged "
                         "patches); set equal to --patch-size to disable")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--keep-blank", action="store_true", help="also run the model on white patches")
    ap.add_argument("--white-level", type=int, default=220)
    ap.add_argument("--blank-fraction", type=float, default=0.95)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--overlay-max-side", type=int, default=4000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not 2 <= len(class_names) <= len(COLORS):
        ap.error(f"--classes needs between 2 and {len(COLORS)} names")
    if args.stride < 1 or args.patch_size < 1:
        ap.error("patch-size and stride must be positive")
    if not args.images_dir.is_dir():
        ap.error(f"images dir not found: {args.images_dir}")
    if args.masks_dir and not args.masks_dir.is_dir():
        ap.error(f"masks dir not found: {args.masks_dir}")

    repo = args.repo.resolve()
    config = repo / "configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml"
    if not config.is_file():
        ap.error(f"Mask2Former config not found: {config}")
    sys.path.insert(0, str(repo))

    images = sorted(list(args.images_dir.glob("*.tiff")) + list(args.images_dir.glob("*.tif")))
    if not images:
        sys.exit(f"No .tif/.tiff slides directly inside {args.images_dir}")

    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run inference on a GPU node.")
    train_dir = args.train_dir or Path(".")
    weights = resolve_weights(args.weights, train_dir)

    print("=" * 70 + "\nMask2Former WHOLE-SLIDE INFERENCE\n" + "=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Weights: {weights}\nSlides: {len(images)} in {args.images_dir}")
    print(f"Classes: {class_names}   patch {args.patch_size}, stride {args.stride}, "
          f"model input {args.infer_size}px")
    print(f"Ground truth: {args.masks_dir or 'none (no metrics)'}")

    out = args.output_dir
    pred_dir, overlay_dir = out / "predictions", out / "overlays"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    predict_fn = build_predictor(args, config, class_names, weights)
    lut = parse_label_map(args.label_map)

    rows, total = [], None
    for path in images:
        row, conf = run_slide(path, predict_fn, args, class_names, lut, pred_dir, overlay_dir)
        rows.append(row)
        if conf is not None:
            total = conf if total is None else total + conf

    n = len(class_names)
    if total is not None:
        iou, dice = scores(total)
        all_row = {"slide": "ALL_SLIDES (pixels pooled)", "status": "ok"}
        for c, name in enumerate(class_names):
            all_row[f"iou_{name}"] = "" if np.isnan(iou[c]) else round(float(iou[c]), 4)
            all_row[f"dice_{name}"] = "" if np.isnan(dice[c]) else round(float(dice[c]), 4)
        rows.append(all_row)
        print("\n" + "=" * 70 + "\nOVERALL (all slides pooled)")
        for c, name in enumerate(class_names):
            print(f"   {name:<12} IoU {iou[c]:.3f}   Dice {dice[c]:.3f}")
        print("   confusion matrix (rows = truth, columns = prediction):")
        print("   " + str(total).replace("\n", "\n   "))

    fields = ["slide", "status", "seconds", "detail"]
    fields += [f"frac_{c}" for c in class_names]
    fields += [f"iou_{c}" for c in class_names] + [f"dice_{c}" for c in class_names]
    summary = out / "inference_summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["status"] not in ("ok", "already_done")]
    print(f"\nDone. {len(rows) - len(bad)} ok, {len(bad)} problems. Summary: {summary}")
    if bad and len(bad) == len([r for r in rows if r["slide"] and "ALL" not in r["slide"]]):
        sys.exit(1)


if __name__ == "__main__":
    main()
