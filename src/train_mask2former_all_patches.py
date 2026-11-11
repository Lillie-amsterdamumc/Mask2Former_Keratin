#!/usr/bin/env python3
"""Train Mask2Former (semantic segmentation) on paired image/mask patches.

Expects the layout produced by adamantinoma_patcher.py:

    DATA_ROOT/images/patches/<slide>/*.png
    DATA_ROOT/masks/patches/<slide>/*.png      (same names; pixel values = class ids,
                                               255 = ignore)
"""

import argparse
import copy
import itertools
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DATASET_NAME = "wsi_train"
STUFF_COLORS = [(255, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]


# ----------------------------------------------------------------------------------
# dataset (no detectron2 needed for these functions)
# ----------------------------------------------------------------------------------

def load_dataset(data_root):
    """Detectron2 semantic-segmentation records for every image/mask patch pair."""
    image_root = data_root / "images" / "patches"
    mask_root = data_root / "masks" / "patches"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(
            f"Patches not found. Run adamantinoma_patcher.py first "
            f"(looked in {image_root} and {mask_root})."
        )

    records = []
    size = None  # all patches come from the same patcher run: read the size once
    for slide_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        mask_dir = mask_root / slide_dir.name
        if not mask_dir.is_dir():
            continue
        mask_names = {m.name for m in mask_dir.glob("*.png")}
        for img_path in sorted(slide_dir.glob("*.png")):
            if img_path.name not in mask_names:
                continue
            if size is None:
                with Image.open(img_path) as im:   # reads only the header
                    size = im.size                 # (width, height)
            records.append({
                "file_name": str(img_path),
                "sem_seg_file_name": str(mask_dir / img_path.name),
                "height": size[1],
                "width": size[0],
                "image_id": len(records),
            })
    print(f"Loaded {len(records)} image/mask pairs")
    return records


def check_labels(records, num_classes, sample=300, seed=0):
    """Fail early if mask pixel values are not valid class ids (0..N-1 or 255)."""
    if not records:
        raise SystemExit("No image/mask pairs found. Did the patcher run?")
    rng = random.Random(seed)
    chosen = rng.sample(records, min(sample, len(records)))
    counts = np.zeros(256, dtype=np.int64)
    for rec in chosen:
        arr = np.array(Image.open(rec["sem_seg_file_name"]))
        counts += np.bincount(arr.reshape(-1), minlength=256)[:256]
    present = [int(v) for v in np.nonzero(counts)[0]]
    valid = set(range(num_classes)) | {255}
    bad = sorted(set(present) - valid)
    total = counts.sum()
    print(f"Label check on {len(chosen)} random masks, values present: {present}")
    for c in sorted(set(present) & valid):
        print(f"   value {c:>3}: {100.0 * counts[c] / total:6.2f}% of pixels")
    if bad:
        raise SystemExit(
            f"ERROR: mask values {bad} are not valid class ids for {num_classes} classes "
            f"(allowed: 0..{num_classes - 1} and 255=ignore).\n"
            f"Re-run the patcher with --label-map (e.g. \"151:1,225:1,255:0\")."
        )
    missing = sorted(set(range(num_classes)) - set(present))
    if missing:
        print(f"WARNING: class ids {missing} do not occur in the sampled masks.")


# ----------------------------------------------------------------------------------
# detectron2 parts
# ----------------------------------------------------------------------------------

def register_dataset(records, class_names):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    if DATASET_NAME in DatasetCatalog:
        DatasetCatalog.remove(DATASET_NAME)
    DatasetCatalog.register(DATASET_NAME, lambda: records)
    MetadataCatalog.get(DATASET_NAME).set(
        stuff_classes=class_names,
        stuff_colors=STUFF_COLORS[:len(class_names)],
        evaluator_type="sem_seg",
        ignore_label=255,
    )


def setup_config(args, config_file, class_names):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)      # defines MODEL.RESNETS.STEM_TYPE etc. used by the yaml
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_file))

    cfg.DATASETS.TRAIN = (DATASET_NAME,)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(class_names)
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = args.num_queries
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    # learning-rate drops at 2/3 and 5/6 of training (20000/25000 for 30000 iterations)
    cfg.SOLVER.STEPS = (int(args.max_iter * 2 / 3), int(args.max_iter * 5 / 6))
    cfg.SOLVER.WARMUP_ITERS = min(500, args.max_iter // 10)
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    if args.disable_grad_clip:
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
    # otherwise the setting of the Mask2Former yaml is kept

    cfg.TEST.EVAL_PERIOD = 0
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def build_mask2former_optimizer(cfg, model):
    """Optimizer as in the official Mask2Former train_net.py.

    Plain detectron2 does not know the yaml's CLIP_TYPE 'full_model' and always builds
    SGD; Mask2Former uses AdamW, a 0.1x learning rate for the backbone, no weight decay
    for norm/embedding layers, and gradient clipping on the whole model.
    """
    defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
    norm_types = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params, memo = [], set()
    for module_name, module in model.named_modules():
        for param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)
            hp = copy.copy(defaults)
            if "backbone" in module_name:
                hp["lr"] = hp["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
            if "relative_position_bias_table" in param_name or "absolute_pos_embed" in param_name:
                hp["weight_decay"] = 0.0
            if isinstance(module, norm_types):
                hp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
            if isinstance(module, torch.nn.Embedding):
                hp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
            params.append({"params": [value], **hp})

    clip = cfg.SOLVER.CLIP_GRADIENTS
    clip_on = clip.ENABLED and clip.CLIP_TYPE == "full_model" and clip.CLIP_VALUE > 0.0

    def with_clipping(optim_cls):
        if not clip_on:
            return optim_cls

        class Clipped(optim_cls):
            def step(self, closure=None):
                all_params = itertools.chain(*[g["params"] for g in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip.CLIP_VALUE)
                return super().step(closure=closure)
        return Clipped

    name = cfg.SOLVER.OPTIMIZER
    if name == "SGD":
        return with_clipping(torch.optim.SGD)(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    if name == "ADAMW":
        return with_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
    raise NotImplementedError(f"no optimizer type {name}")


def make_trainer_class():
    from detectron2.data import build_detection_train_loader
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import SemSegEvaluator

    class WSITrainer(DefaultTrainer):
        @classmethod
        def build_optimizer(cls, cfg, model):
            return build_mask2former_optimizer(cfg, model)

        @classmethod
        def build_train_loader(cls, cfg):
            from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
                MaskFormerSemanticDatasetMapper,
            )
            mapper = MaskFormerSemanticDatasetMapper(cfg, is_train=True)
            return build_detection_train_loader(cfg, mapper=mapper)

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            return SemSegEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR)

    return WSITrainer


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Mask2Former on paired patches.")
    ap.add_argument("--repo", type=Path, required=True, help="path to the Mask2Former repository")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--classes", default="Background,Stroma,Keratin",
                    help="comma-separated class names, in the order of the mask values 0,1,2,...")
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--checkpoint-period", type=int, default=2000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weights", default="", help="initial weights (default: from the yaml, ImageNet R-50)")
    ap.add_argument("--disable-grad-clip", action="store_true",
                    help="turn gradient clipping off (the first version of this script did)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last checkpoint in --output-dir")
    args = ap.parse_args()

    repo = args.repo.resolve()
    config = repo / "configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml"
    if not config.is_file():
        ap.error(f"Mask2Former config not found: {config}")
    sys.path.insert(0, str(repo))

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not 2 <= len(class_names) <= len(STUFF_COLORS):
        ap.error(f"--classes needs between 2 and {len(STUFF_COLORS)} names")

    print("=" * 60)
    print("Mask2Former Training - WSI Segmentation")
    print("=" * 60)
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. Run training on a GPU node.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = load_dataset(args.data_root.resolve())
    check_labels(records, len(class_names))
    register_dataset(records, class_names)
    cfg = setup_config(args, config, class_names)

    seen = cfg.SOLVER.MAX_ITER * cfg.SOLVER.IMS_PER_BATCH
    print(f"Classes: {class_names}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}   LR: {cfg.SOLVER.BASE_LR}   "
          f"Max iter: {cfg.SOLVER.MAX_ITER}")
    print(f"Training sees about {seen} patches = {seen / len(records):.2f} passes over the data")
    if seen < len(records):
        print("WARNING: less than one pass over the data. Consider a larger --max-iter.")
    print(f"Optimizer: {cfg.SOLVER.OPTIMIZER}   backbone LR x{cfg.SOLVER.BACKBONE_MULTIPLIER}   "
          f"gradient clipping: {'OFF' if args.disable_grad_clip else cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE}")
    print(f"Output: {cfg.OUTPUT_DIR}")

    trainer = make_trainer_class()(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    print("Training DONE!")


if __name__ == "__main__":
    main()