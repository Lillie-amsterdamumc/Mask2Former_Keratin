#!/usr/bin/env python3
"""Cut whole-slide images and their masks into paired, identically named patches.

Expected layout under --data-root:

    images/<slide>.tif|.tiff
    masks/<slide>-mask.tif        (also accepted: -mask.tiff, _mask.tif, _mask.tiff,
                                   or the same name as the image)

Output (next to the inputs):

    images/patches/<slide>/0000000.png ...
    masks/patches/<slide>/0000000.png  ...

Safety features compared with the first version:
  * mask pixel values are checked; only {0, 1, 2, 255} are accepted unless you give
    --label-map (for example "151:1,225:1,255:0")
  * several mask file names are tried; slides without a mask are reported, not hidden
  * a slide with an image/mask size mismatch is skipped and reported (the run goes on)
  * blank (white glass) patches that contain no foreground class are skipped
  * finished slides are skipped on re-run (use --overwrite to redo them)
  * a summary CSV (patch_summary.csv) is written into --data-root
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Whole-slide TIFFs are far larger than Pillow's default safety limit. The data is
# your own, so the limit is switched off.
Image.MAX_IMAGE_PIXELS = None

ALLOWED_VALUES = {0, 1, 2, 255}
DONE_MARKER = ".done"
MASK_SUFFIXES = ("-mask", "_mask", "")
MASK_EXTENSIONS = (".tif", ".tiff")


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------

def parse_label_map(text):
    """'151:1,225:1,255:0' -> 256-entry uint8 lookup table (identity elsewhere)."""
    lut = np.arange(256, dtype=np.uint8)
    if not text:
        return lut, False
    for pair in text.split(","):
        src, dst = pair.split(":")
        src, dst = int(src), int(dst)
        if not (0 <= src <= 255 and 0 <= dst <= 255):
            raise ValueError(f"label-map values must be 0..255, got {pair!r}")
        lut[src] = dst
    return lut, True


def find_mask(image_path, mask_root):
    for suffix in MASK_SUFFIXES:
        for ext in MASK_EXTENSIONS:
            candidate = mask_root / f"{image_path.stem}{suffix}{ext}"
            if candidate.exists() and candidate.resolve() != image_path.resolve():
                return candidate
    return None


def read_mask(path):
    """Read a mask as a 2-D uint8 array of raw label values (no resampling)."""
    with Image.open(path) as img:
        mode = img.mode
        if mode in ("L", "P", "1"):
            # "P" = palette image: use the stored indices, NOT the palette colours.
            arr = np.array(img)
        elif mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.array(img)
            if arr.max() > 255:
                raise ValueError(
                    f"16-bit mask with values up to {int(arr.max())}; "
                    "this script expects 8-bit label masks"
                )
        else:
            raise ValueError(
                f"mask mode {mode!r} is not a single-channel label image "
                "(expected L, P, 1 or 16-bit)"
            )
    return arr.astype(np.uint8)


def read_image_rgb(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def is_blank(image_patch, white_level, blank_fraction):
    """True if almost the whole patch is near-white (empty glass)."""
    near_white = image_patch.min(axis=2) >= white_level
    return near_white.mean() >= blank_fraction


# ----------------------------------------------------------------------------------
# one slide
# ----------------------------------------------------------------------------------

def process_slide(image_path, mask_root, image_patch_root, mask_patch_root, args, lut, has_map):
    slide = image_path.stem
    row = {"slide": slide, "status": "", "saved": 0, "skipped_blank": 0,
           "mask_values": "", "detail": ""}

    print("\n" + "=" * 70)
    print(f"Slide: {slide}")
    print("=" * 70)

    image_out = image_patch_root / slide
    mask_out = mask_patch_root / slide

    if (image_out / DONE_MARKER).exists() and not args.overwrite:
        print("Already done (use --overwrite to redo). Skipping.")
        row["status"] = "already_done"
        return row

    mask_path = find_mask(image_path, mask_root)
    if mask_path is None:
        print(f"WARNING: no mask found for {slide} in {mask_root}")
        row["status"] = "no_mask"
        return row
    print(f"Image: {image_path}\nMask:  {mask_path}")

    try:
        image = read_image_rgb(image_path)
        raw_mask = read_mask(mask_path)
    except Exception as exc:  # unreadable file, wrong mask mode, ...
        print(f"WARNING: cannot read {slide}: {exc}")
        row["status"], row["detail"] = "read_error", str(exc)
        return row

    H, W = image.shape[:2]
    mH, mW = raw_mask.shape[:2]
    print(f"Image size: {W} x {H}   Mask size: {mW} x {mH}")
    if (H, W) != (mH, mW):
        print("WARNING: image and mask sizes differ. The mask is NOT resized; slide skipped.")
        row["status"] = "size_mismatch"
        row["detail"] = f"image {W}x{H} vs mask {mW}x{mH}"
        return row

    raw_values = sorted(int(v) for v in np.unique(raw_mask))
    mask = lut[raw_mask]
    del raw_mask
    final_values = sorted(int(v) for v in np.unique(mask))
    row["mask_values"] = " ".join(map(str, final_values))
    print(f"Mask values: raw {raw_values} -> used {final_values}")

    unexpected = set(final_values) - ALLOWED_VALUES
    if unexpected:
        print(f"ERROR: mask contains values {sorted(unexpected)} that are not valid class ids.")
        print("       Use --label-map, e.g.  --label-map \"151:1,225:1,255:0\"")
        row["status"] = "bad_mask_values"
        row["detail"] = f"unexpected values {sorted(unexpected)}"
        return row

    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    for old in list(image_out.glob("*.png")) + list(mask_out.glob("*.png")):
        old.unlink()  # remove leftovers of an earlier (interrupted or different) run
    (image_out / DONE_MARKER).unlink(missing_ok=True)

    P, S = args.patch_size, args.stride
    saved = skipped = 0
    patch_id = -1
    for y in range(0, H - P + 1, S):
        for x in range(0, W - P + 1, S):
            patch_id += 1  # id = grid position, so names stay stable if blanks are skipped
            mask_patch = mask[y:y + P, x:x + P]
            image_patch = image[y:y + P, x:x + P]

            if (not args.keep_blank
                    and not ((mask_patch == 1) | (mask_patch == 2)).any()
                    and is_blank(image_patch, args.white_level, args.blank_fraction)):
                skipped += 1
                continue

            name = f"{patch_id:07d}.png"
            Image.fromarray(image_patch).save(image_out / name)
            Image.fromarray(mask_patch).save(mask_out / name)
            saved += 1

    del image, mask
    (image_out / DONE_MARKER).touch()
    print(f"Saved {saved} pairs, skipped {skipped} blank patches.")
    row.update(status="ok", saved=saved, skipped_blank=skipped)
    return row


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate paired image and mask patches.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128, help="128 with 256 patches = 50%% overlap")
    ap.add_argument("--label-map", default="",
                    help='remap raw mask values, e.g. "151:1,225:1,255:0"')
    ap.add_argument("--keep-blank", action="store_true",
                    help="also save blank (white) patches without foreground")
    ap.add_argument("--white-level", type=int, default=220,
                    help="a pixel is 'white' if all channels are >= this value")
    ap.add_argument("--blank-fraction", type=float, default=0.95,
                    help="a patch is blank if at least this fraction of pixels is white")
    ap.add_argument("--overwrite", action="store_true", help="redo slides that are already done")
    ap.add_argument("--strict", action="store_true",
                    help="exit with an error if ANY slide failed (default: only if none worked)")
    args = ap.parse_args()

    if args.stride < 1 or args.patch_size < 1:
        ap.error("patch-size and stride must be positive")
    lut, has_map = parse_label_map(args.label_map)

    root = args.data_root.resolve()
    image_root, mask_root = root / "images", root / "masks"
    if not image_root.is_dir() or not mask_root.is_dir():
        ap.error("data root must contain images/ and masks/")
    image_patch_root, mask_patch_root = image_root / "patches", mask_root / "patches"
    image_patch_root.mkdir(parents=True, exist_ok=True)
    mask_patch_root.mkdir(parents=True, exist_ok=True)

    # only directly inside images/, so generated patches are never picked up
    image_files = sorted(list(image_root.glob("*.tiff")) + list(image_root.glob("*.tif")))

    print("=" * 70)
    print("ADAMANTINOMA PATCH EXTRACTION")
    print("=" * 70)
    print(f"Root: {root}\nFound {len(image_files)} TIFF images")
    print(f"Patch {args.patch_size}px, stride {args.stride}px, "
          f"label-map: {args.label_map or 'none'}, blank patches: "
          f"{'kept' if args.keep_blank else 'skipped'}")
    if not image_files:
        sys.exit("No .tif/.tiff images found directly inside images/")

    rows = [process_slide(p, mask_root, image_patch_root, mask_patch_root, args, lut, has_map)
            for p in image_files]

    summary = root / "patch_summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide", "status", "saved", "skipped_blank",
                                          "mask_values", "detail"])
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] in ("ok", "already_done")]
    bad = [r for r in rows if r["status"] not in ("ok", "already_done")]
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(ok)} slides ready, {len(bad)} problems  (details: {summary})")
    for r in bad:
        print(f"  - {r['slide']}: {r['status']} {r['detail']}")
    print("=" * 70)

    if not ok or (bad and args.strict):
        sys.exit(1)


if __name__ == "__main__":
    main()
