#!/usr/bin/env python3
"""Convert predicted label masks (from infer_mask2former_wsi.py) into QuPath GeoJSON.

Input : <pred-dir>/<slide>-pred.tif   (pixel value = class id, same size as the slide)
Output: <output-dir>/<slide>.geojson  (one FeatureCollection per slide)

Every connected region of a class becomes one polygon annotation (holes are kept).
Coordinates are slide pixels, so the file can be imported directly in QuPath
(File > Import objects from file...) on the same image.

Needs only numpy, Pillow and OpenCV (cv2).
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DEFAULT_COLORS = [(128, 128, 128), (0, 200, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]


def ring(points):
    """cv2 contour (N,1,2) -> closed GeoJSON ring [[x, y], ...]."""
    pts = points.reshape(-1, 2).astype(float).tolist()
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def class_polygons(binary, min_area, min_hole_area, epsilon, kernel):
    """Polygons (outer ring + holes) of one class from a 0/1 uint8 mask."""
    import cv2

    if kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)    # remove specks
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)   # fill pin-holes

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # [next, previous, first_child, parent]

    def simplify(c):
        return cv2.approxPolyDP(c, epsilon, True) if epsilon > 0 else c

    polygons = []
    for i, c in enumerate(contours):
        if hierarchy[i][3] != -1:          # a hole: handled with its parent
            continue
        outer = simplify(c)
        if len(outer) < 3 or cv2.contourArea(c) < min_area:
            continue
        rings = [ring(outer)]
        child = hierarchy[i][2]
        while child != -1:
            h = contours[child]
            if cv2.contourArea(h) >= min_hole_area:
                hs = simplify(h)
                if len(hs) >= 3:
                    rings.append(ring(hs))
            child = hierarchy[child][0]
        polygons.append(rings)
    return polygons


def convert_slide(pred_path, out_path, class_names, export_ids, colors, args):
    arr = np.array(Image.open(pred_path))
    if arr.ndim != 2:
        raise ValueError(f"{pred_path.name}: expected a single-channel mask, got shape {arr.shape}")
    features, counts = [], {}
    for cid in export_ids:
        binary = (arr == cid).astype(np.uint8)
        if not binary.any():
            continue
        polys = class_polygons(binary, args.min_area, args.min_hole_area, args.simplify, args.smooth)
        counts[class_names[cid]] = len(polys)
        for rings in polys:
            if args.offset_x or args.offset_y:
                rings = [[[x + args.offset_x, y + args.offset_y] for x, y in r] for r in rings]
            features.append({
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {"type": "Polygon", "coordinates": rings},
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": class_names[cid], "color": list(colors[cid])},
                },
            })
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    return counts, arr.shape


def main():
    ap = argparse.ArgumentParser(description="Predicted masks -> QuPath GeoJSON.")
    ap.add_argument("--pred-dir", type=Path, required=True, help="folder with <slide>-pred.tif files")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--classes", default="Background,Stroma,Keratin",
                    help="class names in the order of the mask values 0,1,2,...")
    ap.add_argument("--export", default="",
                    help="comma-separated class names to export (default: all except class 0)")
    ap.add_argument("--min-area", type=float, default=200.0,
                    help="drop regions smaller than this many pixels (default 200)")
    ap.add_argument("--min-hole-area", type=float, default=200.0,
                    help="fill holes smaller than this many pixels (default 200)")
    ap.add_argument("--simplify", type=float, default=1.0,
                    help="polygon simplification tolerance in pixels; 0 = off (default 1.0)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="morphological open+close kernel size in pixels; 0 = off (e.g. 5)")
    ap.add_argument("--offset-x", type=float, default=0.0, help="shift all x coordinates")
    ap.add_argument("--offset-y", type=float, default=0.0, help="shift all y coordinates")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.exit("OpenCV is missing. Install it with:  pip install opencv-python-headless")

    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(class_names) > len(DEFAULT_COLORS):
        ap.error(f"at most {len(DEFAULT_COLORS)} classes supported")
    colors = DEFAULT_COLORS[:len(class_names)]
    if args.export:
        wanted = [c.strip() for c in args.export.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in class_names]
        if unknown:
            ap.error(f"--export names not in --classes: {unknown}")
        export_ids = [class_names.index(c) for c in wanted]
    else:
        export_ids = list(range(1, len(class_names)))

    preds = sorted(list(args.pred_dir.glob("*-pred.tif")) + list(args.pred_dir.glob("*-pred.png")))
    if not preds:
        sys.exit(f"No *-pred.tif files in {args.pred_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(preds)} masks -> GeoJSON; exporting classes: {[class_names[i] for i in export_ids]}")
    problems = 0
    for p in preds:
        slide = p.name.rsplit("-pred", 1)[0]
        out = args.output_dir / f"{slide}.geojson"
        if out.exists() and not args.overwrite:
            print(f"{slide}: exists, skipping (use --overwrite)")
            continue
        try:
            counts, shape = convert_slide(p, out, class_names, export_ids, colors, args)
            print(f"{slide}: {shape[1]}x{shape[0]} px   polygons: {counts or 'none'}   -> {out.name}")
        except Exception as exc:
            problems += 1
            print(f"WARNING {slide}: {exc}")
    print(f"Done. {len(preds) - problems} converted, {problems} problems. Output: {args.output_dir}")
    if problems == len(preds):
        sys.exit(1)


if __name__ == "__main__":
    main()
