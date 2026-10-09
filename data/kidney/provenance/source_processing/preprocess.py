"""Convert KiTS19 3D CT volumes into a 2D axial-slice dataset.

For every fully-downloaded case that ships with a segmentation label we:
  1. clip Hounsfield units to a soft-tissue window and scale to [0, 255] (uint8),
  2. slice along the axial axis (axis 0, the KiTS19 convention),
  3. keep every slice containing kidney/tumor plus a random fraction of pure
     background slices (for context / true negatives),
  4. resize in-plane to SLICE_SIZE (bilinear for image, nearest for mask),
  5. write image/mask PNGs and accumulate a manifest row per slice.

Runs case-parallel with a process pool. Safe to re-run: it skips cases that
are still downloading (an aria2 .aria2 control file is present).
"""
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

import config as C


def _window_to_uint8(vol):
    vol = np.clip(vol, C.HU_MIN, C.HU_MAX)
    vol = (vol - C.HU_MIN) / (C.HU_MAX - C.HU_MIN)
    return (vol * 255.0).astype(np.uint8)


def _case_is_ready(case_dir):
    img = case_dir / "imaging.nii.gz"
    seg = case_dir / "segmentation.nii.gz"
    ctrl = case_dir / "imaging.nii.gz.aria2"
    return img.exists() and seg.exists() and not ctrl.exists()


def process_case(case_name, rng_seed):
    case_dir = C.KITS_DATA_DIR / case_name
    if not _case_is_ready(case_dir):
        return case_name, "skip_incomplete", []
    try:
        img_nii = nib.load(str(case_dir / "imaging.nii.gz"))
        seg_nii = nib.load(str(case_dir / "segmentation.nii.gz"))
        vol = np.asanyarray(img_nii.dataobj).astype(np.float32)
        seg = np.asanyarray(seg_nii.dataobj).astype(np.uint8)
    except Exception as e:  # noqa: BLE001
        return case_name, "error:%s" % e, []

    if vol.shape != seg.shape:
        return case_name, "shape_mismatch %s vs %s" % (vol.shape, seg.shape), []

    spacing = tuple(float(x) for x in img_nii.header.get_zooms()[:3])
    vol_u8 = _window_to_uint8(vol)
    n_slices = vol.shape[0]
    rng = np.random.default_rng(rng_seed)

    rows = []
    for z in range(n_slices):
        m = seg[z]
        has_fg = bool(m.max() > 0)
        if not has_fg and rng.random() > C.NEG_SLICE_FRAC:
            continue
        img2d = cv2.resize(vol_u8[z], (C.SLICE_SIZE, C.SLICE_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        mask2d = cv2.resize(m, (C.SLICE_SIZE, C.SLICE_SIZE),
                            interpolation=cv2.INTER_NEAREST)
        stem = "%s_z%03d" % (case_name, z)
        ip = C.IMG_DIR / (stem + ".png")
        mp = C.MASK_DIR / (stem + ".png")
        cv2.imwrite(str(ip), img2d)
        cv2.imwrite(str(mp), mask2d)
        rows.append({
            "case": case_name,
            "z": z,
            "image": os.path.relpath(ip, C.PROC_DIR),
            "mask": os.path.relpath(mp, C.PROC_DIR),
            "has_kidney": int((mask2d == 1).any()),
            "has_tumor": int((mask2d == 2).any()),
            "kidney_px": int((mask2d == 1).sum()),
            "tumor_px": int((mask2d == 2).sum()),
            "n_slices": n_slices,
            "sp_z": spacing[0], "sp_y": spacing[1], "sp_x": spacing[2],
        })
    return case_name, "ok", rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="process only first N cases (0=all)")
    args = ap.parse_args()

    C.ensure_dirs()
    all_cases = sorted(p.name for p in C.KITS_DATA_DIR.iterdir()
                       if p.is_dir() and p.name.startswith("case_"))
    ready = [c for c in all_cases if _case_is_ready(C.KITS_DATA_DIR / c)]
    if args.limit:
        ready = ready[:args.limit]
    print("cases total=%d ready=%d" % (len(all_cases), len(ready)))

    manifest = []
    summary = {"ok": 0, "skip_incomplete": 0, "other": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_case, c, C.SEED + i): c
                for i, c in enumerate(ready)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cases"):
            name, status, rows = fut.result()
            if status == "ok":
                summary["ok"] += 1
                manifest.extend(rows)
            elif status == "skip_incomplete":
                summary["skip_incomplete"] += 1
            else:
                summary["other"] += 1
                print("  [%s] %s" % (name, status))

    import pandas as pd
    df = pd.DataFrame(manifest).sort_values(["case", "z"]).reset_index(drop=True)
    df.to_csv(C.MANIFEST_CSV, index=False)
    print("manifest rows=%d saved-> %s" % (len(df), C.MANIFEST_CSV))
    print("summary:", summary)
    print("cases with slices:", df["case"].nunique())


if __name__ == "__main__":
    main()
