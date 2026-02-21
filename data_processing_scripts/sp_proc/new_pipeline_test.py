#!/usr/bin/env python3
"""
sp_cloud_diagnostic.py

Minimal site-percolation diagnostic script.
Generates a lattice, runs the preprocessing pipeline, 
and reports how many clouds (connected components) were found.

Use this to sanity-check:
  - Whether clouds appear at a given fill probability (p)
  - Whether order/connectivity choices behave as expected
"""

# python -m clouds.data_processing_scripts.sp_proc.new_pipeline_test

import numpy as np
from pathlib import Path
from PIL import Image

from clouds.utils import cloud_utils
from clouds.utils.image_utils import save_lattice_png  # kept for compatibility

# -------------------------------------------------------------
# === USER CONFIGURATION ===
# (Set these before running)
# -------------------------------------------------------------
WIDTH  = 4000         # lattice width
HEIGHT = 2666         # lattice height
P_VAL  = 1-0.592746     # fill probability (e.g., site percolation threshold)
SEED   = 42           # RNG seed for reproducibility

ORDER  = "LF_bbox"    # "FL" or "LF_bbox"
CL     = 8            # label connectivity (foreground): 4 or 8
CF     = 4            # flood connectivity (background): 4 or 8  (primary run)
MIN_AREA  = 2000      # minimum cloud area (pixels)
MAX_AREA  = 10_000_000  # maximum cloud area (pixels)
BBOX_PAD  = 1         # padding around bbox for per-bbox fill

OUTPUT_DIR = Path("scratch/presentation_mats")  # where PNGs will be saved

# -------------------------------------------------------------
# === COLOR CONFIGURATION (3-COLOR OUTPUT) ===
# Colors are (R, G, B) tuples in 0–255.
# -------------------------------------------------------------
CLOUD_COLOR      = (255, 255, 255)    # color for the main cloud (cf=4)
BACKGROUND_COLOR = (0,   0,   0)    # color for background (outside cf=8 cloud)
FJORD_COLOR      = (210,   180, 180)     # color for fjords (in cf=8 cloud, not in cf=4)

# -------------------------------------------------------------
# === RUN DIAGNOSTIC ===
# -------------------------------------------------------------
print("=== Site Percolation Cloud Diagnostic ===")
print(f"Size: {HEIGHT}x{WIDTH}")
print(f"p = {P_VAL:.6f}   seed = {SEED}")
print(f"Order: {ORDER}   Label conn: {CL}   Flood conn (primary): {CF}")
print(f"Area range: [{MIN_AREA}, {MAX_AREA}]   BBox pad: {BBOX_PAD}")
print("=========================================")

# 1) Generate lattice
lattice = cloud_utils.generate_site_percolation_lattice(
    width=WIDTH, height=HEIGHT, fill_prob=P_VAL, seed=SEED
)

# 2) Run preprocessing + cropping twice (cf=4 and cf=8)
#    We use both versions later to color fjords / background differently.
cropped_cf4 = cloud_utils.preprocess_and_crop_clouds(
    lattice,
    order=ORDER,
    cl=CL,
    cf=4,
    min_area=MIN_AREA,
    max_area=MAX_AREA,
    bbox_pad=BBOX_PAD,
)

cropped_cf8 = cloud_utils.preprocess_and_crop_clouds(
    lattice,
    order=ORDER,
    cl=CL,
    cf=8,
    min_area=MIN_AREA,
    max_area=MAX_AREA,
    bbox_pad=BBOX_PAD,
)

# Sanity check: both runs should produce the same number of clouds
if len(cropped_cf4) != len(cropped_cf8):
    print("[WARN] preprocess_and_crop_clouds produced different cloud counts for cf=4 vs cf=8.")
    print(f"       cf=4: {len(cropped_cf4)}   cf=8: {len(cropped_cf8)}")

# Choose the "primary" set for reporting, based on CF
if CF == 4:
    cropped_clouds = cropped_cf4
else:
    cropped_clouds = cropped_cf8

# 3) Report results
n_clouds = len(cropped_clouds)
areas = [np.count_nonzero(c) for c in cropped_clouds]

print(f"[OK] Clouds found: {n_clouds}")
if n_clouds > 0:
    print(f"    Mean area: {np.mean(areas):.1f}")
    print(f"    Min area:  {np.min(areas)}")
    print(f"    Max area:  {np.max(areas)}")
else:
    print("    No clouds found within given area range.")

# -------------------------------------------------------------
# === SAVE SAMPLE CROPS AS 3-COLOR PNGs ===
#   - Start with BACKGROUND_COLOR everywhere
#   - Color all cf=8 cloud pixels as FJORD_COLOR
#   - Then color all cf=4 cloud pixels as CLOUD_COLOR
#     (so cf=4 overrides cf=8 where they overlap)
# -------------------------------------------------------------
if n_clouds > 0:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_count = min(3, n_clouds)

    print(f"\nSaving {sample_count} sample cropped clouds to: {OUTPUT_DIR}")
    for i in range(sample_count):
        # Masks from cf=4 and cf=8 runs
        mask4 = cropped_cf4[i].astype(bool)
        mask8 = cropped_cf8[i].astype(bool)

        H, W = mask4.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # 1) Start with background color everywhere
        rgb[:, :] = BACKGROUND_COLOR

        # 2) Color cf=4 cloud pixels as main fjord color
        rgb[mask4] = FJORD_COLOR

        # 3) Color cf=8 cloud pixels as cloud color (overrides cf=4 where overlapping)
        rgb[mask8] = CLOUD_COLOR



        area = np.count_nonzero(mask4)
        png_path = (
            OUTPUT_DIR
            / f"cl{CL}_cf{CF}_p{P_VAL:.6f}_cloud_{i:02d}_area{area}_seed{SEED}_fjord_highlight.png"
        )

        Image.fromarray(rgb, mode="RGB").save(png_path)
        print(f"  -> {png_path}")

        png_path = (
            OUTPUT_DIR
            / f"cl{CL}_cf{CF}_p{P_VAL:.6f}_cloud_{i:02d}_area{area}_seed{SEED}.png"
        )

        cloud = cropped_clouds[i].astype(bool)

        cloud_image = np.zeros((H, W, 3), dtype=np.uint8)
        cloud_image[cloud] = CLOUD_COLOR
        cloud_image[~cloud] = BACKGROUND_COLOR

        Image.fromarray(cloud_image, mode="RGB").save(png_path)
        print(f"  -> {png_path}")

print("=========================================")
