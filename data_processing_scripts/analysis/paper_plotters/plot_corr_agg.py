#!/usr/bin/env python3
"""
plot_corr_agg.py

Reads CSV files with columns:
    r, agg_num, agg_den (or agg_denom), Cr

Per-CSV behavior (path points to a CSV or directory):
    For each CSV, produces 6 PNGs:
        <stem>__agg_num_linear.png
        <stem>__agg_den_linear.png
        <stem>__C_r_linear.png
        <stem>__agg_num_loglog.png
        <stem>__agg_den_loglog.png
        <stem>__C_r_loglog.png
    saved under:
        <csv_dir>/plots/

Overlay behavior (path points to a .txt file):
    The .txt file should contain one CSV path per line
    (relative to the .txt location, or absolute).
    For each metric, we overlay all CSVs in a single plot:
        <txt_stem>__agg_num_linear_overlay.png
        <txt_stem>__agg_den_linear_overlay.png
        <txt_stem>__C_r_linear_overlay.png
        <txt_stem>__agg_num_loglog_overlay.png
        <txt_stem>__agg_den_loglog_overlay.png
        <txt_stem>__C_r_loglog_overlay.png
    saved under:
        <txt_dir>/plots/

Usage:
    python plot_corr_agg.py /path/to/file_or_dir_or_list.txt
    python -m clouds.data_processing_scripts.analysis.paper_plotters.plot_corr_agg scratch/big_data_dump/new_data/analysis/autocorr/cl4_cf8_ordLF_bbox
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REQUIRED = {"r", "agg_num", "Cr"}


# -------------------------
#  Helpers: CSV structure
# -------------------------

def has_required_columns(csv_path: Path) -> bool:
    try:
        cols = csv_path.open().readline().strip().split(",")
    except OSError:
        return False
    cols = {c.strip() for c in cols}
    if not REQUIRED.issubset(cols):
        return False
    return ("agg_den" in cols) or ("agg_denom" in cols)


def load_data(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    r = data["r"]
    agg_num = data["agg_num"]

    if "agg_den" in data.dtype.names:
        agg_den = data["agg_den"]
    else:
        agg_den = data["agg_denom"]

    Cr = data["Cr"]
    return r, agg_num, agg_den, Cr


# -------------------------
#  Helpers: legend label
# -------------------------

def _pretty_p_value(raw: str) -> str:
    """
    Try to turn something like '0592746' into '0.592746'.
    If that fails, just return the raw string.
    """
    if raw and raw[0] == "0" and len(raw) > 1:
        return "0." + raw[1:]
    return raw


def legend_label_from_path(csv_path: Path) -> str:
    """
    Build a descriptive legend label from a filename like:

        sp__src_png__cl_LF_bbox__cf_cloud_bbox__order_4__p_0592746__win_03-04__run_abc123.csv

    → "src=png, cl=LF_bbox, cf=cloud_bbox, order=4, p=0.592746, win=03-04"

    Rough semantics:
        src   : data source (e.g., 'png', 'sp')
        cl    : cluster labeling / connectivity + bounding rule
        cf    : cloud filter (e.g., cloud_bbox, size_threshold, etc.)
        order : connectivity order (4, 8, etc., if encoded)
        p     : site-perc probability, pretty-printed
        win   : time window or other selection window
    """
    stem = csv_path.with_suffix("").name
    parts = stem.split("__")

    pieces = []

    for p in parts:
        if p.startswith("src_"):
            pieces.append("src=" + p[len("src_"):])
        elif p.startswith("cl_"):
            pieces.append("cl=" + p[len("cl_"):])
        elif p.startswith("cf_"):
            pieces.append("cf=" + p[len("cf_"):])
        elif p.startswith("order_"):
            pieces.append("order=" + p[len("order_"):])
        elif p.startswith("p_"):
            raw_p = p[len("p_"):]
            pieces.append("p=" + _pretty_p_value(raw_p))
        elif p.startswith("win_"):
            pieces.append("win=" + p[len("win_"):])

    if not pieces:
        return stem
    return ", ".join(pieces)


# -------------------------
#  Plotting primitives
# -------------------------

def plot_single_linear(r, y, ylabel: str, out_path: Path, legend_label: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, y, linewidth=1.0, label=legend_label)
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs r (linear)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_single_loglog(r, y, ylabel: str, out_path: Path, legend_label: str):
    mask = (r > 0) & (y > 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(r[mask], y[mask], linewidth=1.0, label=legend_label)
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs r (log-log)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def overlay_linear(curves, y_key: str, ylabel: str, out_path: Path):
    """
    curves: list of dicts with keys: 'r', y_key, 'label'
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in curves:
        r = c["r"]
        y = c[y_key]
        ax.plot(r, y, linewidth=1.0, label=c["label"])
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs r (linear, overlay)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def overlay_loglog(curves, y_key: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in curves:
        r = c["r"]
        y = c[y_key]
        mask = (r > 0) & (y > 0)
        ax.loglog(r[mask], y[mask], linewidth=1.0, label=c["label"])
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs r (log-log, overlay)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


# -------------------------
#  Per-CSV behavior
# -------------------------

def process_csv(csv_path: Path):
    if not has_required_columns(csv_path):
        return

    r, agg_num, agg_den, Cr = load_data(csv_path)

    plots_dir = csv_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.with_suffix("").name
    label = legend_label_from_path(csv_path)

    # Linear
    plot_single_linear(r, agg_num, "agg_num",
                       plots_dir / f"{stem}__agg_num_linear.png",
                       label)
    plot_single_linear(r, agg_den, "agg_den",
                       plots_dir / f"{stem}__agg_den_linear.png",
                       label)
    plot_single_linear(r, Cr, "C(r)",
                       plots_dir / f"{stem}__C_r_linear.png",
                       label)

    # Log-log
    plot_single_loglog(r, agg_num, "agg_num",
                       plots_dir / f"{stem}__agg_num_loglog.png",
                       label)
    plot_single_loglog(r, agg_den, "agg_den",
                       plots_dir / f"{stem}__agg_den_loglog.png",
                       label)
    plot_single_loglog(r, Cr, "C(r)",
                       plots_dir / f"{stem}__C_r_loglog.png",
                       label)


# -------------------------
#  Overlay behavior (.txt)
# -------------------------

def read_csv_list_from_txt(txt_path: Path):
    csv_paths = []
    with txt_path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            if not p.is_absolute():
                p = txt_path.parent / p
            csv_paths.append(p)
    return csv_paths


def process_txt(txt_path: Path):
    csv_paths = read_csv_list_from_txt(txt_path)

    curves = []
    for csv_p in csv_paths:
        if not csv_p.is_file():
            continue
        if not has_required_columns(csv_p):
            continue
        r, agg_num, agg_den, Cr = load_data(csv_p)
        label = legend_label_from_path(csv_p)
        curves.append({
            "r": r,
            "agg_num": agg_num,
            "agg_den": agg_den,
            "Cr": Cr,
            "label": label,
        })

    if not curves:
        return

    plots_dir = txt_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stem = txt_path.with_suffix("").name

    # Linear overlays
    overlay_linear(curves, "agg_num", "agg_num",
                   plots_dir / f"{stem}__agg_num_linear_overlay.png")
    overlay_linear(curves, "agg_den", "agg_den",
                   plots_dir / f"{stem}__agg_den_linear_overlay.png")
    overlay_linear(curves, "Cr", "C(r)",
                   plots_dir / f"{stem}__C_r_linear_overlay.png")

    # Log-log overlays
    overlay_loglog(curves, "agg_num", "agg_num",
                   plots_dir / f"{stem}__agg_num_loglog_overlay.png")
    overlay_loglog(curves, "agg_den", "agg_den",
                   plots_dir / f"{stem}__agg_den_loglog_overlay.png")
    overlay_loglog(curves, "Cr", "C(r)",
                   plots_dir / f"{stem}__C_r_loglog_overlay.png")


# -------------------------
#  CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="CSV file, directory, or .txt list of CSVs")
    args = parser.parse_args()

    root = Path(args.path)

    if root.is_file():
        suffix = root.suffix.lower()
        if suffix == ".txt":
            process_txt(root)
        else:
            process_csv(root)
    elif root.is_dir():
        # Dir behavior unchanged: per-CSV plots
        for csv_path in root.glob("*.csv"):
            process_csv(csv_path)
    else:
        raise SystemExit(f"Path does not exist: {root}")


if __name__ == "__main__":
    main()
