#!/usr/bin/env python3
"""
plot_bdd_density.py

Boundary-distance distribution (BDD) plotter.

Expected CSV columns:
    r_norm, density

Interpretation:
    r_norm  : distance from site to cloud boundary, normalized by the
              maximum such distance within the cloud (0 = boundary,
              1 = deepest bulk sites, or vice versa depending on your
              convention — the label is symmetric to that choice).
    density : normalized site-count density (probability density) at
              that normalized boundary distance.

Per-CSV behavior (path = CSV or directory):
    For each CSV, produce 2 PNGs:
        <stem>__density_linear.png
        <stem>__density_loglog.png
    saved under:
        <csv_dir>/plots/

Overlay behavior (path = .txt):
    .txt file lists CSVs (one per line, relative or absolute).
    For each scaling (linear/log-log), overlay all curves:
        <txt_stem>__density_linear_overlay.png
        <txt_stem>__density_loglog_overlay.png
    saved under:
        <txt_dir>/plots/

Usage:
    python plot_bdd_density.py /path/to/file_or_dir_or_list.txt
    python -m clouds.data_processing_scripts.analysis.paper_plotters.plot_bdd_agg scratch/big_data_dump/new_data/analysis/bdd/p0p592746_cl4_cf8_ordLF_bbox/bdd_2025-11-14_siteperc_valid_rows_siteperc_argmax__1e5bb256.csv
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REQUIRED = {"r_norm", "density"}


# -------------------------
#  Helpers: CSV structure
# -------------------------

def has_required_columns(csv_path: Path) -> bool:
    try:
        cols = csv_path.open().readline().strip().split(",")
    except OSError:
        return False
    cols = {c.strip() for c in cols}
    return REQUIRED.issubset(cols)


def load_data(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    r_norm = data["r_norm"]
    density = data["density"]
    return r_norm, density


# -------------------------
#  Helpers: legend label
# -------------------------

def legend_label_from_path(csv_path: Path) -> str:
    """
    Similar semantics to the correlation script:
        src   : png/sp/etc.
        cl    : connectivity / cluster labeling
        cf    : cloud filter
        order : connectivity order (4/8/...)
        p     : p-value (if encoded)
        win   : window (if encoded)
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
            pieces.append("p=" + p[len("p_"):])
        elif p.startswith("win_"):
            pieces.append("win=" + p[len("win_"):])

    if not pieces:
        return stem
    return ", ".join(pieces)


# -------------------------
#  Plotting primitives
# -------------------------

X_LABEL = "normalized distance to cloud boundary, r/R_max"
Y_LABEL = "boundary-distance density ρ(r/R_max)"


def plot_density_linear(r_norm, density, out_path: Path, legend_label: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_norm, density, linewidth=1.0, label=legend_label)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Boundary-distance distribution ρ(r/R_max) vs r/R_max (linear)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_density_loglog(r_norm, density, out_path: Path, legend_label: str):
    mask = (r_norm > 0) & (density > 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(r_norm[mask], density[mask], linewidth=1.0, label=legend_label)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Boundary-distance distribution ρ(r/R_max) vs r/R_max (log-log)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def overlay_density_linear(curves, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in curves:
        r_norm = c["r_norm"]
        density = c["density"]
        ax.plot(r_norm, density, linewidth=1.0, label=c["label"])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Boundary-distance distribution (linear, overlay)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


def overlay_density_loglog(curves, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in curves:
        r_norm = c["r_norm"]
        density = c["density"]
        mask = (r_norm > 0) & (density > 0)
        ax.loglog(r_norm[mask], density[mask], linewidth=1.0, label=c["label"])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Boundary-distance distribution (log-log, overlay)")
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

    r_norm, density = load_data(csv_path)

    plots_dir = csv_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.with_suffix("").name
    label = legend_label_from_path(csv_path)

    plot_density_linear(
        r_norm,
        density,
        plots_dir / f"{stem}__density_linear.png",
        label,
    )
    plot_density_loglog(
        r_norm,
        density,
        plots_dir / f"{stem}__density_loglog.png",
        label,
    )


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
        r_norm, density = load_data(csv_p)
        label = legend_label_from_path(csv_p)
        curves.append({
            "r_norm": r_norm,
            "density": density,
            "label": label,
        })

    if not curves:
        return

    plots_dir = txt_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    stem = txt_path.with_suffix("").name

    overlay_density_linear(
        curves,
        plots_dir / f"{stem}__density_linear_overlay.png",
    )
    overlay_density_loglog(
        curves,
        plots_dir / f"{stem}__density_loglog_overlay.png",
    )


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
        for csv_path in root.glob("*.csv"):
            process_csv(csv_path)
    else:
        raise SystemExit(f"Path does not exist: {root}")


if __name__ == "__main__":
    main()
