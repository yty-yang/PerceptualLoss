#!/usr/bin/env python3
"""
Analyze s2 experiment results from the Outputs directory.

Usage:
    python analyze.py                  # summary table + RD plots
    python analyze.py --no-plots       # table only
    python analyze.py --outputs /path  # custom Outputs dir
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(__file__).parent.parent / "Outputs"
PLOTS_DIR = Path(__file__).parent / "plots"


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_name(name: str) -> dict:
    """Extract structured fields from an experiment folder name."""
    stage_m = re.search(r"_(s\d+)$", name)
    config_m = re.search(r"_(x\d+x\d+)_", name)
    lr_m = re.search(r"_lr-([^_]+)_", name)
    lamb_m = re.search(r"_lamb-([^_]+)_", name)
    if "_wd-saliency_" in name:
        loss = "wd-saliency"
    elif "_wd_" in name:
        loss = "wd"
    else:
        loss = "baseline"
    return {
        "stage": stage_m.group(1) if stage_m else None,
        "loss": loss,
        "config": config_m.group(1) if config_m else None,
        "lr": lr_m.group(1) if lr_m else None,
        "lambda": lamb_m.group(1) if lamb_m else None,
    }


def read_all_txt(path: Path) -> dict | None:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[0]
    ms_ssim_key = next((k for k in r if k.startswith("ms-ssim")), None)
    return {
        "bpp": float(r["bpp_avg"]),
        "psnr": float(r["psnr_avg"]),
        "ms_ssim": float(r[ms_ssim_key]) if ms_ssim_key else None,
        "ms_ssim_metric": ms_ssim_key,
        "size_bytes": int(r["size"]),
        "train_time_s": float(r["train_time_avg"]),
    }


# ── Collection ────────────────────────────────────────────────────────────────

def collect(outputs_dir: Path) -> pd.DataFrame:
    records = []
    for codec_dir in sorted(outputs_dir.iterdir()):
        if not codec_dir.is_dir():
            continue
        for video_dir in sorted(codec_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for exp_dir in sorted(video_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                parsed = parse_name(exp_dir.name)
                if parsed["stage"] != "s2":
                    continue
                results_file = exp_dir / "results" / "all.txt"
                if not results_file.exists():
                    continue
                metrics = read_all_txt(results_file)
                if metrics is None:
                    continue
                records.append({
                    "codec": codec_dir.name,
                    "video": video_dir.name,
                    "experiment": exp_dir.name,
                    **parsed,
                    **metrics,
                })
    return pd.DataFrame(records)


# ── Display ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    show = ["video", "config", "loss", "lr", "bpp", "psnr", "ms_ssim", "size_bytes"]
    cols = [c for c in show if c in df.columns]
    fmt = {
        "bpp": "{:.4f}".format,
        "psnr": "{:.4f}".format,
        "ms_ssim": "{:.4f}".format,
        "size_bytes": "{:,}".format,
    }
    styled = df[cols].copy()
    for col, fn in fmt.items():
        if col in styled.columns:
            styled[col] = styled[col].map(fn)
    print(styled.to_string(index=False))


# ── Plotting ──────────────────────────────────────────────────────────────────

METRICS = {
    "psnr": "PSNR (dB)",
    "ms_ssim": "MS-SSIM",
}


def plot_rd(df: pd.DataFrame, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    for video, vdf in sorted(df.groupby("video")):
        for metric, ylabel in METRICS.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            for (config, loss), grp in sorted(vdf.groupby(["config", "loss"])):
                grp = grp.sort_values("bpp")
                ax.plot(grp["bpp"], grp[metric], marker="o", label=f"{config} / {loss}")
            ax.set_xlabel("BPP")
            ax.set_ylabel(ylabel)
            ax.set_title(f"RD Curve — {ylabel} — {video}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            out = plots_dir / f"rd_{metric}_{video}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default=str(OUTPUTS_DIR))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    df = collect(Path(args.outputs))
    if df.empty:
        print("No s2 results found.")
        return

    print_summary(df)

    csv_out = Path(__file__).parent / "results.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved {csv_out}")

    if not args.no_plots:
        plot_rd(df, PLOTS_DIR)


if __name__ == "__main__":
    main()
