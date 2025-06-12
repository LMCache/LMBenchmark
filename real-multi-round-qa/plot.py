#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

def main():
    parser = argparse.ArgumentParser(
        description="Visualize TTFT_95 across (C,S) without interpolation."
    )
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("output",    help="Output image path")
    args = parser.parse_args()

    summary_records = []
    fixed_params_seen = []

    for path in glob.glob(os.path.join(args.input_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        if "params" not in data or "results" not in data:
            print(f"Skip {path}: missing section")
            continue

        p   = data["params"]
        res = pd.DataFrame(data["results"])
        res = res[res["turn"] != 0]          # pre-fill を除外
        if res.empty:
            continue

        summary_records.append({
            "c"       : p["concurrent"],
            "s"       : p["session_depth"],
            "ttft_95" : res["ttft"].quantile(0.95),
        })
        fixed_params_seen.append({k:v for k,v in p.items()
                                  if k not in ("concurrent","session_depth","output")})

    if not summary_records:
        print("No valid data")
        return
    if any(fp != fixed_params_seen[0] for fp in fixed_params_seen):
        raise ValueError("Inconsistent fixed parameters across files")

    df = pd.DataFrame(summary_records)
    print(df)

    grid = df.pivot(index="s", columns="c", values="ttft_95").sort_index(ascending=True)
    S_vals = grid.index.values
    C_vals = grid.columns.values
    Z      = np.ma.masked_invalid(grid.values)

    fig, ax = plt.subplots(figsize=(9, 7))

    pcm = ax.pcolormesh(
        C_vals,
        S_vals,
        Z,
        shading="nearest",
        cmap="plasma",
        norm=LogNorm(vmin=0.01, vmax=100),
    )

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("TTFT_95 (s)")
    cbar.set_ticks([0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 100])
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))

    C_mesh, S_mesh = np.meshgrid(C_vals, S_vals)
    ax.contour(
        C_mesh, S_mesh, Z,
        levels=[2.0],
        colors="white",
        linewidths=2,
        linestyles="dashed",
    )

    ax.scatter(df["c"], df["s"], s=40, c="black", marker="o", label="measured")

    ax.set_xlabel("Concurrent (C)")
    ax.set_ylabel("Session Depth (S)")
    ax.set_title("TTFT_95 Heatmap across (C, S) — no interpolation")
    ax.set_xticks(C_vals)
    ax.set_yticks(S_vals)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    ok = df[df["ttft_95"] <= 2.0].copy()
    if not ok.empty:
        ok["hmean"] = 2 * ok["c"] * ok["s"] / (ok["c"] + ok["s"])
        best = ok.loc[ok["hmean"].idxmax()]
        ax.scatter(best["c"], best["s"],
                   s=160, c="cyan", edgecolors="black", marker="*", label="Best (C,S)")
        print(f"Best (C,S) with TTFT_95 ≤ 2 s → C={best.c}, S={best.s}, "
              f"HarmonicMean={best.hmean:.2f}, C×S={best.c*best.s}")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
