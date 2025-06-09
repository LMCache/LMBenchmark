import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    parser = argparse.ArgumentParser(description="Analyze number of user sessions from benchmark results.")
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("output", help="Output path for the contour plot image")
    args = parser.parse_args()

    json_files = glob.glob(f"{args.input_dir}/*.json")
    all_params = []
    summary_records = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if "params" not in data or "results" not in data:
                print(f"Skipping {file}: missing 'params' or 'results'")
                continue
            params = data["params"]
            results = data["results"]

            params_fixed = {k: v for k, v in params.items()
                            if k not in ["concurrent", "session_depth", "output"]}
            all_params.append(params_fixed)

            df = pd.DataFrame(results)
            df = df[df["turn"] != 0]
            if df.empty:
                continue

            ttft_95 = df["ttft"].quantile(0.95)
            summary_records.append({
                "c": params["concurrent"],
                "s": params["session_depth"],
                "ttft_95": ttft_95
            })

    if all_params:
        first_params = all_params[0]
        assert all(p == first_params for p in all_params), "Inconsistent fixed parameters"

    summary_df = pd.DataFrame(summary_records)
    print(summary_df)

    if summary_df.empty:
        print("No valid TTFT data to visualize.")
        return

    C_vals = summary_df["c"].values
    S_vals = summary_df["s"].values
    TTFT_95_vals = summary_df["ttft_95"].values

    plt.figure(figsize=(9, 7))
    contour = plt.tricontourf(
        C_vals, S_vals, TTFT_95_vals,
        levels=np.logspace(np.log10(0.01), np.log10(100), 30),
        cmap='plasma',
        norm=LogNorm(vmin=0.01, vmax=100)
    )

    cbar = plt.colorbar(contour)
    ticks = [0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 100]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    cbar.set_label('TTFT_95 (s)', fontsize=12)

    plt.tricontour(C_vals, S_vals, TTFT_95_vals, levels=[2.0], colors='white', linewidths=2, linestyles='dashed')

    plt.xlabel('Concurrent (C)', fontsize=12)
    plt.ylabel('Session Depth(S)', fontsize=12)
    plt.title('TTFT_95 Contour across (C, S)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    summary_under_2s = summary_df[summary_df["ttft_95"] <= 2].copy()
    if not summary_under_2s.empty:
        summary_under_2s["harmonic_mean"] = 2 * summary_under_2s["c"] * summary_under_2s["s"] / (
            summary_under_2s["c"] + summary_under_2s["s"]
        )
        best_row = summary_under_2s.sort_values("harmonic_mean", ascending=False).iloc[0]
        product = best_row["c"] * best_row["s"]
        print(f"Max harmonic mean (C,S) where TTFT_95 <= 2s: {best_row['harmonic_mean']:.2f}")
        print(f"  => C={best_row['c']}, S={best_row['s']}, CxS={product}")
        plt.scatter([best_row['c']], [best_row['s']], c='cyan', edgecolors='black', s=120, marker='*', label='Best (C,S)')
        plt.legend()
        plt.autoscale()
    else:
        print("No data points with TTFT_95 <= 2s.")
    plt.savefig(args.output)

if __name__ == "__main__":
    main()
