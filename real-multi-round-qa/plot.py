import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Analyze number of user sessions from benchmark results.")
    parser.add_argument("input_dir", help="Directory containing JSON files")
    parser.add_argument("output", help="Output path for the 3D bar plot image")
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
                            if k not in ["num_users_concurrent", "num_users_sequential", "output"]}
            all_params.append(params_fixed)

            df = pd.DataFrame(results)
            df = df[df["turn"] != 0]
            if df.empty:
                continue

            ttft_95 = df["ttft"].quantile(0.95)
            summary_records.append({
                "num_users_concurrent": params["num_users_concurrent"],
                "num_users_sequential": params["num_users_sequential"],
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

    summary_df_sorted = summary_df.sort_values(by=['num_users_concurrent', 'num_users_sequential'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = summary_df_sorted['num_users_concurrent']
    y = summary_df_sorted['num_users_sequential']
    z = np.zeros_like(x)
    dz = summary_df_sorted['ttft_95']

    colors = ['blue' if v <= 2 else 'red' for v in dz]

    ax.bar3d(x, y, z, dx=0.5, dy=0.5, dz=dz, color=colors, shade=True)
    ax.set_xlabel('Concurrent Users (C)')
    ax.set_ylabel('Sequential Users (S)')
    ax.set_zlabel('95% Tail TTFT (s)')
    plt.title('TTFT 95% Tail vs Concurrent/Sequential Users')
    ax.invert_xaxis()
    plt.savefig(args.output)

    # Max harmonic mean under 2s TTFT
    summary_under_2s = summary_df[summary_df["ttft_95"] <= 2].copy()
    if not summary_under_2s.empty:
        summary_under_2s["harmonic_mean"] = 2 * summary_under_2s["num_users_concurrent"] * summary_under_2s["num_users_sequential"] / (
            summary_under_2s["num_users_concurrent"] + summary_under_2s["num_users_sequential"]
        )
        best_row = summary_under_2s.sort_values("harmonic_mean", ascending=False).iloc[0]
        product = best_row["num_users_concurrent"] * best_row["num_users_sequential"]
        print(f"Max harmonic mean (C,S) where TTFT_95 <= 2s: {best_row['harmonic_mean']:.2f}")
        print(f"  => C={best_row['num_users_concurrent']}, S={best_row['num_users_sequential']}, CxS={product}")
    else:
        print("No data points with TTFT_95 <= 2s.")

if __name__ == "__main__":
    main()
