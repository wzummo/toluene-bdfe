"""
plot_free_energy.py
-------------------
Plots G, H, and TS as a function of temperature from
results/free_energy_vs_T.csv

Produces:
  results/free_energy_vs_T.png   — G, H, -TS vs T (main result)
  results/entropy_vs_T.png       — S vs T (both methods)
  results/convergence_grid.png   — H time series per T (sanity check)

Usage:
    python analysis/plot_free_energy.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
TRAJ_DIR    = os.path.join(REPO_ROOT, "md", "trajectories")

EV_TO_KJ_MOL = 96.485


def plot_main(df: pd.DataFrame):
    """G, H, TS vs T — the headline result."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    T = df["T_K"].values

    # ── Top panel: G, H, TS ───────────────────────────────────
    ax = axes[0]
    ax.plot(T, df["G_2pt_kJ_mol"],       "o-", color="seagreen",
            linewidth=2, markersize=6,   label="G = H − TS  (2PT)")
    ax.plot(T, df["G_schlitter_kJ_mol"], "s--", color="lightgreen",
            linewidth=1.5, markersize=5, label="G (Schlitter, upper bound)", alpha=0.7)
    ax.plot(T, df["H_kJ_mol"],           "^-", color="steelblue",
            linewidth=2, markersize=6,   label="H (enthalpy)")
    ax.plot(T, -df["TS_2pt_eV"] * EV_TO_KJ_MOL, "v-", color="tomato",
            linewidth=2, markersize=6,   label="−TS (entropy term)")

    # Error bars on H
    if "H_sem_eV" in df.columns:
        ax.errorbar(T, df["H_kJ_mol"],
                    yerr=df["H_sem_eV"] * EV_TO_KJ_MOL,
                    fmt="none", color="steelblue", alpha=0.4, capsize=3)

    ax.set_ylabel("Energy (kJ/mol)", fontsize=12)
    ax.set_title("Toluene Gibbs Free Energy vs Temperature\n"
                 f"P = {df['P_atm'].iloc[0]} atm  |  MACE-MP-0  |  NPT MD",
                 fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    # ── Bottom panel: dG/dT = -S ─────────────────────────────
    ax = axes[1]
    if len(T) > 2:
        dG_dT = np.gradient(df["G_2pt_kJ_mol"].values, T)
        ax.plot(T, -dG_dT, "o-", color="purple",
                linewidth=2, markersize=6,
                label=r"$-\partial G/\partial T$ = S (numerical)")
        ax.plot(T, df["S_2pt_eV_K"] * EV_TO_KJ_MOL, "s--", color="orchid",
                linewidth=1.5, markersize=5,
                label="S (2PT direct)", alpha=0.8)
        ax.set_ylabel("Entropy S (kJ/mol·K)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax.set_xlabel("Temperature (K)", fontsize=12)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "free_energy_vs_T.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_entropy(df: pd.DataFrame):
    """S vs T for both methods."""
    fig, ax = plt.subplots(figsize=(9, 5))
    T = df["T_K"].values

    ax.plot(T, df["S_2pt_eV_K"] * EV_TO_KJ_MOL,
            "o-", color="steelblue", linewidth=2, markersize=6,
            label="S — 2PT (VACF power spectrum)")
    ax.plot(T, df["S_schlitter_eV_K"] * EV_TO_KJ_MOL,
            "s--", color="tomato", linewidth=1.5, markersize=5,
            label="S — Schlitter (upper bound)", alpha=0.8)

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Entropy S (kJ/mol·K)", fontsize=12)
    ax.set_title("Toluene Entropy vs Temperature — MACE MD", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "entropy_vs_T.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_convergence_grid(df: pd.DataFrame):
    """Grid of H(t) time series — one panel per temperature."""
    temps = df["T_K"].values
    n = len(temps)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*2.8),
                              sharex=False, sharey=False)
    axes = axes.flatten()

    for i, T in enumerate(temps):
        thermo_path = os.path.join(TRAJ_DIR, f"{int(T)}K", "thermo.csv")
        ax = axes[i]
        ax.set_title(f"{int(T)} K", fontsize=9)

        if not os.path.exists(thermo_path):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_visible(True)
            continue

        try:
            thermo = pd.read_csv(thermo_path)
            t  = thermo["time_ps"].values
            H  = thermo["enthalpy_eV"].values * EV_TO_KJ_MOL

            ax.plot(t, H, alpha=0.5, linewidth=0.5, color="steelblue")
            # Running mean
            w = max(1, len(H)//20)
            rm = np.convolve(H, np.ones(w)/w, mode="valid")
            ax.plot(t[:len(rm)], rm, color="darkblue", linewidth=1.5)
            ax.axhline(np.mean(H[len(H)//5:]), color="red",
                       linewidth=1.0, linestyle="--")
            ax.set_ylabel("H (kJ/mol)", fontsize=7)
            ax.set_xlabel("t (ps)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")

    # Hide unused panels
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Enthalpy Convergence per Temperature — Toluene MD",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "convergence_grid.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_table(df: pd.DataFrame):
    print("\n  ── Free Energy Summary ─────────────────────────────────")
    print(f"  {'T (K)':>7}  {'G/2PT (kJ/mol)':>16}  {'H (kJ/mol)':>12}  "
          f"{'TS (kJ/mol)':>12}  {'S (J/mol·K)':>12}")
    print("  " + "-"*68)
    for _, row in df.iterrows():
        S_J = row["S_2pt_eV_K"] * EV_TO_KJ_MOL * 1000  # kJ/mol·K → J/mol·K
        TS  = row["TS_2pt_eV"] * EV_TO_KJ_MOL
        print(f"  {row['T_K']:7.0f}  {row['G_2pt_kJ_mol']:16.2f}  "
              f"{row['H_kJ_mol']:12.2f}  {TS:12.2f}  {S_J:12.2f}")


def main():
    csv_path = os.path.join(RESULTS_DIR, "free_energy_vs_T.csv")
    if not os.path.exists(csv_path):
        print("ERROR: free_energy_vs_T.csv not found. "
              "Run analysis/collect_results.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path).sort_values("T_K")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  Plotting free energy vs temperature")
    print("=" * 55)

    print_table(df)
    plot_main(df)
    plot_entropy(df)
    plot_convergence_grid(df)

    print("\n  Done. Outputs:")
    print("    results/free_energy_vs_T.png   — main result")
    print("    results/entropy_vs_T.png       — S vs T")
    print("    results/convergence_grid.png   — H(t) per temperature")


if __name__ == "__main__":
    main()
