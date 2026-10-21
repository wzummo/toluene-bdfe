"""
collect_results.py
------------------
Walks md/trajectories/<T>K/results.json for all temperature points
and assembles them into a single results/free_energy_vs_T.csv

Usage:
    python analysis/collect_results.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_DIR    = os.path.join(REPO_ROOT, "md", "trajectories")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def main():
    cfg = load_config()
    temps = cfg["temperatures_K"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 55)
    print("  Collecting results across temperature points")
    print("=" * 55)

    rows = []
    missing = []

    for T in temps:
        result_path = os.path.join(TRAJ_DIR, f"{T}K", "results.json")
        if not os.path.exists(result_path):
            print(f"  MISSING: {T} K — {result_path}")
            missing.append(T)
            continue

        with open(result_path) as f:
            r = json.load(f)

        fe  = r["free_energy"]
        ens = r["ensemble"]
        ent = r["entropy"]
        blk = r.get("block_avg", {})

        rows.append({
            "T_K":                T,
            "P_atm":              r["pressure_atm"],
            # Enthalpy
            "H_eV":               ens["H_mean_eV"],
            "H_kJ_mol":           ens["H_mean_eV"] * 96.485,
            "H_sem_eV":           ens.get("H_sem_eV", 0.0),
            # Entropy
            "S_2pt_eV_K":         ent["S_2pt_eV_K"],
            "S_schlitter_eV_K":   ent["S_schlitter_eV_K"],
            "TS_2pt_eV":          T * ent["S_2pt_eV_K"],
            "TS_schlitter_eV":    T * ent["S_schlitter_eV_K"],
            # Free energy
            "G_2pt_eV":           fe["G_2pt_eV"],
            "G_2pt_kJ_mol":       fe["G_2pt_kJ_mol"],
            "G_schlitter_eV":     fe["G_schlitter_eV"],
            "G_schlitter_kJ_mol": fe["G_schlitter_kJ_mol"],
            # Ensemble averages
            "T_mean_K":           ens["T_mean_K"],
            "P_mean_bar":         ens["P_mean_bar"],
            "V_mean_ang3":        ens["V_mean_ang3"],
            "n_frames":           r["n_frames_prod"],
        })

        print(f"  {T:5.0f} K  |  G(2PT) = {fe['G_2pt_kJ_mol']:10.2f} kJ/mol  |  "
              f"H = {ens['H_mean_eV']*96.485:10.2f} kJ/mol  |  "
              f"TS = {T*ent['S_2pt_eV_K']*96.485:8.2f} kJ/mol")

    if missing:
        print(f"\n  WARNING: Missing results for T = {missing}")
        print(f"  Check job logs in job_scripts/logs/")

    if not rows:
        print("\n  No results found. Have the MD jobs finished?")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("T_K")
    out = os.path.join(RESULTS_DIR, "free_energy_vs_T.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    print(f"  Temperatures collected: {len(rows)}/{len(temps)}")

    # Quick convergence check: flag T points where H_sem > 0.1 eV
    bad = df[df["H_sem_eV"] > 0.1]
    if not bad.empty:
        print(f"\n  WARNING: Poor convergence at T = {bad['T_K'].tolist()} K")
        print(f"  Consider longer prod_ps for these points")

    print("\n  Next: python analysis/plot_free_energy.py")


if __name__ == "__main__":
    main()
