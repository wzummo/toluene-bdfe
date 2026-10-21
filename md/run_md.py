"""
run_md.py
---------
Runs a single NPT MD simulation for toluene at a given temperature
using ASE + MACE universal potential.

Stores trajectory every 1 fs. Computes Gibbs free energy via
thermodynamic integration from ensemble averages + 2PT entropy
from the velocity autocorrelation function (VACF).

Usage (called by SLURM job script):
    python md/run_md.py --temp 300 --outdir md/trajectories/300K

Output files:
    <outdir>/toluene_<T>K.traj      ASE trajectory (every dump_every_fs steps)
    <outdir>/thermo.csv             time-series: T, P, E_pot, E_kin, H, V
    <outdir>/results.json           G, H, S, TS, convergence diagnostics
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config

# ── ASE imports ───────────────────────────────────────────────────────────────
try:
    from ase.io import read, write, Trajectory
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                              Stationary, ZeroRotation)
    from ase import units
except ImportError:
    print("ERROR: ASE not found. Activate your conda environment.")
    sys.exit(1)

try:
    from mace.calculators import mace_mp
except ImportError:
    print("ERROR: MACE not found. Install with: pip install mace-torch")
    sys.exit(1)

EV_TO_KJ_MOL = 96.485
KB_EV        = 8.617333e-5   # eV/K
ATM_TO_EV_ANG3 = 1.01325e5 * 1e-30 / 1.60217663e-19  # 1 atm in eV/Ang^3


# ── Thermodynamic logger ──────────────────────────────────────────────────────

class ThermoLogger:
    """Attaches to ASE dynamics and records thermodynamic quantities each step."""

    def __init__(self, atoms, dynamics, outdir: str, T_target: float, P_target_atm: float):
        self.atoms      = atoms
        self.dyn        = dynamics
        self.T_target   = T_target
        self.P_target   = P_target_atm * units.Pascal * 1e5  # Pa in ASE units
        self.outdir     = outdir
        self.log_path   = os.path.join(outdir, "thermo.csv")
        self.steps      = []
        self.times_ps   = []
        self.temps      = []
        self.pressures  = []
        self.e_pots     = []
        self.e_kins     = []
        self.enthalpies = []
        self.volumes    = []

        # Write CSV header
        with open(self.log_path, "w") as f:
            f.write("step,time_ps,temp_K,press_bar,e_pot_eV,e_kin_eV,"
                    "enthalpy_eV,volume_ang3\n")

    def __call__(self):
        atoms  = self.atoms
        step   = self.dyn.get_number_of_steps()
        dt_ps  = self.dyn.dt / units.fs / 1000.0   # ASE dt is in fs internally
        t_ps   = step * dt_ps

        e_pot  = atoms.get_potential_energy()          # eV
        e_kin  = atoms.get_kinetic_energy()            # eV
        T_inst = atoms.get_temperature()               # K

        # Stress → pressure (eV/Ang^3 → bar)
        try:
            stress = atoms.get_stress(voigt=True)      # eV/Ang^3, Voigt
            P_bar  = -np.mean(stress[:3]) * 1e4 / 1.01325  # eV/Ang^3 → bar
        except Exception:
            P_bar  = 0.0

        V_ang3 = atoms.get_volume()                    # Ang^3
        PV_eV  = P_bar * 1.01325e-4 * V_ang3 * 1.60217663e-19 / 1.60217663e-19
        # Simpler: PV in eV = P[bar]*1e5[Pa/bar] * V[Ang^3]*1e-30[m^3/Ang^3] / e[J/eV]
        PV_eV  = P_bar * 1e5 * V_ang3 * 1e-30 / 1.60217663e-19
        H_eV   = e_pot + e_kin + PV_eV                # enthalpy

        self.steps.append(step)
        self.times_ps.append(t_ps)
        self.temps.append(T_inst)
        self.pressures.append(P_bar)
        self.e_pots.append(e_pot)
        self.e_kins.append(e_kin)
        self.enthalpies.append(H_eV)
        self.volumes.append(V_ang3)

        # Append to CSV every step
        with open(self.log_path, "a") as f:
            f.write(f"{step},{t_ps:.6f},{T_inst:.4f},{P_bar:.4f},"
                    f"{e_pot:.8f},{e_kin:.8f},{H_eV:.8f},{V_ang3:.4f}\n")

        # Print to stdout every 100 steps
        if step % 100 == 0:
            print(f"  step {step:6d} | t={t_ps:.3f} ps | "
                  f"T={T_inst:.1f} K | P={P_bar:.1f} bar | "
                  f"H={H_eV:.4f} eV", flush=True)


# ── Free energy calculation ───────────────────────────────────────────────────

def compute_vacf(velocities: np.ndarray, dt_fs: float) -> tuple:
    """
    Compute normalised VACF from velocity array.
    velocities: shape (n_frames, n_atoms, 3)  in Ang/ps (ASE units)
    Returns (time_fs, vacf)
    """
    n_frames, n_atoms, _ = velocities.shape
    max_lag = n_frames // 2

    # Flatten to (n_frames, 3*n_atoms)
    V = velocities.reshape(n_frames, -1)

    vacf = np.zeros(max_lag)
    for lag in range(max_lag):
        vacf[lag] = np.mean(V[:n_frames - lag] * V[lag:n_frames]) if lag == 0 \
                    else np.mean(np.sum(V[:n_frames - lag] * V[lag:], axis=1))

    # Simpler vectorised version
    vacf = np.array([
        np.mean(np.einsum('ij,ij->', V[:n_frames-lag], V[lag:]) / ((n_frames - lag) * V.shape[1]))
        if lag > 0 else np.mean(V**2)
        for lag in range(max_lag)
    ])
    vacf /= vacf[0]
    time_fs = np.arange(max_lag) * dt_fs
    return time_fs, vacf


def entropy_2pt(velocities: np.ndarray, dt_fs: float, T: float,
                masses_amu: np.ndarray) -> float:
    """
    Estimate vibrational entropy via the 2PT (two-phase thermodynamics) method.
    Uses the power spectrum of the VACF and harmonic oscillator formula.

    Returns S in eV/K (per simulation box).
    """
    from scipy.fft import rfft, rfftfreq
    from scipy.integrate import trapezoid

    n_frames, n_atoms, _ = velocities.shape
    dt_s = dt_fs * 1e-15

    KB   = 1.380649e-23
    HBAR = 1.054571817e-34
    EV   = 1.60217663e-19

    S_total = 0.0

    # Per-atom entropy contributions
    for a in range(n_atoms):
        m_kg = masses_amu[a] * 1.66053906660e-27

        for dim in range(3):
            v_series = velocities[:, a, dim]  # Ang/ps → m/s
            v_si = v_series * 1e-10 / 1e-12  # Ang/ps to m/s

            # Power spectrum
            psd = np.abs(rfft(v_si)) ** 2
            freqs = rfftfreq(n_frames, d=dt_s)  # Hz

            # Normalise PSD to get density of states
            psd_norm = psd / (np.trapz(psd, freqs) + 1e-300)

            # Entropy integral: S = kB * integral g(nu) * s_HO(nu,T) dnu
            for i, (nu, g) in enumerate(zip(freqs[1:], psd_norm[1:]), 1):
                if nu < 1e9:  # skip DC / near-DC
                    continue
                x = HBAR * 2 * np.pi * nu / (KB * T)
                if x > 500:
                    continue
                ex = np.exp(x)
                # Harmonic oscillator entropy per mode
                s_ho = KB * (x / (ex - 1) - np.log(1 - np.exp(-x)))
                dnu  = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                S_total += g * s_ho * dnu

    return S_total / EV  # J/K → eV/K


def schlitter_entropy(positions: np.ndarray, T: float,
                      masses_amu: np.ndarray) -> float:
    """
    Upper bound entropy via Schlitter's formula.
    positions: (n_frames, n_atoms, 3) in Angstrom
    Returns S in eV/K per box.
    """
    KB   = 1.380649e-23
    HBAR = 1.054571817e-34
    EV   = 1.60217663e-19
    AMU  = 1.66053906660e-27
    ANG  = 1e-10
    e    = np.e

    n_frames, n_atoms, _ = positions.shape
    pos_flat = positions.reshape(n_frames, -1) * ANG   # m

    # Covariance
    dpos  = pos_flat - pos_flat.mean(axis=0)
    sigma = (dpos.T @ dpos) / (n_frames - 1)           # m^2

    # Mass weighting
    m_kg = np.repeat(masses_amu * AMU, 3)
    M    = np.outer(np.sqrt(m_kg), np.sqrt(m_kg))

    # Schlitter matrix
    prefactor = KB * T * e**2 / HBAR**2
    A = np.eye(3 * n_atoms) + prefactor * M * sigma

    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return 0.0

    S_J_K = 0.5 * KB * logdet
    return S_J_K / EV


def compute_free_energy(thermo: ThermoLogger, traj_path: str,
                        T: float, P_atm: float,
                        cfg: dict) -> dict:
    """
    Compute G(T,P) from the production MD run.
    Uses:
      H    — ensemble average of enthalpy from thermo log
      S    — 2PT entropy from VACF (primary) + Schlitter (cross-check)
      G    = <H> - T*S
    """
    print("\n  Computing free energy...")

    # Drop first 20% of frames as additional burn-in
    n     = len(thermo.enthalpies)
    start = n // 5
    H_arr = np.array(thermo.enthalpies[start:])
    T_arr = np.array(thermo.temps[start:])
    P_arr = np.array(thermo.pressures[start:])
    V_arr = np.array(thermo.volumes[start:])

    H_mean = np.mean(H_arr)
    H_std  = np.std(H_arr)
    T_mean = np.mean(T_arr)
    P_mean = np.mean(P_arr)
    V_mean = np.mean(V_arr)

    # Block averaging for error estimate
    n_blocks = 5
    block_size = len(H_arr) // n_blocks
    block_means = [H_arr[i*block_size:(i+1)*block_size].mean()
                   for i in range(n_blocks)]
    H_sem = np.std(block_means) / np.sqrt(n_blocks)

    print(f"  <H>  = {H_mean:.6f} ± {H_sem:.6f} eV")
    print(f"  <T>  = {T_mean:.2f} K  (target {T} K)")
    print(f"  <P>  = {P_mean:.2f} bar  (target {P_atm * 1.01325:.2f} bar)")
    print(f"  <V>  = {V_mean:.2f} Å³")

    # Load trajectory for entropy
    print("  Loading trajectory for entropy calculation...")
    traj = read(traj_path, index=f"{start}:")

    positions  = np.array([atoms.get_positions()          for atoms in traj])
    velocities = np.array([atoms.get_velocities()         for atoms in traj])
    masses     = traj[0].get_masses()   # amu

    # Convert ASE velocities (Ang/fs) to consistent units
    dt_fs = cfg["timestep_fs"] * cfg.get("dump_every_fs", 1)

    # ── 2PT entropy ───────────────────────────────────────────
    S_2pt = 0.0
    try:
        S_2pt = entropy_2pt(velocities, dt_fs, T, masses)
        print(f"  S (2PT)       = {S_2pt:.6f} eV/K")
        print(f"  TS (2PT)      = {T * S_2pt:.6f} eV")
    except Exception as ex:
        print(f"  WARNING: 2PT entropy failed: {ex}")

    # ── Schlitter entropy (cross-check) ───────────────────────
    S_schlitter = 0.0
    try:
        S_schlitter = schlitter_entropy(positions, T, masses)
        print(f"  S (Schlitter) = {S_schlitter:.6f} eV/K  [upper bound]")
    except Exception as ex:
        print(f"  WARNING: Schlitter entropy failed: {ex}")

    # ── Gibbs free energy ─────────────────────────────────────
    G_2pt       = H_mean - T * S_2pt
    G_schlitter = H_mean - T * S_schlitter

    print(f"\n  G (2PT)       = {G_2pt:.6f} eV  ({G_2pt * EV_TO_KJ_MOL:.2f} kJ/mol)")
    print(f"  G (Schlitter) = {G_schlitter:.6f} eV  ({G_schlitter * EV_TO_KJ_MOL:.2f} kJ/mol)")

    return {
        "temperature_K":   T,
        "pressure_atm":    P_atm,
        "n_frames_prod":   len(H_arr),
        "ensemble": {
            "H_mean_eV":   float(H_mean),
            "H_std_eV":    float(H_std),
            "H_sem_eV":    float(H_sem),
            "T_mean_K":    float(T_mean),
            "P_mean_bar":  float(P_mean),
            "V_mean_ang3": float(V_mean),
        },
        "entropy": {
            "S_2pt_eV_K":       float(S_2pt),
            "S_schlitter_eV_K": float(S_schlitter),
        },
        "free_energy": {
            "G_2pt_eV":           float(G_2pt),
            "G_2pt_kJ_mol":       float(G_2pt * EV_TO_KJ_MOL),
            "G_schlitter_eV":     float(G_schlitter),
            "G_schlitter_kJ_mol": float(G_schlitter * EV_TO_KJ_MOL),
        },
        "block_avg": {
            "H_blocks_eV": [float(b) for b in block_means],
            "H_sem_eV":    float(H_sem),
        }
    }


# ── Main MD runner ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp",   type=float, required=True, help="Temperature in K")
    parser.add_argument("--outdir", type=str,   required=True, help="Output directory")
    args = parser.parse_args()

    cfg    = load_config()
    T      = args.temp
    P_atm  = cfg["pressure_atm"]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    dt_fs        = cfg["timestep_fs"]
    equil_steps  = int(cfg["equil_ps"]  * 1000 / dt_fs)
    prod_steps   = int(cfg["prod_ps"]   * 1000 / dt_fs)
    dump_every   = int(cfg["dump_every_fs"] / dt_fs)
    traj_path    = os.path.join(outdir, f"toluene_{int(T)}K.traj")

    print("=" * 55)
    print(f"  ASE + MACE MD — Toluene at {T} K")
    print(f"  Equil: {cfg['equil_ps']} ps  |  Prod: {cfg['prod_ps']} ps")
    print(f"  Dump every {cfg['dump_every_fs']} fs")
    print("=" * 55)

    # ── Load structure ────────────────────────────────────────
    box_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "structures", "toluene_box.xyz")
    if not os.path.exists(box_path):
        print(f"ERROR: {box_path} not found. Run structures/build_box.py first.")
        sys.exit(1)

    atoms = read(box_path)
    print(f"  Loaded box: {len(atoms)} atoms, "
          f"cell = {atoms.cell.diagonal().round(2)} Å")

    # ── MACE calculator ───────────────────────────────────────
    print(f"  Loading MACE-MP-0 ({cfg['mace_model']})...")
    t0 = time.time()
    calc = mace_mp(
        model=cfg["mace_model"],
        device=cfg["mace_device"],
        default_dtype="float32",
    )
    atoms.calc = calc
    print(f"  MACE loaded in {time.time()-t0:.1f} s")

    # ── Initial velocities ────────────────────────────────────
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(42))
    Stationary(atoms)       # zero net momentum
    ZeroRotation(atoms)     # zero net angular momentum

    # ── Phase 1: NVT equilibration ────────────────────────────
    print(f"\n  Phase 1: NVT equilibration ({cfg['equil_ps']} ps)...")
    dt_ase = dt_fs * units.fs

    nvt = NVTBerendsen(
        atoms,
        timestep=dt_ase,
        temperature_K=T,
        taut=100 * units.fs,    # thermostat time constant
        logfile=os.path.join(outdir, "equil.log"),
        loginterval=100,
    )
    nvt.run(equil_steps)
    print(f"  Equilibration complete. T = {atoms.get_temperature():.1f} K")

    # ── Phase 2: NPT production ───────────────────────────────
    print(f"\n  Phase 2: NPT production ({cfg['prod_ps']} ps)...")

    # ASE NPTBerendsen: pressure in bar
    P_bar = P_atm * 1.01325

    npt = NPTBerendsen(
        atoms,
        timestep=dt_ase,
        temperature_K=T,
        pressure_au=P_bar * units.bar,
        taut=100  * units.fs,
        taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,  # ~water compressibility
        logfile=os.path.join(outdir, "prod.log"),
        loginterval=100,
    )

    # Attach trajectory writer
    traj_writer = Trajectory(traj_path, "w", atoms)
    npt.attach(traj_writer.write, interval=dump_every)

    # Attach thermo logger
    logger = ThermoLogger(atoms, npt, outdir, T, P_atm)
    npt.attach(logger, interval=1)

    t_start = time.time()
    npt.run(prod_steps)
    t_end = time.time()
    traj_writer.close()

    print(f"\n  Production complete in {(t_end-t_start)/60:.1f} min")
    print(f"  Trajectory: {traj_path}")

    # ── Free energy ───────────────────────────────────────────
    results = compute_free_energy(logger, traj_path, T, P_atm, cfg)

    out_json = os.path.join(outdir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {out_json}")
    print(f"  G(2PT) = {results['free_energy']['G_2pt_kJ_mol']:.2f} kJ/mol")


if __name__ == "__main__":
    main()
