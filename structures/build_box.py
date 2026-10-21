"""
build_box.py
------------
Builds a simulation box of N toluene molecules at a target density
and writes it to structures/toluene_box.xyz

Usage:
    python structures/build_box.py
    python structures/build_box.py --nmol 10 --density 0.867
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config

try:
    from ase import Atoms
    from ase.io import write
except ImportError:
    print("ERROR: ASE not available. Activate your conda environment.")
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH  = os.path.join(REPO_ROOT, "structures", "toluene_box.xyz")

MW_TOLUENE   = 92.14        # g/mol
NA           = 6.02214076e23
ANG3_PER_CM3 = 1e24


def single_toluene() -> Atoms:
    """One toluene molecule centred at origin."""
    positions = [
        ( 0.000,  1.397,  0.000),  # C1 (ipso)
        ( 1.210,  0.698,  0.000),  # C2
        ( 1.210, -0.698,  0.000),  # C3
        ( 0.000, -1.397,  0.000),  # C4
        (-1.210, -0.698,  0.000),  # C5
        (-1.210,  0.698,  0.000),  # C6
        ( 0.000,  2.507,  0.000),  # C7 (methyl)
        ( 2.156,  1.243,  0.000),  # H2
        ( 2.156, -1.243,  0.000),  # H3
        ( 0.000, -2.484,  0.000),  # H4
        (-2.156, -1.243,  0.000),  # H5
        (-2.156,  1.243,  0.000),  # H6
        ( 0.513,  2.876,  0.890),  # H7a
        ( 0.513,  2.876, -0.890),  # H7b
        (-1.026,  2.876,  0.000),  # H7c
    ]
    mol = Atoms(['C']*7 + ['H']*8, positions=positions)
    mol.positions -= mol.positions.mean(axis=0)
    return mol


def random_rotation(seed: int = None) -> np.ndarray:
    """Random 3D rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(v)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def build_box(n_mol: int, density_gcc: float, seed: int = 42) -> Atoms:
    """
    Place n_mol toluene molecules randomly in a cubic periodic box.
    Uses a simple grid placement with random rotations and small random
    displacements — good enough for a hackathon start.
    For production, replace with Packmol.
    """
    # Box size from density
    mass_g   = n_mol * MW_TOLUENE / NA
    vol_cm3  = mass_g / density_gcc
    vol_ang3 = vol_cm3 * ANG3_PER_CM3
    L        = vol_ang3 ** (1/3)

    print(f"  Box length:  {L:.3f} Å  ({L/10:.3f} nm)")
    print(f"  Volume:      {vol_ang3:.1f} Å³")
    print(f"  Density:     {density_gcc} g/cm³")

    single = single_toluene()
    n_side = int(np.ceil(n_mol ** (1/3)))
    spacing = L / n_side

    rng = np.random.default_rng(seed)
    all_pos = []
    all_sym = []
    placed  = 0

    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if placed >= n_mol:
                    break
                # Grid centre + small random jitter (up to 10% of spacing)
                centre = np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * spacing
                centre += rng.uniform(-0.1, 0.1, 3) * spacing

                # Random rotation
                R = random_rotation(seed=seed + placed)
                rotated = single.positions @ R.T + centre

                all_pos.extend(rotated.tolist())
                all_sym.extend(single.get_chemical_symbols())
                placed += 1

    box = Atoms(
        symbols=all_pos and all_sym,
        positions=all_pos,
        cell=[L, L, L],
        pbc=True,
    )
    box = Atoms(
        symbols=all_sym,
        positions=all_pos,
        cell=[L, L, L],
        pbc=True,
    )
    return box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmol",    type=int,   default=None)
    parser.add_argument("--density", type=float, default=None)
    args = parser.parse_args()

    cfg     = load_config()
    n_mol   = args.nmol    or cfg["n_molecules"]
    density = args.density or cfg["density_gcc"]

    print("=" * 50)
    print("  Building toluene simulation box")
    print(f"  Molecules: {n_mol}")
    print("=" * 50)

    box = build_box(n_mol, density)
    print(f"  Total atoms: {len(box)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    write(OUT_PATH, box)
    print(f"  Written: {OUT_PATH}")
    print(f"\n  Next: bash job_scripts/submit_all.sh")


if __name__ == "__main__":
    main()
