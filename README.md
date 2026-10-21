# Toluene Free Energy Workflow — ASE + MACE

Computes the Gibbs free energy of toluene G(T) across 15 temperature points
(100 K → 1500 K, 100 K increments) using NPT molecular dynamics with the
MACE-MP-0 universal machine learning potential.

## Science

```
G(T) = H(T) - T·S(T)
```

- **H(T)** — ensemble average of enthalpy from NPT MD log
- **S(T)** — computed via two methods:
  - **2PT** (two-phase thermodynamics): entropy from the power spectrum of
    the velocity autocorrelation function (VACF). More physical.
  - **Schlitter**: upper bound from positional covariance matrix. Fast cross-check.

All 15 temperature points run simultaneously as a SLURM job array.

## Workflow

```
Step 1  structures/build_box.py
        → builds 10-molecule toluene box → structures/toluene_box.xyz

Step 2  bash job_scripts/submit_all.sh
        → submits SLURM array job (one task per temperature)
        → each task runs md/run_md.py:
            - 2 ps NVT equilibration
            - 10 ps NPT production, dump every 1 fs
            - computes G(T) and writes md/trajectories/<T>K/results.json

Step 3  python analysis/collect_results.py
        → gathers all results.json → results/free_energy_vs_T.csv

Step 4  python analysis/plot_free_energy.py
        → results/free_energy_vs_T.png   (main result: G, H, TS vs T)
        → results/entropy_vs_T.png
        → results/convergence_grid.png
```

## Quickstart

```bash
# 1. Clone and configure
git clone https://github.com/wzummo/toluene-bdfe.git
cd toluene-bdfe
# Edit config.yaml: set conda_env and partition

# 2. Activate your environment
conda activate your_conda_env_name

# 3. Build simulation box
python structures/build_box.py

# 4. Submit all 15 MD jobs in parallel
bash job_scripts/submit_all.sh

# 5. After jobs finish (~1 hr):
python analysis/collect_results.py
python analysis/plot_free_energy.py
```

## Configuration

Edit `config.yaml`:
```yaml
conda_env: your_conda_env_name    # your conda env with ASE + MACE
partition: your_partition         # SLURM partition
mace_device: cuda                 # cuda or cpu
n_molecules: 10                   # molecules in simulation box
prod_ps: 10.0                     # production run length
temperatures_K: [100,200,...,1500] # temperature grid
```

## Output

```
results/
├── free_energy_vs_T.csv     — G, H, S, TS at each temperature
├── free_energy_vs_T.png     — main plot
├── entropy_vs_T.png         — entropy comparison
└── convergence_grid.png     — H(t) convergence per T

md/trajectories/<T>K/
├── toluene_<T>K.traj        — ASE trajectory (every 1 fs)
├── thermo.csv               — step, T, P, E_pot, E_kin, H, V
└── results.json             — G, H, S for this temperature
```

## Requirements

- Python >= 3.9
- ASE >= 3.22
- MACE-torch >= 0.3.0
- numpy, scipy, matplotlib, pandas, pyyaml
- SLURM scheduler
- GPU recommended (MACE runs ~10x faster on GPU)
