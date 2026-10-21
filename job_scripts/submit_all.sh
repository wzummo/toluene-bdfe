#!/bin/bash
# submit_all.sh
# -------------
# Submits all 15 temperature MD jobs in parallel as a SLURM job array.
# Each job runs run_md.py at one temperature point.
#
# Usage:
#   bash job_scripts/submit_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Read config ───────────────────────────────────────────────
PARTITION=$(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(c['partition'])")

ACCOUNT=$(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(c.get('account', ''))")

CONDA_ENV=$(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(c['conda_env'])")

WALLTIME=$(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(c['walltime'])")

CPUS=$(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(c['cpus_per_task'])")

GPUS=$(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(c['gpus_per_task'])")

# ── Temperature list from config ──────────────────────────────
TEMPS=($(python3 -c "
import yaml
c = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(' '.join(str(t) for t in c['temperatures_K']))"))

N_TEMPS=${#TEMPS[@]}

echo "================================================="
echo "  Toluene Free Energy — MACE MD"
echo "  Submitting ${N_TEMPS} temperature points"
echo "  Temps: ${TEMPS[*]}"
echo "================================================="

# ── Build SLURM account line ──────────────────────────────────
ACCT_LINE=""
if [ -n "${ACCOUNT}" ]; then
    ACCT_LINE="#SBATCH --account=${ACCOUNT}"
fi

# ── Build GPU line ────────────────────────────────────────────
GPU_LINE=""
if [ "${GPUS}" -gt 0 ]; then
    GPU_LINE="#SBATCH --gres=gpu:${GPUS}"
fi

# ── Write array job script ────────────────────────────────────
JOB_SCRIPT="${REPO_ROOT}/job_scripts/array_job.sh"

# Write temperature array as bash array inside the script
TEMP_ARRAY_STR="TEMPS=(${TEMPS[*]})"

cat > "${JOB_SCRIPT}" << SBATCH
#!/bin/bash
#SBATCH --job-name=toluene_mace
#SBATCH --partition=${PARTITION}
${ACCT_LINE}
${GPU_LINE}
#SBATCH --array=0-$((N_TEMPS - 1))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${REPO_ROOT}/job_scripts/logs/slurm_%A_%a.out
#SBATCH --error=${REPO_ROOT}/job_scripts/logs/slurm_%A_%a.err

# ── Temperature for this array task ──────────────────────────
${TEMP_ARRAY_STR}
TEMP=\${TEMPS[\$SLURM_ARRAY_TASK_ID]}
OUTDIR="${REPO_ROOT}/md/trajectories/\${TEMP}K"
mkdir -p "\${OUTDIR}"

echo "============================================"
echo " Task \${SLURM_ARRAY_TASK_ID} — T = \${TEMP} K"
echo " Node: \$(hostname)"
echo " Started: \$(date)"
echo "============================================"

# ── Activate conda ────────────────────────────────────────────
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# ── Verify MACE is available ──────────────────────────────────
python3 -c "from mace.calculators import mace_mp; print('MACE OK')"

# ── Run MD ───────────────────────────────────────────────────
python3 "${REPO_ROOT}/md/run_md.py" \
    --temp   \${TEMP} \
    --outdir "\${OUTDIR}"

EXIT_CODE=\$?
if [ \${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: run_md.py failed with exit code \${EXIT_CODE}"
    exit \${EXIT_CODE}
fi

echo "Finished T=\${TEMP}K at \$(date)"
SBATCH

mkdir -p "${REPO_ROOT}/job_scripts/logs"

# ── Submit ────────────────────────────────────────────────────
JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")

echo ""
echo "  Submitted job array: ${JOB_ID}"
echo "  Array tasks: 0–$((N_TEMPS - 1))  (one per temperature)"
echo ""
echo "  Monitor:  squeue -j ${JOB_ID}"
echo "  Logs:     ${REPO_ROOT}/job_scripts/logs/"
echo "  Trajs:    ${REPO_ROOT}/md/trajectories/<T>K/"
echo ""
echo "  When all jobs finish, run:"
echo "    python analysis/collect_results.py"
echo "    python analysis/plot_free_energy.py"

# ── Save job ID for downstream dependency ─────────────────────
echo "${JOB_ID}" > "${REPO_ROOT}/job_scripts/array_job_id.txt"
