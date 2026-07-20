#!/bin/bash

#SBATCH --job-name=fa2fwd
#SBATCH --output=fa2fwd.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:H100:1

set -eo pipefail

if [[ -f /home/spack/spack/share/spack/setup-env.sh ]]; then
  source /home/spack/spack/share/spack/setup-env.sh
  spack load cmake@3.28.6
  spack load cuda@12.9.0
  spack load python@3.13.0
fi

# Keep nounset/pipefail for diagnostics, but allow individual benchmark commands
# to return nonzero so their CSV failures can be recorded and later scans continue.
set +e
set -u

# Submit this file from the repository root. FA2_PYTHON takes precedence; when
# it is unset, automatically use the environment created by the setup job.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set; submit this script with sbatch}"

if [[ -n "${FA2_PYTHON:-}" ]]; then
    FA2_PYTHON="${FA2_PYTHON}"
elif [[ -x "${SLURM_SUBMIT_DIR}/.venv-fa2/bin/python" ]]; then
    FA2_PYTHON="${SLURM_SUBMIT_DIR}/.venv-fa2/bin/python"
else
    echo "ERROR: no experiment Python environment was found."
    echo "Run: sbatch setup_flash_attention_env.sbatch"
    echo "Or submit with: sbatch --export=ALL,FA2_PYTHON=/path/to/python run_flash_attention_experiment.sbatch"
    exit 2
fi

if [[ "${FA2_PYTHON}" == */* ]]; then
    if [[ ! -x "${FA2_PYTHON}" ]]; then
        echo "ERROR: FA2_PYTHON is not executable: ${FA2_PYTHON}"
        exit 2
    fi
else
    FA2_PYTHON_COMMAND=$(command -v "${FA2_PYTHON}" || true)
    if [[ -z "${FA2_PYTHON_COMMAND}" ]]; then
        echo "ERROR: FA2_PYTHON was not found on PATH: ${FA2_PYTHON}"
        exit 2
    fi
    FA2_PYTHON="${FA2_PYTHON_COMMAND}"
fi

FA2_RESULT_DIR="results/slurm_${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
FA2_FAILURES=0

mkdir -p "${FA2_RESULT_DIR}"
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "repository=${SLURM_SUBMIT_DIR}"
echo "python=${FA2_PYTHON}"
FA2_PYTHON_RESOLVED=$(readlink -f "${FA2_PYTHON}")
echo "python_resolved=${FA2_PYTHON_RESOLVED}"
echo "results=${FA2_RESULT_DIR}"

# Fail before starting the experiment if the selected environment is unusable.
"${FA2_PYTHON}" -c \
  'import torch, triton; assert torch.cuda.is_available(), "CUDA is unavailable"; p=torch.cuda.get_device_properties(torch.cuda.current_device()); print(f"torch={torch.__version__} triton={triton.__version__} device={p.name} sm_count={p.multi_processor_count}")'
FA2_PREFLIGHT_STATUS=$?
if (( FA2_PREFLIGHT_STATUS != 0 )); then
    echo "Preflight failed with exit code ${FA2_PREFLIGHT_STATUS}; no experiment was run."
    exit "${FA2_PREFLIGHT_STATUS}"
fi

# Correctness is a gate: do not benchmark a kernel that fails its reference test.
echo "===== correctness ====="
"${FA2_PYTHON}" test_flash_attention_correctness.py
FA2_CORRECTNESS_STATUS=$?
if (( FA2_CORRECTNESS_STATUS != 0 )); then
    echo "Correctness failed with exit code ${FA2_CORRECTNESS_STATUS}; benchmarks were skipped."
    exit "${FA2_CORRECTNESS_STATUS}"
fi

run_benchmark() {
    FA2_STEP_NAME=$1
    shift
    echo "===== ${FA2_STEP_NAME} ====="
    "${FA2_PYTHON}" benchmark_flash_attention.py "$@" \
      --warmup 10 \
      --repeat 50 \
      --quantile 0.5 \
      --output-dir "${FA2_RESULT_DIR}"
    FA2_STEP_STATUS=$?
    if (( FA2_STEP_STATUS != 0 )); then
        # A benchmark returns nonzero when one point is OOM/compile_error/runtime_error.
        # It still writes all completed and failed rows, so continue with later scans.
        echo "${FA2_STEP_NAME} completed with exit code ${FA2_STEP_STATUS}; inspect its CSV status/error_message columns."
        FA2_FAILURES=$((FA2_FAILURES + 1))
    fi
}

# Experiment A: B=1, H=32, N sweep, D={64,128}, FP16/BF16, all providers.
run_benchmark sequence \
  --experiment sequence \
  --providers torch_explicit torch_sdpa triton_fa2 \
  --config-mode fixed \
  --block-m 64 --block-n 64 --num-warps 4 --num-stages 3 \
  --measure-memory

# Experiment B1: fixed Triton configuration plus both PyTorch baselines.
run_benchmark head_dim_fixed \
  --experiment head_dim \
  --providers torch_explicit torch_sdpa triton_fa2 \
  --config-mode fixed \
  --block-m 64 --block-n 64 --num-warps 4 --num-stages 3

# Experiment B2: select the best Triton configuration independently per shape.
run_benchmark head_dim_tuned \
  --experiment head_dim \
  --provider triton_fa2 \
  --config-mode tuned

# Experiment C: H={1,2,4,8,16,32,64} at N=512, D=64.
run_benchmark parallelism \
  --experiment parallelism \
  --providers torch_explicit torch_sdpa triton_fa2 \
  --config-mode fixed \
  --block-m 64 --block-n 64 --num-warps 4 --num-stages 3

# Experiment D: all requested tile configurations on the three representative shapes.
run_benchmark tile \
  --experiment tile

# Plot only the CSV files produced by this Slurm job. Missing NCU columns are
# intentionally skipped by the plotting script.
shopt -s nullglob
FA2_CSV_FILES=("${FA2_RESULT_DIR}"/*.csv)
if (( ${#FA2_CSV_FILES[@]} > 0 )); then
    echo "===== plots ====="
    "${FA2_PYTHON}" plot_flash_attention_results.py \
      "${FA2_CSV_FILES[@]}" \
      --output-dir "${FA2_RESULT_DIR}/plots"
    FA2_PLOT_STATUS=$?
    if (( FA2_PLOT_STATUS != 0 )); then
        echo "Plotting failed with exit code ${FA2_PLOT_STATUS}."
        FA2_FAILURES=$((FA2_FAILURES + 1))
    fi
else
    echo "No benchmark CSV files were produced; plotting was skipped."
    FA2_FAILURES=$((FA2_FAILURES + 1))
fi

echo "===== summary ====="
echo "result_dir=${FA2_RESULT_DIR}"
echo "nonzero_benchmark_or_plot_steps=${FA2_FAILURES}"
echo "Nsight Compute was not run. Generate its commands separately with generate_ncu_commands.py."

# Partial benchmark failures are represented in CSV and reported above. Keep the
# Slurm allocation successful so later results are not mistaken for missing data.
exit 0
