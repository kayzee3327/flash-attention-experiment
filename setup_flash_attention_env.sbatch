#!/bin/bash

#SBATCH --job-name=fa2env
#SBATCH --output=fa2env.out
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

set -u

# Submit from the repository root. Override FA2_BASE_PYTHON when the cluster
# requires a module-provided Python, for example:
#   FA2_BASE_PYTHON=/path/to/python3 sbatch setup_flash_attention_env.sbatch
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set; submit this file with sbatch}"

FA2_BASE_PYTHON="${FA2_BASE_PYTHON:-python3}"
FA2_ENV_DIR="${FA2_ENV_DIR:-${SLURM_SUBMIT_DIR}/.venv-fa2}"

echo "job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
echo "host=$(hostname)"
echo "repository=${SLURM_SUBMIT_DIR}"
echo "base_python=${FA2_BASE_PYTHON}"
echo "environment=${FA2_ENV_DIR}"

"${FA2_BASE_PYTHON}" --version
"${FA2_BASE_PYTHON}" -m venv "${FA2_ENV_DIR}"

FA2_ENV_PYTHON="${FA2_ENV_DIR}/bin/python"
"${FA2_ENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

# By default pip selects PyTorch from its configured package index. For a
# cluster-specific CUDA wheel index, submit with for example:
#   FA2_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#     sbatch setup_flash_attention_env.sbatch
if [[ -n "${FA2_TORCH_INDEX_URL:-}" ]]; then
    echo "Installing PyTorch from FA2_TORCH_INDEX_URL=${FA2_TORCH_INDEX_URL}"
    "${FA2_ENV_PYTHON}" -m pip install \
      --index-url "${FA2_TORCH_INDEX_URL}" \
      torch
else
    echo "Installing PyTorch from pip's configured/default index"
    "${FA2_ENV_PYTHON}" -m pip install torch
fi

# Linux CUDA PyTorch wheels normally declare a compatible Triton dependency.
# Install Triton explicitly only when that wheel did not provide it.
if ! "${FA2_ENV_PYTHON}" -c 'import triton' >/dev/null 2>&1; then
    echo "PyTorch did not install Triton; installing Triton explicitly"
    "${FA2_ENV_PYTHON}" -m pip install triton
fi

"${FA2_ENV_PYTHON}" -m pip install pandas matplotlib

# Record an exact package snapshot inside the environment for later diagnosis.
"${FA2_ENV_PYTHON}" -m pip freeze > "${FA2_ENV_DIR}/requirements.lock.txt"

echo "===== environment verification ====="
"${FA2_ENV_PYTHON}" -c \
  'import matplotlib, pandas, torch, triton; assert torch.cuda.is_available(), "CUDA is unavailable inside the Slurm allocation"; index=torch.cuda.current_device(); props=torch.cuda.get_device_properties(index); print(f"python environment OK\ntorch={torch.__version__}\ntriton={triton.__version__}\npandas={pandas.__version__}\nmatplotlib={matplotlib.__version__}\ncuda_runtime={torch.version.cuda}\ndevice={props.name}\nsm_count={props.multi_processor_count}")'

echo "===== setup complete ====="
echo "Environment Python: ${FA2_ENV_PYTHON}"
echo "Package snapshot: ${FA2_ENV_DIR}/requirements.lock.txt"
echo "Submit the experiment with:"
echo "FA2_PYTHON=${FA2_ENV_PYTHON} sbatch run_flash_attention_experiment.sbatch"
echo "No correctness test, benchmark, or Nsight Compute command was run by this setup job."
