#!/usr/bin/env bash
# =============================================================================
# Resume pi05 (40k) or pi0 (20k) RoboTwin ICL arx5 finetune on H200.
#
# Differs from train_pi0[5]_robotwin_icl_arx5_8gpu.sh in three ways:
#   1. EXP_NAME is pinned to the original run name (no fresh DATE_TAG), so
#      train_pytorch.py's checkpoint_dir lookup finds the existing ckpt and
#      load_checkpoint() resumes from there.
#   2. --resume is passed to train_pytorch.py.
#   3. Paths (DATASET_DIR, CHECKPOINT_BASE_DIR) come from env so the same
#      script works on a cluster with completely different storage layout.
#
# train_pytorch.py:442 has been patched to skip the base ckpt load when
# resuming -- so the pi0_base_pytorch / pi05_base_pytorch dirs do NOT need
# to exist on H200.
#
# Usage:
#   bash resume_robotwin_icl_arx5_h200.sh pi05
#   bash resume_robotwin_icl_arx5_h200.sh pi0
#
# Required env (paths on H200):
#   DATASET_DIR             absolute path to robotwin-arx5-lerobot
#   CHECKPOINT_BASE_DIR     where hf_pull_resume_bundle.py downloaded into
#   WANDB_API_KEY           wandb key
# =============================================================================
set -euo pipefail

MODEL="${1:-}"
case "${MODEL}" in
    pi05)
        CONFIG_NAME="pi05_robotwin_icl_arx_x5"
        EXP_NAME="pi05_robotwin_arx5_icl_20260525_044230"
        EXPECTED_STEP=40000
        ;;
    pi0)
        CONFIG_NAME="pi0_robotwin_icl_arx_x5"
        EXP_NAME="pi0_robotwin_arx5_icl_20260525_121906"
        EXPECTED_STEP=20000
        ;;
    *)
        echo "usage: $0 {pi05|pi0}" >&2
        exit 2
        ;;
esac

# === Distributed shape ======================================================
export NUM_MACHINES="${MLP_WORKER_NUM:-${WORLD_SIZE:-${NUM_MACHINES:-1}}}"
export MACHINE_RANK="${MLP_ROLE_INDEX:-${RANK:-${MACHINE_RANK:-0}}}"
export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-localhost}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
export GPUS_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
TOTAL_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))

if [ "${TOTAL_PROCESSES}" -ne 8 ] && [ "${ALLOW_NON_8:-0}" != "1" ]; then
    echo "ERROR: this launcher targets 8 GPUs, got ${TOTAL_PROCESSES}." >&2
    exit 1
fi

# === Schedule ===============================================================
# Same as original launchers; resume continues the existing schedule.
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
export KEEP_PERIOD="${KEEP_PERIOD:-20000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

# === Paths ==================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON=${WORK_DIR}/.venv/bin/python
TORCHRUN=${WORK_DIR}/.venv/bin/torchrun

: "${DATASET_DIR:?DATASET_DIR env var required (absolute path to robotwin-arx5-lerobot on H200)}"
: "${CHECKPOINT_BASE_DIR:?CHECKPOINT_BASE_DIR env var required (where hf_pull_resume_bundle.py landed)}"

CKPT_RUN_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
CKPT_STEP_DIR="${CKPT_RUN_DIR}/${EXPECTED_STEP}"

# === wandb ==================================================================
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" != "disabled" && "${WANDB_MODE}" != "offline" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "ERROR: WANDB_API_KEY env var is required when WANDB_MODE=${WANDB_MODE}." >&2
        exit 3
    fi
fi

# === Runtime env ============================================================
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

# === Preflight ==============================================================
echo "=== resume ${MODEL} on H200 ==="
echo "  work_dir          = ${WORK_DIR}"
echo "  nodes=${NUM_MACHINES}  gpus_per_node=${GPUS_PER_NODE}  total=${TOTAL_PROCESSES}  rank=${MACHINE_RANK}"
echo "  master            = ${MASTER_ADDR}:${MASTER_PORT}"
echo "  config            = ${CONFIG_NAME}"
echo "  exp_name (pinned) = ${EXP_NAME}"
echo "  expected step     = ${EXPECTED_STEP}"
echo "  num_train_steps   = ${NUM_TRAIN_STEPS}"
echo "  save_interval     = ${SAVE_INTERVAL}   keep_period=${KEEP_PERIOD}"
echo "  log_interval      = ${LOG_INTERVAL}"
echo "  global_batch      = ${GLOBAL_BATCH_SIZE}  (per-device $((GLOBAL_BATCH_SIZE/TOTAL_PROCESSES)))"
echo "  num_workers       = ${NUM_WORKERS}"
echo "  dataset           = ${DATASET_DIR}"
echo "  checkpoint_base   = ${CHECKPOINT_BASE_DIR}"
echo "  ckpt_step_dir     = ${CKPT_STEP_DIR}"
echo "  wandb_mode        = ${WANDB_MODE}"

test -x "${VENV_PYTHON}" || { echo "ERROR: missing venv python: ${VENV_PYTHON}" >&2; exit 1; }
test -x "${TORCHRUN}"   || { echo "ERROR: missing torchrun: ${TORCHRUN}" >&2; exit 1; }
test -d "${DATASET_DIR}" || { echo "ERROR: missing dataset: ${DATASET_DIR}" >&2; exit 1; }
test -f "${CKPT_RUN_DIR}/wandb_id.txt" || {
    echo "ERROR: missing ${CKPT_RUN_DIR}/wandb_id.txt -- did hf_pull_resume_bundle.py run?" >&2
    exit 1
}
test -f "${CKPT_STEP_DIR}/model.safetensors" || {
    echo "ERROR: missing ${CKPT_STEP_DIR}/model.safetensors" >&2
    exit 1
}
test -f "${CKPT_STEP_DIR}/optimizer.pt" || {
    echo "ERROR: missing ${CKPT_STEP_DIR}/optimizer.pt -- resume without Adam moments restarts the optimizer" >&2
    exit 1
}
test -f "${CKPT_STEP_DIR}/assets/robotwin-icl-arx-x5/norm_stats.json" || {
    echo "ERROR: missing norm_stats.json inside ${CKPT_STEP_DIR}/assets/" >&2
    exit 1
}

# Quick H200 sm_90 sanity check (warn only, don't block).
if "${VENV_PYTHON}" -c "import torch; cc=torch.cuda.get_device_capability(); print('sm', cc); exit(0 if cc==(9,0) else 1)" 2>/dev/null; then
    echo "  gpu_capability   = sm_90 (Hopper, expected on H200)"
else
    echo "  WARNING: GPU capability != sm_90; H200 should be (9, 0). Continuing anyway." >&2
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1, not launching."
    exit 0
fi

cd "${WORK_DIR}"

exec "${TORCHRUN}" \
    --nnodes="${NUM_MACHINES}" \
    --node_rank="${MACHINE_RANK}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${WORK_DIR}/scripts/train_pytorch.py" \
    "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --resume \
    --batch-size="${GLOBAL_BATCH_SIZE}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --keep-period="${KEEP_PERIOD}" \
    --log-interval="${LOG_INTERVAL}" \
    --num-workers="${NUM_WORKERS}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
