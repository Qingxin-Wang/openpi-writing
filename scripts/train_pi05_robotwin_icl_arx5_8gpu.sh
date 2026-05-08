#!/usr/bin/env bash
# =============================================================================
# Pi0.5 finetune on RoboTwin ICL paired-v3 (arx-x5) -- baseline for ReCamMaster.
#
# 8 GPU single-node. Style mirrors
#   ReCamMaster-starvla/scripts/baseline_starvla/_launch.sh
# (PAI-DLC env injection, env-overridable knobs, preflight, dry-run).
#
# Schedule aligns with reference VAM launcher
#   src/vam/examples/wanvideo/human2robot/launch_icl_paired_v2_alltasks_16g_50k.sh
# (global batch 16, lr 5e-5 constant after 500-step warmup, run-forever sentinel).
# =============================================================================
set -euo pipefail

# === Distributed shape ======================================================
export NUM_MACHINES="${MLP_WORKER_NUM:-${WORLD_SIZE:-${NUM_MACHINES:-1}}}"
export MACHINE_RANK="${MLP_ROLE_INDEX:-${RANK:-${MACHINE_RANK:-0}}}"
export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-localhost}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
export GPUS_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
TOTAL_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))

if [ "${TOTAL_PROCESSES}" -ne 8 ] && [ "${ALLOW_NON_8:-0}" != "1" ]; then
    echo "ERROR: this launcher targets 8 GPUs, got ${TOTAL_PROCESSES} (${NUM_MACHINES}x${GPUS_PER_NODE})." >&2
    echo "       Set ALLOW_NON_8=1 only for smoke tests." >&2
    exit 1
fi

# === Schedule (knobs env-overridable, defaults match TrainConfig) ===========
# global_bs = 8 GPU x per_device 2 = 16 (TrainConfig.batch_size).
# pi0.5 LoRA + grad checkpointing fits per-device 2 trivially on H20-96GB.
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
export KEEP_PERIOD="${KEEP_PERIOD:-20000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

# === Paths (all absolute) ===================================================
# Derive WORK_DIR from BASH_SOURCE so the launcher works regardless of where
# the upstream openpi tree is mounted (now lives at .../openpi-pi05/external/openpi
# under the ReCamMaster baseline/openpi worktree; mirrors starvla _launch.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON=${WORK_DIR}/.venv/bin/python
TORCHRUN=${WORK_DIR}/.venv/bin/torchrun
DATASET_DIR=/wuji-vepfs/wuji-il/huangsiqiao/data/robotwin-arx5-lerobot
NORM_STATS_DIR=${WORK_DIR}/assets/pi05_robotwin_icl_arx_x5/robotwin-icl-arx-x5
PI05_BASE_CKPT=/wuji-vepfs/wuji-il/lzicong/cache/openpi/openpi-assets/checkpoints/pi05_base/params
CHECKPOINT_BASE_DIR=/wuji-vepfs/wuji-il/huangsiqiao/data/checkpoints

# === Run identity ===========================================================
# Run name encodes model + dataset + date so the wam-baseline wandb project
# (shared with other baselines: oft, fast, gr00t, ...) stays legible.
DATE_TAG=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${RUN_TAG:-pi05_robotwin_arx5_icl}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}_${DATE_TAG}}"

# === wandb ==================================================================
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" != "disabled" && "${WANDB_MODE}" != "offline" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "ERROR: WANDB_API_KEY env var is required when WANDB_MODE=${WANDB_MODE}." >&2
        echo "       Set WANDB_API_KEY=<key>, or export WANDB_MODE=disabled." >&2
        exit 3
    fi
    export WANDB_API_KEY
fi

# === Runtime env (LeRobot abs path + JAX mem fraction) ======================
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# === NCCL ===================================================================
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

# Note: do NOT unset http_proxy/https_proxy here. PAI-DLC injects them and
# wandb relies on them to reach api.wandb.ai from inside the job sandbox.

# === Preflight ==============================================================
echo "=== pi05_robotwin_icl_arx5 8G launch preflight ==="
echo "  work_dir         = ${WORK_DIR}"
echo "  nodes=${NUM_MACHINES}  gpus_per_node=${GPUS_PER_NODE}  total=${TOTAL_PROCESSES}  rank=${MACHINE_RANK}"
echo "  master           = ${MASTER_ADDR}:${MASTER_PORT}"
echo "  exp_name         = ${EXP_NAME}"
echo "  num_train_steps  = ${NUM_TRAIN_STEPS} (run-forever sentinel)"
echo "  save_interval    = ${SAVE_INTERVAL}   keep_period=${KEEP_PERIOD}"
echo "  log_interval     = ${LOG_INTERVAL}"
echo "  global_batch     = ${GLOBAL_BATCH_SIZE}  (per-device $((GLOBAL_BATCH_SIZE/TOTAL_PROCESSES)))"
echo "  num_workers      = ${NUM_WORKERS}"
echo "  dataset          = ${DATASET_DIR}"
echo "  norm_stats       = ${NORM_STATS_DIR}"
echo "  pi05_base_ckpt   = ${PI05_BASE_CKPT}"
echo "  ckpt_out         = ${CHECKPOINT_BASE_DIR}/pi05_robotwin_icl_arx_x5/${EXP_NAME}"
echo "  wandb_mode       = ${WANDB_MODE}"

test -x "${VENV_PYTHON}" || { echo "ERROR: missing venv python: ${VENV_PYTHON}" >&2; exit 1; }
test -x "${TORCHRUN}" || { echo "ERROR: missing torchrun: ${TORCHRUN}" >&2; exit 1; }
test -d "${DATASET_DIR}" || { echo "ERROR: missing dataset: ${DATASET_DIR}" >&2; exit 1; }
test -f "${NORM_STATS_DIR}/norm_stats.json" || {
    echo "ERROR: missing norm stats: ${NORM_STATS_DIR}/norm_stats.json" >&2
    echo "       Run: ${VENV_PYTHON} ${WORK_DIR}/scripts/compute_norm_stats.py --config-name=pi05_robotwin_icl_arx_x5" >&2
    exit 1
}
test -d "${PI05_BASE_CKPT}" || { echo "ERROR: missing pi05_base ckpt: ${PI05_BASE_CKPT}" >&2; exit 1; }

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
    pi05_robotwin_icl_arx_x5 \
    --exp-name="${EXP_NAME}" \
    --batch-size="${GLOBAL_BATCH_SIZE}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --keep-period="${KEEP_PERIOD}" \
    --log-interval="${LOG_INTERVAL}" \
    --num-workers="${NUM_WORKERS}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
