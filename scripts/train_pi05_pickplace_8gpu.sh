#!/usr/bin/env bash
# =============================================================================
# Pi0.5 finetune on Wuji pick_and_place bundle (action_dim=54) -- 54-D dual
# ARX-5 + dual dex hands, 3 cams (head + L wrist + R wrist), 179 teleop ep,
# all trained (no val split). Sibling of train_pi05_writing_8gpu.sh.
#
# Schedule mirrors writing/RoboTwin pi05 baselines: global batch 16, lr 5e-5
# constant after 500-step warmup, run-forever sentinel.
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

# === Schedule (knobs env-overridable; pick-and-place runs target 20k step) ==
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-4000}"
# KEEP_PERIOD == SAVE_INTERVAL => every saved ckpt is also flagged "keep"
# (4k/8k/12k/16k/20k all preserved across the 20k-step run).
export KEEP_PERIOD="${KEEP_PERIOD:-4000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"

# === Paths (all absolute) ===================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON=${WORK_DIR}/.venv/bin/python
TORCHRUN=${WORK_DIR}/.venv/bin/torchrun
DATASET_DIR=/wuji-vepfs/wuji-il/huangsiqiao/data/pick_and_place_bundle/teleop
NORM_STATS_DIR=${WORK_DIR}/assets/pi05_pickplace/pickplace-bundle-teleop
PI05_BASE_PT_CKPT=/wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi05_base_pytorch_a54
CHECKPOINT_BASE_DIR=/wuji-vepfs/wuji-il/huangsiqiao/data/checkpoints

# === Run identity ===========================================================
DATE_TAG=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${RUN_TAG:-pi05_pickplace}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}_${DATE_TAG}}"

# === wandb ==================================================================
export WANDB_MODE="${WANDB_MODE:-online}"
# Bake the team entity so DLC workers (which don't inherit local shell env)
# always land runs in the wuji team workspace, not whoever's API key owner.
export WANDB_ENTITY="${WANDB_ENTITY:-better_guidance}"
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
echo "=== pi05_pickplace 8G launch preflight ==="
echo "  work_dir         = ${WORK_DIR}"
echo "  nodes=${NUM_MACHINES}  gpus_per_node=${GPUS_PER_NODE}  total=${TOTAL_PROCESSES}  rank=${MACHINE_RANK}"
echo "  master           = ${MASTER_ADDR}:${MASTER_PORT}"
echo "  exp_name         = ${EXP_NAME}"
echo "  num_train_steps  = ${NUM_TRAIN_STEPS}"
echo "  save_interval    = ${SAVE_INTERVAL}   keep_period=${KEEP_PERIOD}"
echo "  log_interval     = ${LOG_INTERVAL}"
echo "  global_batch     = ${GLOBAL_BATCH_SIZE}  (per-device $((GLOBAL_BATCH_SIZE/TOTAL_PROCESSES)))"
echo "  num_workers      = ${NUM_WORKERS}"
echo "  dataset          = ${DATASET_DIR}"
echo "  norm_stats       = ${NORM_STATS_DIR}"
echo "  pi05_base_pt_ckpt= ${PI05_BASE_PT_CKPT}  (action_dim=54 preprocessed)"
echo "  ckpt_out         = ${CHECKPOINT_BASE_DIR}/pi05_pickplace/${EXP_NAME}"
echo "  wandb_mode       = ${WANDB_MODE}"

test -x "${VENV_PYTHON}" || { echo "ERROR: missing venv python: ${VENV_PYTHON}" >&2; exit 1; }
test -x "${TORCHRUN}" || { echo "ERROR: missing torchrun: ${TORCHRUN}" >&2; exit 1; }
test -d "${DATASET_DIR}" || { echo "ERROR: missing dataset: ${DATASET_DIR}" >&2; exit 1; }
test -f "${DATASET_DIR}/meta/episodes_stats.jsonl" || {
    echo "ERROR: missing episodes_stats.jsonl under ${DATASET_DIR}/meta/" >&2
    exit 1
}
test -f "${NORM_STATS_DIR}/norm_stats.json" || {
    echo "ERROR: missing norm stats: ${NORM_STATS_DIR}/norm_stats.json" >&2
    echo "       Run: ${VENV_PYTHON} ${WORK_DIR}/scripts/compute_norm_stats.py --config-name=pi05_pickplace" >&2
    exit 1
}
test -f "${PI05_BASE_PT_CKPT}/model.safetensors" || {
    echo "ERROR: missing preprocessed pi05_base ckpt: ${PI05_BASE_PT_CKPT}/model.safetensors" >&2
    echo "       Run: ${VENV_PYTHON} ${WORK_DIR}/scripts/convert_base_ckpt_action_dim.py \\" >&2
    echo "              --src /wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi05_base_pytorch \\" >&2
    echo "              --dst ${PI05_BASE_PT_CKPT} --action-dim-dst 54 --pi05" >&2
    exit 1
}

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
    pi05_pickplace \
    --exp-name="${EXP_NAME}" \
    --batch-size="${GLOBAL_BATCH_SIZE}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --keep-period="${KEEP_PERIOD}" \
    --log-interval="${LOG_INTERVAL}" \
    --num-workers="${NUM_WORKERS}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}"
