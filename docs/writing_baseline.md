# Wuji 毛笔 Writing Baseline (pi0.5 / pi0)

Pi0.5 / Pi0 finetune on the Wuji 毛笔 writing dataset, action_dim=54
(dual ARX-5 + dual dex hands).  Sister of the RoboTwin ICL ARX-X5
baseline; trains on the same shared-vepfs storage and uses identical
data-pipeline scaffolding.

## TL;DR

```bash
# 0. Recreate the venv (if not already there)
cd external/openpi
uv sync

# 1. (one-shot, ~5 min)   Preprocess base ckpts to action_dim=54
.venv/bin/python scripts/convert_base_ckpt_action_dim.py \
  --src /wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi05_base_pytorch \
  --dst /wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi05_base_pytorch_a54 \
  --action-dim-dst 54 --pi05
.venv/bin/python scripts/convert_base_ckpt_action_dim.py \
  --src /wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi0_base_pytorch \
  --dst /wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi0_base_pytorch_a54 \
  --action-dim-dst 54

# 2. (one-shot, ~15 min)  Build episodes_stats.jsonl for the writing bundle
.venv/bin/python scripts/build_writing_bundle_episodes_stats.py \
  --root /wuji-vepfs/wuji-il/huangsiqiao/data/writing_bundle/teleop \
  --workers 8

# 3. (one-shot, ~5 min)   Compute norm stats
HF_LEROBOT_HOME=/ .venv/bin/python scripts/compute_norm_stats.py \
  --config-name=pi05_writing --max-frames 10000
# pi0_writing reuses pi05's norm_stats -- copy it across:
mkdir -p assets/pi0_writing/writing-bundle-teleop
cp assets/pi05_writing/writing-bundle-teleop/norm_stats.json \
   assets/pi0_writing/writing-bundle-teleop/norm_stats.json

# 4. Train
WANDB_API_KEY=<key> NUM_TRAIN_STEPS=100000 \
  bash scripts/train_pi05_writing_16gpu.sh
WANDB_API_KEY=<key> NUM_TRAIN_STEPS=100000 \
  bash scripts/train_pi0_writing_16gpu.sh

# 5. Serve a trained ckpt (no writing-specific code needed)
.venv/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_writing \
  --policy.dir=/wuji-vepfs/wuji-il/huangsiqiao/data/checkpoints/pi05_writing/<exp>/100000 \
  --port=8000
```

## Dataset

Shared-vepfs bundle: `/wuji-vepfs/wuji-il/huangsiqiao/data/writing_bundle/`
(local mirror of HuggingFace `yeeeiii111/wuji-writing`).

| Subset | Episodes | Frames | Cameras | Notes |
|---|---|---|---|---|
| `teleop/` | 487 | 438,986 | stereo_left, stereo_right, cam_left_wrist, cam_right_wrist | LeRobot v2.1, 30 fps, 54-D state/action |
| `ego_ref/` | 896 | 357,427 | observation.images.head | Used by VAM for reference-video conditioning; **not consumed by pi0/pi05 baseline** (no native ego-ref support) |

Tasks: `task_index 0..9` -> "the robot writes digit zero..nine".

**Train/val split** (byte-for-byte aligned with VAM's
`wuji_writing_dataset._apply_split`):

- `random.Random(42).shuffle` over `meta/episodes.jsonl` order
- `val_ratio=0.1`, `n_val = max(1, int(487*0.1)) = 48`
- Result: **train 439 / val 48**; first 10 val ep_idx = `[4, 7, 21, 41, 45, 50, 59, 89, 94, 96]`
- The full lists are committed at `assets/writing_split/{train,val}_episodes.json`
  + `manifest.json` so downstream offline eval shares the exact same split.

Implementation: `compute_writing_episode_lists()` in
`src/openpi/policies/writing_policy.py`.  `LeRobotWritingDataConfig.create()`
filters `train_eps` into `DataConfig.episodes`, and raises if the split sizes
ever drift from `(439, 48)` (sanity guard against bundle changes).

## action_dim=54 (B-path) — why and how

Pi0 / pi0.5 base checkpoints are trained with **action_dim=32** (semantic
14-D dual-arm + 18-D zero-pad). The writing task is 54-D (14-D arm + 40-D
dex hands), which can't fit into 32. The two clean paths:

| Path | Description | Chosen |
|---|---|---|
| A | `action_dim=32`, drop dex hands, only use 14-D arms, zero-pad to 32. Loses brush control. | ❌ |
| B | `action_dim=54`, rebuild the 3 (pi05) / 6 (pi0) action_dim-bound proj layers; keep the ~3B remaining weights. | ✅ |

`safetensors.torch.load_model` raises on shape mismatch regardless of
`strict=`, and the PyTorch trainer in this repo has no
`PartialCheckpointWeightLoader` equivalent of the JAX path. So we do the
B path as a one-shot offline ckpt rewrite:

`scripts/convert_base_ckpt_action_dim.py`:
- Loads source `model.safetensors`
- Replaces `action_in_proj.{weight,bias}`, `action_out_proj.{weight,bias}`,
  and (pi0 only) `state_proj.{weight,bias}` with the new-shape PyTorch
  `nn.Linear` default init (`kaiming_uniform_(a=sqrt(5))` for weights,
  `uniform_(-1/sqrt(fan_in), +1/sqrt(fan_in))` for biases)
- Saves a sibling ckpt dir (`pi05_base_pytorch_a54`, `pi0_base_pytorch_a54`)
- Copies `config.json` unchanged

Validation: 808 / 771 non-proj tensors byte-for-byte identical to source;
new proj layers have correct shape and all-finite bf16 values.

## Cameras

Writing teleop has 4 cameras and no head cam; pi0 expects 3 (base + 2
wrists). `WritingInputs` (in `writing_policy.py`) maps:

| Source (LeRobot) | Target (pi0 inputs) |
|---|---|
| `observation.images.stereo_left` | `base_0_rgb` |
| `observation.images.cam_left_wrist` | `left_wrist_0_rgb` |
| `observation.images.cam_right_wrist` | `right_wrist_0_rgb` |
| `observation.images.stereo_right` | dropped |

`stereo_left` mirrors VAM's `WRITING_TARGET_CAMERA` default.

## max_token_len = 256

Pi0.5 default is 200 tokens. The 54-D state tokenises to ~150-160 tokens
and combined with prompt overshoots 200, triggering truncation warnings
every batch (seen at 203-213 in smoke test). `pi05_writing` sets
`max_token_len=256` (config.py:1163-1165). Pi0 doesn't tokenise state so
its default 48 is fine.

## Training schedule

| Param | 8-GPU launcher | 16-GPU launcher |
|---|---|---|
| GPUs | 8 (1 node) | 16 (2 node x 8) |
| Global batch | 16 | 32 |
| Per-device batch | 2 | 2 |
| LR | 5e-5 cosine, warmup 500, flat after | same |
| Save interval | 2500 steps | same |
| Keep period | 20000 steps | same |
| Default num_train_steps | 10M (run-forever sentinel) | same |

Suggested step count: **100,000** (~8 epochs at bs=32, mirrors VAM's
`MAX_STEPS=100000` default).

Output: `/wuji-vepfs/wuji-il/huangsiqiao/data/checkpoints/{pi05,pi0}_writing/<exp>/<step>/`

Wandb project: `writing-pi`

## Deployment

Server side (this repo, zero extra code):

```bash
.venv/bin/python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_writing \
  --policy.dir=/wuji-vepfs/.../pi05_writing/<exp>/<step> \
  --port=8000
```

Speed up the first call: `OPENPI_DISABLE_COMPILE=1` env var bypasses
`torch.compile` (10-30 min max-autotune cost) -- useful for short eval
rollouts.

Client side (robotic-arm team writes this):

```python
from openpi_client import WebsocketClientPolicy
policy = WebsocketClientPolicy(host="<server-ip>", port=8000)

obs = {
    "observation/image/base":        <stereo_left RGB HxWx3 uint8>,
    "observation/image/left_wrist":  <left_wrist RGB HxWx3 uint8>,
    "observation/image/right_wrist": <right_wrist RGB HxWx3 uint8>,
    "observation/state":             <54-D float32>,
    "prompt":                        "the robot writes digit zero",
}
action_chunk = policy.infer(obs)["actions"]    # (action_horizon, 54)
# execute first N actions, then re-query (receding horizon, e.g. N=16)
```

**Action / state dim layout** must match the training data column order
(unchanged from the original teleop collection). Get the layout from the
data owners before wiring real-robot drivers.

## Files added by this baseline

```
src/openpi/policies/writing_policy.py
src/openpi/training/config.py                                  (+~200 lines)
scripts/convert_base_ckpt_action_dim.py
scripts/build_writing_bundle_episodes_stats.py
scripts/train_pi05_writing_8gpu.sh
scripts/train_pi05_writing_16gpu.sh
scripts/train_pi0_writing_8gpu.sh
scripts/train_pi0_writing_16gpu.sh
assets/writing_split/train_episodes.json
assets/writing_split/val_episodes.json
assets/writing_split/manifest.json
docs/writing_baseline.md   (this file)
```

External shared-vepfs artifacts (NOT in git):

```
/wuji-vepfs/wuji-il/huangsiqiao/data/writing_bundle/teleop/      (dataset)
/wuji-vepfs/wuji-il/huangsiqiao/data/writing_bundle/teleop/meta/episodes_stats.jsonl  (~5 MB, generated by build_*)
/wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi05_base_pytorch_a54/  (~7.5 GB)
/wuji-vepfs/wuji-il/huangsiqiao/data/openpi-assets/checkpoints/pi0_base_pytorch_a54/   (~7.5 GB)
assets/pi05_writing/writing-bundle-teleop/norm_stats.json        (~13 KB, gitignored by upstream)
assets/pi0_writing/writing-bundle-teleop/norm_stats.json         (same)
```
