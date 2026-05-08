"""Smoke test for the RoboTwin ICL paired-v3 (arx-x5) LeRobot dataset wiring.

Loads one train sample and one val sample directly via LeRobotDataset, asserts:
- shapes / camera keys after the RepackTransform
- state and action width = 14 (the per-step action; horizon-stacked actions take action_horizon)
- episode_index ranges agree with info.json["splits"]
- no episode whose task_index is in ROBOTWIN_ICL_HOLDOUT_TASK_INDICES leaks through
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from openpi.policies.robotwin_policy import (
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ICL_HOLDOUT_TASK_INDICES,
    compute_icl_episode_lists,
)

REPO_ROOT = "/wuji-vepfs/wuji-il/huangsiqiao/data/robotwin-arx5-lerobot"
EXPECTED_TRAIN = 3000
EXPECTED_VAL = 1000


def _task_name_to_index(repo_root: pathlib.Path) -> dict[str, int]:
    name_to_idx: dict[str, int] = {}
    for line in (repo_root / "meta" / "tasks.jsonl").open():
        row = json.loads(line)
        name_to_idx[row["task"]] = row["task_index"]
    return name_to_idx


def _ep_to_taskidx(repo_root: pathlib.Path, name_to_idx: dict[str, int]) -> dict[int, int]:
    ep_to_taskidx: dict[int, int] = {}
    for line in (repo_root / "meta" / "episodes.jsonl").open():
        e = json.loads(line)
        ep_to_taskidx[e["episode_index"]] = name_to_idx[e["tasks"][0]]
    return ep_to_taskidx


def main() -> int:
    import openpi.training.data_loader  # noqa: F401  -- applies the sparse-episodes patch
    from lerobot.common.datasets import lerobot_dataset

    repo_root = pathlib.Path(REPO_ROOT)
    info = json.loads((repo_root / "meta" / "info.json").read_text())
    fps = info["fps"]
    splits = info["splits"]
    train_lo, train_hi = (int(x) for x in splits["train"].split(":"))
    val_lo, val_hi = (int(x) for x in splits["val"].split(":"))
    print(f"info.json: fps={fps}, splits=train[{train_lo}:{train_hi}), val[{val_lo}:{val_hi})")

    train_eps, val_eps = compute_icl_episode_lists(REPO_ROOT)
    print(f"helper output: train_eps={len(train_eps)}, val_eps={len(val_eps)}")
    assert len(train_eps) == EXPECTED_TRAIN, f"train_eps={len(train_eps)} != {EXPECTED_TRAIN}"
    assert len(val_eps) == EXPECTED_VAL, f"val_eps={len(val_eps)} != {EXPECTED_VAL}"

    name_to_idx = _task_name_to_index(repo_root)
    ep_to_taskidx = _ep_to_taskidx(repo_root, name_to_idx)
    leak_train = [i for i in train_eps if ep_to_taskidx[i] in ROBOTWIN_ICL_HOLDOUT_TASK_INDICES]
    leak_val = [i for i in val_eps if ep_to_taskidx[i] in ROBOTWIN_ICL_HOLDOUT_TASK_INDICES]
    assert not leak_train and not leak_val, f"holdout leaked: train={leak_train[:5]}, val={leak_val[:5]}"
    assert all(train_lo <= i < train_hi for i in train_eps), "train ep outside info.json train span"
    assert all(val_lo <= i < val_hi for i in val_eps), "val ep outside info.json val span"

    print("loading 1 train ep + 1 val ep via LeRobotDataset (pyav)...")
    train_ds = lerobot_dataset.LeRobotDataset(
        REPO_ROOT,
        episodes=[train_eps[0]],
        delta_timestamps={"action": [t / fps for t in range(2)]},
        tolerance_s=1.0 / fps,
        video_backend="pyav",
    )
    val_ds = lerobot_dataset.LeRobotDataset(
        REPO_ROOT,
        episodes=[val_eps[0]],
        delta_timestamps={"action": [t / fps for t in range(2)]},
        tolerance_s=1.0 / fps,
        video_backend="pyav",
    )

    for tag, ds, eps in (("train", train_ds, train_eps), ("val", val_ds, val_eps)):
        print(f"--- {tag}: ep={eps[0]}, num_frames={len(ds)} ---")
        sample = ds[0]
        cam_keys = [k for k in sample.keys() if k.startswith("observation.images.")]
        print(f"  camera keys: {sorted(cam_keys)}")
        for cam in ("observation.images.head_camera", "observation.images.left_camera", "observation.images.right_camera"):
            assert cam in sample, f"missing {cam} in sample"
            img = np.asarray(sample[cam])
            assert img.ndim == 3, f"{cam} expected 3D, got shape {img.shape}"
        state = np.asarray(sample["observation.state"])
        action = np.asarray(sample["action"])
        assert state.shape[-1] == ROBOTWIN_ACTION_DIM, f"state last-dim {state.shape} != {ROBOTWIN_ACTION_DIM}"
        assert action.shape[-1] == ROBOTWIN_ACTION_DIM, f"action last-dim {action.shape} != {ROBOTWIN_ACTION_DIM}"
        ep_idx = int(np.asarray(sample["episode_index"]).item())
        if tag == "train":
            assert train_lo <= ep_idx < train_hi, f"train ep_idx={ep_idx} outside [{train_lo}, {train_hi})"
        else:
            assert val_lo <= ep_idx < val_hi, f"val ep_idx={ep_idx} outside [{val_lo}, {val_hi})"
        task_idx = ep_to_taskidx[ep_idx]
        assert task_idx not in ROBOTWIN_ICL_HOLDOUT_TASK_INDICES, f"{tag} ep {ep_idx} in holdout (task_idx={task_idx})"
        print(f"  state.shape={state.shape}, action.shape={action.shape}, ep_idx={ep_idx}, task_idx={task_idx}")

    print("smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
