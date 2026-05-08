"""Policy transforms for RoboTwin (ICL paired-v3, arx-x5 embodiment).

Dataset structure (after Repack from LeRobot keys):
- observation/state: 14 dims = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
- observation/image/{head,left,right}: (H, W, 3) uint8
- actions: (action_horizon, 14)
"""

import dataclasses
import json
import pathlib

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ROBOTWIN_ACTION_DIM = 14

# ICL holdout task indices in the on-disk robotwin-arx5-lerobot dataset.
# Source: ReCamMaster/.../robotwin_utils.py:_FALLBACK_ICL_HOLDOUT, mapped through
# meta/tasks.jsonl of /wuji-vepfs/wuji-il/huangsiqiao/data/robotwin-arx5-lerobot.
# 1=beat_block_hammer, 3=click_bell, 9=move_stapler_pad, 13=place_container_plate, 18=place_object_stand.
ROBOTWIN_ICL_HOLDOUT_TASK_INDICES = frozenset({1, 3, 9, 13, 18})


def compute_icl_episode_lists(repo_root: str | pathlib.Path) -> tuple[list[int], list[int]]:
    """Return (train_episodes, val_episodes) for the RoboTwin ICL split.

    Reads ``meta/info.json["splits"]`` for the train/val episode index ranges, then drops
    any episode whose task_index is in ``ROBOTWIN_ICL_HOLDOUT_TASK_INDICES``. Task indices
    are looked up by mapping each episode's ``tasks[0]`` string against ``meta/tasks.jsonl``.

    Expected output for the canonical paired-v3 25-task dataset: 3000 train, 1000 val.
    """
    repo_root = pathlib.Path(repo_root)

    info = json.loads((repo_root / "meta" / "info.json").read_text())
    splits = info.get("splits") or {}
    if "train" not in splits or "val" not in splits:
        raise ValueError(f"info.json missing 'splits.train' / 'splits.val' at {repo_root}")

    def _parse_span(span: str) -> range:
        a, b = span.split(":")
        return range(int(a), int(b))

    train_span = _parse_span(splits["train"])
    val_span = _parse_span(splits["val"])

    task_name_to_index: dict[str, int] = {}
    with open(repo_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            task_name_to_index[row["task"]] = row["task_index"]

    train_eps: list[int] = []
    val_eps: list[int] = []
    with open(repo_root / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            tasks = ep.get("tasks") or []
            if not tasks:
                continue
            task_idx = task_name_to_index.get(tasks[0])
            if task_idx is None or task_idx in ROBOTWIN_ICL_HOLDOUT_TASK_INDICES:
                continue
            ep_idx = ep["episode_index"]
            if ep_idx in train_span:
                train_eps.append(ep_idx)
            elif ep_idx in val_span:
                val_eps.append(ep_idx)

    return train_eps, val_eps


def make_robotwin_example() -> dict:
    """Creates a random input example for the RoboTwin policy."""
    return {
        "observation/state": np.random.rand(ROBOTWIN_ACTION_DIM).astype(np.float32),
        "observation/image/head": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/image/left": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/image/right": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "prompt": "adjust bottle",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Convert RoboTwin observations to the format expected by Pi0 / Pi0.5."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/image/head"])
        left_image = _parse_image(data["observation/image/left"])
        right_image = _parse_image(data["observation/image/right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Strip action padding back to RoboTwin's 14-DoF."""

    action_dim: int = ROBOTWIN_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
