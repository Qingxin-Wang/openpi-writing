"""Policy transforms for Wuji pick-and-place (dual ARX-5 + dual dex hands).

Dataset structure (after Repack from LeRobot keys in pick_and_place_bundle/teleop):
- observation/state: 54 dims = [dual_arm(14) + dual_dex_hands(40)]
- observation/image/{base,left_wrist,right_wrist}: (H, W, 3) uint8
- actions: (action_horizon, 54)

Camera mapping (pick_and_place teleop has 3 cams: real head + dual wrists):
- base_0_rgb        <- observation.images.head             (head camera; 720x1280)
- left_wrist_0_rgb  <- observation.images.cam_left_wrist   (480x848)
- right_wrist_0_rgb <- observation.images.cam_right_wrist  (480x848)
Server-side ``ResizeImages(224, 224)`` (from ``ModelTransformFactory``) handles
the resize+pad to 224x224, so no per-camera-size handling here.

action_dim is 54: the model is configured with action_dim=54 (preprocessed
base ckpts at pi05_base_pytorch_a54 / pi0_base_pytorch_a54), so no padding or
cropping is needed -- state/actions pass through.
"""

import dataclasses
import json
import pathlib

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


PICKPLACE_ACTION_DIM = 54

# Episodes flagged at LeRobot timestamp-sync check with >0.1s drift (multi-frame
# drops during teleop recording, NOT recoverable by tolerance bump). All three
# have a single >=1.4s gap mid-episode that would break action_horizon=50
# chunking. Excluding them: 179 - 3 = 176 episodes for training.
PICKPLACE_BAD_EPISODES = (137, 154, 163)


def compute_pickplace_train_episode_list(
    repo_root: str | pathlib.Path,
    exclude: tuple[int, ...] | None = None,
) -> list[int]:
    """Return all episode_index in meta/episodes.jsonl minus ``exclude``.

    ``exclude=None`` defaults to PICKPLACE_BAD_EPISODES (original pick_and_place
    bundle's 3 timestamp-glitch episodes). Pass ``exclude=()`` to keep all.
    """
    if exclude is None:
        exclude = PICKPLACE_BAD_EPISODES
    repo_root = pathlib.Path(repo_root)
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    keep = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = int(json.loads(line)["episode_index"])
            if ep not in exclude:
                keep.append(ep)
    return keep


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PickPlaceInputs(transforms.DataTransformFn):
    """Convert pick-and-place observations to the format expected by Pi0 / Pi0.5."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image/base"])
        left_image = _parse_image(data["observation/image/left_wrist"])
        right_image = _parse_image(data["observation/image/right_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
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
class PickPlaceOutputs(transforms.DataTransformFn):
    """Identity-shape strip: model action_dim already matches pick_and_place's 54-D."""

    action_dim: int = PICKPLACE_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
