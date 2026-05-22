"""Policy transforms for Wuji 毛笔 writing (dual ARX-5 + dual dex hands).

Dataset structure (after Repack from LeRobot keys in writing_bundle/teleop):
- observation/state: 54 dims = [dual_arm(14) + dual_dex_hands(40)]
- observation/image/{base,left_wrist,right_wrist}: (H, W, 3) uint8
- actions: (action_horizon, 54)

Camera mapping (writing teleop has 4 cams, no head; pi0 expects 3):
- base_0_rgb       <- observation.images.stereo_left   (matches VAM WRITING_TARGET_CAMERA)
- left_wrist_0_rgb <- observation.images.cam_left_wrist
- right_wrist_0_rgb<- observation.images.cam_right_wrist
- (stereo_right is dropped)

action_dim is 54: the model is configured with action_dim=54 (preprocessed
base ckpts at pi05_base_pytorch_a54 / pi0_base_pytorch_a54), so no padding or
cropping is needed -- state/actions pass through.
"""

import dataclasses
import json
import pathlib
import random

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


WRITING_ACTION_DIM = 54

# VAM `wuji_writing_dataset.py:_apply_split` uses these exact two constants:
# random.Random(seed=42).shuffle, then val_ratio=0.1 of the shuffled order
# becomes val. We mirror them so the openpi baseline trains on exactly the
# same 443 episodes VAM does (and the same 44 val ep are held out, available
# for off-policy comparison via writing_val_episodes.json).
WRITING_DEFAULT_SPLIT_SEED = 42
WRITING_DEFAULT_VAL_RATIO = 0.1


def compute_writing_episode_lists(
    repo_root: str | pathlib.Path,
    val_ratio: float = WRITING_DEFAULT_VAL_RATIO,
    seed: int = WRITING_DEFAULT_SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Reproduce VAM's split for the wuji-writing teleop subset.

    Mirrors ``WujiWritingDataset._apply_split`` (ReCamMaster
    src/vam/examples/wanvideo/human2robot/wuji_writing_dataset.py:446-457):
    list all episode indices in ``meta/episodes.jsonl`` order, shuffle with
    ``random.Random(seed)``, then take the first ``n_val`` indices as val
    and the rest as train. ``n_val = max(1, int(N * val_ratio))`` for
    ``0 < val_ratio < 1``; full train/val if ``val_ratio`` is 0 or >=1.

    Returned lists are sorted ascending (the order the train/val loaders see).
    """
    repo_root = pathlib.Path(repo_root)
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)

    episode_indices: list[int] = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode_indices.append(int(json.loads(line)["episode_index"]))

    indices = list(range(len(episode_indices)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if val_ratio <= 0.0:
        n_val = 0
    elif val_ratio >= 1.0:
        n_val = len(episode_indices)
    else:
        n_val = max(1, int(len(episode_indices) * val_ratio))
    val_positions = sorted(indices[:n_val])
    train_positions = sorted(indices[n_val:])
    return [episode_indices[i] for i in train_positions], [episode_indices[i] for i in val_positions]


def make_writing_example() -> dict:
    """Random input example for the Writing policy."""
    return {
        "observation/state": np.random.rand(WRITING_ACTION_DIM).astype(np.float32),
        "observation/image/base": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/image/left_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/image/right_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "prompt": "the robot writes digit zero",
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
class WritingInputs(transforms.DataTransformFn):
    """Convert Writing observations to the format expected by Pi0 / Pi0.5."""

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
class WritingOutputs(transforms.DataTransformFn):
    """Identity-shape strip: model action_dim already matches Writing's 54-D."""

    action_dim: int = WRITING_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
