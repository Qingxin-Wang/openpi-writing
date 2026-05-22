"""Rewrite the 3 action_dim-bound proj layers of a pi0/pi0.5 PyTorch base ckpt
to a new action_dim, keeping every other tensor identical.

Why: ``Pi0Config.action_dim`` is a config parameter, not a model-architecture
constant, but the base pi05/pi0 ckpts on disk are 32-D. Loading a 32-D ckpt
into a model built with a different action_dim hard-fails at
``safetensors.torch.load_model`` time (PyTorch ``load_state_dict`` raises on
shape mismatch regardless of ``strict``). This script does a one-shot rewrite
of the 3 affected layers so the resulting ckpt is compatible with the new
action_dim while leaving the ~3B gemma + action-expert weights untouched.

Layers replaced:

- ``action_in_proj.weight``  (width, action_dim_src)  →  (width, action_dim_dst)
- ``action_in_proj.bias``    unchanged shape (width,) — kept as-is
- ``action_out_proj.weight`` (action_dim_src, width)  →  (action_dim_dst, width)
- ``action_out_proj.bias``   (action_dim_src,)        →  (action_dim_dst,)
- ``state_proj.weight``      (width, action_dim_src)  →  (width, action_dim_dst)   [pi0 only]
- ``state_proj.bias``        unchanged shape (width,) — kept as-is                   [pi0 only]

Replacement uses PyTorch ``nn.Linear`` default init (``kaiming_uniform_`` for
weights, ``uniform_`` for biases). This matches what a freshly constructed
``nn.Linear(action_dim_dst, width)`` would have, so the post-conversion model
behaves identically to "load_model + reinit those 3 layers".
"""

import argparse
import math
import os
import pathlib
import shutil

import safetensors.torch as st
import torch
import torch.nn as nn


_PROJ_KEYS_COMMON = ("action_in_proj", "action_out_proj")
_PROJ_KEYS_PI0_ONLY = ("state_proj",)


def _kaiming_uniform_like(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """PyTorch nn.Linear weight default init."""
    t = torch.empty(shape, dtype=torch.float32)
    nn.init.kaiming_uniform_(t, a=math.sqrt(5))
    return t.to(dtype)


def _bias_uniform_like(shape: tuple[int, ...], fan_in: int, dtype: torch.dtype) -> torch.Tensor:
    """PyTorch nn.Linear bias default init: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    t = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound)
    return t.to(dtype)


def convert(
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    action_dim_src: int,
    action_dim_dst: int,
    is_pi05: bool,
    seed: int,
) -> None:
    torch.manual_seed(seed)

    src_safetensors = src_dir / "model.safetensors"
    assert src_safetensors.is_file(), f"missing {src_safetensors}"
    dst_dir.mkdir(parents=True, exist_ok=False)

    state = st.load_file(str(src_safetensors), device="cpu")

    proj_keys = list(_PROJ_KEYS_COMMON) + ([] if is_pi05 else list(_PROJ_KEYS_PI0_ONLY))
    touched: list[str] = []

    for proj in proj_keys:
        w_key = f"{proj}.weight"
        b_key = f"{proj}.bias"
        if w_key not in state:
            raise KeyError(f"expected key {w_key} not found in {src_safetensors}")
        if b_key not in state:
            raise KeyError(f"expected key {b_key} not found in {src_safetensors}")

        w = state[w_key]
        b = state[b_key]
        dtype = w.dtype

        # nn.Linear convention: weight shape = (out_features, in_features).
        # - action_in_proj:  in=action_dim, out=width  → weight (width, action_dim)
        # - action_out_proj: in=width, out=action_dim  → weight (action_dim, width)
        # - state_proj:      in=action_dim, out=width  → weight (width, action_dim)  (pi0 only)
        if proj == "action_out_proj":
            assert w.shape[0] == action_dim_src, f"{w_key} expected first dim {action_dim_src}, got {w.shape}"
            width = w.shape[1]
            new_w = _kaiming_uniform_like((action_dim_dst, width), dtype)
            new_b = _bias_uniform_like((action_dim_dst,), fan_in=width, dtype=dtype)
        else:
            # action_in_proj / state_proj
            assert w.shape[1] == action_dim_src, f"{w_key} expected last dim {action_dim_src}, got {w.shape}"
            width = w.shape[0]
            new_w = _kaiming_uniform_like((width, action_dim_dst), dtype)
            new_b = _bias_uniform_like((width,), fan_in=action_dim_dst, dtype=dtype)
            assert b.shape == new_b.shape, f"{b_key} shape {b.shape} != {new_b.shape}"

        state[w_key] = new_w
        state[b_key] = new_b
        touched.append(f"  {w_key}: {tuple(w.shape)} -> {tuple(new_w.shape)}")
        touched.append(f"  {b_key}: {tuple(b.shape)} -> {tuple(new_b.shape)}")

    dst_safetensors = dst_dir / "model.safetensors"
    st.save_file(state, str(dst_safetensors))

    # copy config.json (and anything else that isn't the safetensors blob)
    for entry in src_dir.iterdir():
        if entry.name == "model.safetensors":
            continue
        target = dst_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    print(f"wrote: {dst_safetensors}")
    print(f"tensors total: {len(state)} (touched {len(touched)//2} layers)")
    for line in touched:
        print(line)
    print(f"action_dim: {action_dim_src} -> {action_dim_dst}  (pi05={is_pi05}, seed={seed})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=pathlib.Path, help="source ckpt dir (contains model.safetensors)")
    ap.add_argument("--dst", required=True, type=pathlib.Path, help="destination ckpt dir (must not exist)")
    ap.add_argument("--action-dim-src", type=int, default=32)
    ap.add_argument("--action-dim-dst", type=int, required=True)
    ap.add_argument("--pi05", action="store_true", help="set if source is pi0.5 (no state_proj); pi0 has state_proj")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    convert(args.src, args.dst, args.action_dim_src, args.action_dim_dst, args.pi05, args.seed)


if __name__ == "__main__":
    main()
