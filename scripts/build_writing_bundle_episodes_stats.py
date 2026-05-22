"""Generate ``meta/episodes_stats.jsonl`` for the Wuji writing bundle teleop subset.

The bundle was built without per-episode stats, but LeRobot v2.1 requires the
file to instantiate ``LeRobotDataset`` (when missing, LeRobot falls back to
querying HF for the repo, which fails for absolute-path repo_ids).

Reproduces LeRobot's ``compute_episode_stats`` semantics:
- numerical features (state, action, timestamp, frame_index, ...): full-axis-0
  min/max/mean/std, ``count = [num_frames]``
- video features: sparse-frame sampling via ``sample_indices``, normalise to
  [0,1], reduce over (frames, H, W) keeping channel dim → shape ``(3, 1, 1)``,
  ``count = [num_sampled_frames]``

This file is a one-shot fixup for the existing bundle on disk. After it runs,
``LeRobotDatasetMetadata`` opens the bundle without trying to reach HF.
"""

import argparse
import json
import multiprocessing as mp
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices


def _read_video_frames(video_path: pathlib.Path, indices: list[int]) -> np.ndarray:
    """Read the requested frame indices from a video, return (N, 3, H, W) uint8."""
    import av

    wanted = set(int(i) for i in indices)
    out = [None] * len(indices)
    idx_by_frame = {f: i for i, f in enumerate(indices)}

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # Pin PyAV / libavcodec to single-thread per video. Default ("auto" = ncores)
        # spawns ~ncores threads per stream, and we open 4 streams per episode --
        # the resulting oversubscription dominates wall clock when a Pool also runs.
        stream.thread_count = 1
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                arr = frame.to_ndarray(format="rgb24")  # HWC uint8
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError(f"unexpected frame shape from {video_path}: {arr.shape}")
                out[idx_by_frame[frame_idx]] = arr.transpose(2, 0, 1)  # HWC -> CHW
                wanted.discard(frame_idx)
                if not wanted:
                    break

    if any(o is None for o in out):
        raise ValueError(f"missing frames in {video_path}")
    return np.stack(out, axis=0)  # (N, 3, H, W)


def _stats_for_video(video_path: pathlib.Path, num_frames: int) -> dict[str, np.ndarray]:
    indices = sample_indices(num_frames)
    frames = _read_video_frames(video_path, indices)
    arr = frames.astype(np.float32) / 255.0  # normalise
    stats = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
    # squeeze the (0,) singleton frames axis so shape becomes (3, 1, 1) per the schema
    return {k: (v if k == "count" else np.squeeze(v, axis=0)) for k, v in stats.items()}


def _stats_for_numerical(array: np.ndarray) -> dict[str, np.ndarray]:
    axes = 0
    keepdims = array.ndim == 1
    return get_feature_stats(array, axis=axes, keepdims=keepdims)


def _stack_object_column(series: pd.Series) -> np.ndarray:
    return np.stack(series.values).astype(np.float32)


def _atomic_column(series: pd.Series) -> np.ndarray:
    return series.to_numpy()


def _np_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _np_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_np_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _process_single_episode(args: tuple) -> tuple[int, int, dict]:
    """Worker entrypoint: compute stats for one episode. Returns (episode_index, n_frames, ep_stats_jsonable)."""
    (
        bundle_subset_root,
        episode_index,
        chunks_size,
        data_template,
        video_template,
        numerical_features,
        video_features,
    ) = args
    bundle_subset_root = pathlib.Path(bundle_subset_root)
    chunk_index = episode_index // chunks_size
    fmt = dict(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        chunk_index=chunk_index,
        file_index=episode_index,
    )
    parquet_path = bundle_subset_root / data_template.format(**fmt)
    df = pd.read_parquet(parquet_path)
    n_frames = len(df)
    ep_stats: dict[str, dict[str, np.ndarray]] = {}
    for key in numerical_features:
        if key not in df.columns:
            raise KeyError(f"feature '{key}' missing in {parquet_path}")
        col = df[key]
        if col.dtype == object:
            arr = _stack_object_column(col)
        else:
            arr = _atomic_column(col)
        ep_stats[key] = _stats_for_numerical(arr)
    for key in video_features:
        video_path = bundle_subset_root / video_template.format(video_key=key, **fmt)
        ep_stats[key] = _stats_for_video(video_path, n_frames)
    row = {"episode_index": episode_index, "stats": _np_to_jsonable(ep_stats)}
    return episode_index, n_frames, row


def _worker_init() -> None:
    """Pin BLAS/OMP envs inside workers (parent env vars may not always carry through)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def build(
    bundle_subset_root: pathlib.Path,
    output_path: pathlib.Path | None = None,
    resume: bool = False,
    workers: int = 1,
) -> None:
    meta_dir = bundle_subset_root / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    features = info["features"]
    data_template = info["data_path"]
    video_template = info["video_path"]

    numerical_features = [
        k for k, v in features.items()
        if v.get("dtype") not in ("video", "image", "string")
    ]
    video_features = [k for k, v in features.items() if v.get("dtype") == "video"]
    print(f"numerical features: {numerical_features}")
    print(f"video features: {video_features}")

    episodes_meta_path = meta_dir / "episodes.jsonl"
    if not episodes_meta_path.exists():
        raise FileNotFoundError(episodes_meta_path)
    episode_meta = [json.loads(line) for line in episodes_meta_path.read_text().splitlines() if line.strip()]
    print(f"episodes to process: {len(episode_meta)}")

    out_path = output_path or (meta_dir / "episodes_stats.jsonl")
    already_done: set[int] = set()
    if out_path.exists():
        if resume:
            for line in out_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    already_done.add(int(json.loads(line)["episode_index"]))
                except (json.JSONDecodeError, KeyError):
                    pass
            print(f"resume: {len(already_done)} episodes already in {out_path}")
        else:
            out_path.unlink()
            print(f"removed pre-existing {out_path}")

    pending = [int(ep["episode_index"]) for ep in episode_meta if int(ep["episode_index"]) not in already_done]
    print(f"pending episodes: {len(pending)} (workers={workers})", flush=True)

    work_items = [
        (
            str(bundle_subset_root),
            ep_idx,
            chunks_size,
            data_template,
            video_template,
            numerical_features,
            video_features,
        )
        for ep_idx in pending
    ]

    open_mode = "a" if resume and already_done else "w"
    n_done = 0
    with out_path.open(open_mode) as out_f:
        if workers <= 1:
            for item in work_items:
                ep_idx, n_frames, row = _process_single_episode(item)
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 10 == 0 or n_done == len(work_items):
                    print(f"  done {n_done}/{len(work_items)} (ep_idx={ep_idx}, n_frames={n_frames})", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_worker_init) as pool:
                for ep_idx, n_frames, row in pool.imap_unordered(_process_single_episode, work_items, chunksize=1):
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    n_done += 1
                    if n_done % 10 == 0 or n_done == len(work_items):
                        print(f"  done {n_done}/{len(work_items)} (ep_idx={ep_idx}, n_frames={n_frames})", flush=True)

    print(f"wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("/wuji-vepfs/wuji-il/huangsiqiao/data/writing_bundle/teleop"),
        help="LeRobot dataset root (the directory containing meta/, data/, videos/)",
    )
    ap.add_argument("--output", type=pathlib.Path, default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="append to existing output and skip episodes already present",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel episode workers (uses multiprocessing.spawn; each worker pinned to BLAS/OMP/PyAV thread_count=1)",
    )
    args = ap.parse_args()
    build(args.root, args.output, resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
