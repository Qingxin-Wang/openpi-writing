"""Push the pi05 + pi0 RoboTwin ICL resume bundles to a HF Hub repo.

We're moving training to a different cluster (H200) that doesn't share the
vepfs storage with the current pods, so the only transfer channels are
HF Hub (for ~40 GB of ckpt) and GitHub (for code).

What gets uploaded (kept as-is so the H200 side can drop straight into
CHECKPOINT_BASE_DIR/<config-name>/<exp-name>/):

  pi05_robotwin_icl_arx_x5/pi05_robotwin_arx5_icl_20260525_044230/
      wandb_id.txt                  # so resume continues the same wandb run
      40000/
          model.safetensors         # ~7.5 GB
          optimizer.pt              # ~13.5 GB (AdamW moments)
          metadata.pt               # global_step etc.
          assets/robotwin-icl-arx-x5/norm_stats.json
  pi0_robotwin_icl_arx_x5/pi0_robotwin_arx5_icl_20260525_121906/
      wandb_id.txt
      20000/
          ... same layout

Only the single step we want to resume from is uploaded (--keep-step), so
earlier ckpts on disk (e.g. pi05's 20000) are skipped.

Usage:
    huggingface-cli login   # or set HF_TOKEN
    python scripts/hf_push_resume_bundle.py --repo-id <user>/recammaster-resume-20260525
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hf_push")

CKPT_BASE = Path("/wuji-vepfs/wuji-il/huangsiqiao/data/checkpoints")

# (config_name, exp_name, step_to_keep)
BUNDLES = [
    ("pi05_robotwin_icl_arx_x5", "pi05_robotwin_arx5_icl_20260525_044230", 40000),
    ("pi0_robotwin_icl_arx_x5", "pi0_robotwin_arx5_icl_20260525_121906", 20000),
]


def preflight(bundles: list[tuple[str, str, int]]) -> None:
    missing = []
    for cfg, exp, step in bundles:
        exp_dir = CKPT_BASE / cfg / exp
        for required in (
            exp_dir / "wandb_id.txt",
            exp_dir / str(step) / "model.safetensors",
            exp_dir / str(step) / "optimizer.pt",
            exp_dir / str(step) / "metadata.pt",
            exp_dir / str(step) / "assets" / "robotwin-icl-arx-x5" / "norm_stats.json",
        ):
            if not required.exists():
                missing.append(required)
    if missing:
        log.error("preflight failed, missing files:")
        for m in missing:
            log.error("  %s", m)
        sys.exit(2)
    log.info("preflight ok (%d bundles)", len(bundles))


def upload(repo_id: str, bundles: list[tuple[str, str, int]], private: bool, dry_run: bool) -> None:
    api = HfApi()
    if not dry_run:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        log.info("repo ready: https://huggingface.co/%s", repo_id)

    for cfg, exp, step in bundles:
        local_root = CKPT_BASE / cfg / exp
        # Hub path mirrors the on-disk layout so the H200 side can extract into
        # CHECKPOINT_BASE_DIR without any rewriting.
        hub_prefix = f"{cfg}/{exp}"
        # Only upload the step we want + the run-level wandb_id.txt.
        allow = [
            "wandb_id.txt",
            f"{step}/**",
        ]
        log.info("uploading %s -> %s (step=%d, allow=%s)", local_root, hub_prefix, step, allow)
        if dry_run:
            for p in sorted(local_root.rglob("*")):
                rel = p.relative_to(local_root)
                if rel.parts[0] in ("wandb_id.txt", str(step)) and p.is_file():
                    log.info("  WOULD upload: %s (%.1f MB)", rel, p.stat().st_size / 1e6)
            continue
        api.upload_folder(
            folder_path=str(local_root),
            repo_id=repo_id,
            path_in_repo=hub_prefix,
            allow_patterns=allow,
            commit_message=f"add {exp} step {step}",
        )
    log.info("done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="HF Hub repo id, e.g. user/recammaster-resume-20260525")
    ap.add_argument("--private", action="store_true", help="create the repo as private (default: public)")
    ap.add_argument("--dry-run", action="store_true", help="list files that would be uploaded, do not upload")
    args = ap.parse_args()

    preflight(BUNDLES)
    upload(args.repo_id, BUNDLES, private=args.private, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
