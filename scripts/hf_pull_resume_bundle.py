"""Pull the resume bundles from HF Hub into the H200-side CHECKPOINT_BASE_DIR.

Symmetric to hf_push_resume_bundle.py: same hub layout, so we just download
everything in the repo into CHECKPOINT_BASE_DIR and the on-disk path becomes
CHECKPOINT_BASE_DIR/<config-name>/<exp-name>/{wandb_id.txt, <step>/...}.

Usage on H200:
    python scripts/hf_pull_resume_bundle.py \\
        --repo-id <user>/recammaster-resume-20260525 \\
        --checkpoint-base-dir /path/on/h200/checkpoints
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hf_pull")

EXPECTED = [
    ("pi05_robotwin_icl_arx_x5", "pi05_robotwin_arx5_icl_20260525_044230", 40000),
    ("pi0_robotwin_icl_arx_x5", "pi0_robotwin_arx5_icl_20260525_121906", 20000),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--checkpoint-base-dir", required=True, type=Path)
    ap.add_argument("--max-workers", type=int, default=8, help="parallel HF download workers")
    args = ap.parse_args()

    args.checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", args.repo_id, args.checkpoint_base_dir)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.checkpoint_base_dir),
        max_workers=args.max_workers,
    )

    missing = []
    for cfg, exp, step in EXPECTED:
        for required in (
            args.checkpoint_base_dir / cfg / exp / "wandb_id.txt",
            args.checkpoint_base_dir / cfg / exp / str(step) / "model.safetensors",
            args.checkpoint_base_dir / cfg / exp / str(step) / "optimizer.pt",
            args.checkpoint_base_dir / cfg / exp / str(step) / "metadata.pt",
        ):
            if not required.exists():
                missing.append(required)
    if missing:
        log.error("post-download check failed, missing:")
        for m in missing:
            log.error("  %s", m)
        sys.exit(2)
    log.info("ok, ready to resume.")


if __name__ == "__main__":
    main()
