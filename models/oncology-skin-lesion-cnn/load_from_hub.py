"""Download and load the oncology skin-lesion classifier checkpoint from Hugging Face Hub.

Requires the source repo cloned alongside this script (for the exact model
definition in src/classification/model.py):

    git clone https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN
    pip install huggingface_hub torch torchvision

Usage:
    python load_from_hub.py --source-repo ./Oncology-diagnosis-using-CNN
"""

from __future__ import annotations

import argparse
import sys


def _unwrap(obj):
    """Checkpoints saved mid-training wrap the weights under a key such as
    'model_state_dict', alongside bookkeeping like epoch and val_loss. Accept
    both that shape and a bare state dict."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner:
                return inner
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        default="./Oncology-diagnosis-using-CNN",
        help="Path to a local clone of motazalqaoud/Oncology-diagnosis-using-CNN (for src/classification/model.py)",
    )
    parser.add_argument(
        "--repo-id",
        default="motazalqaoud/oncology-skin-lesion-weights",
        help="Hugging Face Hub model repo id to download the checkpoint from",
    )
    parser.add_argument("--filename", default="best_model.pth")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    import torch

    sys.path.insert(0, args.source_repo)
    try:
        from src.classification.model import build_model
    except ImportError as exc:
        raise SystemExit(
            f"Could not import src.classification.model from {args.source_repo!r}. "
            f"Clone https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN "
            f"and pass --source-repo pointing at it."
        ) from exc

    checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)
    print(f"Downloaded checkpoint to {checkpoint_path}")

    model = build_model()
    state_dict = _unwrap(torch.load(checkpoint_path, map_location="cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded EfficientNet-B0 classifier ({n_params:,} parameters), ready for inference.")


if __name__ == "__main__":
    main()
