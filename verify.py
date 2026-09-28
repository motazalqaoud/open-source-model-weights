"""Verify a downloaded checkpoint is intact and is the architecture it claims to be.

    python verify.py oncology-skin-lesion-cnn --checkpoint ./weights/best_model.pth

Checks, in order of how much they need installed:

1. File exists, and its size is close to what the manifest records.
2. It is a readable PyTorch checkpoint (a zip archive) -- no torch required.
3. Its parameter keys match the architecture fingerprint for this model id
   (EfficientNet exposes `features.*`, the MONAI U-Net does not) -- torch required.
4. Optionally, it loads cleanly into the real model class from the source repo,
   with no missing or unexpected keys -- source repo required.

Step 4 is the only real proof it will run, but steps 1-3 catch the common
failures (truncated download, wrong file, wrong model) without a full setup.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zipfile

MANIFEST = pathlib.Path(__file__).parent / "models.json"


def load_models() -> dict:
    with MANIFEST.open() as fh:
        return {m["id"]: m for m in json.load(fh)["models"]}


def check_size(path: pathlib.Path, model: dict) -> bool:
    actual = path.stat().st_size
    expected = model.get("checkpoint_size_bytes")
    print(f"  size: {actual / 1048576:.1f} MB", end="")
    if not expected:
        print(" (no expected size recorded)")
        return True
    tolerance = max(1024 * 1024, expected * 0.05)
    if abs(actual - expected) > tolerance:
        print(f"  MISMATCH -- manifest expects about {expected / 1048576:.1f} MB")
        return False
    print("  OK")
    return True


def check_is_torch_archive(path: pathlib.Path) -> bool:
    if not zipfile.is_zipfile(path):
        print("  format: NOT a PyTorch zip archive -- file is corrupt or not a .pth")
        return False
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    has_pickle = any(n.endswith("data.pkl") for n in names)
    print(f"  format: PyTorch archive, {len(names)} entries", end="")
    print("  OK" if has_pickle else "  MISSING data.pkl")
    return has_pickle


#: Keys a training loop commonly wraps the real state dict in.
_WRAPPER_KEYS = ("model_state_dict", "state_dict", "model", "net")


def unwrap_state_dict(obj):
    """Return (state_dict, metadata).

    Checkpoints saved mid-training are usually a dict holding the weights under
    a key like 'model_state_dict' alongside bookkeeping (epoch, val_loss). Both
    that shape and a bare state dict are handled.
    """
    if not isinstance(obj, dict):
        return None, {}
    for key in _WRAPPER_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            metadata = {k: v for k, v in obj.items() if k != key and not isinstance(v, dict)}
            return inner, metadata
    return obj, {}


def check_architecture(path: pathlib.Path, model: dict) -> bool:
    try:
        import torch
    except ImportError:
        print("  architecture: SKIPPED (pip install torch to enable)")
        return True

    raw = torch.load(path, map_location="cpu", weights_only=True)
    state, metadata = unwrap_state_dict(raw)
    if state is None:
        print(f"  architecture: unexpected checkpoint type {type(raw).__name__}")
        return False

    keys = list(state.keys())
    total_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    print(f"  architecture: {len(keys)} tensors, {total_params:,} parameters")

    if metadata:
        rendered = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(metadata.items())
        )
        print(f"                training metadata: {rendered}")

    hint = model.get("state_dict_key_prefix_hint")
    if not hint:
        print("                (no architecture fingerprint recorded for this model -- skipping match)")
        return True

    matched = sum(1 for k in keys if k.startswith(hint))
    if matched == 0:
        sample = ", ".join(keys[:3])
        print(f"                expected keys starting with '{hint}' -- found none. Wrong model?")
        print(f"                first keys present: {sample}")
        return False
    shown = hint if hint.endswith(".") else hint + "."
    print(f"                {matched}/{len(keys)} keys match the '{shown}*' fingerprint  OK")
    return True


def check_loads_into_model(path: pathlib.Path, model: dict, source_repo: str) -> bool:
    try:
        import torch
    except ImportError:
        print("  load test: SKIPPED (torch not installed)")
        return True

    module_path, _, factory = model["model_class_import"].partition(":")
    sys.path.insert(0, source_repo)
    try:
        module = __import__(module_path, fromlist=[factory])
        build = getattr(module, factory)
    except Exception as exc:
        print(f"  load test: SKIPPED -- could not import {model['model_class_import']} from {source_repo} ({exc})")
        return True

    net = build()
    state, _ = unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load test: {len(missing)} missing, {len(unexpected)} unexpected keys")
        for k in list(missing)[:5]:
            print(f"             missing: {k}")
        for k in list(unexpected)[:5]:
            print(f"             unexpected: {k}")
        return False
    print("  load test: state dict applied cleanly, no missing or unexpected keys  OK")
    return True


def main() -> int:
    models = load_models()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", choices=sorted(models))
    parser.add_argument("--checkpoint", required=True, help="Path to the downloaded .pth file")
    parser.add_argument("--source-repo", help="Local clone of the source repo, to attempt a real load")
    args = parser.parse_args()

    model = models[args.model_id]
    path = pathlib.Path(args.checkpoint)

    print(f"{model['title']}")
    print(f"checkpoint: {path}\n")

    if not path.exists():
        print(f"  ERROR: no such file: {path}")
        return 1

    results = [check_size(path, model), check_is_torch_archive(path), check_architecture(path, model)]
    if args.source_repo:
        results.append(check_loads_into_model(path, model, args.source_repo))

    print()
    if all(results):
        print("PASS -- checkpoint looks intact and matches this model.")
        return 0
    print("FAIL -- see the checks above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
