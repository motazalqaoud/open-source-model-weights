"""Download a published checkpoint by model id, from Hugging Face or the GitHub release.

    python download.py --list
    python download.py brain-tumor-segmentation
    python download.py oncology-skin-lesion-cnn --source github --out ./weights

Hugging Face is the default source (resumable, cached). The GitHub release
mirror needs no extra dependency beyond the standard library, which makes it a
useful fallback in restricted environments.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

MANIFEST = pathlib.Path(__file__).parent / "models.json"


def load_models() -> dict:
    with MANIFEST.open() as fh:
        return {m["id"]: m for m in json.load(fh)["models"]}


def print_catalog(models: dict) -> None:
    print(f"{'ID':<28} {'SIZE':>9}  HEADLINE METRIC")
    print("-" * 78)
    for model in models.values():
        print(f"{model['id']:<28} {model['checkpoint_size_human']:>9}  {model['headline_metric']}")


def download_from_hf(model: dict, out_dir: pathlib.Path) -> pathlib.Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Either:\n"
            "  pip install huggingface_hub\n"
            "or download from the GitHub release instead:\n"
            f"  python download.py {model['id']} --source github"
        )
    path = hf_hub_download(
        repo_id=model["huggingface_repo"],
        filename=model["checkpoint_filename"],
        local_dir=str(out_dir) if out_dir else None,
    )
    return pathlib.Path(path)


def download_from_github(model: dict, out_dir: pathlib.Path) -> pathlib.Path:
    import urllib.request

    url = (
        f"https://github.com/{model['github_source_repo']}/releases/download/"
        f"{model['github_release_tag']}/{model['checkpoint_filename']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / model["checkpoint_filename"]

    print(f"downloading {url}")
    expected = model.get("checkpoint_size_bytes")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        print(f"\r  {done / 1048576:.1f} / {total_size / 1048576:.1f} MB ({pct:.0f}%)", end="")

    urllib.request.urlretrieve(url, target, reporthook=_progress)
    print()

    actual = target.stat().st_size
    if expected and abs(actual - expected) > max(1024 * 1024, expected * 0.05):
        print(
            f"warning: downloaded {actual} bytes but the manifest expects about {expected}. "
            "The release asset may have been replaced.",
            file=sys.stderr,
        )
    return target


def main() -> int:
    models = load_models()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", nargs="?", choices=sorted(models), help="Which checkpoint to fetch")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--source", choices=["hf", "github"], default="hf", help="Where to fetch from")
    parser.add_argument("--out", default="./weights", help="Directory to download into")
    args = parser.parse_args()

    if args.list or not args.model_id:
        print_catalog(models)
        return 0

    model = models[args.model_id]
    out_dir = pathlib.Path(args.out)

    print(f"{model['title']}")
    print(f"  {model['headline_metric']}")
    print(f"  {model['checkpoint_size_human']} from {args.source}")
    print()

    path = download_from_hf(model, out_dir) if args.source == "hf" else download_from_github(model, out_dir)

    print(f"\nsaved to {path}")
    print(f"model card: {model['model_card']}")
    print(f"verify it loaded correctly:  python verify.py {model['id']} --checkpoint {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
