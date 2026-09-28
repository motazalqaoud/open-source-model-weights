# open-source-model-weights

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-motazalqaoud-black)](https://github.com/motazalqaoud)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-motazalqaoud-yellow)](https://huggingface.co/motazalqaoud)

> Model cards, evaluation results, and loader scripts for trained checkpoints
> from this account's medical-imaging projects. The checkpoints themselves
> (`.pth` files) are hosted on Hugging Face Hub, not in this git repository —
> see each model card for the exact Hub repo id.

## Why weights live on Hugging Face, not here

Trained checkpoints are tens to hundreds of megabytes of binary data that
don't belong in git history, and Hugging Face Hub already provides versioned
storage, download stats, and a standard model-card format for exactly this.
This repo is the **documentation and loading layer**: what each checkpoint is,
how it was trained, what it actually scores on a held-out test set, and a
runnable script to fetch and load it — not a duplicate copy of the binary.

## Models

| Model | Task | Key metric | Source repo | Hugging Face |
|---|---|---|---|---|
| [Brain Tumor Segmentation](models/brain-tumor-segmentation/MODEL_CARD.md) | 3D multi-region MRI segmentation | Mean Dice 0.8069 (TC/WT/ET) | [Brain-Tumor-Segmentation](https://github.com/motazalqaoud/Brain-Tumor-Segmentation) | `motazalqaoud/brain-tumor-segmentation-weights` |
| [Oncology Skin Lesion Classifier](models/oncology-skin-lesion-cnn/MODEL_CARD.md) | 7-class dermatoscopic classification | 64.4% malignant sensitivity | [Oncology-diagnosis-using-CNN](https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN) | `motazalqaoud/oncology-skin-lesion-weights` |

## Where to download the weights

Each checkpoint is published in two places. Hugging Face Hub is the primary
source (versioned, resumable, and what `load_from_hub.py` uses); the matching
GitHub Release asset is a mirror for anyone who would rather pull the file
straight from the source repository.

| Model | Hugging Face | GitHub Release asset |
|---|---|---|
| Brain Tumor Segmentation (90.2 MB) | [brain-tumor-segmentation-weights](https://huggingface.co/motazalqaoud/brain-tumor-segmentation-weights) | [`v2.1.0`](https://github.com/motazalqaoud/Brain-Tumor-Segmentation/releases/tag/v2.1.0) |
| Oncology Skin Lesion (16.8 MB) | [oncology-skin-lesion-weights](https://huggingface.co/motazalqaoud/oncology-skin-lesion-weights) | [`v1.0.0`](https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN/releases/tag/v1.0.0) |

Both files are named `best_model.pth`.

**A note on older Brain-Tumor-Segmentation release assets:** the `v1.0.0` and
`v2.0.0` releases of that repo carry assets from *earlier, different* pipelines
(an 8-class WHO tumor-type model and an earlier segmentation run). They do not
correspond to the Dice 0.8069 3D Attention U-Net documented here -- use the
`v2.1.0` asset or the Hugging Face copy for that.

Each model card is a complete, standalone Hugging Face-style model card:
architecture, training data and procedure, full evaluation numbers (not just
a headline accuracy), intended use, and limitations.

## Repository structure

```
open-source-model-weights/
├── models.json                 Machine-readable index of every published checkpoint
├── download.py                 Fetch a checkpoint by id (Hugging Face or GitHub release)
├── verify.py                   Check a downloaded file is intact and the right architecture
├── models/
│   ├── brain-tumor-segmentation/
│   │   ├── MODEL_CARD.md      Architecture, training data, Dice/HD95 results, limitations
│   │   └── load_from_hub.py   Downloads the checkpoint from HF Hub and loads it
│   └── oncology-skin-lesion-cnn/
│       ├── MODEL_CARD.md      Architecture, training data, accuracy/sensitivity results, limitations
│       └── load_from_hub.py   Downloads the checkpoint from HF Hub and loads it
└── README.md
```

## Quickstart

List what's published, fetch one, and confirm it's intact:

```bash
python download.py --list
python download.py oncology-skin-lesion-cnn
python verify.py oncology-skin-lesion-cnn --checkpoint ./weights/best_model.pth
```

`verify.py` on a real checkpoint prints:

```
  size: 16.8 MB  OK
  format: PyTorch archive, 368 entries  OK
  architecture: 362 tensors, 4,379,348 parameters
                training metadata: epoch=14, val_acc=0.8000, val_loss=0.6155
                358/362 keys match the 'backbone.features.*' fingerprint  OK

PASS -- checkpoint looks intact and matches this model.
```

`download.py --source github` pulls from the GitHub release instead of Hugging
Face using only the standard library -- useful where `huggingface_hub` can't be
installed or the Hub isn't reachable.

### Loading it in code

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(
    repo_id="motazalqaoud/oncology-skin-lesion-weights",
    filename="best_model.pth",
)
ckpt = torch.load(checkpoint_path, map_location="cpu")

# NOTE: these checkpoints were saved mid-training, so the weights sit under
# "model_state_dict" alongside epoch/val_loss -- not at the top level.
state_dict = ckpt.get("model_state_dict", ckpt)
```

To actually run inference you also need the model class from the corresponding
source repo -- see each model card's "How to use" section, or run that model's
`load_from_hub.py`, which handles the unwrapping for you.

## Files

| Path | What it does |
|---|---|
| `models.json` | Machine-readable manifest: sizes, metrics, HF/GitHub locations, architecture fingerprints |
| `download.py` | Fetch any published checkpoint by id, from Hugging Face or the GitHub release mirror |
| `verify.py` | Confirm a downloaded file is intact and is the architecture it claims to be |
| `models/*/MODEL_CARD.md` | Full model card per checkpoint |
| `models/*/load_from_hub.py` | Download and load into the real model class |

## Limitations and disclaimer

**These are research and portfolio checkpoints, not medical devices.** None
are validated prospectively, reviewed by a regulatory body, or intended to
make or defer an actual diagnosis. See each model card for dataset-specific
caveats.

## About the author

**Motaz Alqaoud, PhD**
- PhD in Biomedical Engineering with focus on medical image analysis and deep learning
- Senior AI/ML Engineer specializing in medical imaging, segmentation models, and clinical AI systems
- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- Hugging Face: [@motazalqaoud](https://huggingface.co/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

## License

MIT — see [LICENSE](LICENSE).
