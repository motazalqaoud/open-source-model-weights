---
license: mit
tags:
  - medical-imaging
  - segmentation
  - 3d
  - mri
  - brain-tumor
  - monai
  - pytorch
  - attention-unet
datasets:
  - Medical Segmentation Decathlon (Task01_BrainTumour)
metrics:
  - dice
  - hausdorff-distance
library_name: pytorch
pipeline_tag: image-segmentation
---

# Brain Tumor Segmentation (3D Attention U-Net)

Multi-class, full-volume 3D segmentation of brain tumors from 4-modality MRI
(FLAIR, T1, T1-contrast, T2). Segments three clinically standard, nested BraTS
regions: **Tumor Core (TC)**, **Whole Tumor (WT)**, and **Enhancing Tumor (ET)**.

Source code: [motazalqaoud/Brain-Tumor-Segmentation](https://github.com/motazalqaoud/Brain-Tumor-Segmentation)

## Model description

- **Architecture:** 3D Attention U-Net (MONAI `AttentionUnet`, `spatial_dims=3`),
  4 input channels (the 4 MRI modalities) → 3 output channels (TC/WT/ET), with
  attention gates on every skip connection.
- **Task framing:** 3-channel multi-label segmentation, not 4-class softmax — TC,
  WT, and ET are nested regions (ET ⊂ TC ⊂ WT), matching how the official BraTS
  leaderboard evaluates submissions.
- **Input:** a single 4D NIfTI volume in MSD layout, shape `(H, W, D, 4)`.
- **Output:** a 3-channel segmentation volume (one channel per region).

## Intended use

Research and portfolio demonstration of a full 3D (not slice-wise) medical
image segmentation pipeline. Suitable for: benchmarking against other
architectures on the same public dataset, as a teaching example of
attention-gated U-Nets on volumetric medical data, or as a starting checkpoint
for further fine-tuning experiments.

**Not intended for, and not validated for, clinical or diagnostic use.** See
Limitations below.

## Training data

[Medical Segmentation Decathlon, Task01_BrainTumour](http://medicaldecathlon.com/)
— 750 4D MRI volumes (484 train / 266 test) sourced from the BraTS 2016/2017
challenge, four co-registered modalities per subject (FLAIR, T1w, T1gd, T2w).
Split by subject (not by image), so no patient appears in both train and
validation sets.

> Simpson, A. L. et al. A large annotated medical image dataset for the
> development and evaluation of segmentation algorithms. Medical Segmentation
> Decathlon (2019). http://medicaldecathlon.com/

## Training procedure

- Mixed precision (AMP)
- Dice loss (sigmoid, applied per nested region independently)
- Cosine learning-rate annealing
- Checkpoint selection: best mean validation Dice across all three regions
- Full-volume validation via sliding-window inference every 5 epochs (not
  patch-level metrics) — the model is checked against the same full-scan
  setting it will be evaluated and used in

## Evaluation results

Full sliding-window inference on held-out volumes (`src/evaluate.py` in the
source repo), the two official BraTS metrics:

| Region | Dice | HD95 (mm) |
|---|---|---|
| TC (Tumor Core) | 0.7915 | 14.18 |
| WT (Whole Tumor) | 0.8712 | 15.80 |
| ET (Enhancing Tumor) | 0.7582 | 7.53 |
| **Mean Dice** | **0.8069** | -- |

For context, published results on similar architectures on this dataset
family often land around Dice 0.85–0.90 (WT), 0.80–0.85 (TC), 0.70–0.80 (ET) —
ET is consistently the hardest region field-wide, being the smallest. Those
are literature ranges from heavily-tuned, ensembled pipelines, not a claim
about what this single-model, single-GPU checkpoint should be expected to hit
on a different dataset or split.

## Download

The checkpoint (`best_model.pth`, 90.2 MB) is published in two places:

- **Hugging Face Hub (primary):** [`motazalqaoud/brain-tumor-segmentation-weights`](https://huggingface.co/motazalqaoud/brain-tumor-segmentation-weights)
- **GitHub Release mirror:** the `v2.1.0` release of [Brain-Tumor-Segmentation](https://github.com/motazalqaoud/Brain-Tumor-Segmentation/releases/tag/v2.1.0)

> Note: the older `v1.0.0` and `v2.0.0` release assets in that repo come from
> earlier, different pipelines (an 8-class WHO tumor-type model, and an earlier
> segmentation run) and do **not** match the metrics on this card.

## How to use

```bash
pip install huggingface_hub torch monai nibabel
git clone https://github.com/motazalqaoud/Brain-Tumor-Segmentation
```

```python
from huggingface_hub import hf_hub_download
import torch
import sys
sys.path.insert(0, "Brain-Tumor-Segmentation")  # the cloned source repo, for model.py
from src.model import build_model  # defines the exact AttentionUnet configuration used

checkpoint_path = hf_hub_download(
    repo_id="motazalqaoud/brain-tumor-segmentation-weights",
    filename="best_model.pth",
)
ckpt = torch.load(checkpoint_path, map_location="cpu")
# the checkpoint stores weights under "model_state_dict", alongside epoch/val_loss
state_dict = ckpt.get("model_state_dict", ckpt)

model = build_model()
model.load_state_dict(state_dict)
model.eval()
```

See `load_from_hub.py` in this directory for a complete, runnable version of
the snippet above, and the source repo's `src/predict.py` for full
single-subject inference (NIfTI in, segmentation NIfTI out).

## Limitations and disclaimer

**This is a research and portfolio checkpoint, not a medical device.** Not
validated prospectively, not reviewed by a regulatory body, and must never be
used to make or defer an actual diagnosis. BraTS-derived training data comes
from a specific set of institutions and scanner protocols; this checkpoint
should not be assumed to generalize to arbitrary clinical scanners or
populations without further validation. Any actual diagnosis requires a
radiologist and the full clinical picture.

## Citation

If you use this checkpoint, please cite the underlying dataset:

```bibtex
@misc{simpson2019medicalsegmentationdecathlon,
  title={A large annotated medical image dataset for the development and evaluation of segmentation algorithms},
  author={Simpson, A. L. and others},
  year={2019},
  howpublished={Medical Segmentation Decathlon},
  url={http://medicaldecathlon.com/}
}
```

## Author

**Motaz Alqaoud, PhD** — [@motazalqaoud](https://github.com/motazalqaoud) ·
[LinkedIn](https://linkedin.com/in/motazalqaoud)

## License

MIT
