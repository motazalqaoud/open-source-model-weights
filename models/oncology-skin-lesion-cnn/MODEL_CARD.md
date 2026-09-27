---
license: mit
tags:
  - medical-imaging
  - classification
  - dermatology
  - skin-cancer
  - efficientnet
  - pytorch
  - ham10000
datasets:
  - HAM10000
metrics:
  - accuracy
  - roc-auc
  - sensitivity
  - specificity
library_name: pytorch
pipeline_tag: image-classification
---

# Oncology Skin Lesion Classifier (EfficientNet-B0 on HAM10000)

A 7-class dermatoscopic skin lesion classifier, fine-tuned on HAM10000, that
reports the clinically-actionable metric — **malignant-class sensitivity** —
that overall accuracy alone hides.

Source code: [motazalqaoud/oncology-diagnosis-using-cnn](https://github.com/motazalqaoud/oncology-diagnosis-using-cnn)

## Model description

- **Architecture:** EfficientNet-B0 (ImageNet-pretrained), classification head
  replaced with a dropout → linear → ReLU → dropout → linear stack, fine-tuned
  on HAM10000.
- **Task:** whole-image 7-class classification (not pixel-level segmentation).
- **Classes:** `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` — three of
  which (`akiec`, `bcc`, `mel`) are malignant or pre-malignant and are flagged
  as such by the demo app.

## Intended use

Research and portfolio demonstration of a clinically-aware evaluation
methodology (lesion-level splitting, class-weighted loss, malignant-specific
sensitivity/specificity reporting) applied to a standard dermatology imaging
benchmark.

**Not intended for, and not validated for, clinical or diagnostic use.** See
Limitations below.

## Training data

[HAM10000](https://doi.org/10.7910/DVN/DBW86T) ("Human Against Machine with
10000 training images"), Tschandl et al., 2018 — 10,015 dermatoscopic images
across seven diagnostic categories, each label verified by histopathology,
expert consensus, confocal microscopy, or follow-up. Heavily imbalanced
(`nv` ≈ 67% of all images; `df` and `vasc` each under 2%).

Split: 7,055 train / 1,475 val / 1,485 test, using **lesion-level** stratified
splitting (not naive random image splitting) — HAM10000 contains repeat
photographs of the same lesion, so splitting by image instead of by lesion
silently leaks information between train and test sets.

> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin
> lesions. *Sci Data* 5, 180161 (2018). https://doi.org/10.7910/DVN/DBW86T

## Training procedure

Inverse-frequency class-weighted loss (rather than naive oversampling) to
address the ~67%/2% class imbalance.

## Evaluation results

Full evaluation on the held-out test split (1,485 images, lesion-level split),
via `src/evaluate.py` in the source repo:

| Metric | Value |
|---|---|
| 7-class accuracy | 83.1% |
| Macro ROC-AUC | 0.9647 |
| **Malignant-class sensitivity** | **64.4%** |
| Malignant-class specificity | 94.1% |
| Binary (malignant/benign) ROC-AUC | 0.920 |
| False negatives (missed malignant) | 101 of 284 |
| Confusion (binary) | TP=183, FN=101, FP=71, TN=1130 |

**The 101 missed malignant lesions is the number to scrutinize** before
considering any downstream use — a model that is 83% accurate overall but
misses 36% of malignant cases is not yet usable as a screening aid without
further improvement (larger backbone, heavier augmentation, ensemble, or a
higher-recall operating threshold).

## Download

The checkpoint (`best_model.pth`, 16.8 MB) is published in two places:

- **Hugging Face Hub (primary):** [`motazalqaoud/oncology-skin-lesion-weights`](https://huggingface.co/motazalqaoud/oncology-skin-lesion-weights)
- **GitHub Release mirror:** the `v1.0.0` release of [Oncology-diagnosis-using-CNN](https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN/releases/tag/v1.0.0)

## How to use

```bash
pip install huggingface_hub torch torchvision
git clone https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN
```

```python
from huggingface_hub import hf_hub_download
import torch
import sys
sys.path.insert(0, "oncology-diagnosis-using-cnn")
from src.classification.model import build_model  # exact EfficientNet-B0 head config used

checkpoint_path = hf_hub_download(
    repo_id="motazalqaoud/oncology-skin-lesion-weights",
    filename="best_model.pth",
)
model = build_model()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()
```

See `load_from_hub.py` in this directory for a complete, runnable version, and
the source repo's `src/predict.py` for single-image CLI inference.

## Limitations and disclaimer

**This is a research and portfolio project, not a medical device.** Not
validated prospectively, not reviewed by a regulatory body, and must never be
used to make or defer an actual diagnosis. HAM10000 also skews toward lighter
Fitzpatrick skin types, a well-documented limitation of most public
dermatology datasets. See a dermatologist for any concerning skin lesion.

## Citation

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific Data},
  volume={5},
  pages={180161},
  year={2018},
  doi={10.7910/DVN/DBW86T}
}
```

## Author

**Motaz Alqaoud, PhD** — [@motazalqaoud](https://github.com/motazalqaoud) ·
[LinkedIn](https://linkedin.com/in/motazalqaoud)

## License

MIT
