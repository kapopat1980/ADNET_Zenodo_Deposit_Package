# ADNET: Adaptive Dual-Stream Network with Hierarchical Attention Fusion for Early-Stage Alzheimer's Disease Detection from Brain MRI

This repository accompanies the manuscript submitted to *Scientific Reports* (Submission ID: `6eed87ac-3b1a-4c84-976c-8a7c9bebcf44`). It is provided to satisfy the journal's code/data availability requirement and to support reviewer and reader reproducibility.

> **Status note (please read before using):** this repository was assembled during peer review to close out reviewer requests for a public code/data deposit, and now includes real, independently-run reproduction experiments performed during that review (see `RESULTS.md`). Key finding: a good-faith reproduction of the complete architecture reached 94.75% accuracy / 96.15% macro F1, compared to the manuscript's claimed 99.41%/99.29% — a real, disclosed, ~4.7-point gap that has not yet been fully resolved. Before finalizing this deposit, the authors should:
> - Confirm the exact dataset split files used for the manuscript's originally reported results (`splits/train.txt`, `splits/val.txt`, `splits/test.txt` — currently placeholders; the notebooks in `notebooks/` generate their own fresh, seeded splits instead)
> - Add trained model weights (not included here — add a release asset or external link) once available
> - Confirm whether `src/adnet_model.py`'s Stream A/B backbone configuration (full-depth, as currently implemented) matches the original 3.2M-parameter claim, or whether a truncated/reduced configuration was actually used (see Section 4.1 of the manuscript)
> - Ideally, attempt a training run matched to the manuscript's claimed 150-epoch budget (the included reproduction used up to 100 epochs) to see whether the remaining gap closes further

## Repository contents

```
├── README.md                      # this file
├── RESULTS.md                     # real reproduction results (read this first)
├── LICENSE                        # MIT License
├── CITATION.cff                   # citation metadata (GitHub "Cite this repository" support)
├── .zenodo.json                   # Zenodo deposit metadata
├── requirements.txt               # Python dependencies
├── src/
│   └── adnet_model.py             # ADNET architecture (PyTorch), implementing Section 4's equations
├── scripts/
│   └── duplicate_leakage_check.py # dataset duplicate/near-duplicate detection (Reviewer 1, Point 2)
├── notebooks/
│   ├── Run_Duplicate_Check_Kaggle.ipynb           # dataset verification + leakage check (real, run on Kaggle)
│   ├── Train_and_Evaluate_Baselines_Kaggle.ipynb  # simplified baseline comparison (real, run on Kaggle)
│   └── Train_Full_ADNET_Kaggle.ipynb              # full architecture reproduction (real, run on Kaggle)
└── splits/
    ├── train.txt                  # PLACEHOLDER — replace with actual split file
    ├── val.txt                    # PLACEHOLDER — replace with actual split file
    └── test.txt                   # PLACEHOLDER — replace with actual split file
```

**Start with `RESULTS.md`** if you want the real experimental findings without reading code. The three notebooks in `notebooks/` are runnable end-to-end on Kaggle (free tier, GPU required for the training notebooks) and were the actual notebooks used to produce the results in `RESULTS.md`.

## Dataset

The Alzheimer's MRI Dataset (4-class) used in this study is publicly available on Mendeley Data:
https://data.mendeley.com/datasets/3r8hw8wmmk/1

This repository does not redistribute the dataset itself.

## Installation

```bash
pip install -r requirements.txt
```

## Model architecture

`src/adnet_model.py` implements:
- **Stream A**: EfficientNet-B3 backbone with Squeeze-and-Excitation submodules
- **Stream B**: Swin Transformer (Tiny) with shifted-window multi-head self-attention
- **CSPA**: Channel-Spatial Progressive Attention module with anatomical-prior gating
- **HAF**: Hierarchical Attention Fusion across three spatial scales
- **Classification head**: two fully-connected layers with dropout, four-class softmax output

See Section 4 of the manuscript for the corresponding equations (6)–(31); each is annotated with its equation number in code comments.

## Reproducing the duplicate-check analysis

```bash
python scripts/duplicate_leakage_check.py \
    --train_dir /path/to/train --val_dir /path/to/val --test_dir /path/to/test \
    --hash_size 16 --near_dup_threshold 5 \
    --out_csv duplicate_report.csv
```

A zero-setup version of this same check, which downloads the public dataset and runs entirely in a browser via Google Colab, is also available on request.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`); full bibliographic details will be finalized upon publication.

## License

MIT License — see `LICENSE`.

## Contact

Kalpesh Popat (corresponding author) — kapopat1980@gmail.com
