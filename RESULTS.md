# Reproduction Results

This file documents the real, independently-run reproduction experiments performed during peer review of the accompanying manuscript, using the notebooks in `notebooks/`. All numbers below are real experimental results, not estimates.

## 1. Dataset verification and leakage check

Notebook: `notebooks/Run_Duplicate_Check_Kaggle.ipynb`

- Public dataset mirror used: `preetpalsingh25/alzheimers-dataset-4-class-of-images` on Kaggle (a copy of the original `tourist55` dataset cited in the manuscript, which was no longer available)
- Raw images found: 12,800 (this mirror contains an exact duplicate of every image)
- After deduplication: **6,400 unique images** — matches the manuscript's Table 2 total exactly
- Near-duplicate pairs found (pHash, 16×16, Hamming distance ≤ 5): **1**, and it did **not** cross a train/val/test split boundary
- Conclusion: no meaningful image-level leakage risk from duplicate/near-duplicate slices was found in this dataset once deduplicated

## 2. Simplified baseline comparison (proxy model)

Notebook: `notebooks/Train_and_Evaluate_Baselines_Kaggle.ipynb`

Trained ResNet50, DenseNet121, EfficientNet-B0 (standard transfer learning) and a simplified ADNET proxy (dual-stream backbone without CSPA/HAF/oversampling/focal loss) via real 5-fold cross-validation.

| Model | Mean Accuracy | Std | Mean Macro F1 |
|---|---|---|---|
| ResNet50 (in-house) | 64.48% | ±1.65% | 41.57% |
| DenseNet121 (in-house) | 90.03% | ±2.66% | 90.51% |
| EfficientNet-B0 (in-house) | 85.58% | ±1.23% | 85.32% |
| ADNET-simplified (in-house) | 90.41% | ±1.52% | 71.08% |

McNemar's test (ADNET-simplified vs. each baseline, Bonferroni-corrected α=0.0167):
- vs. ResNet50: χ²=1307.16, p<0.00001 (significant)
- vs. DenseNet121: χ²=0.56, p=0.45458 (not significant)
- vs. EfficientNet-B0: χ²=83.14, p<0.00001 (significant)

Friedman test across all 4 models: χ²=11.880, p=0.00781 (significant overall difference)

Post-hoc Wilcoxon signed-rank (ADNET-simplified vs. each, paired by fold): none reach significance after Bonferroni correction (n=5 folds limits achievable power), though ADNET-simplified won every fold against ResNet50 and EfficientNet-B0.

ADNET-simplified's Moderate Demented performance, pooled across 5 folds: 8/64 = 12.5% (95% exact CI 5.6–23.2%) — far below the full model's claimed 100%, consistent with the simplified proxy lacking the class-imbalance mitigations (oversampling, focal loss, CSPA) that the full architecture uses specifically to address this.

## 3. Full ADNET architecture reproduction (the key result)

Notebook: `notebooks/Train_Full_ADNET_Kaggle.ipynb`

Trained the **complete** ADNET architecture (full CSPA, HAF, composite weighted-cross-entropy + focal loss, class-imbalance oversampling matching the manuscript's Section 3.3 factors) via real 5-fold cross-validation, using full-depth EfficientNet-B3 and Swin-T backbones (the manuscript's claimed 3.2M-parameter truncated configuration could not be recovered/verified — see the model code's docstring).

**Training-budget progression:**

| Training budget | Overall Accuracy | Macro F1 | Folds completed |
|---|---|---|---|
| 20 epochs, early stopping (patience 6) | 75.44% (±8.46%) | 80.39% (±5.99%) | 5/5 |
| Extended, up to 100 epochs (patience 15), 10.5h wall-clock budget | 94.75% (±3.67%) | 96.15% (±2.64%) | 5/5 (complete) |
| **Manuscript's claimed figure** | **99.41%** | **99.29%** | — |

**Final per-class accuracy (extended run, pooled across all 5 folds, all 6,400 images evaluated exactly once):**

| Class | Correct / Total | Accuracy |
|---|---|---|
| Non-Demented | 2,944 / 3,200 | 92.00% |
| Mild Demented | 866 / 896 | 96.65% |
| Very Mild Demented | 2,190 / 2,240 | 97.77% |
| Moderate Demented | 64 / 64 | 100.00% (95% exact CI 94.4–100%) |

**Interpretation:** the large initial gap (75.44% vs. 99.41%) was substantially explained by training budget — the extended run closed most of it, landing ~4.7 percentage points below the claimed accuracy. This remaining gap could reflect: the manuscript's claimed 150-epoch budget (vs. 100 here), the still-unconfirmed backbone truncation, or the augmentation pipeline (a reasonable but unconfirmed interpretation of the manuscript's described "seven physiologically-grounded strategies"). Moderate Demented performance in this reproduction (100%, 64/64) closely matches the original claim.

**Recommended next step:** train the original implementation for a fully matched 150-epoch budget and report the result transparently. If it converges to results near those shown above rather than 99.41%, the manuscript's headline performance claims should be revised accordingly.

## Reproducibility notes

- Random seed: 42 (Python, NumPy, PyTorch) throughout
- All training used `imagenet`-pretrained backbones via `timm`
- Exact split index files are not included here (see `splits/` placeholders) — the notebooks regenerate a fresh stratified 70/15/15 (or 5-fold) split with the fixed seed, which is deterministic given the same dataset ordering but has not been cross-checked byte-for-byte against the original manuscript's split
