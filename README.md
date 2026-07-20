# U-Net Medical Reliability Lab

A reproducible deep-learning project for breast-ultrasound lesion segmentation and subgroup reliability analysis using a compact U-Net.

## Overview

This repository implements an end-to-end and reproducible medical-image segmentation pipeline using the Breast Ultrasound Images Dataset (BUSI).

The project goes beyond reporting a single average segmentation score. It includes:

- deterministic dataset auditing and splitting;
- joint image-mask augmentation;
- compact U-Net training with BCE-Dice loss;
- learning-rate scheduling and early stopping;
- validation-only threshold selection;
- final evaluation on an untouched test split;
- image-level and pooled pixel-level metrics;
- bootstrap confidence intervals;
- benign-versus-malignant subgroup analysis;
- inspection of severe segmentation failures.

## Dataset

The project uses a refined version of the Breast Ultrasound Images Dataset (BUSI).

The dataset audit identified:

- 629 image-mask pairs;
- 420 benign samples;
- 209 malignant samples;
- no missing masks;
- no empty masks;
- no image-mask size mismatches;
- one exact duplicated image pair with conflicting class labels and different masks.

The conflicting pair was excluded:

```text
benign_433.png
malignant_145.png
```

After exclusion, 627 clean samples remained:

| Class | Samples |
|---|---:|
| Benign | 419 |
| Malignant | 208 |
| Total | 627 |

A deterministic stratified split was generated using seed `42`:

| Split | Samples |
|---|---:|
| Train | 438 |
| Validation | 94 |
| Test | 95 |
| Excluded | 2 |

The split manifest is stored at:

```text
data/splits/busi_seed42.csv
```

No filename or pixel-hash leakage was detected between the train, validation, and test splits.

The raw BUSI files are not committed to Git. They are expected at:

```text
data/raw/BUSI/BUSI/
├── images/
└── labels/
```

## Model

The baseline model is a compact two-dimensional U-Net with:

- one-channel grayscale input;
- one-channel binary segmentation output;
- base channel width of 16;
- dropout probability of 0.2;
- approximately 1.94 million trainable parameters;
- joint geometric augmentation for images and masks;
- BCE-Dice optimization loss.

## Training Protocol

The reported baseline was trained using the following configuration:

| Setting | Value |
|---|---:|
| Image size | 128 × 128 |
| Batch size | 8 |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Maximum epochs | 50 |
| Scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 3 |
| Minimum learning rate | 0.000001 |
| Early-stopping patience | 10 |
| Early-stopping minimum delta | 0.001 |
| Random seed | 42 |

Training stopped at epoch 45 because of early stopping.

The best checkpoint was selected at epoch 35 using validation Dice:

```text
Validation Dice: 0.8133
Validation IoU:  0.7210
Learning rate:   0.00025
```

## Threshold Selection

The binary segmentation threshold was selected using only the validation split.

The threshold-selection script first performs a coarse search and then a fine search around the best coarse result.

The selected operating threshold was:

```text
Threshold:       0.57
Validation Dice: 0.8138
Validation IoU:  0.7221
Precision:       0.8376
Sensitivity:     0.8356
Specificity:     0.9820
```

The threshold was locked before evaluating the test split.

The test set was not used for:

- model training;
- checkpoint selection;
- early stopping;
- scheduler decisions;
- threshold selection;
- hyperparameter tuning.

## Final Test Results

The final checkpoint was evaluated once on the untouched test split containing 95 images.

The locked segmentation threshold was `0.57`.

Reported confidence intervals are image-level percentile bootstrap intervals.

| Metric | Mean | 95% Confidence Interval |
|---|---:|---:|
| Dice | 0.8178 | 0.7758–0.8548 |
| IoU | 0.7266 | 0.6806–0.7673 |
| Precision | 0.8478 | 0.8033–0.8854 |
| Sensitivity | 0.8155 | 0.7688–0.8572 |
| Specificity | 0.9883 | 0.9840–0.9917 |

Validation and test Dice were closely aligned:

```text
Validation Dice: 0.8138
Test Dice:       0.8178
Difference:      +0.0040
```

This indicates limited performance degradation on the held-out split. It does not, however, constitute external or clinical validation.

## Subgroup Performance

Performance was evaluated separately for benign and malignant lesions.

### Image-level results

| Metric | Benign, n=63 | Malignant, n=32 |
|---|---:|---:|
| Dice | 0.8605 | 0.7335 |
| IoU | 0.7821 | 0.6172 |
| Precision | 0.8644 | 0.8149 |
| Sensitivity | 0.8777 | 0.6930 |
| Specificity | 0.9913 | 0.9823 |

### Pooled pixel-level results

| Metric | Benign | Malignant |
|---|---:|---:|
| Dice | 0.8655 | 0.7484 |
| IoU | 0.7630 | 0.5980 |
| Precision | 0.8864 | 0.8553 |
| Sensitivity | 0.8457 | 0.6653 |
| Specificity | 0.9915 | 0.9824 |

The main malignant-subgroup weakness was reduced sensitivity, indicating that the model frequently missed parts of malignant lesions.

## Bootstrap Reliability Analysis

Direct subgroup comparisons were calculated using independent image-level bootstrap resampling.

The comparison direction was fixed as:

```text
benign − malignant
```

For performance metrics, a positive difference indicates better benign performance.

For failure rates, a negative difference indicates a higher malignant failure rate.

| Reliability comparison | Difference | 95% Confidence Interval |
|---|---:|---:|
| Mean Dice | +0.1270 | 0.0446–0.2186 |
| Mean sensitivity | +0.1847 | 0.0948–0.2808 |
| Rate of Dice < 0.70 | −0.2178 | −0.3904–−0.0605 |
| Rate of Dice ≥ 0.90 | +0.5729 | 0.4147–0.7153 |

All four intervals exclude zero.

The results indicate that:

- benign lesions achieved higher Dice;
- benign lesions achieved higher sensitivity;
- unreliable predictions were more frequent for malignant lesions;
- excellent segmentation was substantially more frequent for benign lesions.

## Failure Distribution

The distribution of image-level test Dice scores revealed behavior hidden by the overall mean.

| Outcome | Benign | Malignant |
|---|---:|---:|
| Dice < 0.05 | 1.6% | 3.1% |
| Dice < 0.50 | 4.8% | 12.5% |
| Dice < 0.70 | 6.3% | 28.1% |
| Dice ≥ 0.90 | 66.7% | 9.4% |

Examples of observed failure modes included:

- predicting a lesion in the wrong image region;
- severe under-segmentation;
- high precision accompanied by low sensitivity;
- over-segmentation of surrounding tissue;
- nearly complete segmentation failure.

For example, some malignant predictions covered only a small portion of the true lesion while maintaining high precision. This means that the predicted pixels were often correct, but much of the actual lesion was missed.

These findings demonstrate why an overall Dice score alone is insufficient for medical-segmentation reliability assessment.

## Reproducible Workflow

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Place the BUSI dataset

The expected structure is:

```text
data/raw/BUSI/BUSI/
├── images/
└── labels/
```

### 4. Audit the dataset

```powershell
python -m scripts.analyze_busi
```

### 5. Create the deterministic split

```powershell
python -m scripts.create_busi_split
```

### 6. Run all automated tests

```powershell
python -m pytest -v
```

### 7. Train the baseline model

```powershell
python -m scripts.train_busi
```

### 8. Select the validation threshold

```powershell
python -m scripts.select_busi_threshold
```

### 9. Evaluate the untouched test split

```powershell
python -m scripts.evaluate_busi_test
```

### 10. Analyze subgroup reliability

```powershell
python -m scripts.analyze_busi_reliability
```

## Generated Artifacts

Generated experiment outputs are stored under:

```text
artifacts/busi_baseline_128_seed42/
```

Important outputs include:

```text
artifacts/busi_baseline_128_seed42/
├── best_unet.pt
├── history.json
├── training.log
├── threshold_selection.json
└── test_evaluation/
    ├── test_results.json
    ├── per_image_results.csv
    └── reliability_analysis.json
```

The `artifacts/` directory is excluded from Git because it contains generated experiment outputs and model weights.

## Repository Structure

```text
unet-medical-reliability-lab/
├── data/
│   ├── raw/
│   └── splits/
├── scripts/
│   ├── analyze_busi.py
│   ├── analyze_busi_reliability.py
│   ├── create_busi_split.py
│   ├── evaluate_busi_test.py
│   ├── select_busi_threshold.py
│   └── train_busi.py
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── reliability.py
│   ├── reproducibility.py
│   └── training.py
├── tests/
│   ├── test_data.py
│   ├── test_evaluation.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_reliability.py
│   └── test_training.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Automated Testing

The repository currently contains 42 automated tests covering:

- BUSI split loading;
- deterministic data-loader behavior;
- binary image and mask processing;
- U-Net forward and backward passes;
- Dice and BCE-Dice losses;
- image-level segmentation statistics;
- Dice and IoU metrics;
- precision, sensitivity, and specificity;
- training and evaluation loops;
- best-checkpoint selection;
- learning-rate scheduling;
- early stopping;
- bootstrap confidence intervals;
- checkpoint-threshold consistency;
- independent subgroup comparison;
- input validation for reliability utilities.

Current test status:

```text
42 passed
```

## Limitations

- Results are based on a single deterministic BUSI split.
- No external dataset was used for independent validation.
- Images were resized to 128 × 128 pixels.
- Resizing may remove fine lesion-boundary details.
- The malignant test subgroup contains only 32 samples.
- Subgroup confidence intervals remain relatively wide because of the sample size.
- High specificity is influenced by the large proportion of background pixels.
- The bootstrap intervals quantify sampling variability within this dataset and split.
- The reported results do not establish clinical validity.
- The model must not be used for diagnosis, treatment, or clinical decision-making.

## Main Conclusion

The compact U-Net achieved strong overall held-out segmentation performance, with a test Dice score of `0.8178`.

However, subgroup analysis revealed a substantial reliability gap. Malignant lesions had:

- lower Dice;
- lower IoU;
- lower sensitivity;
- greater result variability;
- more frequent unreliable predictions;
- far fewer excellent segmentations.

The project demonstrates that medical-image segmentation evaluation should include subgroup analysis, confidence intervals, and failure-case inspection rather than relying only on a single overall average score.