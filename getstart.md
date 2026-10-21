# Getting Started with WallVoice24BD

Welcome to **WallVoice24BD**! This guide walks you step-by-step through setting up the environment, verifying and splitting the dataset, extracting image features, training the baseline machine learning models, evaluating on the frozen test set, and inspecting interactive plots and visual charts for accuracy, loss/error rate, and per-class performance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quickstart in 5 Minutes](#2-quickstart-in-5-minutes)
3. [Environment Setup](#3-environment-setup)
4. [Step 1: Verify and Audit the Dataset](#4-step-1-verify-and-audit-the-dataset)
5. [Step 2: Generate Leakage-Aware Data Splits](#5-step-2-generate-leakage-aware-data-splits)
6. [Step 3: Extract Image Features](#6-step-3-extract-image-features)
7. [Step 4: Train the Models](#7-step-4-train-the-models)
8. [Step 5: Evaluate on the Frozen Test Set](#8-step-5-evaluate-on-the-frozen-test-set)
9. [Step 6: View Performance Plots and Charts](#9-step-6-view-performance-plots-and-charts)
10. [Reference: Scripts and Outputs](#10-reference-scripts-and-outputs)
11. [Troubleshooting & FAQs](#11-troubleshooting--faqs)

---

## 1. Project Overview

WallVoice24BD is a curated dataset of **1,225 high-resolution PNG images** of wall graffiti collected across six prominent areas of Dhaka, Bangladesh:

| Area Folder | Code | Images | Description |
|---|:---:|:---:|---|
| `DU-Shahbagh/` | **DUS** | 428 | Dhaka University and Shahbagh area |
| `Dhanmondi/` | **DMD** | 312 | Dhanmondi residential & commercial walls |
| `Azimpur-Nilkhet/` | **AZN** | 172 | Azimpur and Nilkhet book market area |
| `BaileyRoad-Viqarunnisa/` | **BRV** | 136 | Bailey Road and Viqarunnisa school area |
| `Buet-Palashi/` | **BUP** | 90 | BUET campus and Palashi intersection |
| `Mirpur-Uttara-Tejgaon/` | **MUT** | 87 | Mirpur, Uttara, and Tejgaon areas |
| **Total** | | **1,225** | |

### The Machine Learning Task
The primary baseline task is **Area Classification**: predicting which of the six collection areas a graffiti image belongs to based on visual patterns, colours, and textures.

The pipeline compares three feature representations:
- **`raw`**: 32×32 raw RGB pixel values.
- **`hog`**: Colour histograms (RGB + HSV) concatenated with OpenCV Histogram of Oriented Gradients (HOG).
- **`resnet`**: 512-dimensional penultimate embeddings from a pretrained ResNet-18 deep neural network.

The classification baseline uses a **StandardScaler → PCA (whitening) → K-Nearest Neighbours (KNN)** pipeline optimized with grouped cross-validation, and compares against balanced Logistic Regression and Linear SVM baselines.

---

## 2. Quickstart in 5 Minutes

For users who want to run the full end-to-end pipeline in one go:

### On Windows (PowerShell):
```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Verify dataset integrity and audit images
python scripts/verify_checksums.py
python scripts/audit.py --checksums

# 4. Generate splits and extract features
python scripts/make_splits.py
python scripts/extract_features.py --extractor all

# 5. Train model and evaluate
python scripts/train_knn.py
python scripts/evaluate.py

# 6. Generate and view plots
python scripts/plot_results.py
```

### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python scripts/verify_checksums.py
python scripts/audit.py --checksums
python scripts/make_splits.py
python scripts/extract_features.py --extractor all
python scripts/train_knn.py
python scripts/evaluate.py
python scripts/plot_results.py
```

---

## 3. Environment Setup

### Prerequisites
- **Python**: Version 3.12 or newer (tested and verified on Python 3.14.5).
- **OS**: Windows, macOS, or Linux.
- **Git** and command terminal.

### 1. Image Folders Check
Ensure the six image folders are placed in the root of the project directory alongside `metadata/images.csv`:
```text
WallVoice24BD/
├── Azimpur-Nilkhet/
├── BaileyRoad-Viqarunnisa/
├── Buet-Palashi/
├── Dhanmondi/
├── DU-Shahbagh/
├── Mirpur-Uttara-Tejgaon/
├── metadata/
│   ├── checksums.sha256
│   └── images.csv
├── scripts/
└── requirements.txt
```

### 2. Virtual Environment Setup
Create an isolated environment:
```powershell
python -m venv .venv
```

Activate it:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If you encounter a `PSSecurityException` regarding script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry)*
- **Windows (Command Prompt)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Required Libraries
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include `numpy`, `pandas`, `scikit-learn`, `opencv-python-headless`, `torch`, `torchvision`, `pillow`, `matplotlib`, `seaborn`, `tqdm`, and `joblib`.

---

## 4. Step 1: Verify and Audit the Dataset

### Verify File Integrity
Check that all 1,225 images match the official SHA-256 cryptographic hashes:
```bash
python scripts/verify_checksums.py
```
*Expected output: `Checked 1225 files from …/metadata/checksums.sha256` followed by `All checksums match.`*

### Run Dataset Audit & Near-Duplicate Detection
```bash
python scripts/audit.py --checksums
```
This script inspects image dimensions, color profiles, corrupt files, and detects near-duplicates using perceptual hashes (pHash) and ResNet-18 cosine similarity.
- Audit reports are saved under `outputs/audit/`.
- Near-duplicate groups are saved to `outputs/audit/near_duplicate_groups.csv`.

*(Optional: pass `--sample-sheets` to generate review contact sheets of 8 random graffiti images per area under `outputs/audit/sample-sheets/`)*.

---

## 5. Step 2: Generate Leakage-Aware Data Splits

To ensure scientific validity, near-duplicate images of the same wall or scene must **never** be divided across training and testing sets.

Run the grouping and splitting script:
```bash
python scripts/make_splits.py
```

### How the split works:
- **Ratios**: ~70% Train (854 images), ~15% Validation (186 images), ~15% Test (185 images).
- **Stratification**: Balanced across all six areas while keeping near-duplicate groups intact in the same split.
- **Zero Leakage**: Group overlap between splits is strictly `0 / 0 / 0`.
- The split lists are stored in `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`.

> [!NOTE]
> `make_splits.py` refuses to overwrite existing splits unless you pass `--force`. The test split is frozen to prevent test-set contamination.

---

## 6. Step 3: Extract Image Features

Extract visual features for each data split:

```bash
python scripts/extract_features.py --extractor all
```

You can also extract individual representations:
- `python scripts/extract_features.py --extractor hog` (fast, recommended baseline)
- `python scripts/extract_features.py --extractor resnet` (PyTorch deep embeddings)
- `python scripts/extract_features.py --extractor raw` (raw downsampled pixels)

Features are cached under `features/` (e.g. `hog_train.npy`, `hog_val.npy`, `hog_test.npy`). If you ever need to recompute cached features, pass `--force`.

### Feature Sanity Check (PCA + t-SNE)
You can visualize the feature separability before training:
```bash
python scripts/visualize_features.py
```
This generates `outputs/tsne-features.png`, projecting high-dimensional features down to 2D coloured by collection area.

---

## 7. Step 4: Train the Models

Train the classifier across all feature extractors with grouped cross-validation on `train` + `val`:

```bash
python scripts/train_knn.py
```

### What happens during training:
1. **Grid Search**: Evaluates PCA dimensions (32, 64, 128, 256), k neighbors (1, 3, 5, 7, 9, 15), distance weighting (`uniform` vs `distance`), and distance metrics (`euclidean` vs `cosine`) using 5-fold `StratifiedGroupKFold`.
2. **Model Selection**: Chooses the best configuration using **macro-F1** (protects rare classes from majority class bias).
3. **Baselines**: Trains and compares against simple baselines (`DummyClassifier`, balanced `LogisticRegression`, and balanced `LinearSVC`).
4. **Artifacts Saved**:
   - Model files: `models/knn_best.joblib`, `models/knn_hog.joblib`, `models/knn_raw.joblib`, `models/knn_resnet.joblib`.
   - Grid search results: `outputs/cv-results-<extractor>.csv`.
   - Summary report: `outputs/model-selection.md` and `outputs/model-selection.json`.
5. **Automated Plot Generation**:
   - Automatically generates the training performance dashboard: `outputs/training-performance.png`.

### Useful Training Flags:
- **Fast debug training** (smaller search grid):
  ```bash
  python scripts/train_knn.py --quick
  ```
- **Train only specific extractors**:
  ```bash
  python scripts/train_knn.py --extractors hog
  ```
- **Custom CPU threads / jobs**:
  ```bash
  python scripts/train_knn.py --n-jobs 4
  ```

---

## 8. Step 5: Evaluate on the Frozen Test Set

Once model selection is complete, evaluate the chosen model on the held-out test split:

```bash
python scripts/evaluate.py
```

### Evaluation Highlights:
- Evaluates `models/knn_best.joblib` on `splits/test.csv` (n = 185).
- Expected test result (HOG baseline):
  - **Accuracy**: `0.5892` (58.9%)
  - **Macro-F1**: `0.5252` (52.5%)
  - **Weighted-F1**: `0.5543` (55.4%)
- Writes `outputs/evaluation.json` and `outputs/classification-report.txt`.
- Saves a contact sheet of errors: `outputs/misclassified-test.png`.
- Automatically generates the test evaluation dashboard: `outputs/evaluation-performance.png`.

> [!IMPORTANT]
> The test split is evaluated **once**. `scripts/evaluate.py` will refuse to overwrite `outputs/evaluation.json` unless `--force` is provided. Always tune hyperparameters using validation scores rather than repeated test evaluations.

---

## 9. Step 6: View Performance Plots and Charts

After training and evaluation, you can inspect the visual charts to analyze accuracy, classification loss (error rate), hyperparameter tuning curves, and per-class performance.

### Run the Plotting Tool:
```bash
python scripts/plot_results.py
```
*(If you are running in an interactive desktop environment and want a GUI popup window, pass `--show`: `python scripts/plot_results.py --show`)*

---

### Understanding the Generated Charts

#### 1. Training & Validation Dashboard (`outputs/training-performance.png`)

This 4-panel figure provides an overview of training performance:

```
+------------------------------------------+------------------------------------------+
|  Panel A: Model & Feature Comparison     |  Panel B: Hyperparameter Tuning Curves   |
|  - Validation Accuracy (Blue)            |  - CV Macro-F1 vs k-neighbors (1 to 15)  |
|  - Validation Macro-F1 (Green)           |  - Curves for PCA 32, 64, 128, 256       |
|  - 0-1 Error Rate / Loss (Orange)        |  - Highlights optimal parameter plateau  |
+------------------------------------------+------------------------------------------+
|  Panel C: Per-Class Validation Breakdown |  Panel D: Normalized Confusion Matrix    |
|  - Precision, Recall, F1 for each area   |  - Percentage recall per true class      |
|  - Annotates support count (n) per class |  - Identifies misclassification bias     |
+------------------------------------------+------------------------------------------+
```

1. **Panel A (Model Comparison)**:
   - Compares **KNN (Raw)**, **KNN (HOG)**, **KNN (ResNet)**, **Logistic Regression (balanced)**, **Linear SVM (balanced)**, and **Dummy Baseline**.
   - Displays **Accuracy**, **Macro-F1**, and **0-1 Classification Error Rate (Loss = 1 - Accuracy)**.
   - Shows that HOG features outperform deep ImageNet ResNet embeddings for this domain, while balanced Logistic Regression serves as a strong linear reference model.
2. **Panel B (Hyperparameter Curves)**:
   - Plots cross-validation performance across neighbor values `k ∈ {1, 3, 5, 7, 9, 15}` for each PCA dimension.
   - Illustrates the trade-off: small k overfits noise; large k over-smooths decision boundaries toward the majority class.
3. **Panel C (Per-Class Performance)**:
   - Detailed bar chart of Precision, Recall, and F1-score for each of the six areas: `AZN`, `BRV`, `BUP`, `DMD`, `DUS`, `MUT`.
   - Displays how class imbalance affects recall (e.g. `DUS` has high recall; rarer classes have lower recall).
4. **Panel D (Validation Confusion Matrix)**:
   - Heatmap showing both row-normalized percentage (true class recall) and raw count.
   - Reveals common confusions (e.g., graffiti from smaller areas being classified into the predominant `DUS` class).

---

#### 2. Test Evaluation Dashboard (`outputs/evaluation-performance.png`)

Generated from the held-out test split (n = 185):

- **Panel A (Test Per-Class Performance)**:
  Shows test Precision, Recall, and F1 for each area with sample counts (`AZN: 27`, `BRV: 21`, `BUP: 14`, `DMD: 50`, `DUS: 60`, `MUT: 13`).
- **Panel B (Test Confusion Matrix)**:
  Shows exact predictions vs true labels on the unseen test set, annotated with percentages and raw image counts.
- **Top Summary Banner**:
  Displays overall Test Accuracy (58.92%), Macro-F1 (52.52%), Weighted-F1 (55.43%), and Error Rate (41.08%, 76 misclassified images).

---

#### 3. Visual Error Inspection (`outputs/misclassified-test.png`)
A photographic contact sheet showing misclassified graffiti images from the test set. Each cell shows the image thumbnail, the true label, the predicted label, and the prediction confidence.

---

## 10. Reference: Scripts and Outputs

### Scripts

| Script | Purpose |
|---|---|
| `scripts/verify_checksums.py` | Validates image files against `metadata/checksums.sha256`. |
| `scripts/audit.py` | Audits image dimensions, color channels, and detects near-duplicates. |
| `scripts/make_splits.py` | Generates 70/15/15 leakage-aware train/val/test splits. |
| `scripts/extract_features.py` | Computes and caches raw pixel, HOG, and ResNet-18 features. |
| `scripts/visualize_features.py`| Computes PCA + t-SNE 2D embedding sanity check. |
| `scripts/train_knn.py` | Performs hyperparameter search, fits models, and plots training dashboard. |
| `scripts/evaluate.py` | Evaluates selected model on frozen test set and generates evaluation plots. |
| `scripts/plot_results.py` | Visualizes training and evaluation charts, accuracy, loss, and curves. |
| `scripts/common.py` | Shared utility functions, dataset configurations, and label mappings. |

### Generated Outputs (`outputs/`)

| File | Description |
|---|---|
| `outputs/training-performance.png` | **Multi-panel training chart**: Accuracy, loss, hyperparameter curves, per-class metrics. |
| `outputs/evaluation-performance.png`| **Test evaluation chart**: Test per-class metrics and normalized confusion matrix. |
| `outputs/model-selection.md` | Human-readable markdown report of CV and validation tuning results. |
| `outputs/model-selection.json` | JSON format of all model configurations and metrics. |
| `outputs/cv-results-<extractor>.csv` | Raw grid search table for each feature extractor. |
| `outputs/evaluation.json` | Final frozen test set metrics and configuration. |
| `outputs/classification-report.txt` | Standard scikit-learn test classification report. |
| `outputs/confusion-matrix-test.png` | Standalone test confusion matrix figure. |
| `outputs/misclassified-test.png` | Thumbnail contact sheet of misclassified test images. |
| `outputs/tsne-features.png` | 2D t-SNE projection of extracted features. |

---

## 11. Troubleshooting & FAQs

### Q1: PowerShell says "running scripts is disabled on this system"
Run the following in PowerShell to bypass the execution policy for your current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Q2: How do I retrain after adding new images?
1. Update `metadata/images.csv` with the new records.
2. Re-run `python scripts/audit.py --checksums` to update hash mappings.
3. Re-run `python scripts/make_splits.py --force` to update splits.
4. Re-run `python scripts/extract_features.py --extractor all --force` to recompute cached features.
5. Re-run `python scripts/train_knn.py` and `python scripts/evaluate.py --force`.

### Q3: Why is Macro-F1 (0.525) lower than Accuracy (0.589)?
The dataset has natural collection imbalance: `DUS` accounts for ~35% of all images, whereas `MUT` accounts for ~7%. Standard accuracy is dominated by the majority class. **Macro-F1** weights every area equally, providing an honest measure of performance across all regions.

### Q4: Can I use GPU acceleration?
The baseline KNN and scikit-learn pipelines run on CPU; the full three-extractor grid can take several minutes, while a single extractor or `--quick` is faster. For ResNet-18 feature extraction, PyTorch will automatically utilize CUDA if a compatible GPU and PyTorch CUDA build are installed.

### Q5: How do I display the plots directly on screen?
Run:
```bash
python scripts/plot_results.py --show
```
If you are on a headless remote server without an X11/GUI display, omit `--show` and view the saved `.png` files directly.
