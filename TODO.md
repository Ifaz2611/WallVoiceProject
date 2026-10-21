# TODO — KNN baseline for WallVoice24BD

Goal: build a reproducible k-nearest-neighbours (KNN) baseline over the 1,225 wall-graffiti images, predict a documented label (area first; `content_type` or `topics` later), and report honest results with leakage-aware splits.

Keep originals untouched. All derived files (features, models, predictions) go in ignored folders.

Status: baseline implemented end-to-end; see `results.md` and `getstart.md`. Test accuracy **0.589**, macro-F1 **0.525** on the frozen split (n=185, evaluated once). Remaining unchecked items are manual review tasks, not code.

---

## 0. Setup

- [x] Decide the prediction target: **area** (6 folder-derived classes) — see `docs/label-guide.md`
- [x] Write the label guide in `docs/label-guide.md` (one controlled value per label, include examples)
- [x] Pin dependencies in `requirements.txt` (numpy, pandas, scikit-learn, Pillow, opencv, matplotlib, seaborn, tqdm, joblib, torch, torchvision) and freeze `requirements.lock`
- [x] Decide how the 3 GB of images will be versioned: external archive + `metadata/checksums.sha256` (images stay ignored by Git)

## 1. Data audit

- [x] Verify all 1,225 images open and counts match the README table per folder (`scripts/audit.py` and `scripts/verify_checksums.py`)
- [x] Detect exact duplicates (MD5) and near-duplicates (pHash + ResNet-18 cosine) — 0 exact, 2 pHash pairs, 15 CNN pairs
- [x] Check corrupt files, EXIF rotation, odd colour modes (RGBA/P/CMYK), very small images — all readable RGB PNGs, no EXIF rotation, min 1080×1080
- [ ] Manually review a full sample per area; drop non-graffiti or unreadable shots (record removals, do not delete originals) — six-image spot check done, one per area, all graffiti; full review pending
- [ ] Fill `source` / `license` / `capture_date` where verifiable; leave blanks otherwise — nothing verifiable from the archive; still blank
- [x] Agree on `review_status` values: `unreviewed`, `ai-labelled`, `human-verified`, `disputed`, `excluded`

## 2. Splits (do this before any modelling)

- [x] Group near-duplicate images of the same wall/site into one group (1,211 groups; 8 multi-image, largest 6)
- [x] Split **by group** (near-duplicate groups), not by individual photo: ~70/15/15 per area, seed 42
- [x] Write `splits/train.csv`, `splits/val.csv`, `splits/test.csv` with `image_id,relative_path,label,group_id,area` — paths only, no copied images
- [x] Check per-area counts, class balance, and near-duplicate overlap between splits (zero group overlap)
- [x] Freeze the test split and never tune on it (`scripts/evaluate.py` refuses to overwrite without `--force`)

## 3. Feature extraction

- [x] Baseline A: resize to 128×128 → colour histogram + HOG (OpenCV)
- [x] Baseline B: pretrained CNN embeddings (ResNet-18, 224×224, 512-d)
- [x] L2-normalise CNN and HOG vectors; cache to `features/<extractor>_<split>.npy`
- [x] Save extractor config + library versions to `features/config.json`
- [x] Sanity-check features: PCA + t-SNE plot coloured by area (`outputs/tsne-features.png`); silhouette ≈ -0.01, classes heavily overlap — documented as a limitation

## 4. KNN baseline

- [x] Pipeline: `StandardScaler → PCA (whitening) → KNeighborsClassifier`
- [x] Evaluate `k ∈ {1, 3, 5, 7, 9, 15}`, `weights ∈ {uniform, distance}`, `metric ∈ {euclidean, cosine}`
- [x] Report on validation: accuracy, **macro-F1**, per-class precision/recall, confusion matrix (`outputs/model-selection.md`)
- [x] Inspect misclassified images (contact sheet `outputs/misclassified-test.png`; validation confusion matrix in the selection report)
- [x] Test class imbalance handling: `weights='distance'` (grid) and balanced linear baselines; per-class k/oversampling not implemented because the balanced logistic baseline already beats KNN

## 5. Tuning and comparison

- [x] Tune PCA components (32/64/128/256) + k with grouped 5-fold CV on train+val only
- [x] Compare against simple baselines: `DummyClassifier`, balanced logistic regression, balanced linear SVM
- [x] Note KNN limits in `results.md`: curse of dimensionality, prediction cost, feature scaling, weak with rare classes, over-predicts the majority class
- [x] Ablation table: raw pixels vs HOG+histogram vs CNN embeddings

## 6. Packaging and documentation

- [x] Add scripts: `scripts/audit.py`, `scripts/make_splits.py`, `scripts/extract_features.py`, `scripts/train_knn.py`, `scripts/evaluate.py`, `scripts/plot_results.py` (plus `scripts/visualize_features.py`, `scripts/verify_checksums.py`, and shared `scripts/common.py`)
- [x] Automated plotting of training dashboard (`outputs/training-performance.png`) and evaluation dashboard (`outputs/evaluation-performance.png`)
- [x] Save the fitted pipeline to `models/knn_<extractor>.joblib` and `models/knn_best.joblib` (ignored by Git) and log the config used
- [x] Write `results.md`: dataset version, split method, metrics, confusion matrix, limitations, known collection bias
- [x] Write `getstart.md`: step-by-step onboarding guide for other users to train the model, reproduce splits, and view charts
- [x] Record environment (`python --version` in `results.md`, `pip freeze > requirements.lock`)
- [x] Publish checksums (`metadata/checksums.sha256`, 1,225 entries) so a fresh clone can be verified and rebuilt — the archive itself is external

## 7. Definition of done

- [x] `python scripts/train_knn.py` reproduces the reported numbers from a clean checkout + dataset archive (run with seed 42; model-selection artifacts regenerated)
- [x] No image appears in more than one split, including near-duplicates (0 group overlap)
- [x] Test metrics are reported once, on the frozen split (accuracy 0.589, macro-F1 0.525)
- [ ] Privacy check done: no faces, plates, or precise coordinates published without consent — only aggregate metrics are published and coordinates are blank, but a full face/plate review of all images is still pending

---

## Decisions to make

| Question | Options | Chosen |
|---|---|---|
| Prediction target | area / content_type / topics | **area** (6 folder-derived codes) |
| Feature extractor | HOG+histogram / ResNet-18 / CLIP | **HOG+histogram** (best CV/val); ResNet-18 and raw pixels compared as ablations |
| Split rule | wall/site group / collection route / random (avoid) | **near-duplicate group** (MD5 + pHash + CNN cosine), stratified by area |
| Image storage | Git LFS / external archive + checksums | **external archive + SHA-256 checksums** |

## Implementation summary

| Item | Result |
|---|---|
| Dataset | 1,225 images, 6 areas, all readable RGB PNGs |
| Splits | train 854 / val 186 / test 185, zero group overlap |
| Best KNN | HOG+histogram, PCA 256 whitened, k=9, distance weights, cosine |
| Validation | accuracy 0.548, macro-F1 0.533 |
| Test (frozen, once) | accuracy **0.589**, macro-F1 **0.525**, weighted-F1 0.554 |
| Strongest baseline | balanced logistic regression on HOG, val macro-F1 0.617 |

## Data storage note

Raw images (~3 GB) are excluded by `.gitignore`. Chosen option:

1. **External archive (simplest):** keep images outside Git, publish a zip with `metadata/checksums.sha256`. This repository now includes the checksum file; the zip itself is not committed.

The Git LFS alternative was not chosen.
