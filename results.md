# KNN baseline results — predicting collection area

Status: complete baseline, frozen test evaluated once. Generated on
2026-10-21 from base commit `d7709a6` plus the baseline implementation in this
change, against the ignored image archive verified by
`metadata/checksums.sha256`.

## 1. Task and label

- **Prediction target:** `area`, one of six collection areas (`AZN`, `BRV`,
  `BUP`, `DMD`, `DUS`, `MUT`). See `docs/label-guide.md`.
- The label comes from the image folder. It is a **collection-area** label, not
  a claim about the author, topic, language, or exact site.
- No `content_type`, `topics`, transcription, or sentiment labels were used;
  those fields remain unreviewed.

| Label | Area | Images |
|---|---|---:|
| `DUS` | DU-Shahbagh | 428 |
| `DMD` | Dhanmondi | 312 |
| `AZN` | Azimpur-Nilkhet | 172 |
| `BRV` | BaileyRoad-Viqarunnisa | 136 |
| `BUP` | Buet-Palashi | 90 |
| `MUT` | Mirpur-Uttara-Tejgaon | 87 |
| | **Total** | **1,225** |

## 2. Data audit

`python scripts/audit.py --checksums` checked all 1,225 files.

- Counts match the README table exactly; 1,225 unique IDs and paths.
- All files open, all are RGB PNGs, no corrupt files, no EXIF rotation flags,
  no image smaller than 1080×1080.
- **No exact (MD5) duplicates.**
- 2 pHash near-duplicate pairs (Hamming ≤ 6); 15 ResNet-18 near-duplicate pairs
  (cosine ≥ 0.93). Together they form **1,211 groups**: 8 groups have more than
  one image, the largest has 6 images, and 3 groups cross area folders.
- All rows are `review_status=unreviewed`; a six-image spot check (one per
  area) showed graffiti in every sampled image, but the full manual
  non-graffiti/privacy review is still pending.
- SHA-256 checksums for every original image are in
  `metadata/checksums.sha256`.

Artifacts (git-ignored): `outputs/audit/summary.json`,
`outputs/audit/images_audit.csv`, `outputs/audit/near_duplicate_groups.csv`.

## 3. Splits

Whole near-duplicate groups were assigned to splits, stratified by area, seed
42, target 70/15/15. No group and no image appears in more than one split.

| Split | Images | AZN | BRV | BUP | DMD | DUS | MUT |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 854 | 119 | 94 | 62 | 215 | 303 | 61 |
| val | 186 | 26 | 21 | 14 | 47 | 65 | 13 |
| test | 185 | 27 | 21 | 14 | 50 | 60 | 13 |

Group overlap between splits: **0 / 0 / 0**. The test split was not used for
feature selection, tuning, or model selection, and was evaluated once.

## 4. Features

| Extractor | Dimension | Description |
|---|---:|---|
| `raw` | 3,072 | 32×32 RGB pixels scaled to [0, 1] |
| `hog` | 1,988 | 96-d RGB histogram + 128-d HSV H-S histogram + OpenCV HOG |
| `resnet` | 512 | ResNet-18 ImageNet penultimate embedding, L2-normalised |

Configurations and library versions: `features/config.json`. Features are
cached per split in `features/<extractor>_<split>.npy` (ignored).

Feature sanity check (`scripts/visualize_features.py`,
`outputs/tsne-features.png`): PCA(50) + t-SNE shows heavy overlap between
areas. The silhouette score in the 50-d PCA space is **-0.014** for HOG and
**-0.015** for ResNet-18; HOG does show some local area clustering, ResNet is
nearly uniform. This is the main reason the absolute scores are modest, and it
also explains why the handcrafted features beat the ImageNet embeddings here.

## 5. Model selection (train + val only)

Pipeline: `StandardScaler → PCA(whiten=True) → KNeighborsClassifier`.
Grid: PCA 32/64/128/256 · k 1/3/5/7/9/15 · weights uniform/distance ·
metric euclidean/cosine. Selection metric: macro-F1 from grouped 5-fold CV on
train+val, with `StratifiedGroupKFold`.

| Extractor | Best params | CV macro-F1 | Val accuracy | Val macro-F1 |
|---|---|---:|---:|---:|
| raw | PCA 64, k=7, uniform, cosine | 0.4823 ± 0.0208 | 0.5161 | 0.4672 |
| **hog** | **PCA 256, k=9, distance, cosine** | **0.5170 ± 0.0369** | **0.5484** | **0.5325** |
| resnet | PCA 256, k=7, uniform, cosine | 0.4049 ± 0.0324 | 0.4086 | 0.3820 |

Validation scores above are from a model refitted on `train.csv` only (the CV
score uses train+val, as required by the tuning protocol). Full table:
`outputs/model-selection.md`, `outputs/model-selection.json`.

### Simple baselines (same HOG features, fitted on train, scored on val)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Dummy (most frequent) | 0.3495 | 0.0863 | 0.1810 |
| Logistic regression (balanced) | **0.6559** | **0.6169** | **0.6577** |
| Linear SVM (balanced) | 0.6075 | 0.5424 | 0.6051 |
| Best KNN (HOG) | 0.5484 | 0.5325 | 0.5272 |

The balanced logistic regression beats KNN on validation. KNN is therefore a
useful reproducible baseline, not the best model for this task.

## 6. Frozen test results

Best KNN: HOG features, PCA 256 (whitened), k=9, distance weights, cosine
metric, fitted on train+val (1,040 images). Test n = 185.

| Metric | Value |
|---|---:|
| Accuracy | **0.5892** |
| Macro-F1 | **0.5252** |
| Weighted-F1 | 0.5543 |
| Misclassified | 76 / 185 |

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AZN | 0.500 | 0.185 | 0.270 | 27 |
| BRV | 0.667 | 0.190 | 0.296 | 21 |
| BUP | 0.667 | 0.286 | 0.400 | 14 |
| DMD | 0.627 | 0.740 | 0.679 | 50 |
| DUS | 0.521 | 0.817 | 0.636 | 60 |
| MUT | 1.000 | 0.769 | 0.870 | 13 |

Confusion matrix (rows = true, columns = predicted):

| True ↓ / Pred → | AZN | BRV | BUP | DMD | DUS | MUT |
|---|---:|---:|---:|---:|---:|---:|
| AZN | 5 | 0 | 0 | 5 | 17 | 0 |
| BRV | 2 | 4 | 0 | 3 | 12 | 0 |
| BUP | 2 | 1 | 4 | 3 | 4 | 0 |
| DMD | 0 | 1 | 1 | 37 | 11 | 0 |
| DUS | 1 | 0 | 1 | 9 | 49 | 0 |
| MUT | 0 | 0 | 0 | 2 | 1 | 10 |

Artifacts: `outputs/evaluation.json`,
`outputs/classification-report.txt`, `outputs/confusion-matrix-test.png`,
`outputs/misclassified-test.png`, `outputs/test-predictions.csv`,
`outputs/training-performance.png`, `outputs/evaluation-performance.png`.

## 7. Limitations and KNN-specific issues

- **Curse of dimensionality.** HOG/raw distances become less meaningful as
  dimensions grow; PCA + whitening helps but does not remove the problem.
- **Imbalance.** DUS is 35% of the data. KNN has no class weighting, so it
  over-predicts DUS: 49/60 DUS test images are correct, but 17/27 AZN and 12/21
  BRV images are pulled into DUS. Macro-F1 (0.525) is much lower than accuracy
  (0.589) for this reason.
- **Rare classes.** BUP (14 test images) and MUT (13) have wide confidence
  intervals; MUT's 1.000 precision is based on 13 images and should not be
  over-read.
- **Prediction cost.** KNN stores the whole training set and scans it for every
  prediction (O(N·d)); the saved model is ~1,040 × 1,988 floats plus the PCA
  transform. This is fine at 1,225 images but does not scale to a large corpus.
- **Feature scaling.** Results depend on `StandardScaler` and whitening; the
  distance-weighted vote is sensitive to the whitened covariance estimate.
- **CNN embeddings underperformed here.** ImageNet ResNet-18 features are not
  graffiti-specific and gave the weakest macro-F1 (0.382). A domain-tuned or
  CLIP-style encoder may do better, but that is outside this baseline.
- **Near-duplicate grouping is conservative.** Groups require pHash ≤ 6 or CNN
  cosine ≥ 0.93. Different-angle photos of the same wall below that threshold
  could still cross splits and inflate scores. The true leakage is therefore
  unknown but bounded by 0 only for detected groups.
- **Label quality.** Area labels are folder-derived, not human-verified. The
  collection is uneven across areas and routes, so the task measures
  **collection area**, not graffiti prevalence or public opinion.

## 8. Reproduction

```bash
python -m pip install -r requirements.txt
python scripts/verify_checksums.py
python scripts/audit.py --checksums
python scripts/make_splits.py
python scripts/extract_features.py --extractor all
python scripts/train_knn.py
python scripts/evaluate.py
python scripts/plot_results.py
```

For a detailed setup and training guide for new users, see [`getstart.md`](getstart.md).
Visual performance charts (`outputs/training-performance.png` and
`outputs/evaluation-performance.png`) are automatically generated upon training and
evaluation, and can also be viewed or refreshed at any time with `python scripts/plot_results.py`.

`evaluate.py` refuses to overwrite an existing test evaluation unless
`--force` is passed. All reported numbers above come from the single frozen
run. Random seed is 42 throughout.

Environment: Python 3.14.5, scikit-learn 1.8.0, NumPy 2.4.4, PyTorch 2.12.1
(CPU), torchvision 0.27.1, OpenCV 4.13.0. Exact freeze:
`requirements.lock`; direct pins: `requirements.txt`.

Verify the image archive with the published checksums using
`python scripts/verify_checksums.py` (cross-platform), or
`sha256sum -c metadata/checksums.sha256` on Linux/macOS. The checksum file is
written with LF line endings so both methods work.

## 9. Privacy check

- No new images, coordinates, faces, plates, or contact details are published
  by this repository change; only aggregate metrics and file paths already in
  the inventory are reported.
- `latitude`/`longitude` remain blank in `metadata/images.csv`.
- A full face/plate/privacy review of all 1,225 images has **not** been done.
  Only a six-image spot check was performed. Before any public image release,
  complete that review and record the result in `review_status`.

## 10. What remains manual

The code paths in `TODO.md` are implemented. These items still need a human:

- full manual review of a sample per area and removal/flagging of non-graffiti
  images (`review_status=excluded`, originals untouched);
- filling `source`, `license`, `capture_date` where verifiable;
- full privacy review before publishing images;
- deciding the image-archive hosting policy (this repo uses the external
  archive + checksums option).
