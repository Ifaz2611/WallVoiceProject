# TODO: Improve WallVoice24BD Classification Accuracy from 58.9% to 80%

## Current Performance Analysis

**Best Model:** KNN with HOG features
- Test Accuracy: 58.9%
- Macro-F1: 0.525
- Weighted-F1: 0.554

**Key Findings:**
- Logistic regression baseline already achieves 65.6% validation accuracy (significantly better than KNN)
- ResNet-18 ImageNet features underperform (40.9%) - poor transfer learning for wall graffiti
- Severe class imbalance issues:
  - AZN: 19% recall (27 samples)
  - BRV: 19% recall (21 samples)
  - BUP: 29% recall (14 samples)
  - DMD: 74% recall (50 samples)
  - DUS: 82% recall (60 samples)
  - MUT: 77% recall (13 samples)
- KNN is a simple, non-parametric method struggling with high-dimensional features

---

## Strategy 1: Switch to Better Classifiers (Highest Impact, Lowest Effort)

### 1.1 Random Forest with Class Weighting
- **Why:** Handles non-linear relationships, robust to overfitting with many trees
- **Implementation:**
  - Use `sklearn.ensemble.RandomForestClassifier`
  - Set `class_weight='balanced'` or use `class_weight={...}` with manual weights
  - Hyperparameters to tune: n_estimators (100-500), max_depth (10-30), min_samples_split (2-10)
  - Use existing HOG features (already perform best)
- **Expected gain:** 68-72% accuracy

### 1.2 XGBoost/LightGBM with Focal Loss
- **Why:** Gradient boosting often outperforms RF on tabular data; focal loss handles class imbalance
- **Implementation:**
  - Install: `pip install xgboost lightgbm`
  - Use `scale_pos_weight` parameter for imbalance
  - Try `XGBClassifier` with:
    - `objective='multi:softprob'`
    - `eval_metric='mlogloss'`
    - `scale_pos_weight` based on class ratios
  - Hyperparameters: max_depth (3-10), learning_rate (0.01-0.3), n_estimators (100-1000)
- **Expected gain:** 70-75% accuracy

### 1.3 SVM with RBF Kernel
- **Why:** Your linear SVM already achieves 60.8%; RBF kernel can capture non-linear patterns
- **Implementation:**
  - Use `sklearn.svm.SVC` with `kernel='rbf'`
  - Set `class_weight='balanced'`
  - Hyperparameters: C (0.1-100), gamma (scale/auto/0.001-1)
  - Use `probability=True` for ensemble combination
- **Expected gain:** 65-70% accuracy

### 1.4 Ensemble of Multiple Classifiers
- **Why:** Combining diverse models often improves performance
- **Implementation:**
  - Use `sklearn.ensemble.VotingClassifier` with:
    - Logistic Regression (balanced)
    - Random Forest (balanced)
    - SVM with RBF kernel
    - XGBoost
  - Try both 'hard' and 'soft' voting
  - Also try `StackingClassifier` with a meta-learner
- **Expected gain:** 72-78% accuracy

---

## Strategy 2: Fine-tune CNNs on Your Dataset (Medium Effort, High Impact)

### 2.1 Fine-tune ResNet-50
- **Why:** Your frozen ResNet-18 performs poorly (40.9%); fine-tuning adapts features to graffiti domain
- **Implementation:**
  - Use PyTorch (already installed)
  - Load pretrained ResNet-50 from torchvision
  - Replace final FC layer with 6-class classifier
  - Training strategy:
    - Stage 1: Freeze backbone, train only head for 10 epochs (lr=1e-3)
    - Stage 2: Unfreeze last 2 blocks, train for 20 epochs (lr=1e-4)
    - Stage 3: Unfreeze entire network, train for 30 epochs (lr=1e-5)
  - Use class weights in CrossEntropyLoss based on inverse frequency
  - Batch size: 32-64 (depending on GPU)
- **Expected gain:** 75-82% accuracy

### 2.2 Try EfficientNet-B0/B3
- **Why:** More efficient architecture than ResNet, better feature extraction
- **Implementation:**
  - Install: `pip install efficientnet-pytorch` or use torchvision
  - Same fine-tuning strategy as ResNet-50
  - EfficientNet-B0 is smaller/faster; B3 is more powerful
- **Expected gain:** 76-84% accuracy

### 2.3 Add Custom Classification Head
- **Why:** Standard ImageNet head may not be optimal for 6-class problem
- **Implementation:**
  - Replace FC layer with: Dropout(0.5) → Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, 6)
  - Add batch normalization layers
  - Consider adding attention mechanism
- **Expected gain:** +2-5% over base fine-tuning

---

## Strategy 3: Modern Vision Transformers (High Effort, Very High Impact)

### 3.1 CLIP Embeddings
- **Why:** CLIP captures semantic content, might understand graffiti meaning/context
- **Implementation:**
  - Install: `pip install openai-clip` or use HuggingFace transformers
  - Extract CLIP embeddings for all images
  - Train classifier on top of CLIP features
  - Try both ViT-B/32 and ViT-B/16 variants
- **Expected gain:** 78-85% accuracy

### 3.2 DINOv2 Features
- **Why:** Self-supervised vision transformer, no labels needed for pretraining
- **Implementation:**
  - Install: `pip install timm`
  - Use `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`
  - Extract features (384-d for ViT-S/16, 768-d for ViT-B/16)
  - Train classifier on DINOv2 features
- **Expected gain:** 75-82% accuracy

### 3.3 Vision Transformer (ViT) Fine-tuning
- **Why:** State-of-the-art for image classification
- **Implementation:**
  - Use HuggingFace `transformers` library
  - Load pretrained ViT-base-patch16-224
  - Fine-tune with same strategy as ResNet
  - Use gradient checkpointing to save memory
- **Expected gain:** 80-88% accuracy

---

## Strategy 4: Data Augmentation (Essential for Small Datasets)

### 4.1 Basic Augmentations
- **Why:** 1,225 images is small; augmentation effectively increases dataset size
- **Implementation:**
  - Install: `pip install albumentations`
  - Augmentations to use:
    - Random rotation (±30°)
    - Horizontal flip (p=0.5)
    - Color jitter (brightness, contrast, saturation ±20%)
    - Random Gaussian blur (sigma 0.3-1.0)
    - Normalize with ImageNet stats
  - Apply on-the-fly during training
- **Expected gain:** +3-8% accuracy

### 4.2 Mixup/CutMix Augmentation
- **Why:** Advanced augmentation techniques improve generalization
- **Implementation:**
  - Mixup: blend two images and their labels
  - CutMix: cut patch from one image and paste onto another
  - Use `timm.data.Mixup` or implement custom
  - Apply during training with alpha=0.2 for Mixup, beta=1.0 for CutMix
- **Expected gain:** +2-5% accuracy

### 4.3 Random Erasing
- **Why:** Forces model to learn from partial information, improves robustness
- **Implementation:**
  - Use `torchvision.transforms.RandomErasing`
  - Parameters: p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)
- **Expected gain:** +1-3% accuracy

### 4.4 Synthetic Oversampling (SMOTE)
- **Why:** Addresses class imbalance by generating synthetic minority samples
- **Implementation:**
  - Install: `pip install imbalanced-learn`
  - Apply SMOTE to HOG features in feature space
  - Use `SMOTE(sampling_strategy='auto', k_neighbors=5)`
  - Apply only to training split, not validation/test
- **Expected gain:** +2-4% accuracy (especially for minority classes)

---

## Strategy 5: Advanced Class Imbalance Handling

### 5.1 Focal Loss
- **Why:** Focuses training on hard, misclassified examples
- **Implementation:**
  - Custom PyTorch loss function
  - Formula: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
  - Parameters: α (class weights), γ (focusing parameter, typically 2.0)
  - Replace CrossEntropyLoss in CNN training
- **Expected gain:** +3-6% accuracy (especially for minority classes)

### 5.2 Class-Balanced Loss
- **Why:** Effective number of samples-based weighting for imbalance
- **Implementation:**
  - Use CB loss formula from "Class-Balanced Loss" paper
  - Compute effective number of samples: E_n = (1 - β^n) / (1 - β)
  - β typically 0.999 or 0.99
  - Combine with focal loss
- **Expected gain:** +2-5% accuracy

### 5.3 Oversampling Minority Classes in Batches
- **Why:** Ensures each batch sees minority class examples
- **Implementation:**
  - Custom PyTorch sampler
  - Use `WeightedRandomSampler` with class weights
  - Ensure each batch has balanced class distribution
  - Or use `ImbalancedDatasetSampler` from imbalanced-learn
- **Expected gain:** +2-4% accuracy

### 5.4 Two-Stage Training
- **Why:** First balance dataset, then fine-tune on real distribution
- **Implementation:**
  - Stage 1: Train with oversampled/SMOTE-balanced dataset
  - Stage 2: Fine-tune on original (imbalanced) dataset with lower learning rate
  - Transfer weights from Stage 1 to Stage 2
- **Expected gain:** +2-5% accuracy

---

## Strategy 6: Ensemble Multiple Extractors & Models

### 6.1 Combine HOG + Fine-tuned ResNet Features
- **Why:** Handcrafted features (HOG) and learned features (ResNet) capture different patterns
- **Implementation:**
  - Extract HOG features (already done)
  - Extract fine-tuned ResNet features (from fine-tuned model, penultimate layer)
  - Concatenate features: [HOG_dim + ResNet_dim]
  - Train classifier on combined features
  - Or train separate classifiers and ensemble predictions
- **Expected gain:** +3-7% accuracy

### 6.2 Stack Multiple Classifiers
- **Why:** Meta-learner can learn which classifier to trust for each example
- **Implementation:**
  - Use `sklearn.ensemble.StackingClassifier`
  - Base learners: Logistic Regression, Random Forest, SVM, XGBoost
  - Meta-learner: Logistic Regression or XGBoost
  - Use cross-validation to generate meta-features
- **Expected gain:** +4-8% accuracy

### 6.3 Geometric Mean of Prediction Probabilities
- **Why:** Reduces overconfident predictions, improves calibration
- **Implementation:**
  - Get prediction probabilities from each model
  - Compute geometric mean: (∏ p_i)^(1/n)
  - Use final probabilities for prediction
  - Often outperforms arithmetic mean for classification
- **Expected gain:** +1-3% accuracy

---

## Strategy 7: Domain-Specific Feature Engineering

### 7.1 Text Density Features
- **Why:** Graffiti often contains text; density might be location-specific
- **Implementation:**
  - Use OCR to detect text regions (Tesseract or easyocr)
  - Compute: text coverage ratio, number of text regions, average text size
  - Add as additional features to HOG/ResNet
- **Expected gain:** +1-3% accuracy

### 7.2 Color Palette Histograms
- **Why:** Different areas may have characteristic color palettes
- **Implementation:**
  - Convert to LAB color space (perceptually uniform)
  - Compute histogram with more bins (e.g., 64 bins per channel)
  - Compute color moments (mean, std, skewness per channel)
  - Add to feature vector
- **Expected gain:** +1-2% accuracy

### 7.3 Edge Density Features
- **Why:** Urban vs suburban graffiti patterns differ in edge density
- **Implementation:**
  - Use Canny edge detector
  - Compute: edge density, edge orientation histogram
  - Add to HOG features (or combine with existing edge-based features)
- **Expected gain:** +1-2% accuracy

### 7.4 Texture Features (LBP, Gabor)
- **Why:** Wall texture might vary by location
- **Implementation:**
  - Local Binary Patterns (LBP) with skimage
  - Gabor filter responses at multiple orientations/scales
  - Compute texture statistics (contrast, homogeneity, energy)
  - Add to feature vector
- **Expected gain:** +1-3% accuracy

---

## Strategy 8: Expand Hyperparameter Search

### 8.1 Bayesian Optimization with Optuna
- **Why:** More efficient than grid search; explores continuous parameters
- **Implementation:**
  - Install: `pip install optuna`
  - Define search space:
    - PCA components: 16-512 (continuous)
    - KNN k: 1-50 (integer)
    - SVM C: 0.001-1000 (log-uniform)
    - RF max_depth: 5-50 (integer)
    - XGBoost learning_rate: 0.001-0.3 (log-uniform)
  - Use TPE sampler or CMA-ES
  - Run 100-500 trials
- **Expected gain:** +2-5% accuracy

### 8.2 Expand PCA Grid
- **Why:** Current grid (32, 64, 128, 256) may miss optimal dimensionality
- **Implementation:**
  - Try: 16, 32, 64, 96, 128, 192, 256, 384, 512
  - Use explained variance ratio to determine optimal
  - Consider using incremental PCA for large dimensions
- **Expected gain:** +1-3% accuracy

### 8.3 Expand KNN K Values
- **Why:** Current grid (1, 3, 5, 7, 9, 15) limited
- **Implementation:**
  - Try: 1, 3, 5, 7, 9, 11, 13, 15, 21, 31, 51
  - Use cross-validation to find optimal k
- **Expected gain:** +1-2% accuracy

---

## Implementation Priority Roadmap

### Phase 1: Quick Wins (1-2 days) - Target: 68-72%
- [ ] Implement Random Forest with class weighting on HOG features
- [ ] Implement XGBoost with scale_pos_weight on HOG features
- [ ] Implement SVM with RBF kernel on HOG features
- [ ] Test all three and select best
- [ ] Compare with logistic regression baseline (65.6%)

### Phase 2: Fine-tuning (3-5 days) - Target: 75-82%
- [ ] Implement data augmentation with Albumentations
- [ ] Fine-tune ResNet-50 (3-stage training strategy)
- [ ] Fine-tune EfficientNet-B0 as alternative
- [ ] Add custom classification head with dropout
- [ ] Combine fine-tuned CNN features with HOG features
- [ ] Train classifier on combined features

### Phase 3: Ensemble Methods (2-3 days) - Target: 78-85%
- [ ] Implement VotingClassifier with best models from Phase 1-2
- [ ] Implement StackingClassifier with meta-learner
- [ ] Try geometric mean of prediction probabilities
- [ ] Optimize ensemble weights

### Phase 4: Advanced Techniques (1-2 weeks) - Target: 80-88%
- [ ] Implement focal loss for CNN training
- [ ] Implement class-balanced loss
- [ ] Try CLIP embeddings
- [ ] Try DINOv2 features
- [ ] Implement Bayesian optimization with Optuna
- [ ] Add domain-specific features (text density, color palettes)

### Phase 5: Final Optimization (3-5 days) - Target: 80%+
- [ ] Analyze misclassified examples across all models
- [ ] Implement targeted improvements based on error analysis
- [ ] Final ensemble with best-performing models
- [ ] Generate comprehensive evaluation report
- [ ] Document methodology for research paper

---

## Implementation Checklist

### Environment Setup
- [ ] Install additional packages: `xgboost lightgbm albumentations imbalanced-learn optuna timm transformers`
- [ ] Verify GPU availability for CNN training (optional but recommended)
- [ ] Set up experiment tracking (e.g., Weights & Biases or TensorBoard)

### Code Structure
- [ ] Create `scripts/train_rf.py` for Random Forest
- [ ] Create `scripts/train_xgboost.py` for XGBoost
- [ ] Create `scripts/train_svm_rbf.py` for SVM with RBF
- [ ] Create `scripts/train_ensemble.py` for voting/stacking
- [ ] Create `scripts/finetune_resnet.py` for ResNet fine-tuning
- [ ] Create `scripts/extract_clip_features.py` for CLIP embeddings
- [ ] Create `scripts/extract_dinov2_features.py` for DINOv2 features
- [ ] Create `scripts/hyperopt.py` for Bayesian optimization

### Data Preparation
- [ ] Apply data augmentation to training images
- [ ] Implement balanced batch sampler
- [ ] Create augmented dataset directory (optional, or on-the-fly)

### Evaluation
- [ ] Implement cross-validation for all new models
- [ ] Generate per-class metrics for all models
- [ ] Create confusion matrices for all models
- [ ] Track training/validation curves
- [ ] Compare all models in unified table

### Documentation
- [ ] Document all hyperparameters used
- [ ] Document training procedures
- [ ] Save all model checkpoints
- [ ] Create final model comparison report
- [ ] Prepare results for research paper

---

## Notes

- Always use the frozen test split for final evaluation only
- Use validation split for model selection and hyperparameter tuning
- Keep track of random seeds for reproducibility
- Consider using early stopping to prevent overfitting
- Monitor for overfitting: if training accuracy >> validation accuracy, add regularization
- For research paper, document all failed attempts as well (negative results are valuable)
- Consider data leakage: ensure no duplicate images span train/val/test splits (already handled by group_id)
- If GPU is not available, focus on Phase 1 (classifiers on HOG features) as they run on CPU
