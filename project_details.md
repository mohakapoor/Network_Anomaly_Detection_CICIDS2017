# Project Details — Network Anomaly Detection (CICIDS2017)

## Dataset Overview

**CICIDS2017** — ~2.8M labeled network traffic flows captured over 5 days (Mon–Fri).

| Property | Value |
|----------|-------|
| Total Rows | ~2,556,452 |
| Train Rows | ~1,975,710 (Mon–Thu) |
| Test Rows | ~580,742 (Friday) |
| Features | 69 (after dropping redundant/constant columns) |
| Split Strategy | Temporal — Mon–Thu train, Friday test |

### Attack Type Mapping

| ID | Attack Type | Present In |
|----|-------------|------------|
| 0 | BENIGN | Train + Test |
| 1 | Bot | Test (Friday) |
| 2 | Brute Force | Train (Mon–Thu) |
| 3 | DDoS | Test (Friday) |
| 4 | DoS | Train (Mon–Thu) |
| 5 | PortScan | Test (Friday) |
| 6 | Web Attack | Train (Mon–Thu) |

### Train Class Distribution

| Class | Count |
|-------|-------|
| BENIGN (0) | 1,720,966 (87.1%) |
| DoS (4) | 189,135 |
| Brute Force (2) | 29,999 |
| Web Attack (6) | 25,501 |
| PortScan (5) | 7,417 |
| Other (misc) | 2,692 |

---

## Data Pipeline

### 1. Splitting (`split.ipynb`)
- Loads raw CICIDS2017 CSVs, strips whitespace from column names
- Temporal split: Mon–Thu → train, Friday → test
- Consolidates rare attacks (Infiltration, Heartbleed) into "Other Attacks"
- Maps granular labels to 7 broader categories
- Exports `train_final.parquet`, `test_final.parquet`

### 2. Cleaning (`cleaning.ipynb`)
- Replaces NaN → 0, Inf → 0 via `utils.handle_values()`
- EDA: distribution analysis, class balance inspection

### 3. Preprocessing (`preprocessing.ipynb`)
- Drops redundant column (`Fwd Header Length.1`)
- Downcasts dtypes: float64 → float32, int64 → int32
- Drops 8 constant-value columns (Bwd PSH/URG Flags, 6 bulk-rate columns)

**Supervised path:**
- StandardScaler → IncrementalPCA (69 → 34 features, 98.97% variance retained)
- Random sampling to balance classes for multiclass models
- Exports `final/train_mc.parquet`, `final/test_mc.parquet`

**Unsupervised path:**
- MinMaxScaler (all 69 features, no PCA)
- Drops Friday-only attacks (Bot, DDoS, PortScan) from train
- Drops Mon-Thu-only attacks from test
- Exports `final/train_us.parquet`, `final/test_us.parquet`

---

## Supervised Models

### Multiclass LightGBM (`multiclass_lightgbm.py`)

| Property | Value |
|----------|-------|
| Tree Growth | **Leaf-wise** |
| Tuning | RandomizedSearchCV (20 iters, 5-fold StratifiedKFold) |
| Scoring | F1 macro |
| GPU | Yes |
| Class Weight | Balanced |

**Hyperparameter Search Space:**

| Param | Values |
|-------|--------|
| learning_rate | loguniform(0.01, 0.2) |
| n_estimators | [5, 10] |
| num_leaves | [20, 31, 50] |
| min_child_samples | [10, 30, 50] |
| max_depth | [6, 8, 10] |

**Results:**

|  | Train | Test |
|--|-------|------|
| Accuracy | 0.99 | **0.99** |
| F1 (weighted) | 0.99 | **0.99** |
| F1 (macro) | 0.99 | 0.98 |

Per-class test performance:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| BENIGN (0) | 0.98 | 0.96 | 0.97 |
| Bot (1) | 0.88 | 0.99 | 0.93 |
| Brute Force (2) | 0.99 | 0.99 | 0.99 |
| DDoS (3) | 1.00 | 1.00 | 1.00 |
| DoS (4) | 0.99 | 0.98 | 0.99 |
| PortScan (5) | 1.00 | 1.00 | 1.00 |
| Web Attack (6) | 0.95 | 0.99 | 0.97 |

---

### Multiclass XGBoost (`multiclass_xgboost.ipynb`)

| Property | Value |
|----------|-------|
| Tree Growth | **Level-wise** |
| Tuning | RandomizedSearchCV (20 iters, 5-fold StratifiedKFold) |
| Scoring | F1 macro |
| Class Weight | Balanced (via sample_weight) |

**Hyperparameter Search Space:**

| Param | Values |
|-------|--------|
| learning_rate | loguniform(0.01, 0.2) |
| n_estimators | [5, 10] |
| max_depth | [6, 8, 10] |
| min_child_weight | [10, 30, 50] |

**Results:**

|  | Train | Test |
|--|-------|------|
| Accuracy | 0.98 | **0.97** |
| F1 (weighted) | 0.98 | **0.98** |
| F1 (macro) | 0.98 | 0.96 |

Per-class test performance:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| BENIGN (0) | 0.97 | 0.93 | 0.95 |
| Bot (1) | 0.83 | 0.98 | 0.90 |
| Brute Force (2) | 0.98 | 0.99 | 0.99 |
| DDoS (3) | 0.99 | 1.00 | 0.99 |
| DoS (4) | 0.98 | 0.97 | 0.97 |
| PortScan (5) | 1.00 | 0.99 | 1.00 |
| Web Attack (6) | 0.90 | 0.99 | 0.94 |

---

### Feed-Forward Neural Network (`ffnn.py`)

| Property | Value |
|----------|-------|
| Framework | PyTorch |
| Architecture | 2 hidden layers (128 → 64) |
| Activations | BatchNorm → ReLU → Dropout(0.2) |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | Patience = 8 |
| Batch Size | 256 |

**Results:**

|  | Train | Test |
|--|-------|------|
| Accuracy | 0.98 | **0.98** |
| F1 (weighted) | 0.98 | **0.98** |
| F1 (macro) | 0.97 | 0.96 |

Per-class test performance:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| BENIGN (0) | 0.99 | 0.92 | 0.95 |
| Bot (1) | 0.86 | 0.98 | 0.92 |
| Brute Force (2) | 0.97 | 1.00 | 0.98 |
| DDoS (3) | 0.98 | 1.00 | 0.99 |
| DoS (4) | 0.97 | 1.00 | 0.98 |
| PortScan (5) | 1.00 | 1.00 | 1.00 |
| Web Attack (6) | 0.93 | 0.92 | 0.92 |

---

### Binary SVM (`binary_svm.ipynb`)

| Property | Value |
|----------|-------|
| Framework | cuML (GPU-accelerated) |
| Pipeline | StandardScaler → SVM |
| Tuning | GridSearchCV |
| Test Accuracy | **~96.7%** |

---

### Binary Logistic Regression (`binary_logistic.ipynb`)

| Property | Value |
|----------|-------|
| Framework | cuML (GPU-accelerated) |
| Pipeline | StandardScaler → Logistic Regression |
| Tuning | GridSearchCV |
| Test Accuracy | **~96.7%** |

---

## Supervised Models Comparison

| Model | Test Accuracy | Test F1 (weighted) | Test F1 (macro) |
|-------|--------------|--------------------|-----------------| 
| **LightGBM** | **0.99** | **0.99** | **0.98** |
| FFNN | 0.98 | 0.98 | 0.96 |
| XGBoost | 0.97 | 0.98 | 0.96 |
| SVM (binary) | ~0.97 | — | — |
| Logistic Regression (binary) | ~0.97 | — | — |

**Key takeaway**: LightGBM's leaf-wise tree growth slightly edges out XGBoost's level-wise growth (99% vs 97% accuracy). Both gradient boosting models outperform the neural network, likely because tabular network flow data has clear decision boundaries that trees exploit well.

---

## Unsupervised Models

### Denoising Autoencoder (`autoencoder.py`)

| Property | Value |
|----------|-------|
| Framework | PyTorch |
| Type | Denoising Autoencoder |
| Architecture | Encoder (69→128→64→32) / Decoder (32→64→128→69) |
| Output Activation | Sigmoid (matches MinMaxScaler [0,1] range) |
| Training Data | Benign-only (Attack == 0) |
| Validation | 10% benign holdout |
| Error Metric | Combined: 0.5 × MSE + 0.5 × Max per-feature error |
| Noise | Gaussian noise (σ=0.1) added during training, model learns to reconstruct clean input |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | Patience = 10 |

**Results:**

| Metric | Value |
|--------|-------|
| **ROC AUC** | **0.7801** |
| Best F1 (Anomaly class) | **0.66** |
| F1-Optimal Threshold | 0.001068 |

**Conservative Threshold (mean + 3×std):**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Benign | 0.75 | 0.93 | 0.83 | 395,106 |
| Anomaly | 0.67 | 0.31 | 0.42 | 178,293 |
| **Accuracy** | | | **0.74** | 573,399 |

Per-attack recall (mean + 3×std):

| Attack | Recall | Count |
|--------|--------|-------|
| BENIGN (correctly identified) | 93.2% | 395,106 |
| Bot | 8.2% | 981 |
| DDoS | 52.9% | 98,022 |
| PortScan | 3.8% | 79,290 |

**F1-Optimal Threshold (0.001068):**

|  | Precision | Recall | F1 | Support |
|--|-----------|--------|-----|---------|
| Benign | 0.93 | 0.63 | 0.75 | 395,106 |
| Anomaly | 0.52 | 0.89 | 0.66 | 178,293 |
| **Accuracy** | | | **0.71** | 573,399 |

Per-attack recall (F1-optimal):

| Attack | Recall | Count |
|--------|--------|-------|
| Bot | **73.5%** | 981 |
| DDoS | **83.1%** | 98,022 |
| PortScan | **96.6%** | 79,290 |

**Key findings:**
1. The autoencoder achieves AUC of 0.7801, demonstrating learned separation between benign and attack traffic **without any labeled attack data**
2. At the F1-optimal threshold, it catches **89% of all attacks** (including novel/unseen types) with a tradeoff of 37% false positive rate on benign traffic
3. PortScan (96.6%) and DDoS (83.1%) are detected very effectively; Bot (73.5%) moderately
4. The conservative threshold (mean + 3×std) catches only 31% of attacks — **threshold selection is critical**
5. The combined error metric (MSE + max per-feature) improved detection of subtle attacks (PortScan, Bot) over pure MSE by catching individual feature deviations
6. Denoising approach (adding Gaussian noise during training) forces the model to learn structural benign patterns rather than memorizing values

---

## Observations & Insights

### LightGBM vs XGBoost
- LightGBM (leaf-wise) slightly outperforms XGBoost (level-wise) on this dataset: 99% vs 97% test accuracy
- Both models converge with very few trees (5-10 estimators), suggesting strong signal in the features
- LightGBM's leaf-wise growth is more efficient on this imbalanced dataset as it can focus splits on the most informative regions

### Supervised vs Unsupervised
- Supervised models (LightGBM, XGBoost, FFNN) significantly outperform the autoencoder on known attack types (99%+ recall vs 74-97%)
- The autoencoder's advantage: it detects attacks **without ever seeing attack labels during training** — critical for zero-day attack detection
- The unsupervised approach is complementary, not competitive — ideal for a hybrid pipeline

### Autoencoder Threshold Sensitivity
- The choice of threshold dramatically affects performance (31% vs 89% attack recall)
- ROC AUC (0.7801) is the most reliable metric as it captures performance across all thresholds
- For network security, high attack recall (low missed attacks) is preferred even at the cost of more false positives

### Why ROC AUC & F1-Optimal Threshold?

**The Problem**: Unlike supervised classifiers that output class predictions directly, the autoencoder outputs a continuous **reconstruction error** for each sample. To make a binary decision (benign vs anomaly), we must pick a threshold — but any fixed threshold strategy (like mean + k×std) is arbitrary and gives wildly different results depending on k:

| Threshold Strategy | Attack Recall | Benign Correct |
|-------------------|---------------|----------------|
| mean + 3×std (conservative) | 31% | 93% |
| mean + 1×std | 33% | 94% |
| 95th percentile | 39% | 90% |

No single fixed threshold tells the full story. Reporting just one gives a misleading picture of model quality.

**Why ROC AUC**: AUC (Area Under the ROC Curve) evaluates the model's **ability to separate benign from attack traffic across all possible thresholds**. An AUC of 0.7801 means: if you randomly pick one attack sample and one benign sample, the model assigns a higher reconstruction error to the attack sample 78% of the time. This is a threshold-independent measure of how well the model has learned the distinction — making it the most honest single metric for comparing anomaly detection models (e.g., autoencoder vs Isolation Forest).

**Why F1-Optimal Threshold**: Once we know the model has learned real separation (via AUC), we need to pick an **operating point** — a specific threshold for deployment. The F1-optimal threshold is the point that maximizes the harmonic mean of precision and recall:
- **Too aggressive** (low threshold): catches more attacks but floods analysts with false alarms — low precision
- **Too conservative** (high threshold): few false alarms but misses most attacks — low recall
- **F1-optimal**: the mathematically best **balance** between the two — neither too aggressive nor too conservative

At the F1-optimal threshold (0.001068), the autoencoder catches **89% of all attacks** with a precision of 52%. In a network security context, this means roughly half of flagged traffic is actually malicious — a workable ratio for a first-pass filter that feeds into human review or a supervised model for final classification.


- **Vanilla AE (MSE only)**: DDoS ~44%, PortScan ~0.6%, Bot ~2.5% at k=3
- **+ Denoising**: DDoS improved to ~53%, learned more robust benign representations
- **+ Combined Error (MSE + Max)**: PortScan jumped from ~1.5% to ~3.8% at k=3 by catching individual feature deviations
- **+ F1-Optimal Threshold**: All attacks dramatically improved — PortScan 96.6%, DDoS 83.1%, Bot 73.5%

---

## What's Next

- [ ] **Isolation Forest** — unsupervised anomaly detection for comparison with the autoencoder
- [ ] **Hybrid Pipeline** — combine the best unsupervised model (novel attack detection) with supervised models (known attack classification) in a two-stage system

