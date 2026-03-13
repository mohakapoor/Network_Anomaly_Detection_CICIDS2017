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
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | Patience = 10 |

**Results:**

| Metric | Value |
|--------|-------|
| **ROC AUC** | **0.7858** |
| Best F1 (Anomaly class) | **0.66** |
| F1-Optimal Threshold | 0.000976 |

**Per-Attack Recall (F1-Optimal Threshold):**

| Attack | Recall | Count |
|--------|--------|-------|
| DDoS | **83.6%** | 98,022 |
| PortScan | **98.7%** | 79,290 |
| Bot | **73.5%** | 981 |
| BENIGN (correctly identified) | 61.6% | 395,106 |

**Threshold Comparison:**

| Threshold | Anomaly Recall | Benign Correct | Anomaly F1 |
|-----------|---------------|----------------|------------|
| mean + 3×std | 32% | 93% | 0.43 |
| mean + 1×std | 33% | 94% | 0.44 |
| 95th percentile | 39% | 90% | 0.48 |
| **F1-Optimal** | **90%** | **62%** | **0.66** |

**Key findings:**
1. The autoencoder achieves AUC of 0.7858, demonstrating learned separation between benign and attack traffic without any labeled attack data
2. At the F1-optimal threshold, it catches 90% of attacks (including novel/unseen types) with a tradeoff of 38% false positive rate on benign traffic
3. PortScan (98.7%) and DDoS (83.6%) are detected very effectively; Bot (73.5%) moderately
4. The conservative threshold (k=3) catches only 32% of attacks — threshold selection is critical
5. The combined error metric (MSE + max per-feature) improved detection of subtle attacks (PortScan, Bot) by catching individual feature deviations

---

## Observations & Insights

### LightGBM vs XGBoost
- LightGBM (leaf-wise) slightly outperforms XGBoost (level-wise) on this dataset: 99% vs 97% test accuracy
- Both models converge with very few trees (5-10 estimators), suggesting strong signal in the features
- LightGBM's leaf-wise growth is more efficient on this imbalanced dataset as it can focus splits on the most informative regions

### Supervised vs Unsupervised
- Supervised models (LightGBM, XGBoost, FFNN) significantly outperform the autoencoder on known attack types (99%+ recall vs 74-99%)
- The autoencoder's advantage: it detects attacks **without ever seeing attack labels during training** — critical for zero-day attack detection
- The unsupervised approach is complementary, not competitive — ideal for a hybrid pipeline

### Autoencoder Threshold Sensitivity
- The choice of threshold dramatically affects performance (32% vs 90% attack recall)
- ROC AUC (0.7858) is the most reliable metric as it captures performance across all thresholds
- For network security, high attack recall (low missed attacks) is preferred even at the cost of more false positives

---

## What's Next

- [ ] **Isolation Forest** — unsupervised anomaly detection for comparison with the autoencoder
- [ ] **Hybrid Pipeline** — combine the best unsupervised model (novel attack detection) with supervised models (known attack classification) in a two-stage system
