# Network Anomaly Detection — CICIDS2017

The **detection engine** of a Network Intrusion Detection System (IDS), built using the **CICIDS2017** dataset. Implements both **signature-based** (supervised classification) and **anomaly-based** (unsupervised autoencoder + Isolation Forest) detection — the same two-pronged approach used by modern IDS/IPS platforms like Snort, Suricata, and Darktrace.

```
┌──────────────────────── Intrusion Detection System ────────────────────────┐
│                                                                           │
│   Data Capture          Detection Engine           Response & Alerting    │
│   ┌──────────┐          ┌──────────────┐           ┌─────────────┐        │
│   │ Sniffing │    →     │ This Project │     →     │  Logging    │        │
│   │ Flow     │          │              │           │  Blocking   │        │
│   │ Export   │          │ • Supervised │           │  Dashboard  │        │
│   │ Parsing  │          │ • Anomaly    │           │  SIEM       │        │
│   └──────────┘          └──────────────┘           └─────────────┘        │
│                                                                           │
│   (CICIDS2017 provides                                                    │
│    pre-extracted flows)                                                    │
└───────────────────────────────────────────────────────────────────────────┘
```

**Signature-based detection** (supervised models) identifies known attack patterns with 97-99% accuracy. **Anomaly-based detection** (autoencoder and Isolation Forest) catches novel/zero-day threats by learning what "normal" traffic looks like — the autoencoder achieves AUC 0.7801, outperforming Isolation Forest (AUC 0.7156) as the primary anomaly detector.

## Dataset

**CICIDS2017** — ~2.8M rows of labeled network traffic flows captured over 5 days (Mon–Fri), containing both benign and malicious traffic patterns.

| Split | Days | Rows | Attack Types |
|-------|------|------|--------------|
| Train | Mon–Thu | ~1.97M | BENIGN, DoS, Brute Force, Web Attack, Other Attacks |
| Test | Friday | ~580K | BENIGN, DDoS, PortScan, Bot |

> The temporal train/test split naturally creates a scenario where test attacks (DDoS, PortScan, Bot) are **unseen during training** — ideal for evaluating anomaly detection on novel threats.

## Project Structure

```
├── raw/                              # Raw CICIDS2017 CSV files
├── scalers/                          # Fitted scalers
│   ├── standardscaler.joblib         #   StandardScaler (supervised models)
│   └── minmaxscaler.joblib           #   MinMaxScaler (autoencoder)
├── models/                           # Trained models & evaluation outputs
│   ├── lightgbm/                     #   LightGBM classification reports & plots
│   ├── xgboost/                      #   XGBoost classification reports & plots
│   ├── ffnn/                         #   FFNN classification reports & plots
│   ├── autoencoder/                  #   Autoencoder reports, ROC curve, plots
│   ├── isolation_forest/             #   Isolation Forest reports, ROC curve, plots
│   ├── binary_svm/                   #   SVM classification reports
│   └── logistic_regression/          #   Logistic Regression reports
├── final/                            # Final preprocessed parquet files
├── split.ipynb                       # Stage 1: Data splitting
├── cleaning.ipynb                    # Stage 2: Data cleaning & EDA
├── preprocessing.ipynb               # Stage 3: Feature engineering & scaling
├── multiclass_lightgbm.py            # Signature-based: LightGBM (leaf-wise)
├── multiclass_xgboost.ipynb          # Signature-based: XGBoost (level-wise)
├── ffnn.py                           # Signature-based: Feed-Forward Neural Network
├── autoencoder.py                    # Anomaly-based: Denoising Autoencoder
├── isolation_forest.ipynb            # Anomaly-based: Isolation Forest
├── binary_svm.ipynb                  # Binary SVM classification (cuML GPU)
├── binary_logistic.ipynb             # Binary Logistic Regression (cuML GPU)
├── project_details.md                # Detailed project documentation & results
├── utils.py                          # Helper functions (NaN/Inf handling)
├── config.yaml                       # Project configuration (seed=42)
└── requirements.txt                  # Python dependencies
```

## Pipeline

### Stage 1 — Data Splitting (`split.ipynb`)
- Loads all raw CICIDS2017 CSV files and verifies column consistency
- Splits data temporally: **Mon–Thu → train**, **Friday → test**
- Strips whitespace from column names
- Consolidates rare attack types (Infiltration, Heartbleed) into "Other Attacks"
- Cleans Web Attack label encoding inconsistencies
- Maps granular labels to broader attack categories (e.g., DoS Hulk/GoldenEye/Slowloris → DoS)
- Exports `train_final.parquet` and `test_final.parquet`

### Stage 2 — Cleaning & EDA (`cleaning.ipynb`)
- Handles missing values (NaN → 0) and infinite values (Inf → 0) via `utils.handle_values()`
- Inspects data distributions and class balance

### Stage 3 — Preprocessing (`preprocessing.ipynb`)
- Drops redundant column (`Fwd Header Length.1`)
- Downcasts dtypes (float64 → float32, int64 → int32) for memory efficiency
- Removes constant-value columns (8 columns with single unique value)
- **Supervised path**: StandardScaler → IncrementalPCA (69 → 34 features, 98.97% variance retained)
- **Unsupervised path**: MinMaxScaler (all 69 features, no PCA)
  - Drops Friday-only attacks (Bot, DDoS, PortScan) from training data
  - Drops Mon-Thu-only attacks from test data
- Exports separate datasets for binary and multiclass classification tasks

## Supervised Models

### Multiclass LightGBM (`multiclass_lightgbm.py`)
- RandomizedSearchCV hyperparameter tuning with GPU acceleration
- **Test Accuracy: ~99%** | **F1-Score: ~99%**
- Generates classification report heatmaps

### Feed-Forward Neural Network (`ffnn.py`)
- PyTorch model: 2 hidden layers (128 → 64) with BatchNorm, ReLU, Dropout
- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping (patience=8)
- **Test Accuracy: ~98%** | **F1-Score: ~98%**

### Binary SVM (`binary_svm.ipynb`)
- cuML GPU-accelerated SVM with GridSearchCV
- Pipeline: StandardScaler → SVM
- **Test Accuracy: ~96.7%**

### Binary Logistic Regression (`binary_logistic.ipynb`)
- cuML GPU-accelerated Logistic Regression with GridSearchCV
- Pipeline: StandardScaler → Logistic Regression
- **Test Accuracy: ~96.7%**
### Multiclass XGBoost (`multiclass_xgboost.ipynb`)
- XGBoost multiclass classifier using **level-wise** tree growth (compared against LightGBM's leaf-wise)
- RandomizedSearchCV hyperparameter tuning with balanced sample weights
- **Test Accuracy: ~97%** | **Test F1 (weighted): ~98%**

## Unsupervised Models

### Denoising Autoencoder (`autoencoder.py`)
- Unsupervised anomaly detection — trained on **benign-only** traffic, detects novel attacks via high reconstruction error
- Denoising autoencoder: Encoder (69→128→64→32) / Decoder (32→64→128→69) with Sigmoid output
- Combined error metric: 0.5 × MSE + 0.5 × Max per-feature error
- **ROC AUC: 0.7801** — measures separation quality across all thresholds
- F1-optimal threshold catches **89% of all attacks** including unseen types (DDoS 83.1%, PortScan 96.6%, Bot 73.5%)

### Isolation Forest (`isolation_forest.ipynb`)
- Unsupervised anomaly detection — trained on **benign-only** traffic, detects anomalies by how quickly samples can be isolated via random splits
- Hyperparameter tuning via manual grid search (48 combos) — best: `max_samples=2048, max_features=1.0, n_estimators=200`
- **ROC AUC: 0.7156** — below the autoencoder (0.7801)
- F1-optimal threshold catches **91% of all attacks** (PortScan 99.3%, DDoS 84.0%, Bot 39.3%)
- Excels at detecting structurally distinct attacks (PortScan) but struggles with subtle ones (Bot)

### Unsupervised Comparison

| Metric | Autoencoder | Isolation Forest |
|--------|-------------|------------------|
| **ROC AUC** | **0.7801** | 0.7156 |
| F1 (Anomaly) | **0.66** | 0.63 |
| Bot Recall | **73.5%** | 39.3% |
| DDoS Recall | 83.1% | **84.0%** |
| PortScan Recall | 96.6% | **99.3%** |

The autoencoder is the preferred anomaly detector due to higher AUC and better detection of subtle attacks. See [project_details.md](project_details.md) for full analysis.

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
xgboost
matplotlib
seaborn
joblib
pyyaml
pyarrow
scipy
torch (PyTorch)
cuml (optional, for GPU-accelerated SVM/Logistic Regression)
```

## Usage

1. Place CICIDS2017 CSV files in `raw/`
2. Run notebooks in order: `split.ipynb` → `cleaning.ipynb` → `preprocessing.ipynb`
3. Train supervised models:
   ```bash
   python multiclass_lightgbm.py
   python ffnn.py
   ```
4. Run XGBoost notebook: `multiclass_xgboost.ipynb`
5. Run binary classifiers: `binary_svm.ipynb`, `binary_logistic.ipynb`
6. Train autoencoder:
   ```bash
   python autoencoder.py
   ```
7. Run Isolation Forest notebook: `isolation_forest.ipynb`
8. Check `models/` for evaluation outputs and plots

## What's Next

- [x] **Isolation Forest** — unsupervised anomaly detection for comparison with the autoencoder (AUC 0.7156 vs AE's 0.7801 — AE wins overall)
- [ ] **Hybrid Pipeline** — combine the autoencoder (best unsupervised model) with LightGBM (best supervised model) in a two-stage detection system

## License

For educational and research purposes.
