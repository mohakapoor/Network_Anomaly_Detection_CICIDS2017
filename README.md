# Network Anomaly Detection — CICIDS2017

A machine learning pipeline for detecting network intrusions and cyber attacks using the **CICIDS2017** dataset. Implements both **supervised** (multiclass classification) and **unsupervised** (autoencoder-based anomaly detection) approaches.

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
│   ├── ffnn/                         #   FFNN classification reports & plots
│   ├── binary_svm/                   #   SVM classification reports
│   └── logistic_regression/          #   Logistic Regression reports
├── final/                            # Final preprocessed parquet files
├── split.ipynb                       # Stage 1: Data splitting
├── cleaning.ipynb                    # Stage 2: Data cleaning & EDA
├── preprocessing.ipynb               # Stage 3: Feature engineering & scaling
├── multiclass_lightgbm.py            # Multiclass LightGBM training
├── ffnn.py                           # Feed-Forward Neural Network training
├── binary_svm.ipynb                  # Binary SVM classification (cuML GPU)
├── binary_logistic.ipynb             # Binary Logistic Regression (cuML GPU)
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

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
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
4. Or run binary classifiers: `binary_svm.ipynb`, `binary_logistic.ipynb`
5. Check `models/` for evaluation outputs

## What's Next

### Supervised — XGBoost Multiclass
- Train an **XGBoost** multiclass classifier to compare against LightGBM
- LightGBM uses **leaf-wise** tree growth while XGBoost uses **level-wise** growth — comparing both on the same data will show how these different gradient boosting strategies perform on network intrusion data

### Unsupervised — Autoencoder Anomaly Detection
An **unsupervised autoencoder** for semi-supervised anomaly detection:

- **Objective**: Train on benign-only traffic to learn the "normal" distribution, then detect anomalies via high reconstruction error — enabling detection of **novel/zero-day attacks** without labeled attack data
- **Architecture**: Encoder (69 → 128 → 64 → 32) / Decoder (32 → 64 → 128 → 69) with BatchNorm, ReLU, Dropout, and Sigmoid output
- **Scaling**: MinMaxScaler (already preprocessed) — bounds features to [0, 1] for the autoencoder
- **Threshold**: Reconstruction error threshold tuned on a benign validation set (`mean + k × std`)
- **Evaluation**: Per-attack-type recall, with special focus on unseen Friday attacks (DDoS, PortScan, Bot)
- **Data**: Training on `df[df['Attack'] == 0]` from the MinMaxScaled training set

### Unsupervised — Isolation Forest
- Train an **Isolation Forest** model as a comparison against the autoencoder
- Isolation Forest detects anomalies by isolating observations — anomalous points require fewer random splits to isolate, making it a fundamentally different approach from reconstruction-error-based detection

### Hybrid Approach (Future)
Combine the best unsupervised model (for novel anomaly detection) with supervised models (for known attack classification) into a two-stage pipeline.

## License

For educational and research purposes.