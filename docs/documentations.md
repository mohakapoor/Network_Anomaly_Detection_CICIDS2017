# Intrusion Tracker: Full System Documentation
## Technical Specifications for Network Anomaly Detection (CICIDS2017)

```
┌──────────────────────── Intrusion Detection System ────────────────────────┐
│                                                                            │
│   Data Capture          Detection Engine           Response & Alerting     │
│   ┌──────────┐          ┌──────────────┐           ┌─────────────┐         │
│   │ Sniffing │    →     │ This Project │     →     │  Logging    │         │
│   │ Flow     │          │              │           │  Blocking   │         │
│   │ Export   │          │ • Supervised │           │  Dashboard  │         │
│   │ Parsing  │          │ • Anomaly    │           │  SIEM       │         │
│   └──────────┘          └──────────────┘           └─────────────┘         │
│                                                                            │
│   (CICIDS2017 provides                                                     │
│    pre-extracted flows)                                                    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Project Overview
The **Intrusion Tracker** is an advanced Network Intrusion Detection System (NIDS) developed to integrate academic machine learning research with production-grade security infrastructure. The system utilizes high-fidelity supervised classification for known signatures and unsupervised anomaly detection for zero-day threat identification, providing a comprehensive defense-in-depth architecture.

A core objective of this project is to democratize high-performance network security by ensuring that advanced detection capabilities remain accessible and deployable on resource-constrained hardware. By prioritizing architectural efficiency, the system facilitates deployment in diverse operational environments, ranging from centralized servers to decentralized edge nodes.

---

## 2. Hybrid Detection Architecture
Conventional IDS implementations typically rely on one of two paradigms, both of which possess inherent limitations:
1.  **Signature-based Detection**: High precision for known attacks but incapable of identifying zero-day threats.
2.  **Anomaly-based Detection**: Capable of detecting novel attacks but prone to high False Positive Rates (FPR).

**The Hierarchical Solution**: This system implements a **Hierarchical Hybrid Pipeline** designed to maximize detection coverage while minimizing operational noise.
-   **Detection Phase**: Sensitive unsupervised models (**Denoising Autoencoder** and **Isolation Forest**) monitor for statistical deviations.
-   **Validation Phase**: Flagged anomalies are subjected to a high-precision supervised model (**LightGBM**). If the supervised layer classifies the traffic as benign, the alert is suppressed via a "Veto" mechanism.

---

## 3. Dataset Analysis: CICIDS2017
The architecture is trained on the **CICIDS2017** dataset, an industry-standard collection of network flows spanning five days of activity.

### 3.1 Temporal Splitting Strategy
A **Temporal Split** is utilized to simulate realistic operational scenarios, avoiding the data leakage common in random shuffling:
-   **Training Dataset (Monday–Thursday)**: Establishes the baseline for benign activity and provides signatures for DoS and Brute Force attacks.
-   **Testing Dataset (Friday)**: Evaluates the system against **Unseen Zero-Day Attacks**, including Botnets, DDoS, and PortScans.

### 3.2 Attack Category Mapping
| ID | Category | Training Distribution | Test Distribution |
| :--- | :--- | :--- | :--- |
| 0 | **BENIGN** | 1,720,966 (87.1%) | 395,106 |
| 1 | **Bot** | - | 981 (Friday only) |
| 2 | **Brute Force** | 29,999 | - |
| 3 | **DDoS** | - | 98,022 (Friday only) |
| 4 | **DoS** | 189,135 | - |
| 5 | **PortScan** | 7,417 | 79,290 |
| 6 | **Web Attack** | 25,501 | - |

---

## 4. Analytical Pipeline

### Stage 1: Data Consolidation (`split.ipynb`)
-   Raw CSV files are merged, and column headers are sanitized to remove whitespace.
-   Rare attack labels (e.g., Infiltration) are consolidated into a standardized "Other Attacks" category.
-   Data is partitioned into `train_final.parquet` and `test_final.parquet`.

### Stage 2: Numerical Sanitization (`cleaning.ipynb`)
-   Infinite and missing values are resolved using the `utils.handle_values()` utility, ensuring numerical stability for gradient descent and tree-partitioning algorithms.

### Stage 3: Feature Engineering (`preprocessing.ipynb`)
The system maintains dual feature pipelines optimized for specific model requirements:
1.  **Supervised Path (Dimensionality Reduction)**:
    -   **Scaling**: `StandardScaler`.
    -   **Transformation**: `IncrementalPCA` (reducing 69 features to 34 components while retaining 98.97% variance).
2.  **Unsupervised Path (Feature Preservation)**:
    -   **Scaling**: `MinMaxScaler`.
    -   **Objective**: Maintains the original feature manifold to allow the Autoencoder to detect high-dimensional reconstruction deviations.

---

## 5. Machine Learning Model Specifications

### 5.1 Supervised Layer (Signature Identification)
All supervised models are trained on the 34 PCA-transformed components.

#### **Detailed Performance Matrix**
| Model | Accuracy | F1 (Weighted) | F1 (Macro) |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **0.99** | **0.99** | **0.98** |
| **FFNN** | 0.98 | 0.98 | 0.96 |
| **XGBoost** | 0.97 | 0.98 | 0.96 |
| **Binary SVM** | 96.7% | - | - |

#### **Per-Class Recall Analysis (LightGBM)**
| Category | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| BENIGN | 0.98 | 0.96 | 0.97 |
| Bot | 0.88 | 0.99 | 0.93 |
| Brute Force | 0.99 | 0.99 | 0.99 |
| DDoS | 1.00 | 1.00 | 1.00 |
| DoS | 0.99 | 0.98 | 0.99 |
| PortScan | 1.00 | 1.00 | 1.00 |
| Web Attack | 0.95 | 0.99 | 0.97 |

---

### 5.2 Unsupervised Layer (Anomaly Detection)
Trained exclusively on benign Monday traffic to establish a baseline for "Normal" network behavior.

#### **Denoising Autoencoder Evolution**
The anomaly detection engine underwent three primary iterations to achieve optimal performance:
1.  **Baseline (MSE Only)**: Detected high-volume attacks (DDoS) but struggled with subtle patterns (Bot ~2.5% recall).
2.  **Denoising Integration**: Added 10% Gaussian noise during training to force the model to learn structural benign manifolds rather than memorizing values.
3.  **Hybrid Error Metric**: Implemented a **0.5 × MSE + 0.5 × Max Per-Feature Error** loss function. This modification significantly improved PortScan detection by identifying extreme deviations in individual features (e.g., specific flag counts).

#### **Comparative Unsupervised Results**
| Metric | Autoencoder | Isolation Forest | Winner |
| :--- | :--- | :--- | :--- |
| **ROC AUC** | **0.7801** | 0.7156 | **AE** |
| **F1 (Anomaly)** | **0.66** | 0.63 | **AE** |
| Bot Recall | **73.5%** | 39.3% | **AE** |
| PortScan Recall | 96.6% | **99.3%** | **IF** |

#### **Statistical Metrics: ROC AUC vs. Fixed Thresholds**
Unlike supervised classifiers, the anomaly engine outputs a continuous reconstruction error. Fixed threshold strategies (e.g., mean + 3×std) often provide misleading performance snapshots:
-   **Why ROC AUC?**: It evaluates the model's ability to separate benign from attack traffic across *all possible thresholds*. An AUC of 0.7801 indicates that the model assigns higher error to attacks than to benign samples 78% of the time.
-   **Threshold Sensitivity**: A conservative threshold (mean + 3×std) catches only 31% of attacks. Conversely, the **F1-optimal threshold** (0.001068) catches **89% of attacks**, albeit with a 37% false positive rate—making it a workable first-pass filter for the hybrid pipeline.

---

## 6. Explainability and SHAP Analysis
SHAP analysis was conducted to ensure institutional-grade transparency and to identify the mathematical drivers behind model decisions.

### 6.1 Feature Importance Results
The analysis identified specific network indicators that serve as primary drivers for intrusion detection:

| Model Category | Primary SHAP Features / Components | Detection Indicator |
| :--- | :--- | :--- |
| **Tree Models** | **PC3**, **PC1** | Timing (IAT) and Idle Time variances. |
| **Neural Networks**| **PC1**, **PC7**, **PC5** | Active flow duration and Flag counts. |
| **Autoencoder** | **URG**, **ACK**, **PSH** Flags | Structural deviations in packet headers. |

### 6.2 Technical Conclusion
SHAP results mathematically confirm that **Packet Flags** (URG, ACK, PSH) and **Inter-Arrival Timing (IAT)** variances are the fundamental indicators distinguishing malicious intent within the CICIDS2017 dataset.

---

## 7. Production Infrastructure: Intrusion Tracker Backend
The system is deployed via a high-performance **FastAPI** backend supporting asynchronous real-time inference.

### 7.1 The "LGBM Veto" Mechanism
The hybrid pipeline prioritizes the reduction of False Positives through a hierarchical verification process:
1.  **Initial Flagging**: Unsupervised models identify deviations from the benign baseline.
2.  **Supervised Verification**: LightGBM analyzes the flagged flow. If the signature analysis indicates a benign flow, the "Veto" mechanism suppresses the alert.

### 7.2 PCAP Processing
The backend utilizes `pcap_to_cicids.py` for real-time feature extraction, enabling the system to ingest raw network traffic captures and produce model-ready feature vectors instantaneously.

---

## 8. Directory Structure
```text
├── docs/                     # Technical Docs (documentations, viva, project_details, etc.)
├── training_scripts/         # Development Notebooks & Scripts
├── models/                   # Serialized Weights (.joblib, .pth)
├── scalers/                  # Transformation Models (PCA, Scalers)
├── final/                    # Processed Parquet Datasets
├── raw/                      # Raw CICIDS2017 CSV source
├── router.py                 # FastAPI Gateway
└── predict.py                # Inference Engine
```

---

## 9. Deployment Guidelines

### 9.1 Edge-Optimized Inference
While the models were trained utilizing high-performance GPU acceleration to handle the large-scale CICIDS2017 dataset, the resulting inference engine is optimized for cross-platform compatibility. The system is specifically engineered to run efficiently on **ARM architectures without hardware acceleration**, allowing for seamless deployment on decentralized edge hardware. This versatility ensures that the Intrusion Tracker can provide localized security without requiring specialized data center infrastructure.

### 9.2 Infrastructure Requirements
- Python 3.9+
- Compatible with ARM64 and x86_64 architectures (No GPU required for inference).

### 9.3 Execution
1.  Install dependencies: `pip install -r requirements.txt`.
2.  Configure `.env` environment variables with model and scaler paths.
3.  Launch the gateway: `python router.py`.

---

## 10. Appendix: Hyperparameter Configurations

### 10.1 Supervised Tuning (RandomizedSearchCV)
| Model | Key Parameters | Search Range |
| :--- | :--- | :--- |
| **LightGBM** | `num_leaves` | [20, 31, 50] |
| | `learning_rate` | loguniform(0.01, 0.2) |
| | `n_estimators` | [5, 10] |
| **XGBoost** | `max_depth` | [6, 8, 10] |
| | `min_child_weight` | [10, 30, 50] |

### 10.2 Unsupervised Tuning
| Model | Key Parameters | Selection |
| :--- | :--- | :--- |
| **Autoencoder** | `bottleneck` | 16 |
| | `optimizer` | Adam (lr=1e-3, weight_decay=1e-5) |
| **Isolation Forest** | `max_samples` | 2048 |
| | `n_estimators` | 200 |

---

## 11. External Resources and Repositories
-   **Live Application**: [Intrusion Tracker Project](https://www.mohakapoor.in/projects/IntrusionDetection)
-   **Backend Repository**: [IntrusionBackend GitHub](https://github.com/mohakapoor/IntrusionBackend)
-   **Research & Training Repository**: [Network Anomaly Detection GitHub](https://github.com/mohakapoor/Network_Anomaly_Detection_CICIDS2017)
