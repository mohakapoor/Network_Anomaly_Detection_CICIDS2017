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

| Day | Traffic Category | System Role |
| :--- | :--- | :--- |
| Monday | Benign | Baseline Anomaly Profiling |
| Tuesday | FTP/SSH Brute Force | Supervised Signature Training |
| Wednesday | DoS (Hulk, Slowloris, etc.) | Supervised Signature Training |
| Thursday | Web Attacks, Infiltration | Supervised Signature Training |
| **Friday** | **Botnet, DDoS, PortScan** | **Zero-Day Evaluation** |

---

## 4. Analytical Pipeline

### Stage 1: Data Consolidation (`split.ipynb`)
-   Raw CSV files are merged, and column headers are sanitized to remove whitespace.
-   Rare attack labels are consolidated into a standardized "Other Attacks" category.
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
Trained on PCA-transformed components to identify established attack patterns.

| Model | Architecture | Accuracy (Test) |
| :--- | :--- | :--- |
| **LightGBM** | Leaf-wise Gradient Boosting | **99%** |
| **FFNN** | PyTorch Multi-layer Perceptron | 98% |
| **XGBoost** | Level-wise Gradient Boosting | 97% |
| **Binary SVM** | GPU-accelerated Linear Classifier | 96.7% |

### 5.2 Unsupervised Layer (Anomaly Detection)
Trained exclusively on benign Monday traffic to establish a baseline for "Normal" network behavior.

#### **Denoising Autoencoder**
-   **Structure**: 69 → 128 → 64 → **16 (Bottleneck)** → 64 → 128 → 69.
-   **Error Metric**: Hybrid MSE + Max Per-Feature Deviation.
-   **Key Finding**: Successfully identified **Botnet** traffic with **73.5% recall**, significantly outperforming traditional statistical methods.

#### **Isolation Forest**
-   **Methodology**: Recursive partitioning to isolate anomalous observations.
-   **Result**: Exceptional performance in detecting **PortScans** (**99.3% recall**).

---

## 6. Explainability and SHAP Analysis
To ensure institutional-grade transparency, SHAP (SHapley Additive exPlanations) analysis was conducted to determine the mathematical drivers behind model decisions.

### 6.1 Feature Importance Results
The analysis identified specific network indicators that serve as primary drivers for intrusion detection:

| Model Category | Primary SHAP Features / Components | Detection Indicator |
| :--- | :--- | :--- |
| **Tree Models (LGBM/XGB)** | **PC3**, **PC1** | Timing (IAT) and Idle Time variances. |
| **Neural Networks (FFNN)** | **PC1**, **PC7**, **PC5** | Active flow duration and Flag counts. |
| **Autoencoder (Anomaly)** | **URG**, **ACK**, **PSH** Flags | Structural deviations in packet headers. |
| **Isolation Forest** | **PSH** Flag Count, **Bwd Pkt Len Std** | Structural isolation of scanning traffic. |

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
├── docs/                     # Technical Documentation
├── training_scripts/         # Development Notebooks & Scripts
├── models/                   # Serialized Weights (.joblib, .pth)
├── scalers/                  # Transformation Models (PCA, Scalers)
├── final/                    # Processed Parquet Datasets
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

## 10. External Resources and Repositories

-   **Live Application**: [Intrusion Tracker Project](https://www.mohakapoor.in/projects/IntrusionDetection)
-   **Backend Repository**: [IntrusionBackend GitHub](https://github.com/mohakapoor/IntrusionBackend)
-   **Research & Training Repository**: [Current Project Repository](https://github.com/mohakapoor/Network_Anomaly_Detection_CICIDS2017)
