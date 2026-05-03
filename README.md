<div align="center">

# Intrusion Tracker: Network Anomaly Detection
### Bridging Research and Production for Next-Gen Network Security

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-05998b.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: CICIDS2017](https://img.shields.io/badge/Dataset-CICIDS2017-lightgrey.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Architecture: ARM64](https://img.shields.io/badge/Architecture-ARM64-orange.svg)](#)
[![Model: LightGBM](https://img.shields.io/badge/Model-LightGBM-336791.svg)](#)
[![Model: XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](#)
[![Model: Autoencoder](https://img.shields.io/badge/Model-Autoencoder-magenta.svg)](#)
[![Model: Isolation Forest](https://img.shields.io/badge/Model-Isolation--Forest-purple.svg)](#)
[![Sklearn: Scikit--Learn](https://img.shields.io/badge/Sklearn-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

[**Live Demo**](https://www.mohakapoor.in/projects/IntrusionDetection) | [**Backend Repo**](https://github.com/mohakapoor/IntrusionBackend) | [**Full Documentation**](docs/documentations.md)

</div>

---

## 🚀 Overview

**Intrusion Tracker** is a state-of-the-art Network Intrusion Detection System (NIDS) designed to identify and classify cyber threats in real-time. By implementing a **Hierarchical Hybrid Pipeline**, it combines the surgical precision of **Supervised Signatures** (LightGBM/FFNN) with the zero-day sensitivity of **Unsupervised Anomaly Detection** (Denoising Autoencoders).

### 🛠️ Key Features

-   **Dual-Pronged Detection**: Concurrent signature-based and anomaly-based engines for defense-in-depth (**Hybrid-NIDS**).
-   **Modern Infrastructure**: Built on **FastAPI**, leveraging Pydantic v2 for ultra-low latency data validation and high-throughput flow processing.
-   **The "LGBM Veto"**: A unique hierarchical logic that utilizes a high-precision classifier to validate anomalies, significantly reducing false positives.
-   **Edge-Optimized**: Designed for efficiency—while trained on GPUs, the inference engine is optimized for **ARM architectures** without requiring hardware acceleration.
-   **SHAP-Validated**: Institutional-grade transparency with feature importance confirmed via SHAP analysis.
-   **Real-time Ready**: FastAPI backend with WebSocket streaming for live network monitoring.

---

## 📊 Performance at a Glance

| Metric | Autoencoder (Anomaly) | LightGBM (Signature) |
| :--- | :--- | :--- |
| **Accuracy / AUC** | **0.7801 (AUC)** | **99% (Acc)** |
| **Bot Recall** | **73.5%** | 99% |
| **DDoS Recall** | 83.1% | 100% |
| **PortScan Recall** | 96.6% | 100% |

---

## 🏗️ System Architecture

1.  **Ingestion**: Raw PCAP or CICIDS flow data.
2.  **Detection Layer**: Denoising Autoencoder & Isolation Forest flag deviations from the benign baseline.
3.  **Validation Layer**: Flagged flows are analyzed by LightGBM/FFNN to determine specific attack signatures.
4.  **Actionable Insight**: Final classification results are streamed via WebSockets to the monitoring dashboard.

---

## 🚦 Quick Start

### 1. Prerequisites
- Python 3.9+
- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html) (for training)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Launch the Backend
```bash
python router.py
```

For detailed research methodology, data cleaning steps, and model hyper-parameters, please refer to the [**Full Documentation**](docs/documentations.md).

---

## 📑 Resources

-   **Research Deep-Dive**: [docs/documentations.md](docs/documentations.md)
-   **Production Backend**: [GitHub: IntrusionBackend](https://github.com/mohakapoor/IntrusionBackend)
-   **Live Application**: [mohakapoor.in/projects/IntrusionDetection](https://www.mohakapoor.in/projects/IntrusionDetection)

---

<div align="center">
Built for modern network security and edge-computing resilience.
</div>
