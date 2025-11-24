# Network Anomaly Detection using CICIDS2017

A machine learning pipeline for detecting network anomalies and cyber attacks using the CICIDS2017 dataset.

## Overview

This project implements a three-stage pipeline to detect various types of network attacks including DDoS, PortScan, Bot, and Web attacks using LightGBM classifier.

## Dataset

Uses the **CICIDS2017** dataset containing network traffic data with both benign and malicious patterns.

**Attack Types**: BENIGN, DoS attacks, DDoS, PortScan, Bot, Web Attacks, Infiltration

## Project Structure

```
├── raw/                          # Raw CSV files
├── final_df/                     # Preprocessed data and models
├── split.ipynb                   # Data splitting
├── preprocessing.ipynb           # Feature engineering
├── lightgbm.ipynb               # Model training
├── utils.py                     # Helper functions
└── config.yaml                  # Configuration
```

## Pipeline Workflow

### 1. Data Splitting (`split.ipynb`)
- Splits data into train (Mon-Thu) and test (Fri) sets
- Handles class imbalance through strategic sampling
- Consolidates rare attacks into broader categories

### 2. Preprocessing (`preprocessing.ipynb`)
- Cleans data and handles missing values
- Removes correlated features
- Selects top 20 features using importance scores
- Applies robust scaling

### 3. Model Training (`lightgbm.ipynb`)
- Trains multiclass LightGBM classifier
- Performs hyperparameter tuning
- Evaluates using F1-score and confusion matrices

## Requirements

- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- joblib
- pyyaml

## Usage

1. Place CICIDS2017 CSV files in `raw/` directory
2. Run notebooks in order: split → preprocessing → lightgbm
3. Check results and model performance metrics

## Key Features

- Automated feature selection
- Class imbalance handling
- GPU acceleration support
- Hyperparameter optimization

## License

For educational and research purposes.