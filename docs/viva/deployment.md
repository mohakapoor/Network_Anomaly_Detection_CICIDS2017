# Topic 5: System Structure, Model Serialization, & Cross-Dataset Validation

This document explains the overall end-to-end structure of the intrusion detection system, model and preprocessing serialization, cross-dataset validation, and theoretical resource constraints.

---

## 1. System Structure & End-to-End Flow

To understand where this project fits within a complete Network Intrusion Detection System (NIDS), refer to the end-to-end system flowchart:

```text
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

The dataset provides pre-extracted flows (bypassing the raw packet capture and parsing stage). Within the **Detection Engine**, the project implements a **Hierarchical Hybrid Pipeline** to process these flows:

```text
Incoming Flow
      │
      ▼
[ Stage 1: Anomaly Detection Layer ] ──► (Denoising Autoencoder)
      │
      ├───► [ Below Threshold ] ────────► Safe & Log (Normal Traffic)
      │
      └───► [ Above Threshold (Error > 0.001068) ] 
                  │
                  ▼ (Flagged Anomaly)
[ Stage 2: Signature Verification Layer ] ──► (LightGBM Classifier)
                  │
                  ├───► [ Classified Benign (Class 0) ] ──► [ VETO ] ──► Safe & Log
                  │
                  └───► [ Classified Attack (Class 1-6) ] ──► Generate Alert
```

### 1.1 Stage 1: Anomaly Detection Layer (Autoencoder)
* **Operation**: Incoming flows are normalized via MinMaxScaler and passed through the Denoising Autoencoder.
* **Error Calculation**: The Combined Reconstruction Error is calculated:
  $$\text{Error} = 0.5 \times \text{MSE} + 0.5 \times \max(\text{per-feature error})$$
* **Routing**:
  * If the error is below the F1-optimal threshold ($0.001068$), the flow is logged as safe.
  * If the error is above $0.001068$, the flow is flagged as anomalous and routed to Stage 2.

### 1.2 Stage 2: Signature Verification Layer (LightGBM)
* **Operation**: The flagged flow is standard-scaled, projected down to 34 components using the pre-fit Incremental PCA matrix, and evaluated by the LightGBM classifier.
* **Veto Mechanism**:
  * If LightGBM classifies the signature as `BENIGN` (Class 0), a **Veto** is issued. The alert is suppressed, and the flow is logged as safe.
  * If LightGBM classifies the signature as an active attack (Classes 1–6), an alert is generated.

---

## 2. Model & Scaler Serialization

Once the models and preprocessing pipelines are trained, they must be saved (serialized) to disk so they can be loaded for future offline inference without retraining.

### 2.1 Preprocessing and Scaler Serialization
To ensure mathematical consistency when new unseen flows are evaluated, the parameters calculated during training are saved under the `scalers/` directory:
* **The Scalers**: The mean and standard deviation for `StandardScaler`, and the minimum and maximum ranges for `MinMaxScaler`.
* **The PCA Matrix**: The orthogonal transformation components calculated by `IncrementalPCA`.
* **Tooling**: Standard Python `joblib` serialization is used to dump these estimators:
  ```python
  joblib.dump(scaler, "scalers/scaler.joblib")
  joblib.dump(pca, "scalers/pca.joblib")
  ```

### 2.2 Model Weights Serialization
Different libraries use different serialization formats based on their underlying structures:
* **Tree-Based Models (LightGBM, XGBoost, Isolation Forest)**:
  * Serialized using `joblib` into `.joblib` files. These contain the complete binary representation of the tree split thresholds, leaf paths, and weights.
* **Neural Network Models (FFNN & Autoencoder)**:
  * Serialized using PyTorch's native state dictionary format (`.pth` or `.pt` files).
  * **Mechanism**: PyTorch saves the weights and biases of each layer as a dictionary mapping parameter names to tensors:
    ```python
    torch.save(model.state_dict(), "models/autoencoder/autoencoder_model.pth")
    ```
  * During inference, the model structure is instantiated, and the pre-trained weights are loaded:
    ```python
    model = Autoencoder(NUM_FEATURES)
    model.load_state_dict(torch.load("models/autoencoder/autoencoder_model.pth"))
    model.eval() # Sets dropout and batch normalization to evaluation mode
    ```

---

## 3. Generalization & Cross-Dataset Validation (CICIDS2018)

To evaluate how well the trained models generalize beyond the control testbed of the **CICIDS2017** dataset, the project includes a dedicated validation pipeline under the `cross_dataset_val/` directory.

### 3.1 The Generalization Challenge
* **The Problem**: Machine learning models often suffer from overfitting to the specific background noise, IP address layouts, and network configurations of the training dataset. A model that achieves $99\%$ accuracy on the 2017 test set might fail on different network environments due to slight shifts in packet distributions.
* **The Validation**: The models trained on **CICIDS2017** are tested against data extracted from the **CICIDS2018** dataset.

### 3.2 Preprocessing the 2018 Dataset (`cross_dataset_preprocessing.ipynb`)
To perform a fair cross-dataset evaluation, the 2018 traffic must undergo identical preprocessing:
1. **Feature Matching**: The columns in the 2018 dataset are filtered and aligned to match the exact 69 raw features used by the 2017 model.
2. **Sanitization**: Missing values (NaN) and infinite rate flags ($\infty$) are resolved to `0` using the same `handle_values()` routine.
3. **Scaler & PCA Application**: The 2018 test data is transformed using the **pre-fit 2017 scalers and PCA matrices**. We do not fit new scalers on the 2018 data, as this would violate the assumption of testing on raw unseen traffic.
4. **Evaluation**: The pre-trained LightGBM, Autoencoder, and Isolation Forest models evaluate the transformed 2018 parquet file, providing a realistic measure of how the model generalizes to a new network environment.

---

## 4. Resource Feasibility: Theoretical Edge Optimization

While this repository focuses on training and research validation, the serialized models are mathematically suited for deployment on resource-constrained edge hardware (such as routers or local network gateways) due to several structural factors:

* **Shallow Tree Architectures**:
  * The LightGBM and XGBoost models are trained with a low tree count (`n_estimators = 10`) and limited depth (`max_depth = 8`).
  * In a production environment, evaluating a new flow through these 10 trees does not require matrix calculators or GPU accelerators. The CPU traverses the trees as simple conditional `if-else` branches, resulting in near-instantaneous execution.
* **CPU-Only Neural Inference**:
  * Large deep learning models (like LLMs or heavy computer vision nets) require GPUs for inference.
  * Your Denoising Autoencoder bottleneck is extremely small (bottleneck = 16) and the MLP is shallow. The model can run inference entirely on standard **ARM64 edge CPUs** without a GPU.
* **RAM footprint**:
  * By downcasting the datasets and weights to 32-bit floats during training, the serialized weights files are tiny (a few kilobytes for tree models, under 1MB for the Autoencoder). This allows the entire pipeline to fit into low active RAM environments, which is ideal for decentralized edge gateways.

---

## 🎯 5. 30 Crux Generalization & Flow Questions

### Q1: What is a Hierarchical Hybrid Pipeline, and what are its two stages in your project?
**A**: It is a two-stage classification cascade. Stage 1 is the unsupervised Autoencoder (Anomaly Detection Layer), which acts as a highly sensitive filter to flag deviations from the benign baseline. Stage 2 is the supervised LightGBM (Signature Verification Layer), which verifies the anomaly, identifies the specific attack signature, or suppresses false alarms via a veto.

### Q2: Why is the hybrid pipeline superior to using a supervised-only or unsupervised-only system?
**A**: A supervised-only system is blind to novel zero-day attacks. An unsupervised-only system can detect zero-days but generates high rates of false positives. Combining them allows the system to detect novel anomalies (Stage 1) while keeping false alarm rates extremely low (Stage 2 Veto).

### Q3: What is the "LGBM Veto" and how does it function inside your pipeline flow?
**A**: An anomaly flagged by the Autoencoder is passed to LightGBM. If LightGBM classifies the flow's signature as "Benign" (Class 0), it issues a "Veto". The anomaly alert is suppressed, the false alarm is neutralized, and the flow is logged as safe.

### Q4: How does the pipeline calculate the anomaly score in Stage 1?
**A**: It calculates the **Combined Reconstruction Error** of the flow across the 69 features:
$$\text{Reconstruction Error} = 0.5 \times \text{MSE} + 0.5 \times \max(\text{per-feature error})$$
If this error exceeds the F1-optimal threshold ($0.001068$), it is flagged as anomalous.

### Q5: What happens to a flow that yields a reconstruction error below $0.001068$?
**A**: It is declared normal, logged as safe, and routed immediately without being passed to the supervised LightGBM stage, saving computational overhead.

### Q6: Why is model serialization necessary?
**A**: Training large machine learning models on millions of rows is computationally intensive and slow. Serialization allows us to save the optimized model weights, biases, and tree split thresholds to disk so they can be loaded instantly for offline inference without retraining.

### Q7: What serialization format did you use for your tree-based models, and why?
**A**: I used **`joblib`** to serialize LightGBM, XGBoost, and Isolation Forest models into `.joblib` files. `joblib` is highly optimized for storing large NumPy arrays, which make up the internal weights, split thresholds, and structures of tree estimators.

### Q8: How are your PyTorch neural networks serialized?
**A**: They are serialized using PyTorch's native state dictionary (`state_dict`) format, saved as `.pth` or `.pt` files. This format maps each layer's parameter name (e.g. weights and biases) to its corresponding tensor representation, keeping the saved file highly compact.

### Q9: Why is saving only the `state_dict` preferred over saving the entire PyTorch model?
**A**: Saving the entire model pickles the specific directory paths and class definitions, which easily breaks when the model is loaded on a different machine or within a different folder structure. Saving only the parameter weights (`state_dict`) is robust, portable, and independent of directory locations.

### Q10: What does the command `model.eval()` do, and why is it mandatory before running inference?
**A**: `model.eval()` sets the PyTorch model to evaluation mode. This deactivates **Dropout** (which should only run during training) and freezes **Batch Normalization** layer statistics (forcing the model to use running averages rather than batch statistics), ensuring consistent and stable predictions.

### Q11: What is data leakage in the context of scaling, and how does serialization prevent it?
**A**: Data leakage occurs if you fit your MinMaxScaler or StandardScaler on the test set, as this lets the model "peek" at the test set's distribution (mean, standard deviation, min, and max). Serializing the scalers pre-fit on the training set ensures the test set is scaled strictly using training-set parameters, preventing leakage.

### Q12: Why is it critical to validate your models on the CICIDS2018 dataset?
**A**: Models trained on a single dataset often overfit to that specific environment's background noise, subnet IP ranges, and timing patterns. Validating on the CICIDS2018 dataset evaluates how well the models generalize to a completely separate network environment with different configurations.

### Q13: What does the directory `cross_dataset_val/` contain?
**A**: It contains `cross_dataset_preprocessing.ipynb` (which parses, sanitizes, and aligns the CICIDS2018 dataset to match our 2017 features) and the raw parquet test files used to run generalizability evaluations.

### Q14: How did you preprocess the CICIDS2018 dataset to run cross-dataset validation?
**A**: I aligned the 2018 columns to match the 2017 feature set, removed zero-variance and duplicate columns, resolved NaN and infinite values to 0 using the same `handle_values` function, and applied the pre-fit 2017 scalers and PCA matrices.

### Q15: Why did you apply the pre-fit 2017 scalers to the 2018 dataset instead of fitting new ones?
**A**: Fitting a new scaler on the 2018 dataset would adjust the feature ranges based on the 2018 data distribution, violating the rule of testing on raw unseen traffic. The 2018 data must be scaled using the exact same training boundaries (2017) that the models learned.

### Q16: What is the main generalizability risk for supervised models trained on CICIDS2017?
**A**: Supervised models learn specific, exact feature split thresholds for 2017 attacks. If the 2018 dataset features shift slightly due to different network hardware or traffic volume, the supervised models can easily misclassify the flows because the learned boundaries are rigid.

### Q17: Why is the unsupervised Autoencoder theoretically more robust to cross-dataset drift than supervised classifiers?
**A**: Supervised classifiers rely on exact, rigid split boundaries. The Autoencoder learns the soft statistical manifold of normal benign behavior. While background details shift, the overall structure of normal traffic (like low packet flag rates and normal durations) remains similar, allowing the Autoencoder to flag anomalies robustly.

### Q18: What does the term "inference" mean?
**A**: Inference is the process of feeding new, unseen data points into a pre-trained, serialized model to generate classifications or reconstruction anomaly scores, without performing any backpropagation or updating of weights.

### Q19: Why is inference computationally much cheaper than training?
**A**: Training requires a forward pass, loss calculation, backward pass (calculating derivatives and gradients), and weight updates across millions of rows for many epochs. Inference requires only a single forward pass (basic matrix multiplications or tree branch traversals), which is highly efficient.

### Q20: Why are your trained models suitable for CPU-only edge environments?
**A**: The supervised tree models are highly shallow (`max_depth = 8`, `n_estimators = 10`), and the neural networks have small structures (bottleneck = 16). They do not require heavy parallel matrix computations, allowing them to run efficiently on low-power CPUs without GPU hardware.

### Q21: How does a decision tree execute on a CPU during inference?
**A**: The CPU evaluates the tree as a sequential cascade of conditional `if-else` branches matching the learned thresholds. This requires no floating-point matrix operations, executing almost instantly.

### Q22: What is the benefit of downcasting float64 to float32 for model deployment?
**A**: Standard edge CPUs natively accelerate 32-bit float calculations using register instructions. Downcasting halves the file sizes of the serialized weights and minimizes RAM usage, allowing the models to run on memory-constrained hardware.

### Q23: Why is the Denoising Autoencoder a good fit for low-RAM devices?
**A**: The Denoising Autoencoder uses a lightweight MLP structure with a central bottleneck of 16. The total count of parameter weights is tiny, resulting in a serialized file size of under 1MB, which is easily loaded into the RAM of standard edge gateways.

### Q24: Explain why PCA was used for the supervised models but omitted for the unsupervised Autoencoder.
**A**: For supervised models, PCA reduces the feature dimensions from 69 to 34, speeding up decision tree training and preventing overfitting. For the Autoencoder, the bottleneck layer performs non-linear dimensionality reduction; applying linear PCA first would flatten the original manifolds, reducing detection sensitivity.

### Q25: Why is the `contamination` parameter set to 0.01 in the Isolation Forest?
**A**: It acts as a regularizer, indicating that the model should assume up to 1% outlier noise exists within the benign training data, preventing the isolation forest from overfitting to rare normal packets.

### Q26: Gotcha Q: If your pipeline detects an anomaly and routes it to Stage 2, but the network suffers a massive burst of legitimate traffic, does the Stage 2 veto model cause a latency bottleneck?
**A**: Yes, this is a classic **cascade queue congestion** problem. If the Stage-1 Autoencoder threshold is set too low, it flags a high proportion of normal network bursts as anomalous. This floods the Stage-2 LightGBM model with verification workloads. Since LightGBM must now evaluate thousands of flows, it creates a processing bottleneck, inflating joint latency and causing packet queues to drop on the interface. This highlights why the Stage-1 threshold must be tuned carefully using the F1-optimal contour to balance detection sensitivity with cascade throughput.

### Q27: Gotcha Q: When deploying on the 2018 dataset, if you fit a new MinMaxScaler on the 2018 data, why does your model's false positive rate spike immediately?
**A**: Fitting a new scaler on the 2018 dataset recalculates the minimum and maximum boundaries based on the 2018 distribution. This distorts the feature coordinate system. Legitimate 2018 flows are projected onto completely different scaled coordinates than the 2017 benign manifold that the Autoencoder learned, resulting in massive false positive reconstruction errors. The 2018 test data must be scaled strictly using the pre-fit 2017 scaler bounds.

### Q28: Gotcha Q: Can a hacker exploit your deployment by modifying the `.joblib` model file?
**A**: Yes, this represents a severe **de-serialization vulnerability**. The Python `pickle` and `joblib` libraries are unsafe formats; they execute arbitrary Python byte instructions during loading. If an attacker gains write access to the deployment server's filesystem and replaces a model file with a malicious payload, loading the model via `joblib.load()` will execute the payload with the privileges of the NIDS process, compromising the entire network. Production systems must secure model directories and verify cryptographical file hashes (e.g., SHA-256) before loading.

### Q29: Gotcha Q: If your GBDT model achieves a 99% F1-score, why can a hacker bypass it by launching a DDoS flood using extremely large packet sizes?
**A**: This is an evasion attack exploiting **covariate shift**. A GBDT model trained on standard DDoS attacks learns split boundaries based on the training set distribution where attack packets were small (e.g., 64 bytes). If the attacker simply expands their attack packet sizes to 1500 bytes, the flow falls into the "Benign" split, bypassing the model. This is why we must pair the GBDT with an unsupervised Autoencoder that flags any abnormal feature combinations (such as high rates paired with massive sizes) as out-of-distribution anomalies.

### Q30: Gotcha Q: Why does streaming inference flow-by-flow on a CPU cause cache thrashing compared to batch-based inference?
**A**: If we perform inference on a single flow at a time, the CPU must reload the model weights and scaler parameters from RAM into the L1/L2 caches for every single execution cycle, wasting massive clock cycles on memory latency. Batching flows allows the CPU to load the model weights once and apply them to multiple rows in a single contiguous operation, maximizing cache locality and maximizing pipeline throughput.
