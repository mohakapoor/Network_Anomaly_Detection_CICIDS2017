# SHAP Feature Importance & Explanation Guide
### Network Anomaly Detection (CICIDS2017)

This document provides a detailed breakdown of the features and Principal Components (PCs) that drive the predictions of various models in the Intrusion Tracker pipeline, based on empirical SHAP (SHapley Additive exPlanations) analysis.

---

## 1. Model-by-Model SHAP Feature Importance

### 1.1 Tree-Based Models: LightGBM & XGBoost
Both tree-based models were trained on the **PCA-reduced dataset** (35 components) to prevent overfitting and handle feature multicollinearity. Their SHAP importances align closely, showing that their decision trees split heavily on timing, packet cadence, and flag counts.

| Rank | LightGBM Feature (Mean \|SHAP\|) | XGBoost Feature (Mean \|SHAP\|) |
| :--- | :----------------------------- | :------------------------------ |
| **1**| **PC3** (0.227031)               | **PC3** (0.046184)              |
| **2**| **PC1** (0.225945)               | **PC2** (0.032785)              |
| **3**| **PC29** (0.119322)              | **PC1** (0.031266)              |
| **4**| **PC2** (0.109137)               | **PC29** (0.029923)             |
| **5**| **PC4** (0.108425)               | **PC25** (0.018819)             |

* **Core Insight**: The tree models are heavily driven by **PC3** (packet timing frequency) and **PC1** (maximum idle/silence durations), followed closely by **PC29** (TCP flags) and **PC2** (traffic volume).

---

### 1.2 Feed-Forward Neural Network (FFNN)
The FFNN, also trained on the PCA-reduced space, shows a slightly different sensitivity profile, prioritizing physical connection duration and TCP configurations.

| Rank | FFNN Feature | Mean Absolute SHAP Value |
| :--- | :----------- | :----------------------- |
| **1**| **PC1**      | 2.626495                 |
| **2**| **PC7**      | 1.112023                 |
| **3**| **PC5**      | 1.051691                 |
| **4**| **PC4**      | 1.035856                 |
| **5**| **PC13**     | 0.858120                 |

* **Core Insight**: The deep learning model is exceptionally sensitive to **PC1** (idle states/timing), **PC7** (active duration of connection), and **PC5** (TCP connections initialization and ACK handshakes).

---

### 1.3 Unsupervised Model: Isolation Forest
To ensure extreme coordinate deviations (like individual flag anomalies) are not smoothed out by dimensionality reduction, the unsupervised model was trained directly on the **raw physical features**. 

| Rank | Isolation Forest Feature | Mean Absolute SHAP Value |
| :--- | :----------------------- | :----------------------- |
| **1**| **PSH Flag Count**       | 0.263505                 |
| **2**| **URG Flag Count**       | 0.212149                 |
| **3**| **Bwd Packet Length Std**| 0.199398                 |
| **4**| **Bwd Packet Length Max**| 0.178683                 |
| **5**| **Packet Length Variance**| 0.167954                |

* **Core Insight**: The unsupervised Isolation Forest detector is heavily dominated by **TCP Control Flags (URG, PSH)** and **packet size distributions/variances (Standard Deviation & Max of Backward Packets)**.

---

## 2. In-Depth Feature Explanations & Physical Interpretations

To translate the mathematical parameters into physical network behavior, here is an explanation of the underlying network metrics and why they are critical for detecting network anomalies.

### ⏱️ 2.1 Timing and Inter-Arrival Times (IAT)
* **PC3 (Packet Cadence & Frequency - High weight on `Fwd IAT Min`, `Bwd IAT Min`, `Bwd IAT Mean`, `Fwd IAT Mean`)**:
  * **Physical Meaning**: Represents the average and minimum time differences between consecutive packets in a flow.
  * **Anomalous Behavior**: In human-generated traffic, packet transmission is highly irregular and relatively slow. In automated attacks (such as PortScans, DDoS, or script execution), packets are fired at extreme, uniform speeds with tiny, deterministic inter-arrival times. A drop in PC3 (indicating highly compressed, rapid packet intervals) is a strong indicator of automated scans or automated DDoS flows.
* **PC1 (Max Idle/Silence Durations - High weight on `Flow IAT Max`, `Idle Max`, `Fwd IAT Max`, `Idle Mean`)**:
  * **Physical Meaning**: Captures the maximum silence period or inactivity duration within a connection.
  * **Anomalous Behavior**: Botnets and Command & Control (C2) servers utilize periodic, low-frequency heartbeats (beaconing) where a connection stays completely idle for long periods and then transmits small control sequences. Normal web traffic has small, dynamic idle times, whereas beaconing creates massive anomalies in PC1.

### 📊 2.2 Traffic Volume & Payload Sizes
* **PC2 (Traffic Volume - High weight on `Subflow Fwd/Bwd Packets`, `Total Fwd/Bwd Packets`, `Total Length of Bwd Packets`)**:
  * **Physical Meaning**: Directly reflects the overall throughput and packet count in both directions.
  * **Anomalous Behavior**: Normal transactional flows (like DNS requests or brief API calls) contain few packets. Volumetric DDoS attacks, brute-force sweeps, or massive data exfiltration actions lead to a massive spike in packet and byte counts, making PC2 an essential component for high-volume threat detection.
* **PC4 (Payload Profile - High weight on `Fwd Packet Length Max`, `Fwd Packet Length Mean`, `Avg Fwd Segment Size`, `Fwd Packet Length Std`)**:
  * **Physical Meaning**: Represents the size characteristics of forward packets.
  * **Anomalous Behavior**: Attacks like Heartbleed or buffer overflows push unusually large payloads in the forward direction. Conversely, SYN floods and PortScans send empty forward packets with zero payload. The variance and max size captured in PC4 easily separate these anomalous payloads from standard HTTP/HTTPS requests.

### 🚩 2.3 TCP Control Flags & Connection States
* **PC29 (TCP Control State - High weight on `URG Flag Count`, `PSH Flag Count`, negative weight on `ACK Flag Count`)**:
  * **Physical Meaning**: Captures specific TCP control flag combinations.
  * **Anomalous Behavior**: Normal TCP traffic follows standard state handshakes where ACK flags are highly common, and URG (Urgent) or raw PSH (Push) flags without standard packets are rare. Attacks that attempt to bypass protocol boundaries or scan targets (such as Nmap NULL, FIN, or Xmas scans) set abnormal flag configurations to elicit specific responses from operating systems.
* **URG / PSH Flag Counts (in Isolation Forest)**:
  * **Physical Meaning**: Represents the explicit counts of these flags in the packet headers.
  * **Anomalous Behavior**: In anomalous scan patterns (like FIN, PSH, or Urgent scans), these flags are set to values that rarely occur in normal traffic. In the Isolation Forest tree structure, these rare coordinate values allow anomalous flows to be partitioned (isolated) very close to the root of the trees (requiring very few splits), leading to high anomaly scores.

### 📏 2.4 Packet Length Statistics
* **Bwd Packet Length Std / Max / Variance (in Isolation Forest)**:
  * **Physical Meaning**: The standard deviation, maximum, and variance of packet sizes returned from the server to the client.
  * **Anomalous Behavior**: During normal web traffic, servers return varying payload sizes (images, text, documents), creating a high standard deviation in packet size. When a client performs a PortScan or a DDoS attack, the target server either drops the packet or responds with uniform, tiny error packets (e.g., TCP RST). This drives the backward packet length variance close to zero. This drastic shift in variance stands out immediately in unsupervised isolation trees.

---

## 3. Quick Reference PCA Dictionary Mapping (Top Components)

For study and review, this maps the main Principal Components (PCs) identified by SHAP to their highest-weight raw network features:

| Principal Component | Core Concept | Top Contributing Features (Weights) |
| :--- | :--- | :--- |
| **PC1** | Inactivity & Max Idle Timing | **Flow IAT Max** (0.229), **Idle Max** (0.228), **Fwd IAT Max** (0.228) |
| **PC2** | Packet Count & Throughput | **Subflow Fwd Packets** (0.368), **Total Fwd Packets** (0.368) |
| **PC3** | Packet Timing Cadence & Speed | **Fwd IAT Min** (0.326), **Bwd IAT Min** (0.315), **Bwd IAT Mean** (0.314) |
| **PC4** | Forward Payload Sizes | **Fwd Packet Length Max** (0.347), **Fwd Packet Length Mean** (0.342) |
| **PC5** | TCP State & Session Initiation | **ACK Flag Count** (0.343), **Min Packet Length** (-0.321), **Destination Port** (0.319) |
| **PC7** | Active Session Durations | **Active Mean** (0.333), **Active Max** (0.306), **PSH Flag Count** (-0.278) |
| **PC9** | Urgent & Custom Flags | **CWE Flag Count** (0.582), **Fwd URG Flags** (0.582) |
| **PC13** | Packet Transmission Rates | **Flow Packets/s** (0.369), **Bwd Packets/s** (0.332) |
| **PC29** | Custom TCP Flags / Handshake | **URG Flag Count** (0.524), **PSH Flag Count** (0.445), **ACK Flag Count** (-0.378) |
