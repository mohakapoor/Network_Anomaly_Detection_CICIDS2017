# Topic 4: Performance Metrics & SHAP Explainability
### Ultimate In-Depth Viva Study Guide: Network Anomaly Detection (CICIDS2017)

This document provides a detailed theoretical, mathematical, and project-specific analysis of **Evaluation Metrics, Threshold Engineering, and SHAP Explainability** implemented in the Intrusion Tracker pipeline.

---

## 1. Mathematical Foundations of Performance Metrics

Evaluating machine learning models on highly imbalanced network datasets requires a rigorous mathematical framework.

### 1.1 The Confusion Matrix
The foundation of all classification metrics, mapping actual states to predicted states:

| | Predicted Normal (0) | Predicted Anomaly (1) |
| :--- | :--- | :--- |
| **Actual Normal (0)** | **True Negative (TN)** | **False Positive (FP)** (False Alarm) |
| **Actual Anomaly (1)**| **False Negative (FN)** (Undetected Breach) | **True Positive (TP)** |

* **Security Interpretation**:
  * **False Positive (FP)**: Normal, safe network traffic is incorrectly flagged as an attack, causing operational alert noise.
  * **False Negative (FN)**: An actual attack bypasses the NIDS undetected, posing a severe security risk.

---

### 1.2 Core Single-Class Metrics
* **Accuracy**: The ratio of correct predictions to total instances:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  * *The Imbalance Trap*: In a dataset where $99\%$ of flows are benign, a static classifier that predicts `Benign` for all instances achieves $99\%$ accuracy while detecting $0\%$ of attacks. Accuracy is invalid for evaluation here.
* **Precision (Positive Predictive Value)**: The proportion of flagged flows that are actually malicious:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  * *Significance*: Directly correlates to the reliability of alarms. Low precision leads to analyst alert fatigue.
* **Recall (Sensitivity / True Positive Rate)**: The proportion of actual malicious flows that were successfully detected:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  * *Significance*: Measures the security coverage. High recall ensures minimal threats bypass the system.

---

### 1.3 The F1-Score: Harmonic Mean vs. Arithmetic Mean
The F1-Score is the harmonic mean of precision and recall:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Why Harmonic Mean?
The arithmetic mean of precision and recall is:
$$\text{Arithmetic Mean} = \frac{\text{Precision} + \text{Recall}}{2}$$
* If a model has a **Precision of $1.0$ and a Recall of $0.0$** (it flags only 1 flow, which happens to be correct, and misses all other 10,000 attacks):
  * **Arithmetic Mean**: $\frac{1.0 + 0.0}{2} = 0.5$ (suggests moderate performance).
  * **Harmonic Mean (F1)**: $2 \times \frac{1.0 \times 0.0}{1.0 + 0.0} = 0.0$ (correctly flags a completely failed detector).
* **The Math**: The harmonic mean is mathematically dominated by the smaller of the two numbers, forcing the model to achieve high values in *both* metrics to obtain a high F1-Score.

---

### 1.4 Multi-Class Averaging Paradigms
In a multi-class setup ($C$ classes), metrics are aggregated across classes:

* **Macro F1-Score**: Calculates the F1-Score for each class independently and computes a simple arithmetic average:
  $$\text{F1}_{macro} = \frac{1}{C}\sum_{k=1}^C \text{F1}_k$$
  * *Significance*: Treats every class with equal weight. A poor classification rate on the rare "Botnet" class severely impacts the score, making it the most rigorous and honest metric for class imbalance.
* **Weighted F1-Score**: Calculates the F1-Score for each class and averages them weighted by the number of support instances ($N_k$) in each class:
  $$\text{F1}_{weighted} = \sum_{k=1}^C \left(\frac{N_k}{N_{total}}\right) \text{F1}_k$$
  * *Significance*: Heavily dominated by the majority class (Benign). Hides poor classification on minority classes.
* **Micro F1-Score**: Computes global precision and recall by pooling the total TPs, FPs, and FNs across all classes, and then calculates the F1-Score. On standard multi-class classification, Micro F1 is mathematically equivalent to Accuracy.

---

### 1.5 Curve Analyses: ROC AUC vs. PR AUC
* **ROC Curve (Receiver Operating Characteristic)**: Plots the True Positive Rate (Recall) against the False Positive Rate (FPR):
  $$\text{FPR} = \frac{FP}{TN + FP}$$
  * *ROC AUC*: Measures the area under this curve. Bounded between 0.5 (random guessing) and 1.0 (perfect separation). It represents the probability that a randomly selected positive sample will score higher than a randomly selected negative sample.
* **The Trap of ROC AUC under Severe Imbalance**:
  * The denominator of the FPR includes **True Negatives (TN)**. In network traffic, TN represents benign flows, which count in the millions.
  * If the model generates a high count of False Positives ($FP = 10,000$), but the benign dataset is massive ($TN = 1,000,000$), the FPR remains tiny:
    $$\text{FPR} = \frac{10,000}{1,000,000 + 10,000} \approx 0.0099 \ (0.99\%)$$
  * This falsely suggests an excellent FPR, leading to a highly inflated ROC AUC (e.g. $0.98$) despite generating 10,000 false alarms.
* **PR Curve (Precision-Recall)**: Plots Precision against Recall.
  * *PR AUC (Average Precision)*: Bypasses True Negatives entirely. The denominator for precision ($TP + FP$) focuses strictly on the active alerts. Generating 10,000 false alarms immediately tanks the precision, resulting in a low, honest PR AUC. **PR AUC is mathematically superior for evaluations under heavy class imbalance.**

---

## 2. Project-Specific Performance Trade-offs & Thresholds

### 2.1 The Operational Decision: Recall over Precision
In a live Security Operations Center (SOC), the cost of a False Negative (allowing a Botnet or DDoS attack to proceed undetected) is catastrophic, potentially leading to data loss or network downtime. The cost of a False Positive is merely analyst time spent validating the alert.
* **The Project Strategy**: The anomaly detection layer is tuned to maximize **Recall** (catching 89% of attacks).
* **The LGBM Veto**: The resulting high rate of False Positives is handled by routing all anomalies through the **LightGBM signature layer**. If LightGBM classifies the anomaly as benign, it is vetoed, recovering lost Precision without sacrificing Recall.

---

### 2.2 Threshold Engineering: F1-Optimal vs. Conservative
Because the unsupervised Autoencoder outputs a continuous reconstruction error, setting the operational alert boundary (threshold) is critical.

```text
Reconstruction Error Distribution (Test Set):
Benign Density (steelblue) ───────────► █ █ █ 
                                        █ █ █ █ 
                                        █ █ █ █ █    [Optimal Threshold: 0.001068]
                                        █ █ █ █ █ █  │
Attack Density (crimson)  ────────────► ░ ░ ░ ░ ░ █ █│█ █ █ █ █ █ █ █ █ █ █
                                        ░ ░ ░ ░ ░ █ █│█ █ █ █ █ █ █ █ █ █ █
                                                     ▼
                                          Recall = 89%, FPR = 37%
                                          
                                          [Conservative Threshold (Mean + 3*Std): 0.004521]
                                                                                     │
                                                                                     ▼
                                                                          Recall = 31%, FPR = 7%
```

* **Conservative Threshold ($\mu + 3\sigma$)**: 
  * Sets the threshold at the mean plus three standard deviations of the normal validation errors.
  * **Result**: Highly conservative. It yields an FPR of only $7\%$, but misses almost all stealthy attacks, resulting in a **terrible Attack Recall of $31\%$**. It completely missed Botnets ($8.2\%$ recall) and PortScans ($3.8\%$).
* **F1-Optimal Threshold ($0.001068$)**:
  * Sweeps the validation dataset to find the exact reconstruction error threshold that mathematically maximizes the F1-Score.
  * **Result**: Boosts overall **Attack Recall to $89\%$**. Botnet recall jumps to **$73.5\%$** and PortScan recall to **$96.6\%$**, with a tolerable $37\%$ false-positive rate (which is handled by the subsequent LightGBM Veto layer).

---

## 3. Explainability & Model Transparency: SHAP

### 3.1 SHAP Mathematical Theory
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual predictions. It defines the contribution of each feature as a **Shapley Value**, treating features as players in a cooperative game where the "payout" is the model's prediction output.

#### The Shapley Value Formula:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
where:
* $F$ is the complete set of all features.
* $S$ is a subset of features excluding feature $i$.
* $f_x(S)$ is the model's prediction using only the features in subset $S$.
* The marginal contribution of feature $i$ is calculated across all possible feature combinations $S$, weighted by the probability of each subset size.

#### The Three Mathematical Axioms of SHAP:
1. **Local Accuracy (Additivity)**: The sum of the Shapley values of all features must equal the difference between the model's prediction $f(x)$ and the baseline expected prediction value $\mathbb{E}[f(x)]$:
   $$f(x) = \mathbb{E}[f(x)] + \sum_{i=1}^{|F|} \phi_i(x)$$
2. **Missingness**: A feature that is completely absent or has zero impact on the model's prediction must be assigned a Shapley value of zero:
   $$x_i = \text{None} \implies \phi_i(x) = 0$$
3. **Consistency**: If a model changes such that a feature's marginal contribution increases or remains constant across all subsets, its calculated Shapley value cannot decrease.

---

### 3.2 Project Explainability Findings

SHAP analysis mathematically identified the key feature components driving predictions across the models:

* **Tree-Based Models (LightGBM & XGBoost)**:
  * Highly sensitive to **PC3** and **PC1** components.
  * *Network Mapping*: These components represent **Flow IAT (Inter-Arrival Time)** variances and **Active/Idle state ratios**, proving tree splits prioritize connection timing anomalies over static headers.
* **Feed-Forward Neural Networks (FFNN)**:
  * Dependent on **PC1**, **PC7**, and **PC5** components.
  * *Network Mapping*: These map directly to **active connection durations** and **cumulative packet flag counts**.
* **Denoising Autoencoder**:
  * The reconstruction errors are highly driven by deviations in **TCP Control Flags (URG, ACK, and PSH)**.
  * *Interpretation*: This mathematically proves the Autoencoder successfully maps standard TCP handshake expectations, flagging custom-crafted packets with abnormal flag combinations.

---

## 🎯 4. 30 Crux ECE & Mathematical Viva Questions (Topic 4)

### Q1: What is the relationship between ROC Curve analysis in machine learning and Signal Detection Theory in ECE?
**A**: The Receiver Operating Characteristic (ROC) curve originated in ECE during World War II for radar signal detection. In detection theory, the ROC curve plots the **Probability of Detection ($P_d$)** on the y-axis (Recall/Sensitivity) against the **Probability of False Alarm ($P_{fa}$)** on the x-axis (False Positive Rate) across all possible decision thresholds, representing the detector's capability under noise.

### Q2: Why is the denominator of the False Positive Rate ($P_{fa}$) mathematically problematic in large-scale network monitoring?
**A**: The $P_{fa}$ is defined as $\frac{FP}{TN + FP}$. In network monitoring, the number of benign instances (True Negatives, $TN$) is extremely large. Because the $TN$ in the denominator is massive, it suppresses the calculated $P_{fa}$ value close to zero, even when the model generates thousands of false alarms ($FP$). This hides severe alert noise.

### Q3: Why does Average Precision (PR AUC) serve as a superior metric to ROC AUC for class-imbalanced signals?
**A**: ROC AUC's x-axis ($P_{fa}$) includes True Negatives, which inflates the denominator and masks false alarms. PR AUC plots Precision ($\frac{TP}{TP + FP}$) against Recall. By utilizing only the positive prediction space, PR AUC treats $FP$ (false alarms) as a direct penalty, making it highly sensitive to classification noise under severe class imbalance.

### Q4: Explain the mathematical proof of why the F1-Score uses the harmonic mean instead of the arithmetic mean.
**A**: The arithmetic mean of Precision ($P$) and Recall ($R$) is $\frac{P+R}{2}$. If a model has a Precision of $1.0$ (flags 1 flow correctly) and a Recall of $0.0$ (misses 10,000 attacks), the arithmetic mean is $0.5$ (suggesting moderate performance). The harmonic mean is calculated as:
$$\text{F1} = \frac{2}{\frac{1}{P} + \frac{1}{R}} = \frac{2PR}{P+R}$$
If $R = 0$, the numerator becomes $0$, dropping the F1-score to $0.0$, which correctly penalizes single-metric failures.

### Q5: How does the F1-optimal threshold selection mathematically relate to Neyman-Pearson optimization or Bayes Risk minimization?
**A**: Under Neyman-Pearson, we maximize $P_d$ subject to a bound on $P_{fa}$. F1-optimal threshold selection is a form of **Bayes Risk Minimization** with an asymmetric loss function. It sweeps the threshold to minimize the total risk where the cost of a False Negative (missed attack) is weighted heavily against the cost of a False Positive (false alarm), finding the mathematical maximum of their joint harmonic utility.

### Q6: Explain the difference in the mathematical formulation of Macro F1-Score and Weighted F1-Score, and why Macro is preferred.
**A**: Macro F1 averages the F1-scores of each class with equal weight: $\frac{1}{C}\sum F1_k$. Weighted F1 weighs each class score by its support fraction: $\sum \frac{N_k}{N_{total}} F1_k$. On highly imbalanced data, the weighted F1 is dominated by the majority class (Benign), whereas Macro F1 treats the rare classes (e.g. Botnets) as mathematically equal, exposing any failures on minority classes.

### Q7: Mathematically, what does the Area Under the ROC Curve (ROC AUC) represent?
**A**: ROC AUC represents the probability that a randomly chosen positive sample $x_+$ (attack flow) will receive a higher classification score or probability $f(x)$ from the model than a randomly chosen negative sample $x_-$ (benign flow):
$$\text{AUC} = P(f(x_+) > f(x_-))$$

### Q8: How does the Micro F1-Score mathematically simplify in a standard single-label multi-class classification task?
**A**: Micro F1 pools TPs, FPs, and FNs globally across all classes. In a single-label multi-class task where every instance has exactly one ground truth label and one prediction, the sum of False Positives must mathematically equal the sum of False Negatives. This causes Micro F1 to simplify directly to standard **Accuracy**.

### Q9: In the context of game theory, what is a Shapley Value, and how is it used in SHAP?
**A**: A Shapley value is a solution concept in cooperative game theory that defines a unique, fair distribution of payouts among players in a coalition. In SHAP, the "coalition" is the set of all input features, the "players" are individual features, and the "payout" is the change in the model's prediction from the expected baseline.

### Q10: What are the three mathematical axioms that guarantee the uniqueness of SHAP values?
**A**: 
1. **Local Accuracy (Additivity)**: The sum of the Shapley values of all features must equal the difference between the model's prediction $f(x)$ and the baseline expected prediction value $\mathbb{E}[f(x)]$.
2. **Missingness**: A feature that is completely absent or has zero impact on the prediction must be assigned a Shapley value of zero.
3. **Consistency**: If a model changes such that a feature's marginal contribution increases or remains constant across all subsets of features, its calculated Shapley value cannot decrease.

### Q11: Explain the mathematical theorem of "Efficiency" (Additivity) in SHAP.
**A**: The efficiency axiom states that the total gain of a coalition is fully distributed among its members:
$$\sum_{i=1}^{D} \phi_i(x) = f(x) - \phi_0$$
where $\phi_0 = \mathbb{E}[f(x)]$ is the expected model prediction over the background training dataset.

### Q12: How does SHAP calculate the marginal contribution of a feature $i$ across subsets?
**A**: It calculates the difference in the model's output prediction when feature $i$ is included in a subset $S$ versus when it is excluded:
$$\Delta(S, i) = f_x(S \cup \{i\}) - f_x(S)$$
This difference is evaluated across all possible subsets $S \subseteq F \setminus \{i\}$ and weighted by a combinatorial coefficient that accounts for the subset size.

### Q13: What does the SHAP value baseline expectation $\mathbb{E}[f(x)]$ mathematically represent?
**A**: It represents the empirical mean of the model's prediction outputs evaluated across the background training dataset:
$$\mathbb{E}[f(x)] = \frac{1}{N}\sum_{j=1}^N f(x_j)$$
It serves as the starting value (origin) from which the feature Shapley contributions add or subtract.

### Q14: Why is Cross-Entropy loss preferred over Mean Squared Error (MSE) for training the Softmax supervised models?
**A**: Cross-entropy loss is mathematically derived from the **Kullback-Leibler (KL) Divergence** between the true categorical distribution $q(x)$ and the model's predicted probability distribution $p(x)$:
$$\mathcal{L} = -\sum q(x) \log p(x)$$
Unlike MSE, cross-entropy yields large gradients when predictions are far off (avoiding gradient saturation), accelerating optimization.

### Q15: How does the custom Combined Error Metric of the Autoencoder act as an extreme value detector?
**A**: By combining MSE (which averages errors) with the maximum absolute per-feature error:
$$\text{Error} = 0.5 \times \text{MSE} + 0.5 \times \max_{j} |x_j - \hat{x}_j|$$
The $\max$ term acts as an $L_{\infty}$ norm, which is highly sensitive to extreme deviations in individual features, ensuring single-feature anomalies (like PortScans) are not averaged out.

### Q16: Why is the F1-Optimal Threshold computed as a continuous scalar instead of using a standard $0.5$ probability threshold?
**A**: A $0.5$ probability threshold assumes balanced class distributions and symmetric misclassification costs. On highly imbalanced data (87% benign), the reconstruction errors have a highly skewed distribution. Sweeping thresholds continuously to maximize F1-score mathematically adapts the operational boundary to this skewness.

### Q17: What does a ROC AUC score of 1.0 mathematically imply about the feature space?
**A**: An AUC of 1.0 implies that there exists a threshold on the model's prediction space that perfectly separates the positive and negative distributions, meaning the two classes occupy completely disjoint, non-overlapping manifolds in the feature space.

### Q18: Explain the mathematical difference between Macro-Precision and Micro-Precision.
**A**: Macro-precision calculates precision for each class independently and averages them: $\frac{1}{C}\sum \frac{TP_i}{TP_i + FP_i}$. Micro-precision aggregates TPs and FPs globally across all classes to calculate a single ratio: $\frac{\sum TP_i}{\sum TP_i + \sum FP_i}$. Micro-precision is dominated by the majority class, while Macro-precision is sensitive to minority class failures.

### Q19: Why does the SHAP analysis on your Denoising Autoencoder prioritize TCP control flags?
**A**: The Autoencoder was trained strictly on normal benign traffic, which follows standard TCP handshake behaviors. Attacks like SYN floods or PortScans use abnormal flag combinations, producing large reconstruction deviations in the flag features. SHAP identifies these flag features as having the largest positive Shapley contributions to the reconstruction error.

### Q20: Explain the mathematical property of "Dummy Player" (Missingness) in Shapley value allocation.
**A**: If a player (feature $i$) contributes nothing to any coalition (subset $S$):
$$f_x(S \cup \{i\}) = f_x(S), \ \forall S \subseteq F \setminus \{i\}$$
then its allocated Shapley value is strictly zero: $\phi_i(x) = 0$. This ensures the model does not attribute significance to redundant features.

### Q21: How does the combinatorial weight coefficient in the Shapley formula ensure fairness?
**A**: The coefficient is $\frac{|S|!(|F| - |S| - 1)!}{|F|!}$, which represents the probability of choosing subset $S$ out of all possible permutations of feature sets. This ensures that the order in which features are introduced does not bias the contribution calculation.

### Q22: What is the information theory explanation for the "Accuracy Trap"?
**A**: Under heavy class imbalance (87.1% Benign), the dataset's entropy is low. A simple majority-class classifier minimizes empirical risk without extracting any information from the input features, yielding $87\%$ accuracy while conveying zero mutual information about the target attack classes.

### Q23: How does the "LGBM Veto" mathematically restore lost Precision?
**A**: Stage 1 uses a low threshold to maximize Recall, yielding a high False Positive count ($FP_{S1}$). Stage 2 (LightGBM) acts as a conditional check:
$$P(\text{Attack} | \text{Flagged}) = \frac{TP_{S2}}{TP_{S2} + FP_{S2}}$$
By classifying and filtering out benign anomalies, LightGBM dramatically reduces $FP_{S2}$, mathematically driving up the system's overall Precision.

### Q24: Why does a conservative threshold ($\mu + 3\sigma$) on the Autoencoder yield poor recall on stealthy attacks like Botnets?
**A**: Stealthy attacks like Botnets sit close to the normal data manifold. Their reconstruction errors are only slightly larger than benign traffic variance. Setting the threshold at $\mu + 3\sigma$ positions the boundary far out in the tail of the distribution, making the model blind to these subtle, low-error anomalies.

### Q25: Explain how the F1-Optimal Threshold ($0.001068$) balances false alarms with detection coverage.
**A**: By maximizing the harmonic mean of precision and recall (F1-score), the threshold sweep finds the exact intersection where the marginal gain in True Positives (catching more stealthy attacks) is balanced against the marginal cost of False Positives (generating false alarms), optimizing overall system utility.

### Q26: What is the relationship between the Kullback-Leibler (KL) Divergence and Cross-Entropy Loss?
**A**: Cross-entropy loss $\mathcal{H}(q, p)$ is the sum of the entropy of the true distribution $\mathcal{H}(q)$ and the KL divergence between the true distribution $q$ and predicted distribution $p$:
$$\mathcal{H}(q, p) = \mathcal{H}(q) + \mathcal{D}_{KL}(q \| p)$$
Since the true target labels are fixed, $\mathcal{H}(q)$ is a constant, meaning minimizing Cross-Entropy loss is mathematically equivalent to minimizing the KL divergence.

### Q27: How does SHAP TreeExplainer optimize Shapley calculations for decision trees?
**A**: Calculating exact Shapley values requires evaluating $2^{|F|}$ feature combinations, which is computationally expensive. `TreeExplainer` exploits the tree structure to evaluate all feature subsets in polynomial time $\mathcal{O}(T \cdot L \cdot D^2)$ (where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum tree depth) by tracking paths recursively.

### Q28: Explain the mathematical property of "Symmetry" in SHAP.
**A**: The symmetry axiom states that if two features $i$ and $j$ make identical marginal contributions to all feature subsets:
$$f_x(S \cup \{i\}) = f_x(S \cup \{j\}), \ \forall S \subseteq F \setminus \{i, j\}$$
then their calculated Shapley values must be identical: $\phi_i(x) = \phi_j(x)$.

### Q29: Why is the PR AUC metric (Average Precision) robust to changes in the number of True Negatives?
**A**: Average Precision is defined strictly in the Precision-Recall metric space. Precision depends only on $TP$ and $FP$, and Recall depends only on $TP$ and $FN$. Since True Negatives ($TN$) do not appear in any of these formulas, the metric remains stable and unaffected by the size of the benign majority class.

### Q30: If an examiner asks why you used SHAP instead of Gini Importance (feature split counts) from LightGBM, what is your answer?
**A**: Gini Importance only measures the global reduction in impurity provided by a feature across the training set, which is biased toward continuous, high-cardinality features. SHAP is built on game-theoretic Shapley values, providing mathematically consistent, locally accurate feature contributions for each individual prediction.
