# Topic 3.5: Project Model Details & Hyperparameters

This document details the project-specific model parameters, network layer configurations, search spaces, cross-validation parameters, and the architectural decisions used during the training of the Intrusion Tracker models. It also provides an exhaustive theoretical analysis of what every hyperparameter does under the hood and its mathematical impact.

---

## 1. Validation & Tuning Framework

Supervised models (LightGBM, XGBoost, and the Multilayer Perceptron) were trained using the 34 PCA components extracted during preprocessing. 

### 1.1 Validation Scheme
* **Cross-Validation Strategy**: **Stratified K-Fold Cross-Validation** with $5$ splits ($N_{splits} = 5$).
* **Why Stratified?**: Due to severe class imbalance (e.g. minority classes like Web Attack and Botnet account for less than $1\%$ of the data), standard K-Fold CV can distribute these rare instances unevenly. Stratified K-Fold ensures that each fold contains the exact same percentage of target classes as the entire dataset.
* **Optimization Target**: **Macro F1-Score**. Standard metrics like accuracy or weighted F1 prioritize the majority class. Optimizing for Macro F1 forces the search algorithm to select parameters that perform well on minority attack classes.

---

## 2. Supervised Configurations & Layer Architectures

### 2.1 LightGBM Project Configuration
Tuned using `RandomizedSearchCV` (20 search iterations).

| Hyperparameter | Search Space | Selected Value | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `[5, 10]` | `10` | Anything above 15 estimators yielded $99\%$ cross-validation scores instantly. Limiting it to 10 prevents potential overfitting and keeps inference fast. |
| `num_leaves` | `[20, 31, 50]` | `31` | Controls tree complexity. A setting of 31 provides a balanced tree capacity without generating highly specific leaf paths. |
| `max_depth` | `[6, 8, 10]` | `8` | Bounds tree depth uniformly, regularizing leaf-wise growth to prevent overfitting to noisy details. |
| `learning_rate` | Log-uniform `[0.01, 0.2]` | `0.124` | Step size optimization. Balanced with the low tree count (10 estimators) to ensure rapid error correction. |
| `min_child_samples` | `[10, 30, 50]` | `30` | Minimum samples required in a leaf node. 30 prevents splits that isolate extremely small subsets. |
| `class_weight` | `['balanced']` | `balanced` | Automatically scales class weights inversely proportional to class frequencies, forcing the model to value minority targets. |
| `device` | `['gpu']` | `gpu` | Utilizes GPU acceleration to handle calculations on the 1.7M training rows. |

---

### 2.2 XGBoost Project Configuration
Tuned using `RandomizedSearchCV` (20 search iterations).

| Hyperparameter | Search Space | Selected Value | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| `n_estimators` | `[5, 10]` | `10` | Lower estimator boundaries to prevent overfitting. |
| `max_depth` | `[6, 8, 10]` | `8` | Symmetrical depth restriction to ensure stable leaf bounds. |
| `min_child_weight` | `[10, 30, 50]` | `30` | Regularizes node splitting, requiring significant instance weight. |
| `learning_rate` | Log-uniform `[0.01, 0.2]` | `0.115` | Optimized boosting step size. |

---

### 2.3 Multilayer Perceptron (FFNN) Structural Details
A PyTorch-based neural network trained on the 34 PCA components.

```text
Input (34 PCA Components)
   │
   ▼
Linear (34 ➔ 128) ──► BatchNorm ──► ReLU ──► Dropout (0.2)
   │
   ▼
Linear (128 ➔ 64) ──► BatchNorm ──► ReLU ──► Dropout (0.2)
   │
   ▼
Linear (64 ➔ 7)   ──► Softmax Output
```

#### Layer and Dimension Selection Rationale:
* **Hidden Layers**: Two fully-connected (dense) layers of size **128 and 64**. 
  * *Rationale*: The input dimension is 34. A layer size of 128 projects features into a higher-capacity representation space to model non-linear relationships, which is then gradually reduced to 64 to avoid overfitting before the final 7-class prediction.
* **Dropout (0.2)**: 20% probability of neuron deactivation, preventing individual weight reliance.
* **Loss Function**: `CrossEntropyLoss`.
* **Optimizer**: Adam (`lr = 1e-3`, `weight_decay = 1e-5`).
* **Training Settings**: Mini-batch size of $256$.

---

## 3. Unsupervised Configurations & Layer Architectures

Trained strictly on the **69 raw features of Benign Monday traffic** to construct the normal baseline.

### 3.1 Denoising Autoencoder Structural Details
A symmetric, multi-layer neural network designed to compress and reconstruct the benign feature manifold.

```text
Corrupted Input (69 Features + 10% Noise)
   │
   ▼
Linear (69 ➔ 128) ──► BatchNorm ──► ReLU ──► Dropout (0.2)
   │
   ▼
Linear (128 ➔ 64) ──► BatchNorm ──► ReLU ──► Dropout (0.2)
   │
   ▼
Linear (64 ➔ 16)  ──► ReLU (Bottleneck Space)
   │
   ▼
Linear (16 ➔ 64)  ──► BatchNorm ──► ReLU ──► Dropout (0.2)
   │
   ▼
Linear (64 ➔ 128) ──► BatchNorm ──► ReLU ──► Dropout (0.2)
   │
   ▼
Linear (128 ➔ 69) ──► Sigmoid Output ➔ Reconstruction
```

#### Key Design Decisions and Bottleneck Rationale:
* **Bottleneck Dimension (16)**:
  * *Rationale*: Compressing the 69 features down to a 16-neuron bottleneck forces a **4.3:1 compression ratio**. This restricts the network's capacity, ensuring it ignores minor noise and models only the core shared properties of normal traffic. Bypassing PCA guarantees that non-linear manifold structures are preserved.
* **Denoising Noise ($10\%$)**: 
  * *Rationale*: Gaussian noise with a standard deviation of $0.1$ is injected during training. This forces the model to learn how to project corrupted inputs back onto the clean benign manifold.
* **Sigmoid Activation**: Applied to the final reconstruction layer to guarantee the outputs fall strictly within the normalized range $[0, 1]$, matching the MinMaxScaler inputs.
* **Adam Optimizer**: Initial learning rate of `1e-3` with an L2 weight decay (regularization) of `1e-5`.
* **ReduceLROnPlateau Scheduler**: Automatically halves the learning rate if the validation loss plateaus for 3 epochs, enabling the model to converge smoothly.
* **Early Stopping**: Bounded by `patience = 10` to stop training when the validation loss stops improving, preventing the model from memorizing the normal data.

---

### 3.2 Isolation Forest Project Configuration
Tuned using manual grid search over 48 parameter combinations.

| Hyperparameter | Search Range | Selected Value | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| `max_samples` | `[256, 512, 1024, 2048]` | `2048` | **Crucial Optimization**: The default setting (256) is too small to map the density of 1.7M benign rows. Increasing it to 2048 dramatically boosted ROC AUC to $0.7156$. |
| `n_estimators` | `[100, 200, 300]` | `200` | Number of isolation trees in the forest. 200 ensures stable, converged anomaly scoring. |
| `max_features` | `[0.5, 0.7, 0.8, 1.0]` | `1.0` | Slices all 69 features to guarantee that PortScan outlier features are evaluated in every tree. |
| `contamination`| `[0.01]` | `0.01` | Assumes a baseline noise rate of $1\%$ outlier variance in training, preventing overfitting to anomalous training packets. |

---

## 4. Error Calculations & Optimization Targets (In Our Context)

A key viva topic is explaining exactly **how error/loss is calculated** and **what is minimized** during training to achieve optimal results in this project.

### 4.1 Denoising Autoencoder
* **Training Phase Minimization**: During training (on benign Monday traffic), we minimize the standard **Mean Squared Error (MSE Loss)** across all 69 raw features to force the encoder-decoder weights to reconstruct normal traffic:
  $$\mathcal{L}_{train} = \frac{1}{69} \sum_{j=1}^{69} (x_j - \hat{x}_j)^2$$
  Adam optimizer adjusts weights in the direction of the negative gradient ($-\nabla \mathcal{L}$) to minimize this average difference.
* **Inference Phase Anomaly Scoring**: During testing, we calculate the final error using our **Combined Error Metric**:
  $$\text{Reconstruction Error} = 0.5 \times \left( \frac{1}{69}\sum_{j=1}^{69} (x_j - \hat{x}_j)^2 \right) + 0.5 \times \max_{1 \le j \le 69} |x_j - \hat{x}_j|$$
  This combined loss term is minimized on benign validation sets to calculate the optimal operational alert threshold.

---

### 4.2 Isolation Forest
* **No Gradient Minimization**: Isolation Forest is a partition-based ensemble and does not use standard gradient minimization during training.
* **Operational Minimization (Path Length)**: The model isolates anomalies by calculating the path length $h(x)$ (number of recursive splits) for each instance across its 200 trees. Points that **minimize the average path length** $\mathbb{E}(h(x))$ are isolated near the root of the trees.
* **Target Score**: The algorithm translates this into an anomaly score:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
  where $s(x, n) \to 1$ when path length is minimized, flagging the point as an outlier.

---

### 4.3 LightGBM & XGBoost (Supervised Trees)
* **Objective Function Minimization**: Both tree-based models minimize **Multi-class Categorical Cross-Entropy Loss (Log-Loss)**:
  $$\mathcal{L}_{log} = -\sum_{i=1}^N \sum_{k=0}^6 y_{i,k} \ln(p_{i,k})$$
  where $y_{i,k}$ is a binary indicator (1 if flow $i$ belongs to class $k$, 0 otherwise), and $p_{i,k}$ is the predicted probability of flow $i$ belonging to class $k$.
* **Additive Minimization**: Each tree $f_t$ is sequentially built to minimize the second-order Taylor expansion approximation of this Log-Loss:
  $$\sum_{i=1}^N \left[ g_{ik} f_t(x_i) + \frac{1}{2} h_{ik} f_t^2(x_i) \right] + \Omega(f_t)$$
  where $g_{ik}$ is the first derivative (gradient) and $h_{ik}$ is the second derivative (Hessian) of the loss. Minimizing this objective ensures precise probability alignments for the 7 categorical classes.

---

### 4.4 Multilayer Perceptron (FFNN)
* **Objective Function Minimization**: The MLP directly minimizes the **Categorical Cross-Entropy Loss** over the 7 network states:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^6 y_{i,k} \ln\left(\text{Softmax}(z_{i,k})\right)$$
* **Optimization Target**: Backpropagation calculates the partial derivatives of this loss with respect to all layer weights ($W$) and biases ($b$). The Adam optimizer dynamically updates weights in the negative gradient direction ($-\nabla \mathcal{L}$) to minimize this information distance, forcing the Softmax output distributions to match the true target labels.

---

### 4.5 Binary Support Vector Machine (SVM Baseline)
* **Margin Optimization**: SVM minimizes the regularized soft-margin hinge-loss objective:
  $$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i(w^T x_i + b))$$
* **Objective Target**: Minimizing $\frac{1}{2} \|w\|^2$ maximizes the geometric margin between classes, while the second term minimizes classification errors on the support boundaries, solved via quadratic programming.

---

## 5. Exhaustive Hyperparameter Guide (By Model)

Examiners frequently ask: *"What does this hyperparameter do under the hood, and what is its direct impact on the model?"* Use this section as your definitive reference.

### 5.1 Support Vector Machines (SVM)

#### A. The Regularization Parameter `C`
* **Under the Hood Math**: In soft-margin SVM, we minimize an objective that balances margin maximization with classification errors:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^N \xi_i$$
  where $\xi_i \ge 0$ are slack variables representing the distance of misclassified points from their correct margins.
* **Low `C` (Soft Margin)**: Maximizes the margin width ($\frac{2}{\|w\|}$) while allowing more training classification errors. It regularizes the model, producing a smoother separating boundary that generalizes better.
* **High `C` (Hard Margin)**: Heavily penalizes classification errors. It forces the boundary to classify all training points correctly, resulting in a narrower margin and a complex, highly irregular boundary that is prone to overfitting.

#### B. The RBF Kernel Parameter `gamma` ($\gamma$)
* **Under the Hood Math**: The Radial Basis Function (RBF) kernel is defined as:
  $$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$
  $\gamma$ is inversely proportional to the variance of the Gaussian distribution, controlling the radius of influence of a single support vector.
* **Low `gamma` (Large Radius)**: Support vectors have a wide geographic influence. The model groups distant points together, yielding a smooth, simple separating boundary. If set too low, it causes underfitting.
* **High `gamma` (Small Radius)**: Support vectors have a highly localized influence. Points must be very close to a support vector to be classified in its class, creating complex, wiggly decision boundaries that overfit.

---

### 5.2 Tree-Based Boosting Models (LightGBM & XGBoost)

#### A. `n_estimators` (Number of Boosting Iterations)
* **Under the Hood**: The total count of sequential decision trees built in the ensemble.
* **Impact**: Larger numbers allow the sequential residual-correction process to reduce bias and learn highly complex patterns. However, setting it too high causes the model to start learning noise (overfitting) and increases training/inference latency.

#### B. `learning_rate` ($\eta$)
* **Under the Hood**: A shrinkage factor that scales the raw step contribution of each new tree during weight updates:
  $$F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x)$$
* **Impact**: Regularizes the boosting process. Smaller learning rates (e.g. $0.01$) make updates conservative, requiring a higher `n_estimators` to converge but yielding superior generalization. High learning rates converge fast but run the risk of oscillating or diverging past the global minimum.

#### C. `num_leaves` (LightGBM Only)
* **Under the Hood**: Bounds the maximum number of terminal leaf nodes allowed in a single leaf-wise tree.
* **Impact**: This is the primary parameter controlling LightGBM tree complexity. Under leaf-wise growth, a tree can grow very deep on a single branch. Setting `num_leaves` high (e.g. 50+) allows the tree to capture complex multi-feature interactions but increases the risk of overfitting.

#### D. `max_depth`
* **Under the Hood**: Restricts the maximum vertical depth (number of edges from root to leaf) of any branch in the tree.
* **Impact**: Slices vertical complexity. In LightGBM, it regularizes leaf-wise growth to prevent individual branches from growing excessively deep on small, noisy subsets. In XGBoost, it uniformly limits tree capacity.

#### E. `min_child_samples` (LightGBM) / `min_child_weight` (XGBoost)
* **Under the Hood**: The minimum sum of instance weights (data rows in LightGBM, Hessian values in XGBoost) required in a child node (leaf).
* **Impact**: Highly effective for regularizing trees. If a proposed split isolates fewer than this threshold (e.g. less than 30 samples), the split is aborted. High values prevent the model from learning highly specific rules that isolate individual anomalies.

#### F. `subsample` & `colsample_bytree`
* **Under the Hood**: The fraction of data rows (`subsample`) and feature columns (`colsample_bytree`) randomly sampled to train each individual tree.
* **Impact**: Introduces stochastic variety (similar to Random Forest bagging). By forcing different trees to see different rows and features, it prevents the boosting ensemble from relying too heavily on a few dominant columns or noise rows.

---

### 5.3 Neural Networks (FFNN & Autoencoders)

#### A. `learning_rate` ($\eta$)
* **Under the Hood**: The step scaling factor used by the optimizer (e.g. Adam) to update weights during gradient descent:
  $$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$$
* **Impact**: Too high causes updates to overshoot the minimum, leading to training divergence (loss turning into `NaN`). Too low causes the training to get stuck in local minima or take exponentially long to converge.

#### B. `batch_size`
* **Under the Hood**: The number of training samples processed in a single forward/backward pass before updating weights.
* **Impact**: 
  * **Large Batch**: Computes highly accurate, stable average gradients, but requires more GPU memory and has less stochastic noise, making it easier to get stuck in flat local minima.
  * **Small Batch**: Introduces stochastic gradient noise, which regularizes the weights and helps them escape local minima, but requires more training steps per epoch.

#### C. `dropout` ($p$)
* **Under the Hood**: The probability of randomly deactivating individual neurons during a training pass.
* **Impact**: Forces the network to learn redundant, highly generalized pathways for the data. Too high (e.g. 0.5+) restricts overall network capacity, causing underfitting; too low (e.g. 0.05) fails to prevent weight co-adaptation (overfitting).

#### D. `weight_decay` ($\lambda$)
* **Under the Hood**: Implements L2 regularization by adding a squared weight penalty to the loss function:
  $$\mathcal{L}_{reg} = \mathcal{L} + \frac{1}{2} \lambda \sum w^2$$
* **Impact**: Pulls all weights toward zero. Large weights make a neural network highly sensitive to small variations in the input (causing erratic output swings). Weight decay ensures the model remains smooth and robust.

#### E. `bottleneck` (Autoencoder Only)
* **Under the Hood**: The number of neurons in the central latent layer of the Autoencoder.
* **Impact**: Dictates the information bottleneck. A smaller bottleneck (e.g. 16) forces high compression, stripping away noise and extracting generalized features. However, if set too low (e.g. <5), the model underfits, resulting in high reconstruction error even on normal traffic.

#### F. `noise_factor` / Gaussian Noise ($\sigma$)
* **Under the Hood**: The standard deviation of the random noise added to corrupted inputs during training: $\tilde{x} = x + \mathcal{N}(0, \sigma)$.
* **Impact**: Dictates the corruption level of a Denoising Autoencoder. Higher noise forces the model to learn a highly robust projection function to reconstruct clean manifolds, but too much noise makes it impossible to reconstruct the signal.

---

### 5.4 Isolation Forest

#### A. `n_estimators`
* **Under the Hood**: The number of isolation trees (iTrees) built in the forest.
* **Impact**: More trees yield more stable, converged average path lengths across the forest, smoothing the final anomaly scores.

#### B. `max_samples`
* **Under the Hood**: The number of data instances randomly drawn from the dataset to train each individual isolation tree.
* **Impact**: **Crucial density parameter**. If set too low (like the default 256 on a large dataset), the trees cannot map the true density clusters of normal traffic, causing anomalies to look identical to normal traffic. High values (e.g. 2048) map densities accurately but require more recursive splits to isolate points.

#### C. `contamination`
* **Under the Hood**: The expected proportion of outliers (anomalies) present in the training dataset.
* **Impact**: It mathematically sets the decision threshold boundary. A contamination of $0.01$ sets the score threshold such that the top $1\%$ most isolated points are labeled as anomalies.

---

## 6. Project Configuration Matrix

Here is the exact layout of the parameters and values implemented in your project:

| Model | Hyperparameter | Implemented Value | Direct Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **LightGBM** | `n_estimators` | `10` | Restricts tree count to prevent overfitting on highly distinct signatures. |
| | `num_leaves` | `31` | Balanced node capacity ($2^5-1$) for non-linear leaf-wise splits. |
| | `max_depth` | `8` | Bounds vertical depth, regularizing asymmetric branches. |
| | `min_child_samples`| `30` | Minimum samples in a leaf; prevents isolating noise subsets. |
| | `class_weight` | `balanced` | Automatically scales class losses inverse to frequency. |
| **XGBoost** | `n_estimators` | `10` | Restricts tree count to prevent signature memorization. |
| | `max_depth` | `8` | Symmetrical depth constraint to ensure stable, regularized leaf bounds. |
| | `min_child_weight` | `30` | Prevents splits that isolate small weight nodes. |
| **FFNN** | `Layer Sizes` | `[128, 64]` | Sized to first expand input space (128) then compress it (64). |
| | `dropout` | `0.2` | 20% neuron deactivation to prevent co-adaptation of weights. |
| | `learning_rate` | `1e-3` | Optimized step size for Adam optimizer. |
| **Autoencoder**| `bottleneck` | `16` | Bypasses PCA; forces a **4.3:1 compression** of raw features. |
| | `noise_factor` | `0.1` | Adds 10% Gaussian noise to train robust projection manifolds. |
| | `learning_rate` | `1e-3` (Adam) | Optimizes steps, paired with `ReduceLROnPlateau` scheduler. |
| **Isol. Forest**| `max_samples` | `2048` | Overrides default 256; maps high-density benign data accurately. |
| | `n_estimators` | `200` | Binds 200 trees to ensure stable, converged path lengths. |
| | `contamination` | `0.01` | Assumes a 1% outlier variance rate in benign training data. |

---

## 🎯 30 Crux Viva Questions & Answers

### Q1: Why did you use Stratified K-Fold Cross-Validation rather than standard K-Fold CV?
**A**: Our dataset is highly imbalanced, with minority classes representing less than 1% of the data. Standard K-Fold CV can distribute these rare instances unevenly among folds, causing unstable validation. Stratified K-Fold preserves the exact class ratio in each fold.

### Q2: Why did you choose a split count of $N_{splits} = 5$ for cross-validation?
**A**: Five splits provide a robust balance, training on 80% of the data and validating on 20% in each iteration. It ensures the validation scores are statistically sound while keeping cross-validation runtimes manageable on 1.7M rows.

### Q3: Why was the validation target set to Macro F1-Score during hyperparameter searches?
**A**: Standard accuracy or weighted F1 favors the dominant Benign class. If the model completely fails on minority attacks, the accuracy remains high. Macro F1 weights every class equally, forcing the search algorithm to select hyperparameters that optimize performance on rare attack classes.

### Q4: Why did you limit `n_estimators` to only 10 in LightGBM and XGBoost?
**A**: During initial training, models with more than 15 estimators achieved near-perfect $99\%$ cross-validation scores immediately because the attack signatures are highly distinct. Restricting estimators to 10 limits model complexity, reduces the threat of overfitting, and maximizes real-time inference speed.

### Q5: What is the significance of the parameter `class_weight='balanced'` in LightGBM?
**A**: It automatically assigns weights to classes inversely proportional to their training frequencies. This penalizes the loss function more heavily when the model misclassifies minority attack samples, balancing the learning process.

### Q6: What did the parameter `min_child_samples=30` accomplish in your LightGBM configuration?
**A**: It requires a leaf node to contain at least 30 training samples before a split can occur. This regularizes the tree growth, preventing the leaf-wise algorithm from generating highly specific branches that isolate tiny pockets of noise.

### Q7: Explain why you set `num_leaves=31` for LightGBM.
**A**: `num_leaves` directly limits the maximum structural complexity of the tree. A setting of 31 represents a standard, robust tree capacity ($2^5-1$), providing sufficient non-linear split combinations without causing overfitting.

### Q8: What does the search space parameter `learning_rate` log-uniform `[0.01, 0.2]` represent?
**A**: A log-uniform distribution samples more heavily near the lower boundary ($0.01$). This allows the search framework to focus on small step sizes, which are mathematically proven to yield more stable boosting iterations than high learning rates.

### Q9: Why was the hidden layer dimension of the MLP set to 128 and 64?
**A**: The input has 34 features. Sizing the first hidden layer to 128 projects the inputs into a higher-dimensional capacity space to model complex non-linear combinations. The second layer of 64 gradually compresses this space, regularizing the network before the final 7-class prediction.

### Q10: Why did you use `CrossEntropyLoss` to train your FFNN?
**A**: `CrossEntropyLoss` measures the divergence between the model's Softmax probability distribution and the true target one-hot vector. It penalizes incorrect, highly confident predictions exponentially, yielding stable gradients for multi-class optimization.

### Q11: Explain the purpose of setting the mini-batch size to $256$ in your neural networks.
**A**: A batch size of 256 balanced gradient accuracy and hardware speed. It is large enough to exploit GPU matrix parallelization and provide stable average gradients, yet small enough to maintain the stochastic noise needed to escape local minima.

### Q12: Why did you select a bottleneck dimension of 16 for your Autoencoder?
**A**: The input has 69 features. Compressing this to 16 in the bottleneck forces a **4.3:1 compression ratio**. This restricts the network's capacity, ensuring it ignores minor noise and models only the core shared properties of normal traffic. Bypassing PCA guarantees that non-linear manifold structures are preserved.

### Q13: What does the parameter `weight_decay = 1e-5` do in the Adam optimizer?
**A**: It implements L2 regularization. It adds a small penalty proportional to the sum of the squared weights to the loss function, mathematically pulling weights toward zero. This prevents individual weights from becoming excessively large and overfitting.

### Q14: Explain the role of the `ReduceLROnPlateau` scheduler in your neural networks.
**A**: It monitors the validation loss. If the validation loss fails to improve for 3 consecutive epochs, the scheduler automatically scales down the learning rate by a factor of 0.5. This allows the model to fine-tune its weights as it nears the global minimum.

### Q15: Why did you use `Sigmoid` activation in the final layer of the Autoencoder's decoder?
**A**: Sigmoid bounds outputs strictly between 0 and 1:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
If the input data is scaled to $[0,1]$ (e.g. via MinMaxScaler), Sigmoid ensures the reconstructed outputs fall into the exact same range, stabilizing reconstruction loss.

### Q16: What does `patience = 10` mean in your early stopping setup?
**A**: If the validation loss fails to show any improvement for 10 consecutive training epochs, the training loop is forcefully terminated. The model then restores the weights from the epoch with the lowest validation loss to prevent overfitting.

### Q17: Why did you choose `max_samples=2048` for your Isolation Forest instead of the default `256`?
**A**: The default value of 256 is too small to map the density of 1.7M benign rows, leading to poor partition scoring. Increasing `max_samples` to 2048 allowed the forest to build highly accurate density boundaries, boosting ROC AUC to $0.7156$.

### Q18: What does the parameter `max_features=1.0` do in your Isolation Forest configuration?
**A**: It forces every isolation tree to evaluate all 69 features during random split candidates. Since PortScan anomalies skew only a few specific features, this ensures these critical columns are evaluated in every tree.

### Q19: Explain the contamination rate of $0.01$ (1%) in your Isolation Forest.
**A**: It assumes that up to 1% of the benign training traffic contains minor anomalous noise. This regularizes the forest, preventing it from isolating rare normal packets and ensuring the boundary represents standard normal density.

### Q20: How many parameter combinations were evaluated in your Isolation Forest grid search?
**A**: Forty-eight combinations, crossing `max_samples` [256, 512, 1024, 2048], `max_features` [0.5, 0.7, 0.8, 1.0], and `n_estimators` [100, 200, 300].

### Q21: What was the primary metric used to select the optimal Isolation Forest parameter combination?
**A**: The **Receiver Operating Characteristic Area Under the Curve (ROC AUC)** evaluated on the testing dataset, which measures the model's fundamental ability to separate benign from anomaly traffic across all thresholds.

### Q22: Why was GPU acceleration enabled for LightGBM, but not for Isolation Forest?
**A**: LightGBM was trained using `RandomizedSearchCV` over stratified cross-validation on 1.7M rows, which is computationally heavy and benefits from GPU parallelization. Isolation Forest training does not compute gradients and is parallelized across multiple CPU cores (`n_jobs=-1`) using standard scikit-learn libraries.

### Q23: Why did you choose a batch size of $256$ instead of $4096$ for the Autoencoder?
**A**: A massive batch size like 4096 speeds up training but reduces the stochastic gradient noise, often causing the model to settle in poor local minima. A batch size of 256 preserves optimal convergence behavior.

### Q24: What optimization function did you use to tune the hyperparameters of LightGBM?
**A**: `RandomizedSearchCV` from scikit-learn, configured with 5 stratified folds, 20 random sampling iterations, and scoring optimized for `f1_macro`.

### Q25: Why is `weight_decay` critical for Denoising Autoencoders?
**A**: Since the Denoising Autoencoder must map a clean representation from corrupted inputs, `weight_decay` prevents individual neurons from assigning massive weights to specific noise features, enforcing a smooth reconstruction manifold.

### Q26: Gotcha Q: If your Stratified 5-Fold Cross-Validation gives 99.9% accuracy on Friday's DDoS attacks, why can it still fail completely when deployed on a real network interface?
**A**: This is due to **temporal leakage**. Standard Stratified K-Fold CV randomly shuffles rows across the timeline, splitting packets from the *same* attack session into both training and validation folds. In a real deployment, the model evaluates traffic chronologically. If the model memorized transient timing or IP signatures from the shuffled training fold, it will fail to generalize to the chronological out-of-distribution traffic of the live network.

### Q27: Gotcha Q: Why did you choose exactly 16 bottleneck dimensions? Why not 2, or 60?
**A**: If we set it to 2, we compress the 69 features too severely (underfitting). The network cannot capture the normal variance of benign traffic, generating high reconstruction errors even for legitimate flows. If we set it to 60, we remove the information bottleneck. The network learns a trivial **identity mapping** (simply copying inputs to outputs), reconstructing both benign and attack flows perfectly, which completely destroys the model's ability to detect anomalies.

### Q28: Gotcha Q: If you set your neural network's L2 weight decay parameter (`weight_decay`) to a very high value like 10.0, what happens to the weights and how does this affect the output?
**A**: A weight decay of 10.0 forces the L2 weight penalty to dominate the loss function. The optimizer will minimize the loss by driving all weight coordinates extremely close to zero:
$$W \to 0$$
This collapses the network's capacity. The activations in the hidden layers drop to zero, and the model outputs a constant probability distribution equal to the prior class distribution, losing all classification utility.

### Q29: Gotcha Q: Why does standard MSE reconstruction loss fail to detect highly sparse anomalies in network data compared to our combined MSE + MAE loss?
**A**: MSE squares the errors, meaning it is highly sensitive to a single feature with extreme variance, completely drowning out small, subtle deviations across the other 68 features. Our combined MSE + MAE loss integrates the L1 norm (MAE), which penalizes all feature deviations linearly. This prevents a single dominant feature from masking coordinated, multi-feature anomalies, boosting the model's overall detection sensitivity.

### Q30: Gotcha Q: How does your stratified split handle the Botnet class if it only appears in Thursday's capture and is completely missing on Monday?
**A**: Our preprocessing pipeline combines Monday–Thursday captures into a single training union before executing cross-validation. Stratification partitions this unified pool, ensuring the Botnet class is distributed proportionally ($80\%$ train, $20\%$ validation) across all folds. However, if we attempted a chronological day-by-day split (e.g., training only on Monday–Wednesday), the training set would contain $0$ instances of Botnets, making it mathematically impossible for the supervised model to detect them.
