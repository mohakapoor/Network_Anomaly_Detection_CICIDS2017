# Machine Learning Model Architectures: Theoretical Foundations

This document provides a detailed theoretical analysis of the machine learning algorithms, architectural differences, and underlying mathematics of the models covered in this project. 

---

## 1. Supervised Learning Paradigms: Theory and Mechanics

Supervised learning assumes a closed-world setting: given a labeled dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the goal is to learn a mapping function $f: X \to Y$ that minimizes an empirical loss function.

### 1.1 Gradient Boosting Decision Trees (GBDT)
Gradient Boosting is an additive ensemble technique. It models the target function as a sum of base learners (typically shallow decision trees):
$$F_M(x) = \sum_{m=1}^M \gamma_m h_m(x)$$
where $h_m(x)$ are base classifiers and $\gamma_m$ are step sizes.

* **Sequential Optimization**: Unlike bagging (e.g., Random Forest), where trees are trained independently in parallel, GBDT trains trees sequentially.
* **Residual Fitting**: At step $m$, the new tree $h_m(x)$ is trained to predict the pseudo-residuals, which are the negative gradients of the loss function $\mathcal{L}(y, F_{m-1}(x))$ evaluated at the current model's predictions:
  $$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
* **Step Size (Shrinkage)**: The prediction is updated as:
  $$F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x)$$
  where $\eta \in (0, 1]$ is the learning rate, regularizing the contribution of each subsequent tree.

---

### 1.2 LightGBM (Light Gradient Boosting Machine)
LightGBM is a specialized GBDT framework engineered for high efficiency and scalability. It introduces several key algorithmic optimizations:

#### A. Leaf-Wise (Best-First) Tree Growth
Traditional GBDT frameworks grow trees level-by-level (level-wise). LightGBM grows trees by splitting the leaf node that yields the maximum reduction in loss (leaf-wise), regardless of its depth.
* **Level-Wise**: Splits all nodes at a given depth synchronously. Easy to parallelize and prevents overfitting, but can waste calculations on leaves with negligible gradients.
* **Leaf-Wise**: Focuses splits on high-gradient leaves. This yields asymmetric, deeper trees that reduce training loss faster, though it requires strict regularization (e.g. `max_depth` and `min_data_in_leaf`) to prevent overfitting.

#### B. Histogram-Based Decision Tree Algorithm
LightGBM discretizes continuous floating-point features into $K$ discrete bins (typically $256$).
* **Benefit**: Reduces the computational complexity of finding optimal split points from $\mathcal{O}(N \times d)$ to $\mathcal{O}(K \times d)$ (where $N$ is the number of data instances, $d$ is the number of features, and $K$ is the number of bins), while reducing memory consumption because bin indices can be stored as 8-bit integers.


#### C. Gradient-Based One-Side Sampling (GOSS)
To reduce the number of training instances without losing accuracy:
* Instances with larger gradients (errors) contribute more to information gain calculation.
* GOSS keeps all instances with large gradients and performs random sampling on instances with small gradients.
* It compensates for the sample reduction by scaling the sampled small-gradient instances during split calculation.

#### D. Exclusive Feature Bundling (EFB)
High-dimensional tabular datasets are often highly sparse. EFB groups mutually exclusive features (features that rarely take non-zero values simultaneously) into a single feature bundle, reducing the total number of features evaluated during split searches.

---

### 1.3 XGBoost (Extreme Gradient Boosting)
XGBoost is a highly regularized implementation of GBDT.

#### A. Regularized Objective Function
XGBoost adds L1 ($\alpha$) and L2 ($\lambda$) regularization directly into the objective function to penalize model complexity and leaf weights:
$$\mathcal{L}^{(t)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
where:
$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$
* $T$ is the number of leaves.
* $w$ represents the vector of leaf weights.
* $\gamma$, $\lambda$, and $\alpha$ are regularizing hyperparameters.

#### B. Second-Order Taylor Expansion
To optimize custom loss functions, XGBoost approximates the objective using a second-order Taylor expansion:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
where $g_i$ and $h_i$ are the first (gradient) and second-order (Hessian) derivatives of the loss function:
$$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$

#### C. Level-Wise Tree Growth
XGBoost splits nodes level-by-level. This limits the maximum depth of the tree uniformly, ensuring stable leaf distributions and reducing the likelihood of generating highly localized leaf nodes that overfit.

---

### 1.4 Multilayer Perceptron (Feed-Forward Neural Network)
A Multilayer Perceptron (MLP) is a class of feedforward artificial neural network consisting of an input layer, one or more hidden layers, and an output layer.

```
Input Vector x ──► [ W1*x + b1 ] ──► Activation (ReLU) ──► [ W2*h1 + b2 ] ──► Activation (Softmax) ──► Probability y
```

#### A. Layer Transitions and Activations
Each layer computes a linear transformation followed by a non-linear activation:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g(z^{[l]})$$
* **ReLU (Rectified Linear Unit)**: Typically used in hidden layers:
  $$g(z) = \max(0, z)$$
  It avoids vanishing gradients for positive inputs and accelerates optimization.
* **Softmax**: Used in the output layer for multi-class classification:
  $$a_i^{[L]} = \frac{e^{z_i^{[L]}}}{\sum_{j=1}^C e^{z_j^{[L]}}}$$
  It outputs a valid probability distribution across $C$ target classes.

#### B. Optimization Mechanics
* **Backpropagation**: Calculates the partial derivative of the loss function $\mathcal{L}$ with respect to each weight ($W$) and bias ($b$) in the network using the chain rule.
* **Adam (Adaptive Moment Estimation)**: Computes adaptive learning rates for each parameter by maintaining exponential moving averages of both the gradients ($m_t$, first moment) and the squared gradients ($v_t$, second moment):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

#### C. Regularization Layers
* **Batch Normalization**: Normalizes layer inputs across a mini-batch to speed up training, reduce internal covariate shift, and stabilize weight updates.
* **Dropout**: Randomly zeroes out a fraction ($p$) of layer activations during forward passes, preventing co-adaptation of weights and encouraging redundancy.

---

### 1.5 Support Vector Machine (SVM)
SVM constructs an optimal hyperplane in a high-dimensional space to separate classes.

#### A. Margin Maximization
The decision boundary is defined by the hyperplane:
$$w^T x + b = 0$$
The SVM optimization problem aims to maximize the geometric margin $\frac{2}{\|w\|}$, which is equivalent to minimizing:
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \ge 1, \ \forall i$$

#### B. The Kernel Trick
If data is not linearly separable in the original input space, it is mapped to a higher-dimensional space $\phi(x)$. The dual formulation of the optimization problem depends only on the dot product of vectors, allowing the use of a Kernel function to bypass explicit projection:
$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$
* **Linear Kernel**: $K(x_i, x_j) = x_i^T x_j$
* **RBF (Radial Basis Function) Kernel**: $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$

---

## 2. Unsupervised Anomaly Detection: Theory and Mechanics

Unsupervised anomaly detection operates in an open-world setting, assuming the training data consists of normal samples, and the goal is to evaluate if test samples deviate from this baseline.

### 2.1 Autoencoders (Reconstruction-Based)
An Autoencoder is a neural network designed to learn a compressed representation (encoding) of input data and reconstruct it (decoding) with minimal error.

```
Input x ──► [ Encoder (Weights We) ] ──► Bottleneck Latent Space z ──► [ Decoder (Weights Wd) ] ──► Reconstruction x̂
```

#### A. Mathematical Framework
* **Encoder**: Maps the high-dimensional input $x \in \mathbb{R}^D$ to a low-dimensional bottleneck space $z \in \mathbb{R}^d$ (where $d \ll D$):
  $$z = \sigma_e(W_e x + b_e)$$
* **Decoder**: Maps the latent representation $z$ back to the original dimension space $\hat{x} \in \mathbb{R}^D$:
  $$\hat{x} = \sigma_d(W_d z + b_d)$$
* **Loss Function**: The network parameters are optimized by minimizing the Mean Squared Error (MSE) between the input and its reconstruction:
  $$\mathcal{L}_{MSE}(x, \hat{x}) = \frac{1}{D} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$

#### B. Denoising Autoencoder (DAE)
Standard autoencoders can learn the identity function without extracting structural features. A Denoising Autoencoder adds a corruption step:
1. Map the clean input $x$ to a corrupted version $\tilde{x}$ using a conditional distribution $q(\tilde{x}|x)$ (e.g., adding Gaussian noise: $\tilde{x} = x + \epsilon$).
2. The model maps the corrupted input $\tilde{x}$ through the encoder and decoder to generate a reconstruction $\hat{x} = f_d(f_e(\tilde{x}))$.
3. The loss is computed against the **clean target $x$**:
   $$\mathcal{L}_{DAE}(x, \hat{x}) = \|x - \hat{x}\|^2$$
This forces the model to map the manifold of normal data and learn to project out-of-distribution noise back onto this manifold.

---

### 2.2 Isolation Forest (Partition-Based)
Isolation Forest isolates anomalies instead of profiling normal points.

#### A. Recursive Space Partitioning
Anomalies sit in sparse regions and are geometrically distant from dense clusters.
* **Mechanism**: The algorithm constructs an ensemble of Isolation Trees (iTrees). For each tree:
  1. A feature $q$ is selected at random from the feature space.
  2. A split value $p$ is selected at random between the minimum and maximum values of feature $q$.
  3. The dataset is partitioned into two subsets based on whether values are less than or greater than $p$.
  4. This process is repeated recursively until all data points are isolated.

#### B. Path Length and Anomaly Score
* **Path Length $h(x)$**: The number of edges $x$ traverses from the root of an iTree to the terminating leaf node.
* **Expected Path Length**: The average path length $\mathbb{E}(h(x))$ is computed across a forest of $H$ trees.
* **Normalization Factor $c(n)$**: The average path length of an unsuccessful search in a Binary Search Tree (BST) built on $n$ instances:
  $$c(n) = 2\ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$
* **Anomaly Score $s(x, n)$**:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
* **Evaluation**:
  * If $\mathbb{E}(h(x)) \to 0$ (very short path), then $s(x, n) \to 1$: The instance is easily isolated $\to$ **Anomaly**.
  * If $\mathbb{E}(h(x)) \to n-1$ (deep search path), then $s(x, n) \to 0$: The instance is hard to isolate $\to$ **Normal**.

---

## 3. General Theoretical Comparison Matrix

| Algorithmic Paradigm | Model | Core Mathematical Operation | Optimization Method | Primary Structural Sensitivity |
| :--- | :--- | :--- | :--- | :--- |
| **Supervised Tree** | **LightGBM** | Leaf-wise gradient split | Sequential residual reduction | Discretized step-boundaries |
| **Supervised Tree** | **XGBoost** | Level-wise regularized split | Second-order Taylor optimization | Symmetrical regularized partitions |
| **Supervised Neural** | **FFNN** | Dense matrix transformations | Backpropagation (Adam / SGD) | Continuous curved hyperplanes |
| **Supervised Margin** | **SVM** | Hyperplane projection | Quadratic programming (Dual) | Margin separating support vectors |
| **Unsupervised Neural**| **Autoencoder**| Bottleneck dimensional mapping| Backpropagation (Adam / SGD) | Latent manifold reconstruction error |
| **Unsupervised Tree** | **Isol. Forest**| Recursive split profiling | Random bina## 🎯 30 General Crux Viva Questions & Answers

### Q1: What is the fundamental optimization difference between Bagging and Boosting?
**A**: Bagging (e.g. Random Forest) trains multiple base estimators in parallel and averages their predictions to reduce variance. Boosting (e.g. GBDT) trains estimators sequentially, where each new base model is trained to minimize the residual errors of the combined prior models, primarily reducing bias.

### Q2: What is the pseudo-residual in Gradient Boosting, and why does it represent a gradient?
**A**: The pseudo-residual is the negative derivative of the loss function with respect to the model's current prediction:
$$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]$$
For Mean Squared Error, this derivative simplifies to the raw error $(y_i - F(x_i))$, proving that fitting residuals is mathematically equivalent to performing gradient descent in function space.

### Q3: Why does leaf-wise tree growth reduce loss faster than level-wise tree growth?
**A**: Leaf-wise growth splits the single node that provides the maximum information gain (loss reduction) across the entire tree, regardless of its level. Level-wise growth forces all nodes at a given depth level to split, spending computations on nodes that yield negligible loss reductions.

### Q4: What are the regularization methods used in LightGBM to prevent leaf-wise overfitting?
**A**: Overfitting is regularized by setting:
1. `max_depth`: Limits the maximum depth of any branch.
2. `min_data_in_leaf`: Binds the minimum number of data instances a leaf node must contain before a split can occur.
3. `num_leaves`: Directly bounds the maximum number of leaves allowed in a single tree.

### Q5: How does the histogram-based split finder in LightGBM optimize memory usage?
**A**: Standard split algorithms sort all unique values of each continuous feature, requiring 32-bit or 64-bit float storage per row. LightGBM maps values into discrete bins (e.g. 256 bins), allowing the data to be represented as 8-bit integers, which reduces memory usage by up to $80\%$.

### Q6: What is Gradient-Based One-Side Sampling (GOSS)?
**A**: GOSS is a sampling method that reduces training rows. It retains all data points with large gradients (as they represent high training errors) and performs random sampling on points with small gradients. This preserves split accuracy while training on a fraction of the dataset.

### Q7: What is the objective of EFB (Exclusive Feature Bundling)?
**A**: Tabular data is often highly sparse with mutually exclusive features (features that rarely take non-zero values at the same time). EFB bundles these features into a single composite feature, reducing the feature dimension space and accelerating split scans.

### Q8: Why does XGBoost use a second-order Taylor expansion of the loss function?
**A**: It allows the optimization framework to accept any custom, twice-differentiable loss function. By using both the first-order gradient ($g$) and the second-order Hessian ($h$), the algorithm computes exact leaf weights and split scores independently of the specific loss implementation.

### Q9: What is the regularized objective function of XGBoost?
**A**: It adds L1 ($\alpha$) and L2 ($\lambda$) weight penalties, along with a leaf count penalty ($\gamma$), directly into the objective function:
$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum w^2 + \alpha \sum |w|$$
This restricts leaf weight magnitudes and tree complexity, preventing overfitting during boosting iterations.

### Q10: How does a Multilayer Perceptron (MLP) learn non-linear boundaries?
**A**: By inserting non-linear activation functions (such as ReLU or Sigmoid) between successive linear transformation layers ($W^T x + b$). Without non-linear activations, any cascade of dense layers would collapse mathematically into a single linear transformation.

### Q11: Explain the Vanishing Gradient problem and how ReLU resolves it.
**A**: In deep networks using Sigmoid or Tanh activations, the derivative values are very small (maximum 0.25). Multiplying these derivatives during backpropagation causes gradients to diminish exponentially as they route back to early layers, halting training. The ReLU activation ($\max(0, x)$) has a constant gradient of $1$ for all positive inputs, preventing vanishing gradients.

### Q12: Why is the Softmax activation function mandatory for multi-class classification?
**A**: Softmax normalizes raw linear scores ($z_i$) into values between 0 and 1 that sum strictly to 1:
$$P(y=i|x) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
This transforms the network outputs into a mathematically valid probability distribution, allowing categorical cross-entropy loss computation.

### Q13: How does the Adam optimizer combine the principles of Momentum and RMSProp?
**A**: Adam maintains an exponential moving average of past gradients (Momentum, to damp oscillations) and an exponential moving average of past squared gradients (RMSProp, to adapt the learning rate per parameter):
* Momentum: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
* RMSProp: $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
It scales weight updates by dividing by the square root of $v_t$, ensuring sparse features receive larger updates.

### Q14: Explain the operational mechanics of Batch Normalization.
**A**: Batch Normalization normalizes the activation outputs $x$ of a layer across a mini-batch:
$$\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
It then scales and shifts the normalized activation using two learnable parameters ($\gamma$ and $\beta$):
$$y = \gamma \hat{x} + \beta$$
This stabilizes the distribution of inputs to subsequent layers, speeding up convergence.

### Q15: How does Dropout regularize a Neural Network during training?
**A**: During each training pass, Dropout deactivates a random percentage of neurons. This prevents co-adaptation of weights, forcing the network to learn redundant, robust feature representations, and acts as an ensemble method by implicitly training many sub-networks.

### Q16: How does a Support Vector Machine (SVM) define its optimal margin?
**A**: SVM defines the margin as the shortest perpendicular distance between the separating hyperplane ($w^T x + b = 0$) and the closest training points from both classes (support vectors). Maximizing this margin $\frac{2}{\|w\|}$ minimizes generalization error.

### Q17: What is the Dual formulation of SVM, and why is it useful?
**A**: The dual formulation converts the optimization problem into one that depends only on the dot products of the training vectors ($x_i^T x_j$). This allows the use of the Kernel Trick, enabling the model to learn non-linear boundaries in higher dimensions without explicitly computing high-dimensional coordinate mappings.

### Q18: What is the "Kernel Trick" in SVM?
**A**: It is a mathematical method that computes the inner product of data points projected into a higher-dimensional space without physically computing the high-dimensional coordinates of the projection $\phi(x)$:
$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

### Q19: Explain the mathematical difference between a Linear and RBF Kernel in SVM.
**A**: The Linear kernel calculates the simple dot product in the original space: $K(x_i, x_j) = x_i^T x_j$. The RBF (Radial Basis Function) kernel measures similarity using a Gaussian distribution, mapping the data into an infinite-dimensional Hilbert space:
$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

### Q20: What is the role of a bottleneck layer in an Autoencoder?
**A**: The bottleneck is a hidden layer with a smaller dimension than the input layer. It acts as an information constraint, forcing the network to compress the data and discard noise, retaining only the most critical latent features needed to reconstruct the input.

### Q21: What is the loss function of a basic Autoencoder, and how is it optimized?
**A**: Mean Squared Error (MSE Loss) between the original input $x$ and the decoded output $\hat{x}$:
$$\mathcal{L} = \frac{1}{D}\sum_{i=1}^D (x_i - \hat{x}_i)^2$$
It is optimized via backpropagation, updating the encoder and decoder weights to minimize reconstruction error.

### Q22: Why is a Denoising Autoencoder superior to a standard Autoencoder for manifold learning?
**A**: A standard Autoencoder with high capacity can simply copy inputs to outputs without extracting structural representations. Adding noise forces the network to learn the structural manifold of the data, mapping corrupted inputs back onto the clean, normal manifold.

### Q23: Why is a Sigmoid activation function preferred in the final layer of an Autoencoder?
**A**: Sigmoid bounds outputs strictly between 0 and 1:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
If the input data is scaled to $[0,1]$ (e.g. via MinMaxScaler), Sigmoid ensures the reconstructed outputs fall into the exact same range, stabilizing reconstruction loss.

### Q24: What is the core mechanical difference between a reconstruction-based and a partition-based anomaly detector?
**A**: Reconstruction-based models (Autoencoder) learn to profile the normal data manifold and flag deviations based on high reconstruction error. Partition-based models (Isolation Forest) slice the feature space randomly to isolate outliers, flagging points that require very few partitions to isolate.

### Q25: How does Isolation Forest isolate anomalies?
**A**: It constructs a forest of binary search trees by randomly selecting features and random split points. Outliers reside in sparse regions of the input space and require very few random splits to isolate. Normal points sit in dense regions and require many splits to separate.

### Q26: Gotcha Q: If a tree model like LightGBM achieves a 99.9% F1-score on Thursday's attacks, why can it completely fail to detect Botnets on Friday?
**A**: This is due to **feature misalignment** or overfitting to transient signatures. If the model relies heavily on high-frequency identifier features—like `Destination Port` or specific IP fragments—it learns static rules (e.g., "traffic on port 8080 is an attack"). If the botnet on Friday communicates via a different port (e.g., port 443 or dynamic high ports), the model fails to flag it. This is why we must prune non-generalizable network identifiers and force models to rely on invariant statistical timing and volume metrics.

### Q27: Gotcha Q: Why did you choose a bottleneck dimension of 16 for the Autoencoder? What happens if you set it to 1, or to 60?
**A**: Setting it to 1 creates an **extreme information bottleneck**, causing severe underfitting. The network cannot capture the variance of normal benign traffic, resulting in high reconstruction error even for legitimate flows. Setting it to 60 removes the bottleneck constraint. The network learns a trivial **identity mapping** (simply copying inputs to outputs), reconstructing both benign and attack flows perfectly, which completely destroys the model's ability to detect anomalies.

### Q28: Gotcha Q: If Isolation Forest splits features randomly, could it select a zero-variance feature and get stuck?
**A**: It will not get stuck, but it will waste tree depth. A constant, zero-variance feature has $x_{min} = x_{max}$. The random split generator attempts to select a split point $p \in [x_{min}, x_{max}]$, which forces $p = x_{min}$. All instances will fall to the right of this split point, resulting in an empty left child and no reduction in sample size, wasting a tree partition level. This highlights why our preprocessing pipeline must prune all zero-variance columns before training.

### Q29: Gotcha Q: Can the final layer of your Autoencoder reconstruct values outside the range $[0, 1]$ if they appear in the test set?
**A**: No, the final layer uses a Sigmoid activation, which mathematically restricts outputs to $(0, 1)$. If a test sample contains highly anomalous out-of-bounds features (which scale to $> 1$), the decoder outputs will saturate at $1.0$, creating a massive reconstruction error. This limitation is actually a **mathematical advantage**: the saturation guarantees a high reconstruction loss for extreme outliers, making them easily detectable.

### Q30: Gotcha Q: If you train a Deep Neural Network with 15 hidden layers for network flow classification, why does it perform worse than a shallow 2-layer `[128, 64]` network?
**A**: Tabular network flow features lack the local spatial and temporal hierarchical correlations found in images (pixels) or audio (waveforms). A 15-layer deep network suffers from severe overfitting (memorizing training set noise) and suffers from decaying gradients in early layers, whereas a shallow 2-layer network (`[128, 64]`) has sufficient capacity to learn the statistical boundaries of the 69 features without introducing redundant complexity.
