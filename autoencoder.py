import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── Paths ──────────────────────────────────────────────────────
TRAIN_PATH = Path(r"final\train_us.parquet")
TEST_PATH  = Path(r"final\test_us.parquet")
OUT_DIR    = Path("models/autoencoder")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT  = OUT_DIR / "autoencoder_model.pth"

# ── Hyperparameters ────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 256
LR         = 1e-3
EPOCHS     = 100
VALID_RATIO = 0.1          # 10% of benign data for validation
PATIENCE   = 10            # early stopping patience
DROPOUT    = 0.2
NUM_WORKERS = 4
THRESHOLD_K = 3            # threshold = mean + k * std

USE_GPU = torch.cuda.is_available()
DEVICE  = torch.device("cuda" if USE_GPU else "cpu")


# ── Seed ───────────────────────────────────────────────────────
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed()


# ══════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ══════════════════════════════════════════════════════════════
print("=" * 50)
print("LOADING DATA")
print("=" * 50)

train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# Filter benign-only for autoencoder training
benign_df = train_df[train_df['Attack'] == 0].drop(columns=['Attack'])
print(f"Benign samples for training: {len(benign_df)}")

# Split benign into train (90%) and validation (10%)
X_train, X_val = train_test_split(benign_df, test_size=VALID_RATIO, random_state=SEED)
print(f"Benign train: {len(X_train)} | Benign val: {len(X_val)}")

# Prepare test set (keep all — benign + attacks)
X_test = test_df.drop(columns=['Attack'])
y_test = test_df['Attack'].values

NUM_FEATURES = X_train.shape[1]
print(f"Number of features: {NUM_FEATURES}")


# ══════════════════════════════════════════════════════════════
#  2. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════
class AEDataset(Dataset):
    """Dataset for autoencoder — returns only X (target is X itself)."""
    def __init__(self, X):
        self.X = torch.tensor(X.values if hasattr(X, 'values') else X,
                              dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


train_ds = AEDataset(X_train)
val_ds   = AEDataset(X_val)
test_ds  = AEDataset(X_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ══════════════════════════════════════════════════════════════
#  3. AUTOENCODER MODEL
# ══════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    """
    Symmetric autoencoder:
      Encoder: in_dim → 128 → 64 → 32 (bottleneck)
      Decoder: 32 → 64 → 128 → in_dim
    
    Sigmoid output because data is MinMaxScaled to [0, 1].
    """
    def __init__(self, in_dim, hidden1=128, hidden2=64, bottleneck=32, dropout=DROPOUT):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, bottleneck),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, in_dim),
            nn.Sigmoid(),   # output bounded [0, 1] to match MinMaxScaler
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = Autoencoder(NUM_FEATURES).to(DEVICE)
print(f"\nModel on: {DEVICE}")
print(model)


# ══════════════════════════════════════════════════════════════
#  4. TRAINING
# ══════════════════════════════════════════════════════════════
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

history = {'train_loss': [], 'val_loss': []}


def train_one_epoch():
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)

        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)  # target is the input itself

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate(loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            running_loss += loss.item() * batch.size(0)
    return running_loss / len(loader.dataset)


print("\n" + "=" * 50)
print("TRAINING AUTOENCODER")
print("=" * 50)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    val_loss   = evaluate(val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"LR: {current_lr:.2e}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# Load best model
model.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
print(f"\nLoaded best model from {MODEL_OUT}")


# ══════════════════════════════════════════════════════════════
#  5. THRESHOLD COMPUTATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("COMPUTING THRESHOLD")
print("=" * 50)


def compute_reconstruction_errors(loader):
    """Compute per-sample MSE reconstruction errors."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            reconstruction = model(batch)
            # Per-sample MSE: average across features, one error per sample
            mse = torch.mean((batch - reconstruction) ** 2, dim=1)
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


val_errors = compute_reconstruction_errors(val_loader)

threshold_mean = val_errors.mean()
threshold_std  = val_errors.std()
threshold = threshold_mean + THRESHOLD_K * threshold_std

# Also compute percentile-based thresholds for comparison
threshold_95 = np.percentile(val_errors, 95)
threshold_99 = np.percentile(val_errors, 99)

print(f"Benign validation errors — mean: {threshold_mean:.6f}, std: {threshold_std:.6f}")
print(f"Threshold (mean + {THRESHOLD_K}*std): {threshold:.6f}")
print(f"Threshold (95th percentile):  {threshold_95:.6f}")
print(f"Threshold (99th percentile):  {threshold_99:.6f}")

# Save thresholds
with open(OUT_DIR / "threshold.txt", "w") as f:
    f.write(f"mean: {threshold_mean:.6f}\n")
    f.write(f"std: {threshold_std:.6f}\n")
    f.write(f"k: {THRESHOLD_K}\n")
    f.write(f"threshold (mean + k*std): {threshold:.6f}\n")
    f.write(f"threshold (95th percentile): {threshold_95:.6f}\n")
    f.write(f"threshold (99th percentile): {threshold_99:.6f}\n")
print(f"Saved thresholds to {OUT_DIR / 'threshold.txt'}")


# ══════════════════════════════════════════════════════════════
#  6. EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("EVALUATING ON TEST SET")
print("=" * 50)

test_errors = compute_reconstruction_errors(test_loader)

# Binary classification: error > threshold → anomaly (1), else benign (0)
predictions = (test_errors > threshold).astype(int)
true_labels = (y_test != 0).astype(int)  # 0=benign → 0, any attack → 1

# Overall binary report
report = classification_report(true_labels, predictions,
                               target_names=["Benign", "Anomaly"],
                               zero_division=0)
print("\nBinary Classification Report (Benign vs Anomaly):")
print(report)

accuracy = accuracy_score(true_labels, predictions)
print(f"Overall Accuracy: {accuracy:.4f}")

# Per-attack-type recall
print("\nPer-Attack-Type Recall:")
print("-" * 40)

attack_names = {0: "BENIGN", 1: "Bot", 2: "Brute Force", 3: "DDoS",
                4: "DoS", 5: "PortScan", 6: "Web Attack"}

per_attack_results = {}
for attack_id in sorted(test_df['Attack'].unique()):
    mask = y_test == attack_id
    name = attack_names.get(attack_id, f"Attack_{attack_id}")
    count = mask.sum()

    if attack_id == 0:
        # For benign: recall = what % correctly identified as benign (NOT anomaly)
        recall = 1 - predictions[mask].mean()
        print(f"  {name:15s} | Correctly identified: {recall:.4f} | Count: {count}")
    else:
        # For attacks: recall = what % flagged as anomaly
        recall = predictions[mask].mean()
        print(f"  {name:15s} | Recall: {recall:.4f} | Count: {count}")

    per_attack_results[name] = {'recall': recall, 'count': count}

# Save text report
with open(OUT_DIR / "classification_reports.txt", "w") as f:
    f.write("Binary Classification Report (Benign vs Anomaly)\n")
    f.write("=" * 50 + "\n")
    f.write(report + "\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
    f.write("Per-Attack-Type Recall\n")
    f.write("=" * 50 + "\n")
    for name, res in per_attack_results.items():
        f.write(f"  {name:15s} | Recall: {res['recall']:.4f} | Count: {res['count']}\n")
print(f"\nSaved reports to {OUT_DIR / 'classification_reports.txt'}")


# ══════════════════════════════════════════════════════════════
#  7. PLOTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("GENERATING PLOTS")
print("=" * 50)


## Loss Plot
def plot_loss(history):
    """Plots training and validation loss across epochs."""
    plt.figure(figsize=(10, 6))
    epochs_ran = len(history['train_loss'])
    epochs_range = range(1, epochs_ran + 1)

    plt.plot(epochs_range, history['train_loss'], label='Training Loss',
             marker='o', linestyle='--', markersize=3)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss',
             marker='o', markersize=3)

    plt.title('Autoencoder — Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = OUT_DIR / "loss_plot.png"
    plt.savefig(out_path)
    print(f"Saved loss plot to: {out_path}")
    plt.close()


## Reconstruction Error Distribution
def plot_error_distribution(test_errors, y_test, threshold):
    """Plots reconstruction error histograms: benign vs each attack type."""
    plt.figure(figsize=(12, 7))

    benign_errors = test_errors[y_test == 0]
    attack_errors = test_errors[y_test != 0]

    plt.hist(benign_errors, bins=100, alpha=0.6, label='Benign', color='steelblue', density=True)
    plt.hist(attack_errors, bins=100, alpha=0.6, label='Attacks', color='crimson', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.4f})')

    plt.title('Reconstruction Error Distribution — Benign vs Attacks')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUT_DIR / "reconstruction_error_distribution.png"
    plt.savefig(out_path)
    print(f"Saved error distribution plot to: {out_path}")
    plt.close()


## Per-Attack Recall Bar Chart
def plot_per_attack_recall(per_attack_results):
    """Bar chart of recall per attack type."""
    names = [k for k in per_attack_results if k != "BENIGN"]
    recalls = [per_attack_results[k]['recall'] for k in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, recalls, color='steelblue', edgecolor='black')

    for bar, val in zip(bars, recalls):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=10)

    plt.title('Autoencoder — Per-Attack-Type Recall')
    plt.xlabel('Attack Type')
    plt.ylabel('Recall')
    plt.ylim(0, 1.15)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = OUT_DIR / "per_attack_recall.png"
    plt.savefig(out_path)
    print(f"Saved per-attack recall plot to: {out_path}")
    plt.close()


plot_loss(history)
plot_error_distribution(test_errors, y_test, threshold)
plot_per_attack_recall(per_attack_results)

print("\n" + "=" * 50)
print("DONE — All outputs saved to 'models/autoencoder/'")
print("=" * 50)