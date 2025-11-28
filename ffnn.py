#!/usr/bin/env python3
"""
ffnn_tabular_processed.py
Feed-forward NN for multiclass tabular data when your features are already scaled/PCA'd.
"""

import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN_PATH = Path("final/train_mc.parquet")
TEST_PATH  = Path("final/test_mc.parquet")
OUT_DIR    = Path("models/ffnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT  = OUT_DIR / "ffnn_multiclass.pt"

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 100
VALID_RATIO = 0.1        
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
PATIENCE = 8           
DROPOUT = 0.2
NUM_WORKERS = 4          

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed()


train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)

X = train_df.drop(columns=["Attack"])
y = train_df["Attack"].astype(int)

X_test = test_df.drop(columns=["Attack"])   
y_test = test_df["Attack"].astype(int)

# -------------------------
# Train/validation split
# -------------------------
# If your training split is temporal, replace this with a time-based split.
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=VALID_RATIO, stratify=y, random_state=SEED
)

# Convert to numpy arrays (already processed)
X_tr_np = X_tr.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

NUM_FEATURES = X_tr_np.shape[1]
NUM_CLASSES = int(y.nunique())

# -------------------------
# PyTorch Dataset
# -------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TabularDataset(X_tr_np, y_tr)
val_ds   = TabularDataset(X_val_np, y_val)
test_ds  = TabularDataset(X_test_np, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# Model
# -------------------------
class FFNN(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = FFNN(NUM_FEATURES).to(DEVICE)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

# -------------------------
# Training loop
# -------------------------
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for Xb, yb in train_loader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * Xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(loader):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(Xb)
        l = criterion(logits, yb)
        loss += l.item() * Xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    if total == 0:
        return None
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return loss / total, correct / total, all_preds, all_targets


history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}
best_val_f1 = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    val_res = evaluate(val_loader)
    val_loss, val_acc, val_preds, val_targets = val_res
    val_f1 = f1_score(val_targets, val_preds, average="macro")

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    scheduler.step(val_f1)

    print(f"Epoch {epoch:02d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val acc {val_acc:.4f} f1_macro {val_f1:.4f}")

    # early stopping (optional)
    if PATIENCE is not None:
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs — stopping.")
                break

# load best model if early-stopped
if PATIENCE is not None and Path(MODEL_OUT).exists():
    model.load_state_dict(torch.load(MODEL_OUT))

# -------------------------
# Final eval on test set
# -------------------------
test_loss, test_acc, test_preds, test_targets = evaluate(test_loader)
print(f"\nTest acc: {test_acc:.4f}  |  Test f1_macro: {f1_score(test_targets, test_preds, average='macro'):.4f}")
print("\nClassification report (test):")
print(classification_report(test_targets, test_preds))

# save final model
torch.save(model.state_dict(), MODEL_OUT)
print("Saved model to:", MODEL_OUT)

# Get predictions for the training set
_, _, train_preds, train_targets = evaluate(train_loader)

print("\n" + "="*50)
print("CLASSIFICATION REPORT (TRAIN SET)")
print("="*50)
train_report_dict = classification_report(train_targets, train_preds, output_dict=True)
train_classification_report = classification_report(train_targets, train_preds)
print(train_classification_report)

print("\n" + "="*50)
print("CLASSIFICATION REPORT (TEST SET)")
print("="*50)
test_report_dict = classification_report(test_targets, test_preds, output_dict=True)
test_classification_report = classification_report(test_targets, test_preds)
print(test_classification_report)

report_content = (
        "#" * 50 + "\n"
        "CLASSIFICATION REPORT (TRAIN SET)\n"
        "#" * 50 + "\n"
        f"{train_classification_report}\n\n"
        
        "#" * 50 + "\n"
        "CLASSIFICATION REPORT (TEST SET)\n"
        "#" * 50 + "\n"
        f"{test_classification_report}\n"
    )

REPORT_OUT_PATH = OUT_DIR / "classification_reports.txt"
with open(REPORT_OUT_PATH, 'w') as f:
        f.write(report_content)
    
print(f"\nSuccessfully dumped classification reports to: {REPORT_OUT_PATH}")





def plot_classification_report(y_true, y_pred, title,out_dir):
    """Plot classification report heatmap with support shown as plain numbers per row."""
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report_dict).transpose()

    # Extract support
    support = df['support'].fillna(0).astype(int)

    # Drop summary rows
    drop_rows = ['accuracy', 'macro avg', 'weighted avg', 'micro avg']
    df = df.drop(drop_rows, errors='ignore')

    # Keep only metric columns
    df_metrics = df.drop(columns=['support'], errors='ignore').astype(float)

    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        df_metrics,
        annot=True,
        cmap="YlGnBu",
        fmt=".3f",
        linewidths=.5,
        linecolor='black',
        cbar=True
    )

    # Push the figure content slightly left so we have space on right
    plt.subplots_adjust(right=0.88)

    # Place support numbers farther right
    for y, cls in enumerate(df_metrics.index):
        sup_val = support.loc[cls]
        ax.text(
            df_metrics.shape[1] + 0.6,   # shifted right
            y + 0.5,
            str(sup_val),
            va='center',
            ha='left',
            fontsize=10,
            color='black'
        )

    # Support column header
    ax.text(
        df_metrics.shape[1] + 0.6,
        -0.2,
        "support",
        va='bottom',
        ha='left',
        fontsize=10,
        color='black',
        fontweight='bold'
    )

    plt.title(f"Classification Report Heatmap ({title})")
    plt.ylabel("Class")
    plt.xlabel("Metrics")
    plt.tight_layout()

    # Save the plot
    out_path = out_dir / f"classification_report_{title.lower().replace(' ', '_')}.png"
    plt.savefig(out_path)
    print(f"Saved {title} plot to: {out_path}")
    plt.show()




## 2. Loss Plot

def plot_loss(history):
    """Plots the training and validation loss across epochs."""
    plt.figure(figsize=(10, 6))
    
    # Use min(len) in case of early stopping
    epochs_ran = len(history['train_loss'])
    epochs_range = range(1, epochs_ran + 1)
    
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', marker='o', linestyle='--')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Loss Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.savefig(OUT_DIR / "loss_plot.png")


print("\n" + "="*50)
print("GENERATING PLOTS AND SAVING TO 'models/'")
print("="*50)

# Execute the plotting functions
plot_loss(history)
plot_classification_report(train_targets,train_preds, 'Train Set - FFNN Model',OUT_DIR)
plot_classification_report(test_targets,test_preds, 'Test Set - FFNN Model',OUT_DIR)

