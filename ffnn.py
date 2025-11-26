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

# -------------------------
# Settings (tweak)
# -------------------------
TRAIN_PATH = Path("final/train_mc.parquet")
TEST_PATH  = Path("final/test_mc.parquet")
OUT_DIR    = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT  = OUT_DIR / "ffnn_multiclass.pt"

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 100
VALID_RATIO = 0.1        # fraction of train used for validation
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
PATIENCE = 8             # set None to disable early stopping
DROPOUT = 0.2
NUM_WORKERS = 4          # set to 0 if you get issues on your machine

# -------------------------
# Reproducibility
# -------------------------
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed()

# -------------------------
# Load data (assumes processed: scaled + PCA applied)
# -------------------------
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

# -------------------------
# Loss, optimizer, scheduler
# -------------------------
# If classes are imbalanced, optionally use class weights:
# counts = y_tr.value_counts().sort_index().values
# weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
# criterion = nn.CrossEntropyLoss(weight=weights)

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

best_val_f1 = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    val_res = evaluate(val_loader)
    _, val_acc, val_preds, val_targets = val_res
    val_f1 = f1_score(val_targets, val_preds, average="macro")

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
