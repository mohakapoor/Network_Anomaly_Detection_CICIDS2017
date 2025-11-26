#!/usr/bin/env python3
"""
multiclass_lightgbm.py
Clean, full script to run hyperparameter search for LightGBM (multiclass),
WITHOUT early stopping (per your request).

Toggle USE_RANDOM to False to run full GridSearch (may be very slow).
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import loguniform

warnings.filterwarnings("ignore")

# ----------------------------
# Settings
# ----------------------------
TRAIN_PATH = Path("final/train_mc.parquet")
TEST_PATH = Path("final/test_mc.parquet")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "multiclass_lightgbm.joblib"

RANDOM_STATE = 42
VERBOSE = 2

# ----------------------------
# Load data
# ----------------------------
train_mc = pd.read_parquet(TRAIN_PATH)
test_mc = pd.read_parquet(TEST_PATH)

X_train = train_mc.drop("Attack", axis=1)
y_train = train_mc["Attack"].astype(int)   # expecting labels 0..7
X_test = test_mc.drop("Attack", axis=1)
y_test = test_mc["Attack"].astype(int)


NUM_CLASSES = int(y_train.nunique())


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# ----------------------------
# Base estimator
# ----------------------------
param_dist = {
    
    'class_weight': ['balanced'],

    # Controls complexity and speed:
    'learning_rate': loguniform(0.01, 0.2), 
    'n_estimators': [5,10], #anything over 15 gives 0.99 so i made it smaller
    'num_leaves': [20, 31, 50], 

    # Controls stability and overfitting :
    'min_child_samples': [10,30, 50],
    'max_depth': [6, 8, 10]
}

import lightgbm as lgb
# Initialize the base LightGBM model ()
lgbm_base = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=NUM_CLASSES,
    device='gpu',
    n_jobs=1,
    random_state=42,
)

# Initialize Randomized Search CV

random_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_dist,
    n_iter=20, 
    scoring='f1_macro', 
    cv=skf, 
    verbose=2,
    random_state=42,
    n_jobs=1
)

print("Starting Randomized Search CV for LightGBM tuning...")
random_search.fit(X_train, y_train)
print("Tuning complete.")


best_params = random_search.best_params_
print("\n=============================================")
print("Best Hyperparameters Found:")
print(best_params)

# Get the best score achieved
best_score = random_search.best_score_
print(f"\nBest Mean F1-Score (macro) from CV: {best_score:.4f}")
print("=============================================")

# Use the best model found by the search
best = random_search.best_estimator_

joblib.dump(best, OUT_FILE)
print("Saved best model to:", OUT_FILE)
# ---------------------------
# -
# Evaluation on test set
# ----------------------------
y_pred = best.predict(X_test)
y_pred_train = best.predict(X_train)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test F1-weighted: {test_f1:.4f}\n")
print("Classification report (test):")
print(classification_report(y_test, y_pred))


print("Classification report (train):")
print(classification_report(y_train, y_pred_train))

print("Done.")
