import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import sys

train_bc = pd.read_parquet('final/train_bc.parquet')
test_bc = pd.read_parquet('final/test_bc.parquet')

X = train_bc.drop('Attack',axis=1)
y = train_bc['Attack']




# Separate features (X) and target (y)
X_train = train_bc.drop('Attack', axis=1)
y_train = train_bc['Attack']
X_test = test_bc.drop('Attack', axis=1)
y_test = test_bc['Attack']

if X_train.empty or X_test.empty:
    print("ERROR: DataFrames are empty. Exiting script.")
    sys.exit(1)

print(f"Train data loaded: {len(X_train)} samples.")
print(f"Test data loaded: {len(X_test)} samples.")
print("-" * 30)

# --- 1. Define Model Pipeline and EXPANDED Hyperparameter Grid ---

# Define the Pipeline: Scaling is essential for Logistic Regression.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=5000, # Increased max_iter for sag/saga solvers
        random_state=42
    )) 
])

# Define the EXPANDED Hyperparameter Grid:
# We now tune 'C', 'penalty', and 'solver'.
# Note: 'lbfgs' only supports L2. 'saga' supports L1, L2, and none.
param_grid = [
    # Configuration 1: Solvers that support L1 regularization
    {
        'logreg__solver': ['liblinear', 'saga'],
        'logreg__penalty': ['l1', 'l2'],
        'logreg__C': np.logspace(-3, 3, 7) # C values: 0.001 to 1000
    },
    # Configuration 2: Solvers that primarily support L2 regularization
    {
        'logreg__solver': ['lbfgs', 'sag'],
        'logreg__penalty': ['l2'],
        'logreg__C': np.logspace(-3, 3, 7)
    }
]

# --- 2. Initialize and Run Grid Search ---

# Initialize Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    pipe, 
    param_grid, 
    cv=5, 
    scoring='f1', # Using F1-score for evaluation
    verbose=0, 
    n_jobs=-1 
)

print(f"Starting Grid Search across {len(param_grid[0]['logreg__C']) * (len(param_grid[0]['logreg__solver']) * len(param_grid[0]['logreg__penalty'])) + len(param_grid[1]['logreg__C']) * (len(param_grid[1]['logreg__solver']) * len(param_grid[1]['logreg__penalty']))} total parameter combinations...")
grid_search.fit(X_train, y_train)
print("Grid Search complete.")

# --- 3. Evaluation and Print Logging ---

# 3a. Get Best Model and Params
best_logreg = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\n" + "="*50)
print("✨ HYPERPARAMETER TUNING RESULTS ✨")
print(f"Best cross-validation F1 score: {best_score:.4f}")
print(f"Best parameters found: {best_params}")
print("="*50)

# 3b. Evaluate on the Test Set
y_pred = best_logreg.predict(X_test)

joblib.dump(best_logreg,r'models/binary_logistic.joblib')
# Calculate key metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("\n--- FINAL MODEL PERFORMANCE ON TEST SET ---")
print(f"Best Solver: {best_params['logreg__solver']}")
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Test Set F1-Score: {test_f1:.4f}")

# Print full classification report
print("\nClassification Report (Test Set):\n")
print(classification_report(y_test, y_pred))
print("-" * 50)
print("Script execution finished.")