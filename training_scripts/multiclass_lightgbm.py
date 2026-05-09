import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import loguniform

warnings.filterwarnings("ignore")


TRAIN_PATH = Path("final/train_mc.parquet")
TEST_PATH = Path("final/test_mc.parquet")
OUT_DIR = Path("models/lightgbm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "multiclass_lightgbm.joblib"

RANDOM_STATE = 42
VERBOSE = 2


train_mc = pd.read_parquet(TRAIN_PATH)
test_mc = pd.read_parquet(TEST_PATH)

X_train = train_mc.drop("Attack", axis=1)
y_train = train_mc["Attack"].astype(int)   # expecting labels 0..7
X_test = test_mc.drop("Attack", axis=1)
y_test = test_mc["Attack"].astype(int)


NUM_CLASSES = int(y_train.nunique())


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

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

lgbm_base = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=NUM_CLASSES,
    device='gpu',
    n_jobs=1,
    random_state=42,
)



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
print("Best Hyperparameters Found:")
print(best_params)

# Get the best score achieved
best_score = random_search.best_score_
print(f"\nBest Mean F1-Score (macro) from CV: {best_score:.4f}")
best = random_search.best_estimator_

joblib.dump(best, OUT_FILE)
print("Saved best model to:", OUT_FILE)

# Evaluation on test set
y_pred = best.predict(X_test)
y_pred_train = best.predict(X_train)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test F1-weighted: {test_f1:.4f}\n")
print("Classification report (test):")
test_cr = classification_report(y_test, y_pred)
print(test_cr)


print("Classification report (train):")
train_cr = classification_report(y_train, y_pred_train)
print(train_cr)

report_content = (
        "#" * 50 + "\n"
        "CLASSIFICATION REPORT (TRAIN SET)\n"
        "#" * 50 + "\n"
        f"{train_cr}\n\n"
        
        "#" * 50 + "\n"
        "CLASSIFICATION REPORT (TEST SET)\n"
        "#" * 50 + "\n"
        f"{test_cr}\n"
    )
REPORT_OUT_PATH = OUT_DIR / "classification_reports.txt"
with open(REPORT_OUT_PATH, 'w') as f:
        f.write(report_content)
    
print(f"\nSuccessfully dumped classification reports to: {REPORT_OUT_PATH}")


def plot_classification_report(y_true, y_pred, title,out_dir):
    """Plot classification report heatmap with support shown as plain numbers per row."""
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report_dict).transpose()

    support = df['support'].fillna(0).astype(int)

    drop_rows = ['accuracy', 'macro avg', 'weighted avg', 'micro avg']
    df = df.drop(drop_rows, errors='ignore')

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

    plt.subplots_adjust(right=0.88)

    for y, cls in enumerate(df_metrics.index):
        sup_val = support.loc[cls]
        ax.text(
            df_metrics.shape[1] + 0.6,
            y + 0.5,
            str(sup_val),
            va='center',
            ha='left',
            fontsize=10,
            color='black'
        )

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

    out_path = out_dir / f"classification_report_{title.lower().replace(' ', '_')}.png"
    plt.savefig(out_path)
    print(f"Saved {title} plot to: {out_path}")
    plt.show()




print("\n" + "="*50)
print("GENERATING CLASSIFICATION REPORT PLOTS")
print("="*50)

plot_classification_report(
    y_train, 
    y_pred_train, 
    'Train Set - Lightgbm', 
    OUT_DIR
)

plot_classification_report(
    y_test, 
    y_pred, 
    'Test Set - Lightgbm', 
    OUT_DIR
)
