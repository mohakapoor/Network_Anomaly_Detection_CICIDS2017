import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    set_seed()
    
    print("=" * 50)
    print("Starting SHAP Analysis")
    print("=" * 50)
    
    # Load test datasets
    print("Loading test data...")
    try:
        test_mc = pd.read_parquet(os.path.join(BASE_DIR, "final", "test_mc.parquet"))
        test_us = pd.read_parquet(os.path.join(BASE_DIR, "final", "test_us.parquet"))
    except Exception as e:
        print(f"Failed to load test datasets: {e}")
        return

    X_mc = test_mc.drop(columns=["Attack"])
    X_us = test_us.drop(columns=["Attack"])

    # Subsample data because SHAP can be very slow, especially for DeepExplainer
    # We use a representative sample of 500 instances
    sample_size = 500
    X_mc_sample = X_mc.sample(n=sample_size, random_state=42)
    X_us_sample = X_us.sample(n=sample_size, random_state=42)
    
    # Background dataset for DeepExplainer
    background_size = 100
    X_mc_bg = X_mc.sample(n=background_size, random_state=42)
    X_us_bg = X_us.sample(n=background_size, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────────────────────────
    # 1. LightGBM
    # ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Running SHAP for LightGBM...")
    try:
        lgb_model = joblib.load(os.path.join(BASE_DIR, "models", "lightgbm", "multiclass_lightgbm.joblib"))
        # For sklearn API LightGBM, the estimator might be wrapped
        # We can extract the underlying booster or just pass the model
        explainer_lgb = shap.TreeExplainer(lgb_model)
        shap_values_lgb = explainer_lgb.shap_values(X_mc_sample)
        
        plt.figure(figsize=(10, 6))
        # If multiclass, shap_values is a list. summary_plot handles it.
        shap.summary_plot(shap_values_lgb, X_mc_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "models", "lightgbm", "shap_summary.png"), bbox_inches='tight')
        plt.close()
        print("  [OK] Saved LightGBM SHAP summary.")
    except Exception as e:
        print(f"  [ERROR] Error in LightGBM SHAP: {e}")

    # ─────────────────────────────────────────────────────────────────
    # 2. XGBoost
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Running SHAP for XGBoost...")
    try:
        xgb_model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost", "multiclass_xgboost.joblib"))
        explainer_xgb = shap.TreeExplainer(xgb_model)
        shap_values_xgb = explainer_xgb.shap_values(X_mc_sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_xgb, X_mc_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "models", "xgboost", "shap_summary.png"), bbox_inches='tight')
        plt.close()
        print("  [OK] Saved XGBoost SHAP summary.")
    except Exception as e:
        print(f"  [ERROR] Error in XGBoost SHAP: {e}")

    # ─────────────────────────────────────────────────────────────────
    # 3. Isolation Forest
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Running SHAP for Isolation Forest...")
    try:
        if_model = joblib.load(os.path.join(BASE_DIR, "models", "isolation_forest", "isolation_forest.joblib"))
        explainer_if = shap.TreeExplainer(if_model)
        shap_values_if = explainer_if.shap_values(X_us_sample)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_if, X_us_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "models", "isolation_forest", "shap_summary.png"), bbox_inches='tight')
        plt.close()
        print("  [OK] Saved Isolation Forest SHAP summary.")
    except Exception as e:
        print(f"  [ERROR] Error in Isolation Forest SHAP: {e}")

    # ─────────────────────────────────────────────────────────────────
    # 4. FFNN (PyTorch)
    # ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Running SHAP for FFNN...")
    try:
        class FFNN(torch.nn.Module):
            def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=7, dropout=0.2):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden1),
                    torch.nn.BatchNorm1d(hidden1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden1, hidden2),
                    torch.nn.BatchNorm1d(hidden2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden2, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        NUM_FEATURES = X_mc_sample.shape[1]
        
        ffnn_model = FFNN(NUM_FEATURES).to(device)
        ffnn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "ffnn", "ffnn_multiclass.pt"), map_location=device))
        ffnn_model.eval()
        
        bg_tensor = torch.tensor(X_mc_bg.values, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_mc_sample.values, dtype=torch.float32).to(device)
        
        explainer_ffnn = shap.DeepExplainer(ffnn_model, bg_tensor)
        shap_values_ffnn = explainer_ffnn.shap_values(test_tensor)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_ffnn, X_mc_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "models", "ffnn", "shap_summary.png"), bbox_inches='tight')
        plt.close()
        print("  [OK] Saved FFNN SHAP summary.")
    except Exception as e:
        print(f"  [ERROR] Error in FFNN SHAP: {e}")

    # ─────────────────────────────────────────────────────────────────
    # 5. Autoencoder (PyTorch)
    # ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Running SHAP for Autoencoder...")
    try:
        class Autoencoder(torch.nn.Module):
            def __init__(self, in_dim, hidden1=128, hidden2=64, bottleneck=16, dropout=0.2):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden1),
                    torch.nn.BatchNorm1d(hidden1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden1, hidden2),
                    torch.nn.BatchNorm1d(hidden2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden2, bottleneck),
                    torch.nn.ReLU(),
                )
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(bottleneck, hidden2),
                    torch.nn.BatchNorm1d(hidden2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden2, hidden1),
                    torch.nn.BatchNorm1d(hidden1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden1, in_dim),
                    torch.nn.Sigmoid(),
                )
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        NUM_FEATURES_AE = X_us_sample.shape[1]
        
        ae_model = Autoencoder(NUM_FEATURES_AE).to(device)
        ae_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "autoencoder", "autoencoder_model.pth"), map_location=device, weights_only=True))
        ae_model.eval()

        bg_tensor = torch.tensor(X_us_bg.values, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_us_sample.values, dtype=torch.float32).to(device)
        
        # Use KernelExplainer for the Autoencoder to avoid PyTorch hook precision errors
        def ae_predict(x_np):
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                reconstruction = ae_model(x_tensor)
                mse = torch.mean((x_tensor - reconstruction)**2, dim=1, keepdim=True)
            return mse.cpu().numpy()

        # We pass a summary of the background to speed up KernelExplainer
        bg_summary = shap.kmeans(X_us_bg, 10)
        explainer_ae = shap.KernelExplainer(ae_predict, bg_summary)
        shap_values_ae = explainer_ae.shap_values(X_us_sample)
        
        plt.figure(figsize=(10, 6))
        # DeepExplainer on scalar output might return a single array or list of arrays
        if isinstance(shap_values_ae, list):
            shap.summary_plot(shap_values_ae[0], X_us_sample, show=False)
        else:
            shap.summary_plot(shap_values_ae, X_us_sample, show=False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "models", "autoencoder", "shap_summary.png"), bbox_inches='tight')
        plt.close()
        print("  [OK] Saved Autoencoder SHAP summary.")
    except Exception as e:
        print(f"  [ERROR] Error in Autoencoder SHAP: {e}")

    # ─────────────────────────────────────────────────────────────────
    # 6 & 7. Logistic Regression & Binary SVM (CUML)
    # ─────────────────────────────────────────────────────────────────
    print("\n[6/7 & 7/7] Skipping Logistic Regression and Binary SVM...")
    print("  [INFO] These models were trained with CUML (RAPIDS).")
    print("  [INFO] They cannot be loaded or explained in a non-CUML environment.")

    print("\n" + "=" * 50)
    print("SHAP analysis complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
