# federated/generate_eval_metrics_best.py
"""
Pick the best round per household (by minimum MAE) from eval_metrics.csv
and write federated_lstm/<target_name>/eval_metrics_best.csv
"""

import os
import sys
import pandas as pd

# ----- Paths -----
project_root = "/content/drive/MyDrive/elec_forecasting"
target_name = "6h_1h"  # <-- change if needed
base_dir = os.path.join(project_root, "experiments", "federated_lstm", target_name)
eval_csv_path = os.path.join(base_dir, "eval_metrics.csv")
save_csv_path = os.path.join(base_dir, "eval_metrics_best.csv")

# ----- Basic checks -----
if not os.path.exists(eval_csv_path):
    print(f"❌ Missing file: {eval_csv_path}")
    sys.exit(1)

df_all = pd.read_csv(eval_csv_path)

required_cols = {"Household", "Round", "MAE"}
missing = required_cols - set(df_all.columns)
if missing:
    print(f"❌ Missing required columns in eval_metrics.csv: {missing}")
    sys.exit(1)

if df_all.empty:
    print("⚠️ eval_metrics.csv is empty; nothing to do.")
    sys.exit(0)

# ----- Pick best round per household by minimum MAE -----
df_best = (
    df_all.loc[df_all.groupby("Household")["MAE"].idxmin()]
          .reset_index(drop=True)
          .sort_values("Household")
)

# ----- Save -----
os.makedirs(base_dir, exist_ok=True)
df_best.to_csv(save_csv_path, index=False)
print(f"✅ Saved best-round metrics to: {save_csv_path}")
