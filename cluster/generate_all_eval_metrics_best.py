# cluster/generate_all_eval_metrics_best.py

import os
import pandas as pd

# ---- Paths ----
project_root = "/content/drive/MyDrive/elec_forecasting"
base_target = "30min_30min_tw"
base_dir = os.path.join(project_root, "experiments", "clustered_fl", base_target)
cluster_ids = ["0", "1", "2"]

# Keep this map consistent with cluster_result.json
cluster_map = {
    "0": ['H14_Wh', 'H15_Wh', 'H2_Wh', 'H3_Wh', 'H6_Wh', 'H8_Wh'],
    "1": ['H10_Wh', 'H11_Wh', 'H18_Wh', 'H1_Wh', 'H20_Wh', 'H4_Wh', 'H5_Wh', 'H7_Wh', 'H9_Wh'],
    "2": ['H12_Wh', 'H13_Wh', 'H16_Wh', 'H17_Wh', 'H19_Wh']
}

os.makedirs(base_dir, exist_ok=True)

for cluster_id in cluster_ids:
    target_name = f"{base_target}/cluster_{cluster_id}"
    eval_csv_path = os.path.join(project_root, "experiments", "clustered_fl", target_name, "eval_metrics.csv")
    save_csv_path = os.path.join(project_root, "experiments", "clustered_fl", target_name, "eval_metrics_best.csv")

    print(f"\n Processing cluster_{cluster_id}")

    if not os.path.exists(eval_csv_path):
        print(f"[cluster_{cluster_id}] ❌ Missing eval_metrics.csv, skip")
        continue

    # Read all rounds metrics
    df_all = pd.read_csv(eval_csv_path)

    # Normalize household IDs to *_Wh for matching
    df_all["Household"] = df_all["Household"].apply(lambda x: f"{x}_Wh" if not str(x).endswith("_Wh") else x)

    # Keep only clients belonging to this cluster
    client_ids = set(cluster_map[cluster_id])
    df_all = df_all[df_all["Household"].isin(client_ids)]

    if df_all.empty:
        print(f"[cluster_{cluster_id}] ️ No rows after filtering by cluster members.")
        continue

    # Pick the best round per household by minimum MAE
    df_best = (
        df_all.loc[df_all.groupby("Household")["MAE"].idxmin()]
              .reset_index(drop=True)
              .sort_values("Household")
              .copy()
    )

    # Strip _Wh suffix for downstream postprocess
    df_best["Household"] = df_best["Household"].str.replace("_Wh", "", regex=False)

    # Save best-per-household table
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df_best.to_csv(save_csv_path, index=False)
    print(f"[cluster_{cluster_id}]  Saved best-round metrics to: {save_csv_path}")
