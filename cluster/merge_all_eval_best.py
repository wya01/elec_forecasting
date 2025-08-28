# cluster/merge_all_eval_best.py
import os
import pandas as pd

# --------- Path Configuration --------- #
project_root = "/content/drive/MyDrive/elec_forecasting"
base_target = "30min_30min_tw"
base_dir = os.path.join(project_root, "experiments", "clustered_fl", base_target)
cluster_ids = ["0", "1", "2"]

merged_path = os.path.join(base_dir, "merged_eval_metrics_best.csv")
global_all_path = os.path.join(base_dir, "global_all_metrics.csv")
dfs = []

# --------- Merge each cluster (per-household + Cluster-ALL rows) --------- #
for cid in cluster_ids:
    cluster_name = f"cluster_{cid}"
    path = os.path.join(base_dir, cluster_name, "eval_metrics_best.csv")
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        continue

    df = pd.read_csv(path)
    df["Cluster"] = cluster_name  # keep provenance
    dfs.append(df)

# --------- Concatenate --------- #
if not dfs:
    print("⚠️ No CSV files were successfully loaded")
    raise SystemExit(0)

df_merged = pd.concat(dfs, ignore_index=True)

# --------- Optionally append GLOBAL-ALL --------- #
if os.path.exists(global_all_path):
    df_global = pd.read_csv(global_all_path)
    # Expect columns: Scope, R^2, MAE, RMSE
    if {"R^2", "MAE", "RMSE"}.issubset(df_global.columns):
        row = {
            "Household": "ALL",
            "Round": -1,
            "R^2": float(df_global.iloc[0]["R^2"]),
            "MAE": float(df_global.iloc[0]["MAE"]),
            "RMSE": float(df_global.iloc[0]["RMSE"]),
            "Cluster": "GLOBAL",
        }
        df_merged = pd.concat([df_merged, pd.DataFrame([row])], ignore_index=True)
        print("✅ Appended GLOBAL-ALL from global_all_metrics.csv")
    else:
        print("⚠️ global_all_metrics.csv does not contain the expected columns; skip appending GLOBAL-ALL")
else:
    print("ℹ️ global_all_metrics.csv not found; merged file will not include GLOBAL-ALL")

# --------- Sort by household ID (H1–H20), keep ALL rows at the end --------- #
def extract_num(h):
    try:
        return int(str(h).lstrip("H"))
    except Exception:
        return 10**9  # push non-H* (e.g., ALL) to bottom

df_merged["__ord__"] = df_merged["Household"].apply(extract_num)
df_merged = df_merged.sort_values(["__ord__", "Cluster"]).drop(columns="__ord__")

# --------- Save --------- #
os.makedirs(base_dir, exist_ok=True)
df_merged.to_csv(merged_path, index=False)
print("✅ Merged successfully:", merged_path)
