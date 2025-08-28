# cluster/merge_all_eval_best_ukdale.py

import os
import pandas as pd

# --------- 路径配置 --------- #
project_root = "/content/drive/MyDrive/elec_forecasting"
base_target = "ukdale_1h_30min_tuned"
base_dir = os.path.join(project_root, "experiments", "clustered_fl", base_target)
cluster_ids = ["0", "1", "2"]

merged_path = os.path.join(base_dir, "merged_eval_metrics_best.csv")
dfs = []

# --------- 合并每个 cluster 的 best 表 --------- #
for cid in cluster_ids:
    cluster_name = f"cluster_{cid}"
    path = os.path.join(base_dir, cluster_name, "eval_metrics_best.csv")
    if not os.path.exists(path):
        print(f"❌ 缺少：{path}")
        continue

    df = pd.read_csv(path)
    df["Cluster"] = cluster_name
    dfs.append(df)

# --------- 合并并排序输出 --------- #
if dfs:
    df_merged = pd.concat(dfs, ignore_index=True)

    # 提取 building 编号用于排序（如 building1 → 1）
    df_merged["Building_Num"] = df_merged["Household"].str.extract(r"building(\d+)").astype(int)
    df_merged = df_merged.sort_values("Building_Num").drop(columns=["Building_Num"])

    df_merged.to_csv(merged_path, index=False)
    print("✅ 合并完成：", merged_path)
else:
    print("⚠️ 没有成功读取任何 CSV")
