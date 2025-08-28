# cluster/generate_eval_best_ukdale.py

import os
import pandas as pd
import json

# ---- 路径配置 ----
project_root = "/content/drive/MyDrive/elec_forecasting"
base_target = "ukdale_1h_30min_tuned"
base_dir = os.path.join(project_root, "experiments", "clustered_fl", base_target)
cluster_result_path = os.path.join(project_root, "cluster", "cluster_result_ukdale.json")

# ---- 加载聚类分组 ----
with open(cluster_result_path, "r") as f:
    cluster_map = json.load(f)

cluster_ids = sorted([k.replace("cluster_", "") for k in cluster_map.keys()])

for cluster_id in cluster_ids:
    target_name = f"{base_target}/cluster_{cluster_id}"
    eval_csv_path = os.path.join(project_root, "experiments", "clustered_fl", target_name, "eval_metrics.csv")
    save_csv_path = os.path.join(project_root, "experiments", "clustered_fl", target_name, "eval_metrics_best.csv")

    print(f"\n🧪 处理 cluster_{cluster_id}")

    if not os.path.exists(eval_csv_path):
        print(f"[cluster_{cluster_id}] ❌ 缺少 eval_metrics.csv，跳过")
        continue

    df_all = pd.read_csv(eval_csv_path)
    client_ids = cluster_map[f"cluster_{cluster_id}"]

    # 检查缺失的 client
    missing_clients = [c for c in client_ids if c not in df_all["Household"].unique()]
    if missing_clients:
        print(f"[cluster_{cluster_id}] ⚠️ 以下户未参与训练或缺失评估结果: {missing_clients}")

    # 只保留当前 cluster 中存在评估数据的 client
    df_all = df_all[df_all["Household"].isin(client_ids)]

    # 每个 household 按 MAE 最小挑选最佳轮次
    df_best = (
        df_all.loc[df_all.groupby("Household")["MAE"].idxmin()]
              .reset_index(drop=True)
              .sort_values("Household")
    )

    # 保存
    df_best.to_csv(save_csv_path, index=False)
    print(f"[cluster_{cluster_id}] ✅ 已保存 {len(df_best)} 户最佳轮次到: {save_csv_path}")
