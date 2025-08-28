# cluster/cluster_households.py
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === 设置路径 ===
DATA_DIR = "data/community_2024"
OUTPUT_FILE = "cluster/cluster_result.json"
N_CLUSTERS = 3

# === 特征提取函数（可扩展） ===
def extract_features(household_csv):
    df = pd.read_csv(household_csv, parse_dates=['date'])
    df.columns = df.columns.str.strip()  # 去除前后空格
    consumption = df['Consumption(Wh)'].values
    return {
        "mean": np.mean(consumption),
        "std": np.std(consumption),
        "max": np.max(consumption),
        "min": np.min(consumption),
        "range": np.max(consumption) - np.min(consumption),
    }

# === 加载所有 household 特征 ===
households = []
features = []
for filename in sorted(os.listdir(DATA_DIR)):
    if filename.endswith(".csv") and filename.startswith("H"):
        household_id = filename.split(".")[0]
        path = os.path.join(DATA_DIR, filename)
        feat = extract_features(path)
        households.append(household_id)
        features.append([feat["mean"], feat["std"], feat["range"]])

# === KMeans 聚类 ===
features = np.array(features)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(features)
labels = kmeans.labels_

# === 保存结果 ===
from collections import defaultdict
import json

cluster_result = defaultdict(list)
for i, label in enumerate(labels):
    cluster_result[f"cluster_{label}"].append(households[i])

os.makedirs("cluster", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(cluster_result, f, indent=2)

# === 可视化 ===
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="Set2")
for i, name in enumerate(households):
    plt.text(features[i, 0], features[i, 1], name, fontsize=8)
plt.xlabel("Mean Consumption")
plt.ylabel("Std Dev")
plt.title("Household Clustering")
plt.savefig("cluster/cluster_visualization.png")
plt.close()
