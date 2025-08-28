# cluster/cluster_households_ukdale.py
import sys, os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import matplotlib.pyplot as plt

project_root = "/content/drive/MyDrive/elec_forecasting"
sys.path.append(project_root)
from utils.preprocess_ukdale_timefeat import PowerDataset

# === 设置路径 ===
CACHE_DIR = "data/UK-DALE/cached_ukdale"
OUTPUT_JSON = "cluster/cluster_result_ukdale.json"
PLOT_PATH = "cluster/cluster_visualization_ukdale.png"
N_CLUSTERS = 3

# === 加载缓存数据 ===
def load_cached_households(cache_dir):
    for file in os.listdir(cache_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(cache_dir, file), "rb") as f:
                cache = pickle.load(f)
            return cache['train']  # 只需要训练集即可
    raise FileNotFoundError("No .pkl cache found in", cache_dir)

# === 特征提取函数 ===
def extract_household_features(dataset):
    all_values = np.concatenate([ds[0].numpy().flatten() for ds in dataset])
    return {
        "mean": np.mean(all_values),
        "std": np.std(all_values),
        "max": np.max(all_values),
        "min": np.min(all_values),
        "range": np.max(all_values) - np.min(all_values)
    }

# === 主程序 ===
train_datasets = load_cached_households(CACHE_DIR)

households = []
features = []
for household_id, dataset in train_datasets.items():
    feat = extract_household_features(dataset)
    households.append(household_id)
    features.append([feat["mean"], feat["std"], feat["range"]])

features = np.array(features)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(features)
labels = kmeans.labels_

# === 保存聚类结果 ===
cluster_result = defaultdict(list)
for i, label in enumerate(labels):
    cluster_result[f"cluster_{label}"].append(households[i])

os.makedirs("cluster", exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(cluster_result, f, indent=2)
print("聚类结果已保存至：", OUTPUT_JSON)

# === 可视化 ===
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="Set2")
for i, name in enumerate(households):
    plt.text(features[i, 0], features[i, 1], name, fontsize=8)
plt.xlabel("Mean Consumption")
plt.ylabel("Std Dev")
plt.title("UK-DALE Household Clustering")
plt.savefig(PLOT_PATH)
plt.close()
print("聚类图已保存至：", PLOT_PATH)
