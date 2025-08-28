# cluster/postprocess_all_clusters.py
# 生成每户预测 CSV/PNG + 写入 Cluster-ALL 与 Global-ALL

import os, sys, re, glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- 基础路径 --------------------
project_root = "/content/drive/MyDrive/elec_forecasting"
os.chdir(project_root)
sys.path.append(project_root)

from cluster.client_def_cluster import LSTMClient  # 需提供 test_loader 与 inverse_transform_y(可选)

data_dir    = os.path.join(project_root, "data", "community_2024")
base_target = "30min_30min_t"
base_dir    = os.path.join(project_root, "experiments", "clustered_fl", base_target)
cluster_ids = ["0", "1", "2"]

# 候选权重根目录（先找 models_output，再找 experiments，确保与你的存放一致）
WEIGHT_ROOTS = [
    os.path.join(project_root, "models_output", "clustered_fl"),
    os.path.join(project_root, "experiments",   "clustered_fl"),
]

# -------------------- 小工具 --------------------
def find_weight_path(target_name: str, rnd: int, hid_model: str):
    """在候选根目录查找 best_model_{hid_model}.pth"""
    for root in WEIGHT_ROOTS:
        p = os.path.join(root, target_name, f"round_{rnd}", f"best_model_{hid_model}.pth")
        if os.path.exists(p):
            return p
    return None

def plot_prediction(
    y_true,
    y_pred,
    save_path=None,
    title="Prediction vs. Actual",
    sample_range=None,
    time_index=None,
    resample_freq="5min"
):
    import matplotlib.dates as mdates
    import pandas as pd

    def _get_minutes(freq_str):
        try:
            delta = pd.Timedelta(freq_str)
            return int(delta.total_seconds() / 60)
        except Exception:
            return 5

    if sample_range is None:
        freq_minutes = _get_minutes(resample_freq)
        sample_range = max(1, (24 * 60) // freq_minutes)  # 5min -> 288

    n = min(sample_range, len(y_true), len(y_pred))
    x_axis = time_index[:n] if time_index is not None else range(n)

    plt.figure(figsize=(12, 4))
    plt.plot(x_axis, y_true[:n], label='Actual', linewidth=2)
    plt.plot(x_axis, y_pred[:n], label='Predicted', linewidth=2)

    if time_index is not None:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"    🖼  Saved plot: {save_path}")
    plt.close()

def save_pred_csv_png(save_dir, hshort, rnd, y_true, y_pred, time_index=None):
    """保存 per-household 预测 CSV 和 PNG"""
    pred_dir = os.path.join(save_dir, "predictions")
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # CSV
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if time_index is not None and len(time_index) >= len(df):
        df.insert(0, "timestamp", pd.to_datetime(time_index[:len(df)]))
    csv_path = os.path.join(pred_dir, f"{hshort}_round_{rnd}.csv")
    df.to_csv(csv_path, index=False)
    print(f"    📄 Saved CSV : {csv_path}")

    # PNG（标题按要求）
    title = f"Prediction vs. Actual - {hshort} - C-FL-T (30min→30min)"
    png_path = os.path.join(plot_dir, f"{hshort}.png")
    plot_prediction(
        y_true=y_true,
        y_pred=y_pred,
        save_path=png_path,
        title=title,
        time_index=pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else None,
        resample_freq="5min",
    )

# -------------------- 全局累加器 --------------------
global_y_true_all, global_y_pred_all = [], []

# -------------------- 主流程 --------------------
for cluster_id in cluster_ids:
    target_name = f"{base_target}/cluster_{cluster_id}"
    save_dir    = os.path.join(project_root, "experiments", "clustered_fl", target_name)
    best_csv    = os.path.join(save_dir, "eval_metrics_best.csv")

    if not os.path.exists(best_csv):
        print(f"[cluster_{cluster_id}] ❌ Missing eval_metrics_best.csv, skip")
        continue

    print(f"\n▶ Processing cluster_{cluster_id}")
    df_best = pd.read_csv(best_csv)

    # 与训练一致的配置
    config = {
        "target_name": target_name,
        "value_col": "Consumption(Wh)",
        "resample_freq": "5min",
        "sum_target": True,
        "window_size": 6,
        "prediction_horizon": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "batch_size": 32,
        "lr": 0.005,
        "bidirectional": False,
        "normalize_x": True,
        "device": "cpu",
        "cache_suffix": "timefeat",
        "use_time_features": True,
        "use_weather_features": False,
        "project_root": project_root,
    }
    device = torch.device(config["device"])

    # Cluster-ALL 累加器
    cluster_y_true_all, cluster_y_pred_all = [], []

    for _, row in df_best.iterrows():
        hshort = str(row["Household"])
        if hshort.upper() == "ALL":
            continue  # 跳过 ALL 行
        rnd = int(row["Round"])
        hid_model = f"{hshort}_Wh"

        wpath = find_weight_path(target_name, rnd, hid_model)
        if not wpath:
            print(f"  ⚠️ Missing weight: {hid_model} (round {rnd}) in models_output/ or experiments/")
            continue

        print(f"  ✅ {hshort} | round {rnd} | {os.path.relpath(wpath, project_root)}")

        # 构造 client & 加载权重
        client = LSTMClient(
            cid=hshort,
            household_id=hid_model,
            config=config,
            device=device,
            data_dir=data_dir,
            save_dir=save_dir,
        )
        state = torch.load(wpath, map_location=device)
        client.model.load_state_dict(state)

        # 离线推理（test_loader）
        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for X, y in client.test_loader:
                X, y = X.to(device), y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                pred = client.model(X)
                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true_list).flatten()
        y_pred = np.concatenate(y_pred_list).flatten()

        # 反归一化
        ds = client.test_loader.dataset
        if hasattr(ds, "inverse_transform_y"):
            y_true = ds.inverse_transform_y(y_true.reshape(-1, 1)).flatten()
            y_pred = ds.inverse_transform_y(y_pred.reshape(-1, 1)).flatten()

        # 时间索引（若数据集提供）
        time_index = None
        for attr in ["time_index", "timestamps", "get_time_index"]:
            if hasattr(ds, attr):
                try:
                    time_index = getattr(ds, attr)
                    if callable(time_index):
                        time_index = time_index()
                except Exception:
                    time_index = None
                break

        # 保存 CSV + PNG
        save_pred_csv_png(save_dir, hshort, rnd, y_true, y_pred, time_index=time_index)

        # 汇总
        cluster_y_true_all.append(y_true)
        cluster_y_pred_all.append(y_pred)
        global_y_true_all.append(y_true)
        global_y_pred_all.append(y_pred)

    # 写回 Cluster-ALL（先去旧 ALL，再追加新 ALL）
    if cluster_y_true_all:
        ytc = np.concatenate(cluster_y_true_all)
        ypc = np.concatenate(cluster_y_pred_all)
        mae_c  = mean_absolute_error(ytc, ypc)
        rmse_c = np.sqrt(mean_squared_error(ytc, ypc))
        r2_c   = r2_score(ytc, ypc)

        dfb = pd.read_csv(best_csv)
        dfb = dfb[dfb["Household"].astype(str) != "ALL"]  # 去除旧 ALL
        dfb = pd.concat([dfb, pd.DataFrame([{
            "Household":"ALL","Round":-1,"R^2":r2_c,"MAE":mae_c,"RMSE":rmse_c
        }])], ignore_index=True)
        dfb.to_csv(best_csv, index=False)
        print(f"  [cluster_{cluster_id}::ALL]  R²={r2_c:.3f}  MAE={mae_c:.1f}  RMSE={rmse_c:.1f}")
    else:
        print(f"  [cluster_{cluster_id}] No predictions collected; skip ALL row.")

# -------------------- 写 GLOBAL-ALL --------------------
global_all_csv = os.path.join(base_dir, "global_all_metrics.csv")
if global_y_true_all:
    ytg = np.concatenate(global_y_true_all)
    ypg = np.concatenate(global_y_pred_all)
    mae_g  = mean_absolute_error(ytg, ypg)
    rmse_g = np.sqrt(mean_squared_error(ytg, ypg))
    r2_g   = r2_score(ytg, ypg)
    pd.DataFrame([{
        "Scope":"GLOBAL_ALL", "R^2":r2_g, "MAE":mae_g, "RMSE":rmse_g
    }]).to_csv(global_all_csv, index=False)
    print(f"\nGLOBAL-ALL  R²={r2_g:.3f}  MAE={mae_g:.1f}  RMSE={rmse_g:.1f}")
    print(f"Saved: {global_all_csv}")
else:
    print("\nNo global predictions collected; skip GLOBAL-ALL.")
