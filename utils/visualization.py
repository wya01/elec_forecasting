# utils/visualization.py

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np


def plot_prediction(
    y_true,
    y_pred,
    save_path=None,
    title="Prediction vs. Actual",
    sample_range=None,
    time_index=None,
    household_id=None,
    resample_freq="5min"
):
    import matplotlib.dates as mdates
    import pandas as pd
    import matplotlib.pyplot as plt

    def _get_minutes(freq_str):
        try:
            delta = pd.Timedelta(freq_str)
            return int(delta.total_seconds() / 60)
        except Exception:
            print(f"[warn] 无法解析 resample_freq: {freq_str}, 使用默认 5min")
            return 5

    if sample_range is None:
        freq_minutes = _get_minutes(resample_freq)
        sample_range = max(1, (24 * 60) // freq_minutes)

    n = min(sample_range, len(y_true), len(y_pred))
    x_axis = time_index[:n] if time_index is not None else range(n)

    plt.figure(figsize=(12, 4))
    plt.plot(x_axis, y_true[:n], label='Actual', linewidth=2)
    plt.plot(x_axis, y_pred[:n], label='Predicted', linewidth=2)

    if time_index is not None:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()

    # —— 标题改为你需要的格式 ——
    if household_id is not None:
        title = f"Prediction vs. Actual - {household_id} - C-FL-T (30min→30min)"
    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✅ Prediction plot saved to: {save_path}")
    plt.close()


def plot_granularity_effect(metrics_paths, metric="R^2"):
    """
    metrics_paths: dict like {"30min_1min": "path/to/csv", ...}
    """
    results = []
    for label, path in metrics_paths.items():
        df = pd.read_csv(path, index_col=0)
        avg_metric = df[metric].mean()
        results.append((label, avg_metric))

    results.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*results)

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.title(f"Granularity Effect on {metric}")
    plt.xticks(rotation=30)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def save_sorted_r2_plot(eval_csv_path, save_path=None, title="Model R² Comparison (Sorted)"):
    """
    读取评估结果 CSV，按 R² 从小到大排序后绘图并保存
    """
    df = pd.read_csv(eval_csv_path, index_col=0)
    df_sorted = df.sort_values(by="R^2", ascending=True)

    plt.figure(figsize=(12, 4))
    plt.bar(df_sorted.index, df_sorted["R^2"], color="steelblue")
    plt.ylabel("R² Score")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"R² comparison plot saved to: {save_path}")
    plt.show()
    plt.close()


def save_sorted_mae_plot(eval_csv_path, save_path=None, title="Model MAE Comparison (Sorted)"):
    """
    读取评估结果 CSV，按 MAE 从小到大排序后绘图并保存
    """
    df = pd.read_csv(eval_csv_path, index_col=0)
    df_sorted = df.sort_values(by="MAE", ascending=True)

    plt.figure(figsize=(12, 4))
    plt.bar(df_sorted.index, df_sorted["MAE"], color="seagreen")
    plt.ylabel("MAE")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"MAE comparison plot saved to: {save_path}")
    plt.show()
    plt.close()


def save_sorted_rmse_plot(eval_csv_path, save_path=None, title="Model RMSE Comparison (Sorted)"):
    """
    读取评估结果 CSV，按 RMSE 从小到大排序后绘图并保存
    """
    df = pd.read_csv(eval_csv_path, index_col=0)
    df_sorted = df.sort_values(by="RMSE", ascending=True)

    plt.figure(figsize=(12, 4))
    plt.bar(df_sorted.index, df_sorted["RMSE"], color="darkorange")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"RMSE comparison plot saved to: {save_path}")
    plt.show()
    plt.close()


def plot_federated_vs_centralized(
    fed_path,
    cen_path,
    metric="R^2",
    save_path=None,
    title=None
):
    """
    比较 federated 和 centralized 在每个户上的指标表现
    metric: 可选 "R^2", "MAE", "RMSE"
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # 加载 CSV
    df_fed = pd.read_csv(fed_path)
    df_cen = pd.read_csv(cen_path)

    # 明确指定 Household 顺序
    ordered_households = [f"H{i}" for i in range(1, 21)]

    # 设置 index 并 reindex 保证顺序一致
    df_fed = df_fed.set_index("Household").reindex(ordered_households).reset_index()
    df_cen = df_cen.set_index("Household").reindex(ordered_households).reset_index()

    households = df_fed["Household"]
    metric_fed = df_fed[metric]
    metric_cen = df_cen[metric]

    x = range(len(households))
    width = 0.35

    plt.figure(figsize=(14, 5))
    plt.bar([i - width / 2 for i in x], metric_fed, width=width, label="Federated", color="orange")
    plt.bar([i + width / 2 for i in x], metric_cen, width=width, label="Centralized", color="steelblue")

    plt.xticks(x, households, rotation=45)
    plt.ylabel(metric)
    plt.title(title or f"Federated vs. Centralized Comparison ({metric})")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Federated vs. Centralized comparison saved to: {save_path}")
    plt.show()


def plot_federated_vs_clustered(
        fed_path,
        cluster_path,
        metric="R^2",
        save_path=None,
        title=None,
        show_global=True,
):

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ---- Load ----
    df_fed = pd.read_csv(fed_path)
    df_clu = pd.read_csv(cluster_path)

    # ---- Per-household: drop ALL & keep last unique household row if duplicates exist ----
    df_fed_h = df_fed[df_fed["Household"] != "ALL"].drop_duplicates(subset="Household", keep="last")
    df_clu_h = df_clu[df_clu["Household"] != "ALL"].drop_duplicates(subset="Household", keep="last")

    # ---- Order households H1..H20 ----
    ordered_households = [f"H{i}" for i in range(1, 21)]
    df_fed_h = df_fed_h.set_index("Household").reindex(ordered_households).reset_index()
    df_clu_h = df_clu_h.set_index("Household").reindex(ordered_households).reset_index()

    households = df_fed_h["Household"]
    metric_fed = df_fed_h[metric]
    metric_clu = df_clu_h[metric]

    # ---- Prepare figure with optional GLOBAL panel ----
    if show_global:
        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
        ax_main = fig.add_subplot(gs[0])
        ax_global = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=(14, 5))
        ax_main = fig.add_subplot(111)
        ax_global = None

    # ---- Per-household bars ----
    x = range(len(households))
    width = 0.35
    ax_main.bar([i - width / 2 for i in x], metric_fed, width=width, label="Federated", color="orange")
    ax_main.bar([i + width / 2 for i in x], metric_clu, width=width, label="Clustered FL", color="skyblue")

    ax_main.set_xticks(list(x))
    ax_main.set_xticklabels(households, rotation=45)
    ax_main.set_ylabel(metric)
    ax_main.set_title(title or f"Federated vs. Clustered FL ({metric})")
    ax_main.legend()
    ax_main.grid(axis="y")

    # ---- Global ALL panel (if requested & available) ----
    if show_global and ax_global is not None:
        has_all_fed = "ALL" in set(df_fed.get("Household", []))
        has_all_clu = "ALL" in set(df_clu.get("Household", []))

        if has_all_fed and has_all_clu:
            val_fed = df_fed.loc[df_fed["Household"] == "ALL", metric].values[0]
            val_clu = df_clu.loc[df_clu["Household"] == "ALL", metric].values[0]
            ax_global.bar(["FL", "Clustered FL"], [val_fed, val_clu], color=["orange", "skyblue"])
            ax_global.set_ylabel(metric)
            ax_global.set_title("GLOBAL-ALL")
            ax_global.grid(axis="y")
        else:
            ax_global.axis("off")
            print("⚠️ 'ALL' row not found in one or both CSV files; skipped GLOBAL-ALL panel.")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_clustered_vs_timefeat_clustered(
    base_cluster_path,
    timefeat_cluster_path,
    metric="R^2",
    save_path=None,
    title=None
):
    df_base = pd.read_csv(base_cluster_path)
    df_timefeat = pd.read_csv(timefeat_cluster_path)

    df_base = df_base[df_base["Household"] != "ALL"].drop_duplicates(subset="Household", keep="last")
    df_timefeat = df_timefeat[df_timefeat["Household"] != "ALL"].drop_duplicates(subset="Household", keep="last")

    ordered_households = [f"H{i}" for i in range(1, 21)]

    df_base = df_base.set_index("Household").reindex(ordered_households).reset_index()
    df_timefeat = df_timefeat.set_index("Household").reindex(ordered_households).reset_index()

    households = df_base["Household"]
    metric_base = df_base[metric]
    metric_timefeat = df_timefeat[metric]

    x = range(len(households))
    width = 0.35

    plt.figure(figsize=(14, 5))
    plt.bar([i - width / 2 for i in x], metric_base, width=width, label="Clustered FL", color="skyblue")
    plt.bar([i + width / 2 for i in x], metric_timefeat, width=width, label="Clustered + Time Features", color="mediumseagreen")

    plt.xticks(x, households, rotation=45)
    plt.ylabel(metric)
    plt.title(title or f"Clustered FL vs. Time-Enhanced Clustered FL ({metric})")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存：{save_path}")
    plt.show()


def plot_global_all_bar(base_path, timefeat_path, metric="R^2", save_path=None, title=None):
    """
    Compare the GLOBAL-ALL row between Clustered FL and Clustered + Time Features.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    df_base = pd.read_csv(base_path)
    df_timefeat = pd.read_csv(timefeat_path)

    if "ALL" not in df_base["Household"].values or "ALL" not in df_timefeat["Household"].values:
        print("❌ 'ALL' row not found in one or both CSV files. Skipping GLOBAL-ALL comparison.")
        return

    val_base = df_base[df_base["Household"] == "ALL"][metric].values[0]
    val_time = df_timefeat[df_timefeat["Household"] == "ALL"][metric].values[0]

    plt.figure(figsize=(5, 5))
    plt.bar(["Clustered FL", "Clustered + Time Features"], [val_base, val_time], color=["skyblue", "mediumseagreen"])
    plt.ylabel(metric)
    plt.title(title or f"GLOBAL-ALL Comparison ({metric})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"GLOBAL-ALL comparison plot saved to: {save_path}")
    plt.show()

# ===== 配色方案 (Set2) =====
palette_colors = sns.color_palette("Set2", 5)
palette = {
    "CL": palette_colors[0],
    "FL": palette_colors[1],
    "C-FL": palette_colors[2],
    "C-FL+T": palette_colors[3],
    "C-FL+T+W": palette_colors[4],
}


# ===== 1. per-household 柱状图 =====
def plot_per_household_5methods(
    df,
    metric,
    outdir,
    title=None,
    filename_suffix=None,   # e.g., "30min_30min"
):
    """
    在同一张图里画 5 种方法（CL, FL, C-FL, C-FL+T, 可选 C-FL+T+W）的 per-household 分组柱状图
    df: 期望包含列 ["Household", "Method", <metric>]，且可以混有 ALL 行（会自动剔除）
    metric: "R^2" / "R2" / "MAE" / "RMSE"
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # --- 统一指标列名 ---
    metric_alias = {"R2": "R^2", "r2": "R^2", "R_2": "R^2", "R^2 ": "R^2"}
    mcol = metric_alias.get(metric, metric)
    if mcol not in df.columns:
        raise ValueError(f"Metric column '{metric}' / '{mcol}' not found in df.columns={list(df.columns)}")

    # --- 只保留 per-household，去重并按 H1..H20 排序 ---
    ordered_households = [f"H{i}" for i in range(1, 21)]
    dfh_all = (
        df[df["Household"] != "ALL"]
        .drop_duplicates(subset=["Household", "Method"], keep="last")
    )

    base = pd.DataFrame({"Household": ordered_households})

    # --- 需要对齐五种方法，缺失的会自动跳过 ---
    methods_order = ["CL", "FL", "C-FL", "C-FL+T", "C-FL+T+W"]
    available = [m for m in methods_order if m in df["Method"].unique()]
    if len(available) < 2:
        raise ValueError("Not enough methods to compare (need ≥2).")

    # --- 为每种方法构建一个与 H1..H20 对齐的数组 ---
    series = {}
    for m in available:
        cur = dfh_all[dfh_all["Method"] == m][["Household", mcol]]
        # 只跟 base 合并，避免产生 mcol_x/mcol_y
        aligned = base.merge(cur, on="Household", how="left")
        series[m] = aligned[mcol].values

    # --- 分组柱状图：数值 x + 偏移量 ---
    x = np.arange(len(ordered_households))
    total_w = 0.8
    bar_w = total_w / len(available)

    # 颜色：Seaborn Set2
    palette_colors = sns.color_palette("Set2", len(methods_order))
    palette = dict(zip(methods_order, palette_colors))

    plt.figure(figsize=(20, 6))
    for i, m in enumerate(available):
        offset = (i - (len(available) - 1) / 2) * bar_w
        plt.bar(x + offset, series[m], width=bar_w, label=m, color=palette[m], alpha=0.95)

    plt.xticks(x, ordered_households, rotation=45)
    plt.ylabel(mcol)
    if title is None:
        title = f"Per-household {mcol}"
        if filename_suffix:
            title += f" — {filename_suffix}"
    plt.title(title)
    plt.legend(ncol=min(len(available), 5), frameon=False)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = f"per_household_{mcol.replace('^','').replace(' ','').replace('/','')}.png"
    if filename_suffix:
        fname = f"per_household_{filename_suffix}_{mcol.replace('^','').replace(' ','').replace('/','')}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
    plt.close()


# ===== 2. GLOBAL-ALL 柱状图 =====
def plot_global_all_5methods(
    df,
    metric,
    outdir,
    title=None,
    filename_suffix=None,   # e.g., "30min_30min"
):
    """
    在同一张图里画 5 种方法（CL, FL, C-FL, C-FL+T, 可选 C-FL+T+W）的 GLOBAL-ALL 柱状图
    df: 期望包含列 ["Method", "Household", <metric>] 且至少有 ALL 行
    metric: "R^2" / "R2" / "MAE" / "RMSE"
    """

    # --- 统一指标列名 ---
    metric_alias = {"R2": "R^2", "r2": "R^2"}
    mcol = metric_alias.get(metric, metric)
    if mcol not in df.columns:
        raise ValueError(f"Metric column '{metric}' / '{mcol}' not found in df.columns={list(df.columns)}")

    # --- 取 ALL 行 ---
    dfa = df[df["Household"] == "ALL"].copy()
    if dfa.empty:
        raise ValueError("GLOBAL-ALL row not found in the provided DataFrame.")

    methods_order = ["CL", "FL", "C-FL", "C-FL+T", "C-FL+T+W"]
    dfa = dfa[dfa["Method"].isin(methods_order)]
    if dfa.empty:
        raise ValueError("No valid methods in GLOBAL-ALL DataFrame.")

    # 保持固定顺序
    dfa["Method"] = pd.Categorical(dfa["Method"], categories=methods_order, ordered=True)
    dfa = dfa.sort_values("Method")

    # 颜色：Seaborn Set2
    palette_colors = sns.color_palette("Set2", len(methods_order))
    palette = dict(zip(methods_order, palette_colors))

    colors = [palette[m] for m in dfa["Method"]]

    plt.figure(figsize=(8, 5))
    plt.bar(dfa["Method"], dfa[mcol], color=colors, alpha=0.95,width=0.6)
    plt.ylabel(mcol)
    if title is None:
        title = f"GLOBAL-ALL {mcol}"
        if filename_suffix:
            title += f" — {filename_suffix}"
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = f"global_all_{mcol.replace('^','').replace(' ','').replace('/','')}.png"
    if filename_suffix:
        fname = f"global_all_{filename_suffix}_{mcol.replace('^','').replace(' ','').replace('/','')}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
    plt.close()



