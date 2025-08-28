# training_centralized.py
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.visualization import plot_prediction
from utils.visualization import save_sorted_r2_plot
from utils.preprocess import load_or_preprocess_datasets


def build_centralized_splits(config, data_dir, include_households=None):
    """
    加载所有住户的数据，并将 train/val 合并成一个全局数据集；
    test 保留按住户分的字典，用于逐户评估。
    """
    train_dict, val_dict, test_dict, _ = load_or_preprocess_datasets(config, data_dir)

    # 确定要包含的住户
    if include_households is None:
        households = sorted(list(train_dict.keys()))
    else:
        households = include_households

    # 合并 train / val
    train_list = [train_dict[h] for h in households if h in train_dict]
    val_list = [val_dict[h] for h in households if h in val_dict]

    if len(train_list) == 0 or len(val_list) == 0:
        raise ValueError("No datasets to merge. Check include_households and preprocessing outputs.")

    train_merged = ConcatDataset(train_list)
    val_merged = ConcatDataset(val_list)

    # 只保留指定住户的测试集
    test_kept = {h: test_dict[h] for h in households if h in test_dict}

    return train_merged, val_merged, test_kept, households


def train_and_evaluate_centralized(config, model_class, device, data_dir, include_households=None):
    """
    训练一个全局集中式模型（合并所有住户的 train/val 数据），
    然后在每个住户的 test 数据上分别评估。
    """
    if config is None:
        raise ValueError("config is None!")

    # 创建必要的目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['eval_path']), exist_ok=True)
    os.makedirs(config['plot_dir'], exist_ok=True)

    # 合并集中式数据集
    train_merged, val_merged, test_dict, households = build_centralized_splits(
        config=config, data_dir=data_dir, include_households=include_households
    )

    train_loader = DataLoader(train_merged, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_merged,   batch_size=config['batch_size'])

    # 初始化一个全局模型
    output_size = (1 if config.get('sum_target', True) else config['prediction_horizon'])
    model = model_class(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        bidirectional=config.get('bidirectional', False),
        output_size=output_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # 早停参数
    best_loss = float('inf')
    patience_counter = 0
    log_entries = []

    best_model_path = os.path.join(config['save_dir'], "best_model_global.pth")

    # 训练循环
    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / max(1, len(train_loader))

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"[GLOBAL] Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {avg_train:.3f}, Val Loss: {avg_val:.3f}")
        log_entries.append({"household": "GLOBAL", "epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val})

        # 早停检查
        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[GLOBAL] Best model saved to: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"[GLOBAL] Early stopping at epoch {epoch+1}")
                break

    # 保存训练日志
    log_exists = os.path.isfile(config['log_path'])
    with open(config['log_path'], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["household", "epoch", "train_loss", "val_loss"])
        if not log_exists:
            writer.writeheader()
        writer.writerows(log_entries)
    print(f"[GLOBAL] Training log saved to: {config['log_path']}")

    # 加载最佳全局模型
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))

    # 逐户评估
    for h in households:
        test_loader = DataLoader(test_dict[h], batch_size=config['batch_size'])
        evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            household_id=h,
            eval_path=config['eval_path'],
            plot_dir=config['plot_dir'],
            config=config
        )

    # （可选）整体合并评估
    try:
        y_true_all, y_pred_all = [], []
        model.eval()
        with torch.no_grad():
            for h in households:
                dl = DataLoader(test_dict[h], batch_size=config['batch_size'])
                for X, y in dl:
                    X, y = X.to(device), y.to(device)
                    if y.ndim == 1:
                        y = y.unsqueeze(1)
                    pred = model(X)
                    if pred.shape != y.shape:
                        pred = pred.view_as(y)
                    ds = dl.dataset
                    yt = y.cpu().numpy().flatten()
                    yp = pred.cpu().numpy().flatten()
                    if hasattr(ds, "inverse_transform_y"):
                        yt = ds.inverse_transform_y(yt.reshape(-1,1)).flatten()
                        yp = ds.inverse_transform_y(yp.reshape(-1,1)).flatten()
                    y_true_all.append(yt)
                    y_pred_all.append(yp)
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        overall = {
            "Household": "ALL",
            "RMSE": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
            "MAE": mean_absolute_error(y_true_all, y_pred_all),
            "R^2": r2_score(y_true_all, y_pred_all)
        }
        df_overall = pd.DataFrame([overall])
        if os.path.exists(config['eval_path']):
            df_overall.to_csv(config['eval_path'], mode='a', header=False, index=False)
        else:
            df_overall.to_csv(config['eval_path'], index=False)
        print("[GLOBAL] Overall metrics appended as ALL.")
    except Exception as e:
        print(f"[GLOBAL] Skipped ALL row due to: {e}")


def evaluate_model(model, dataloader, device, household_id, eval_path=None, plot_dir=None, config=None):
    if config is None:
        raise ValueError("config is None")
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(X)
            if pred.shape != y.shape:
                try:
                    pred = pred.view_as(y)
                except RuntimeError:
                    raise ValueError(f"Shape mismatch: pred {pred.shape}, y {y.shape}")
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_all).flatten()
    y_pred = np.concatenate(y_pred_all).flatten()

    # 反归一化
    if hasattr(dataloader.dataset, "inverse_transform_y"):
        try:
            y_pred = dataloader.dataset.inverse_transform_y(y_pred.reshape(-1, 1)).flatten()
            y_true = dataloader.dataset.inverse_transform_y(y_true.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"[{household_id}] Inverse transform failed: {e}")

    # 指标
    metrics = {
        "Household": household_id,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R^2": r2_score(y_true, y_pred)
    }

    # 保存指标
    if eval_path:
        df = pd.DataFrame([metrics])
        if os.path.exists(eval_path):
            df.to_csv(eval_path, mode='a', header=False, index=False)
        else:
            df.to_csv(eval_path, index=False)
        print(f"[{household_id}] Evaluation metrics saved to: {eval_path}")

    # 保存预测图和CSV
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        y_time = np.array(dataloader.dataset.y_timestamps)
        plot_path = os.path.join(plot_dir, f"prediction_{household_id}.png")
        plot_prediction(
            y_true,
            y_pred,
            save_path=plot_path,
            time_index=y_time,
            household_id=household_id,
            resample_freq=config.get("resample_freq", "5min")
        )
        csv_path = os.path.join(plot_dir, f"prediction_{household_id}.csv")
        df_pred = pd.DataFrame({
            "timestamp": y_time[:len(y_pred)],
            "y_true": y_true[:len(y_pred)],
            "y_pred": y_pred[:len(y_pred)]
        })
        df_pred.to_csv(csv_path, index=False)
        print(f"[{household_id}] Prediction values saved to: {csv_path}")

    return metrics



