# training_utils.py

import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.visualization import plot_prediction
from utils.visualization import save_sorted_r2_plot
from utils.preprocess import load_or_preprocess_datasets


def train_and_evaluate(household_id, config, model_class, device, data_dir):
    if config is None:
        raise ValueError(f"[{household_id}] config is None! Please check .ipynb for overwritten value.")
    # === Create necessary directories ===
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['log_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['eval_path']), exist_ok=True)
    os.makedirs(config['plot_dir'], exist_ok=True)

    # === Load or preprocess dataset ===
    train_datasets, val_datasets, test_datasets, _ = load_or_preprocess_datasets(config, data_dir)
    train_dataset = train_datasets[household_id]
    val_dataset = val_datasets[household_id]
    test_dataset = test_datasets[household_id]

    best_model_path = os.path.join(config['save_dir'], f"best_model_{household_id}.pth")

    # === If model exists, skip training and load ===
    if os.path.exists(best_model_path):
        print(f"[{household_id}] Model exists. Skipping training and evaluating existing model...")
        model = model_class(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            bidirectional=config.get('bidirectional', False),
            output_size=1
        ).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        return evaluate_model(model, test_loader, device, household_id, config['eval_path'], config['plot_dir'], config)


    train_data = train_dataset
    val_data = val_dataset  # 使用 preprocess 提供的 val_dataset

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # === Initialize model ===
    model = model_class(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        bidirectional=config.get('bidirectional', False),
        output_size=(1 if config.get('sum_target', True) else config['prediction_horizon'])
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_loss = float('inf')
    patience_counter = 0
    log_entries = []

    # === Training loop ===
    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0
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

        avg_loss = total_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[{household_id}] Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {avg_loss:.3f}, Val Loss: {avg_val_loss:.3f}")

        log_entries.append({
            "household": household_id,
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss
        })

        # === Early stopping and model saving ===
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[{household_id}] Best model saved to: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"[{household_id}] Early stopping at epoch {epoch+1}")
                break

    # === Save training logs ===
    log_exists = os.path.isfile(config['log_path'])
    with open(config['log_path'], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["household", "epoch", "train_loss", "val_loss"])
        if not log_exists:
            writer.writeheader()
        writer.writerows(log_entries)
    print(f"[{household_id}] Training log saved to: {config['log_path']}")

    # === Load best model and evaluate ===
    model.load_state_dict(torch.load(best_model_path))
    return evaluate_model(model, test_loader, device, household_id, config['eval_path'], config['plot_dir'], config)


# ==== Evaluation function ====
def evaluate_model(model, dataloader, device, household_id, eval_path=None, plot_dir=None, config=None):
    if config is None:
        raise ValueError("config is None — please check that train_and_evaluate always passes it.")
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = model(X)
            # reshape pred if necessary to match y
            if pred.shape != y.shape:
                try:
                    pred = pred.view_as(y)
                except RuntimeError:
                    raise ValueError(f"Shape mismatch: pred {pred.shape}, y {y.shape}")

            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_all).flatten()
    y_pred = np.concatenate(y_pred_all).flatten()

    # === Inverse transform (if applicable) ===
    if hasattr(dataloader.dataset, "inverse_transform_y"):
        try:
            y_pred = dataloader.dataset.inverse_transform_y(y_pred.reshape(-1, 1)).flatten()
            y_true = dataloader.dataset.inverse_transform_y(y_true.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"[{household_id}] Inverse transform failed: {e}")

    # === Compute metrics ===
    metrics = {
        "Household": household_id,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R^2": r2_score(y_true, y_pred)
    }

    # === Save evaluation metrics ===
    if eval_path:
        df = pd.DataFrame([metrics])
        if os.path.exists(eval_path):
            df.to_csv(eval_path, mode='a', header=False, index=False)
        else:
            df.to_csv(eval_path, index=False)
        print(f"[{household_id}] Evaluation metrics saved to: {eval_path}")

    # === Save prediction plot and values ===
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        y_time = np.array(dataloader.dataset.y_timestamps)

        # Save prediction plot
        plot_path = os.path.join(plot_dir, f"prediction_{household_id}.png")
        plot_prediction(
            y_true,
            y_pred,
            save_path=plot_path,
            time_index=y_time,
            household_id=household_id,
            resample_freq=config.get("resample_freq", "5min")  # 传入分辨率
        )

        print(f"[{household_id}] Prediction plot saved to: {plot_path}")

        # Save prediction values to CSV
        csv_path = os.path.join(plot_dir, f"prediction_{household_id}.csv")
        df_pred = pd.DataFrame({
            "timestamp": y_time[:len(y_pred)],
            "y_true": y_true[:len(y_pred)],
            "y_pred": y_pred[:len(y_pred)]
        })
        df_pred.to_csv(csv_path, index=False)
        print(f"[{household_id}] Prediction values saved to: {csv_path}")

    return metrics


