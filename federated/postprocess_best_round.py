# federated/postprocess_best_round.py

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== Project path setup =====
project_root = "/content/drive/MyDrive/elec_forecasting"
os.chdir(project_root)
sys.path.append(project_root)

# ===== Import modules =====
from federated.client_def import LSTMClient

# ===== Paths =====
target_name = "6h_1h"
save_dir = os.path.join(project_root, "experiments", "federated_lstm", target_name)
data_dir = os.path.join(project_root, "data", "community_2024")
csv_best_path = os.path.join(save_dir, "eval_metrics_best.csv")

# ===== Client config =====
config = {
    "target_name": target_name,
    "value_col": "Consumption(Wh)",
    "resample_freq": "30min",
    "sum_target": True,
    "window_size": 12,
    "prediction_horizon": 2,
    "hidden_size": 64,
    "num_layers": 2,
    "batch_size": 32,
    "lr": 0.005,
    "bidirectional": False,
    "normalize_x": True,
    "device": "cpu",
    "project_root": project_root,
}

device = torch.device(config["device"])
df_best = pd.read_csv(csv_best_path)

y_true_all, y_pred_all = [], []

# ===== Iterate over households =====
for _, row in df_best.iterrows():
    hid, rnd = row["Household"], int(row["Round"])
    cid = str(int(hid[1:]) - 1)

    model_path = os.path.join(
        project_root, "models_output", "federated_lstm", target_name,
        f"round_{rnd}", f"best_model_{hid}.pth"
    )

    if not os.path.exists(model_path):
        print(f"[{hid}] round_{rnd} weight missing, skipped.")
        continue

    print(f"[{hid}] Generating plots (round={rnd})")

    client = LSTMClient(
        cid=cid,
        config=config,
        device=device,
        data_dir=data_dir,
        save_dir=save_dir,
    )
    client.model.load_state_dict(torch.load(model_path, map_location=device))

    eval_cfg = {
        "server_round": rnd,
        "total_rounds": rnd,
        "save_plot": True,
    }
    client.evaluate(client.get_parameters(), eval_cfg)

    # Collect predictions for ALL
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in client.test_loader:
            X, y = X.to(device), y.to(device)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            pred = client.model(X)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    ds = client.test_loader.dataset
    if hasattr(ds, "inverse_transform_y"):
        y_true = ds.inverse_transform_y(y_true.reshape(-1, 1)).flatten()
        y_pred = ds.inverse_transform_y(y_pred.reshape(-1, 1)).flatten()

    y_true_all.append(y_true)
    y_pred_all.append(y_pred)

print("Household plots completed.")

# ===== Compute ALL row =====
if len(y_true_all) > 0:
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae_all  = mean_absolute_error(y_true_all, y_pred_all)
    r2_all   = r2_score(y_true_all, y_pred_all)

    row_all = pd.DataFrame([{
        "Household": "ALL",
        "Round":     -1,
        "R^2":       r2_all,
        "MAE":       mae_all,
        "RMSE":      rmse_all,
    }])

    row_all.to_csv(csv_best_path, mode="a", header=False, index=False)
    print(f"[ALL]  R²={r2_all:.3f}  MAE={mae_all:.1f}  RMSE={rmse_all:.1f}")
else:
    print("No predictions collected, skipping ALL row.")
