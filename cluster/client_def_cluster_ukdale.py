# cluster/client_def_cluster.py

import os, copy, torch, flwr as fl, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.lstm import LSTMForecaster
from utils.preprocess_ukdale_timefeat import load_or_preprocess_datasets
from utils.visualization import plot_prediction


class LSTMClient(fl.client.NumPyClient):
    def __init__(self, cid, household_id, config, device, data_dir, save_dir):
        self.cid = cid
        self.household_id = household_id
        self.cfg_static = config
        self.device = device
        self.data_dir = data_dir
        self.save_dir = save_dir

        # ✅ 指定 h5 文件所在目录和缓存目录
        h5_path = os.path.join(self.data_dir, "ukdale.h5")
        cache_dir = os.path.join(self.data_dir, "cached_ukdale")

        tr, va, te, _ = load_or_preprocess_datasets(config, h5_path=h5_path, cache_dir=cache_dir)

        # ✅ 安全检查 household 是否存在于缓存数据中
        assert self.household_id in tr, f"{self.household_id} not in cached dataset: {list(tr.keys())}"

        self.train_loader = DataLoader(tr[self.household_id], batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(va[self.household_id], batch_size=config["batch_size"])
        self.test_loader = DataLoader(te[self.household_id], batch_size=config["batch_size"])

        input_size = 3 if config.get("use_time_features", False) else 1

        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            bidirectional=config.get("bidirectional", False),
            output_size=1 if config.get("sum_target", True) else config["prediction_horizon"],
        ).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, params):
        state = self.model.state_dict()
        for k, np_v in zip(state.keys(), params):
            state[k] = torch.tensor(np_v)
        self.model.load_state_dict(state)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        patience = self.cfg_static.get("patience", 5)
        local_epochs = self.cfg_static.get("local_epochs", 1)
        best_state, best_val = None, float("inf")
        counter = 0

        for _ in range(local_epochs):
            self.model.train()
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                if y.ndim == 1: y = y.unsqueeze(1)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward(); self.optimizer.step()

            self.model.eval(); v_loss = 0.0
            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if y.ndim == 1: y = y.unsqueeze(1)
                    v_loss += self.criterion(self.model(X), y).item()
            v_loss /= len(self.val_loader)

            if v_loss < best_val:
                best_val, best_state = v_loss, copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience: break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            best_state = copy.deepcopy(self.model.state_dict())

        # Save best model
        target_name = self.cfg_static["target_name"]
        model_dir = os.path.join(
            self.cfg_static["project_root"],
            "models_output", "clustered_fl",
            target_name,
            f"round_{config['server_round']}"
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"best_model_{self.household_id}.pth")
        torch.save(best_state, model_path)

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                if y.ndim == 1: y = y.unsqueeze(1)
                y_true.append(y.cpu().numpy())
                y_pred.append(self.model(X).cpu().numpy())

        y_true = np.concatenate(y_true).flatten()
        y_pred = np.concatenate(y_pred).flatten()

        # Inverse transform
        if hasattr(self.test_loader.dataset, "inverse_transform_y"):
            y_true = self.test_loader.dataset.inverse_transform_y(y_true.reshape(-1,1)).flatten()
            y_pred = self.test_loader.dataset.inverse_transform_y(y_pred.reshape(-1,1)).flatten()

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        hid_short = self.household_id.replace("_Wh", "")
        print(f"[{hid_short}] round={config['server_round']:>2}  R²={r2:.3f}  MAE={mae:.1f}")

        # Save evaluation metrics
        csv_main = os.path.join(self.save_dir, "eval_metrics.csv")
        pd.DataFrame([{
            "Household": hid_short,
            "Round":     config["server_round"],
            "R^2":       r2,
            "MAE":       mae,
            "RMSE":      rmse,
        }]).to_csv(csv_main, mode="a", header=not os.path.exists(csv_main), index=False)

        # Optional: save prediction plot
        if config.get("save_plot", False):
            plot_dir = os.path.join(self.save_dir, "plot_predictions")
            os.makedirs(plot_dir, exist_ok=True)
            ts = np.array(self.test_loader.dataset.y_timestamps)
            plot_prediction(
                y_true, y_pred,
                save_path=os.path.join(plot_dir, f"prediction_{hid_short}.png"),
                time_index=ts,
                household_id=hid_short
            )
            pd.DataFrame({"timestamp": ts[:len(y_pred)],
                          "y_true":   y_true[:len(y_pred)],
                          "y_pred":   y_pred[:len(y_pred)]
                         }).to_csv(os.path.join(plot_dir, f"prediction_{hid_short}.csv"), index=False)

        return 0.0, len(self.test_loader.dataset), {"r2": r2, "mae": mae, "rmse": rmse}
