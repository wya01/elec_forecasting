# federated/client_def.py

import os, copy, torch, flwr as fl, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.lstm import LSTMForecaster
from utils.preprocess import load_or_preprocess_datasets
from utils.visualization import plot_prediction


class LSTMClient(fl.client.NumPyClient):
    # ---------------------- 初始化 ---------------------- #
    def __init__(self, cid, config, device, data_dir, save_dir):
        self.cid = cid
        self.household_id = f"H{int(cid)+1}"
        if "allowed_clients" in config:
            assert self.household_id in config["allowed_clients"], \
                f"Household {self.household_id} not allowed for this cluster"
        self.cfg_static = config              # 保存静态超参
        self.device = device
        self.data_dir = data_dir
        self.save_dir = save_dir

        # ----- 数据 -----
        tr, va, te, _ = load_or_preprocess_datasets(config, data_dir)
        self.train_loader = DataLoader(tr[self.household_id], batch_size=config["batch_size"], shuffle=True)
        self.val_loader = DataLoader(va[self.household_id], batch_size=config["batch_size"])
        self.test_loader = DataLoader(te[self.household_id], batch_size=config["batch_size"])

        # ----- 模型 -----
        self.model = LSTMForecaster(
            input_size = 1,
            hidden_size= config["hidden_size"],
            num_layers = config["num_layers"],
            bidirectional = config.get("bidirectional", False),
            output_size = 1 if config.get("sum_target", True) else config["prediction_horizon"],
        ).to(device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

    # ----------------- Flower 必备接口 ----------------- #
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, params):
        state = self.model.state_dict()
        for k, np_v in zip(state.keys(), params):
            state[k] = torch.tensor(np_v)
        self.model.load_state_dict(state)

    # =========================== fit =========================== #
    def fit(self, parameters, config):
        """
        `config` 来自 on_fit_config_fn，至少包含：
            server_round : 当前轮
            total_rounds : 总轮数
        """
        self.set_parameters(parameters)

        patience = self.cfg_static.get("patience", 5)
        local_epochs = self.cfg_static.get("local_epochs", 1)
        best_state, best_val = None, float("inf")
        counter = 0

        for _ in range(local_epochs):
            # ---- 训练 ----
            self.model.train()
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                if y.ndim == 1: y = y.unsqueeze(1)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward(); self.optimizer.step()

            # ---- 验证 ----
            self.model.eval(); v_loss = 0.0
            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    if y.ndim == 1: y = y.unsqueeze(1)
                    v_loss += self.criterion(self.model(X), y).item()
            v_loss /= len(self.val_loader)

            # ---- Early‑Stopping ----
            if v_loss < best_val:
                best_val, best_state = v_loss, copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience: break

        # ---- 保存本轮最佳权重 ----
        if best_state is not None:
            self.model.load_state_dict(best_state)
        else:
            # 若没有 early-stopping 的最佳状态，也存当前状态（防止返回 None 报错）
            best_state = copy.deepcopy(self.model.state_dict())

        # 构造保存路径
        model_dir = os.path.join(
            self.cfg_static["project_root"],
            "models_output",
            "federated_lstm",
            self.cfg_static["target_name"],
            f"round_{config['server_round']}"
        )
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"best_model_{self.household_id}.pth")
        torch.save(best_state, model_path)

        # 确保返回合法参数（防止 Flower 报 TypeError）
        return self.get_parameters(), len(self.train_loader.dataset), {}

    # ======================== evaluate ========================= #
    def evaluate(self, parameters, config):
        """
        `config` 来自 on_evaluate_config_fn，包含：
            server_round / total_rounds / save_plot
        """
        self.set_parameters(parameters); self.model.eval()

        # ---- 预测 ----
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                if y.ndim == 1: y = y.unsqueeze(1)
                y_true.append(y.cpu().numpy())
                y_pred.append(self.model(X).cpu().numpy())

        y_true = np.concatenate(y_true).flatten()
        y_pred = np.concatenate(y_pred).flatten()

        # ---- 反归一化 ----
        if hasattr(self.test_loader.dataset, "inverse_transform_y"):
            y_true = self.test_loader.dataset.inverse_transform_y(y_true.reshape(-1,1)).flatten()
            y_pred = self.test_loader.dataset.inverse_transform_y(y_pred.reshape(-1,1)).flatten()

        # ---- 指标 ----
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"[{self.household_id}] round={config['server_round']:>2}  R²={r2:.3f}  MAE={mae:.1f}")

        # ---- 追加到主表 ----
        csv_main = os.path.join(self.save_dir, "eval_metrics.csv")
        pd.DataFrame([{
            "Household": self.household_id,
            "Round":     config["server_round"],
            "R^2":       r2,
            "MAE":       mae,
            "RMSE":      rmse,
        }]).to_csv(csv_main, mode="a", header=not os.path.exists(csv_main), index=False)

        # ---- 仅在需要时保存图/户级 CSV ----
        if config.get("save_plot", False):
            plot_dir = os.path.join(self.save_dir, "plot_predictions")
            os.makedirs(plot_dir, exist_ok=True)
            ts = np.array(self.test_loader.dataset.y_timestamps)
            plot_prediction(
                y_true, y_pred,
                save_path=os.path.join(plot_dir, f"prediction_{self.household_id}.png"),
                time_index=ts,
                household_id=self.household_id  # 传入户名
            )

            pd.DataFrame({"timestamp": ts[:len(y_pred)],
                          "y_true":   y_true[:len(y_pred)],
                          "y_pred":   y_pred[:len(y_pred)]
                         }).to_csv(os.path.join(plot_dir, f"prediction_{self.household_id}.csv"),
                                   index=False)

        return 0.0, len(self.test_loader.dataset), {"r2": r2, "mae": mae, "rmse": rmse}
