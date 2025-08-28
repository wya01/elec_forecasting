# federated/run_simulation_lstm_1h_1h.py

import os, sys, torch, flwr as fl
from flwr.simulation import start_simulation

# ---------- 项目根路径 ----------
project_root = "/content/drive/MyDrive/elec_forecasting"
os.chdir(project_root)
sys.path.append(project_root)

# ---------- 导入客户端 ----------
from federated.client_def import LSTMClient

# ---------- 全局配置 ----------
config = {
    "target_name":        "1h_30min",
    "value_col":          "Consumption(Wh)",
    "resample_freq":      "5min",
    "sum_target":         True,
    "window_size":        12,
    "prediction_horizon": 6,
    "hidden_size":        64,
    "num_layers":         2,
    "batch_size":         32,
    "lr":                 0.005,
    "local_epochs":       5,
    "bidirectional":      False,
    "normalize_x":        True,
    "device":             "cpu",
    "patience":           3,
    "project_root":       project_root
}

# ---------- 路径 ----------
data_dir = os.path.join(project_root, "data", "community_2024")
save_dir = os.path.join(project_root, "experiments", "federated_lstm", config["target_name"])
os.makedirs(save_dir, exist_ok=True)
device = torch.device(config["device"])

# ---------- Flower 客户端工厂 ----------
def client_fn(cid: str):
    return LSTMClient(
        cid = cid,
        config = config,
        device = device,
        data_dir = data_dir,
        save_dir = save_dir,
    ).to_client()

# ---------- 向客户端注入 round 等信息 ----------
def fit_config_fn(rnd: int):
    return {"server_round": rnd, "total_rounds": NUM_ROUNDS}

def eval_config_fn(rnd: int):
    """外部指定时才让客户端保存图 / 户级 CSV"""
    return {
        "server_round": rnd,
        "total_rounds": NUM_ROUNDS,
        "save_plot":   False,   # 训练阶段不画图
    }

# ---------- 运行参数 ----------
NUM_CLIENTS = 20
NUM_ROUNDS = 25
FRACTION_FIT = 0.5     # 每轮随机挑 10/20 = 50% 客户


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit = FRACTION_FIT,
        min_fit_clients = int(NUM_CLIENTS * FRACTION_FIT),
        min_available_clients = NUM_CLIENTS,
        on_fit_config_fn = fit_config_fn,
        on_evaluate_config_fn = eval_config_fn,
    )

    start_simulation(
        client_fn = client_fn,
        num_clients = NUM_CLIENTS,
        config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy = strategy,
        client_resources = {"num_cpus": 1},
    )
