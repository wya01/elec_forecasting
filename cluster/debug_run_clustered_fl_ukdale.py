# cluster/debug_run_clustered_fl_ukdale.py

import os, sys, json, torch, flwr as fl
from flwr.simulation import start_simulation

project_root = "/content/drive/MyDrive/elec_forecasting"
os.chdir(project_root)
sys.path.append(project_root)

from cluster.client_def_cluster_ukdale import LSTMClient

# ---- 选择测试 cluster ID ----
cluster_id = "1"
with open("cluster/cluster_result_ukdale.json", "r") as f:
    cluster_map = json.load(f)

client_ids = cluster_map[f"cluster_{cluster_id}"]

base_target_name = "ukdale_1h_30min_t"
target_name = f"{base_target_name}/cluster_{cluster_id}"

config = {
    "target_name":        target_name,
    "value_col":          "Consumption(Wh)",
    "resample_freq":      "5min",
    "sum_target":         True,
    "window_size":        12,
    "prediction_horizon": 6,
    "hidden_size":        64,
    "num_layers":         2,
    "batch_size":         32,
    "lr":                 0.005,
    "local_epochs":       1,
    "bidirectional":      False,
    "normalize_x":        True,
    "cache_suffix":       "timefeat",
    "use_time_features":  True,
    "device":             "cpu",
    "patience":           1,
    "allowed_clients":    client_ids,
    "project_root":       project_root
}

data_dir = os.path.join(project_root, "data", "UK-DALE")
save_dir = os.path.join(project_root, "experiments", "clustered_fl", target_name)
os.makedirs(save_dir, exist_ok=True)

print(f"▶ Testing cluster_{cluster_id} with clients: {client_ids}")
print(f"📂 Save dir: {save_dir}")

def wrapped_client_fn(cid: str):
    idx = int(cid)
    household_id = client_ids[idx]
    print(f"🔁 Creating client: cid={cid} → {household_id}")
    return LSTMClient(
        cid=household_id,
        household_id=household_id,
        config=config,
        device=torch.device(config["device"]),
        data_dir=data_dir,
        save_dir=save_dir,
    )

def fit_config_fn(rnd):
    return {"server_round": rnd, "total_rounds": 1}

def eval_config_fn(rnd):
    return {"server_round": rnd, "total_rounds": 1, "save_plot": False}

NUM_CLIENTS = len(client_ids)

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    fraction_evaluate=1.0,
    min_evaluate_clients=NUM_CLIENTS,
    on_fit_config_fn=fit_config_fn,
    on_evaluate_config_fn=eval_config_fn,
)

start_simulation(
    client_fn=wrapped_client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
    client_resources={"num_cpus": 1},
)
