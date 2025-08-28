# utils/preprocess.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# ==== Dataset class ====
class PowerDataset(Dataset):
    def __init__(self, series, timestamps, window_size, prediction_horizon, sum_target=True):
        self.series = series
        self.timestamps = timestamps
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.sum_target = sum_target
        self.scaler = MinMaxScaler()

        self.X, self.y, self.y_timestamps = self.create_samples()

        # 对 y 做归一化（例如变成 0~1 区间）
        self.y = self.scaler.fit_transform(self.y)

    def create_samples(self):
        X, y, ts = [], [], []
        for i in range(len(self.series) - self.window_size - self.prediction_horizon + 1):
            X.append(self.series[i: i + self.window_size])
            future = self.series[i + self.window_size: i + self.window_size + self.prediction_horizon]
            if self.sum_target:
                y.append([future.sum()])  # scalar (wrapped in list → shape [1])
            else:
                y.append(future)          # vector
            ts.append(self.timestamps[i + self.window_size + self.prediction_horizon - 1])

        assert len(X) == len(y) == len(ts), "Sample alignment error."
        return np.stack(X), np.stack(y), ts

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1),  # [window_size, 1]
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def inverse_transform_y(self, y_scaled):
        """
        将预测结果从归一化状态逆变换为真实能耗值
        y_scaled: shape (n, 1) or (n,)
        """
        y_scaled = np.array(y_scaled).reshape(-1, 1)  # ensure 2D input
        return self.scaler.inverse_transform(y_scaled)



# ==== Load single household CSV ====
def load_household_csv(path, resample_freq=None, value_col="Consumption(Wh)"):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.columns = df.columns.str.strip()
    df = df.sort_index()

    if resample_freq:
        df = df.resample(resample_freq).sum().dropna()

    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in {path}. Available columns: {df.columns.tolist()}")

    return df


# ==== Full preprocessing for all households ====
def preprocess_all_households(data_dir, window_size=12, prediction_horizon=1,
                               train_ratio=0.7, val_ratio=0.1, resample_freq=None, value_col="Consumption(Wh)", sum_target=True):
    train_datasets, val_datasets, test_datasets, lengths = {}, {}, {}, {}

    for file in sorted(os.listdir(data_dir)):
        if file.endswith('_Wh.csv'):
            household_id = file.split('_')[0]
            df = load_household_csv(os.path.join(data_dir, file), resample_freq=resample_freq, value_col=value_col)
            series = df[value_col].astype(np.float32).values
            timestamps = df.index.to_list()

            total_len = len(series)
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))

            train_series = series[:train_end]
            val_series = series[train_end - window_size - prediction_horizon + 1:val_end]
            test_series = series[val_end - window_size - prediction_horizon + 1:]

            train_timestamps = timestamps[:train_end]
            val_timestamps = timestamps[train_end - window_size - prediction_horizon + 1:val_end]
            test_timestamps = timestamps[val_end - window_size - prediction_horizon + 1:]

            train_dataset = PowerDataset(train_series, train_timestamps, window_size, prediction_horizon, sum_target=sum_target)
            val_dataset = PowerDataset(val_series, val_timestamps, window_size, prediction_horizon, sum_target=sum_target)
            test_dataset = PowerDataset(test_series, test_timestamps, window_size, prediction_horizon, sum_target=sum_target)

            train_datasets[household_id] = train_dataset
            val_datasets[household_id] = val_dataset
            test_datasets[household_id] = test_dataset
            lengths[household_id] = {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            }

    return train_datasets, val_datasets, test_datasets, lengths


# ==== Cached version of preprocessing ====
def load_or_preprocess_datasets(config, data_dir, cache_dir="data/cached"):
    os.makedirs(cache_dir, exist_ok=True)
    sum_flag = "sum" if config.get("sum_target", True) else "seq"
    cache_filename = f"dataset_w{config['window_size']}_h{config['prediction_horizon']}_{sum_flag}_resample{config.get('resample_freq', 'none')}.pkl"

    dataset_cache_path = os.path.join(cache_dir, cache_filename)

    # 检查是否已有缓存
    if os.path.exists(dataset_cache_path):
        try:
            print("Loading cached dataset from:", dataset_cache_path)
            with open(dataset_cache_path, "rb") as f:
                cache = pickle.load(f)
            # 检查是否每个数据集都一致长度
            for k in ["train", "val", "test"]:
                for hh_id in cache[k]:
                    dataset = cache[k][hh_id]
                    assert len(dataset.X) == len(dataset.y) == len(dataset.y_timestamps)
            return cache['train'], cache['val'], cache['test'], cache['lengths']
        except Exception as e:
            print("Cache corrupted or inconsistent:", e)
            print("Reprocessing and overwriting the cache...")
            os.remove(dataset_cache_path)

    print("Preprocessing raw data...")
    train_datasets, val_datasets, test_datasets, lengths = preprocess_all_households(
        data_dir=data_dir,
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        resample_freq=config.get('resample_freq', None),
        value_col=config.get('value_col', "Consumption(Wh)"),
        sum_target=config.get("sum_target", True)
    )

    with open(dataset_cache_path, "wb") as f:
        pickle.dump({
            "train": train_datasets,
            "val": val_datasets,
            "test": test_datasets,
            "lengths": lengths
        }, f)
    print("Cached at:", dataset_cache_path)

    return train_datasets, val_datasets, test_datasets, lengths
