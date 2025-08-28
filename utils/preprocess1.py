# preprocess1.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# ==== Dataset class ====
class PowerDataset(Dataset):
    def __init__(self, series, timestamps, window_size, prediction_horizon, sum_target=True, normalize_x=True):
        self.series = series
        self.timestamps = timestamps
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.sum_target = sum_target
        self.normalize_x = normalize_x
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X, self.y, self.y_timestamps = self.create_samples()

        if self.normalize_x:
            num_samples, length = self.X.shape
            self.X = self.scaler_x.fit_transform(self.X.reshape(num_samples, -1)).reshape(num_samples, length)

        self.y = self.scaler_y.fit_transform(self.y)

    def create_samples(self):
        X, y, ts = [], [], []
        for i in range(len(self.series) - self.window_size - self.prediction_horizon + 1):
            X.append(self.series[i: i + self.window_size])
            future = self.series[i + self.window_size: i + self.window_size + self.prediction_horizon]
            if self.sum_target:
                y.append([future.sum()])
            else:
                y.append(future)
            ts.append(self.timestamps[i + self.window_size + self.prediction_horizon - 1])
        return np.stack(X), np.stack(y), ts

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def inverse_transform_y(self, y_scaled):
        y_scaled = np.array(y_scaled).reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled)


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


# ==== Full preprocessing ====
def preprocess_all_households(data_dir, window_size=12, prediction_horizon=1,
                               train_ratio=0.7, val_ratio=0.1, resample_freq=None,
                               value_col="Consumption(Wh)", sum_target=True, normalize_x=True):
    train_datasets, val_datasets, test_datasets, lengths = {}, {}, {}, {}

    for file in sorted(os.listdir(data_dir)):
        if file.endswith('_Wh.csv'):
            household_id = file.replace(".csv", "")  # 保留 "H1_Wh" 格式
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

            train_dataset = PowerDataset(train_series, train_timestamps, window_size, prediction_horizon, sum_target=sum_target, normalize_x=normalize_x)
            val_dataset = PowerDataset(val_series, val_timestamps, window_size, prediction_horizon, sum_target=sum_target, normalize_x=normalize_x)
            test_dataset = PowerDataset(test_series, test_timestamps, window_size, prediction_horizon, sum_target=sum_target, normalize_x=normalize_x)

            train_datasets[household_id] = train_dataset
            val_datasets[household_id] = val_dataset
            test_datasets[household_id] = test_dataset
            lengths[household_id] = {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            }

    return train_datasets, val_datasets, test_datasets, lengths


# ==== Cached version ====
def load_or_preprocess_datasets(config, data_dir, cache_dir="data/cached"):
    os.makedirs(cache_dir, exist_ok=True)

    sum_flag = "sum" if config.get("sum_target", True) else "seq"
    norm_flag = "normX" if config.get("normalize_x", True) else "rawX"
    suffix_flag = config.get("cache_suffix", "")  # 可选，如 "clustered"

    cache_filename = (
        f"dataset_w{config['window_size']}_h{config['prediction_horizon']}_"
        f"{sum_flag}_{norm_flag}_resample{config.get('resample_freq', 'none')}"
        f"{f'_{suffix_flag}' if suffix_flag else ''}.pkl"
    )
    dataset_cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(dataset_cache_path):
        try:
            print("Loading cached dataset from:", dataset_cache_path)
            with open(dataset_cache_path, "rb") as f:
                cache = pickle.load(f)
            for k in ["train", "val", "test"]:
                for hh_id in cache[k]:
                    dataset = cache[k][hh_id]
                    assert len(dataset.X) == len(dataset.y) == len(dataset.y_timestamps)
            return cache['train'], cache['val'], cache['test'], cache['lengths']
        except Exception as e:
            print("Cache corrupted:", e)
            print("Reprocessing...")
            os.remove(dataset_cache_path)

    print("Preprocessing raw data...")
    train_datasets, val_datasets, test_datasets, lengths = preprocess_all_households(
        data_dir=data_dir,
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        resample_freq=config.get('resample_freq', None),
        value_col=config.get('value_col', "Consumption(Wh)"),
        sum_target=config.get("sum_target", True),
        normalize_x=config.get("normalize_x", True)
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
