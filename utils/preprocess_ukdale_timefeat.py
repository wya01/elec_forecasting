# preprocess_ukdale_timefeat.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import h5py

# ==== Dataset class (same as your original) ====
class PowerDataset(Dataset):
    def __init__(self, series, timestamps, time_features, window_size, prediction_horizon, sum_target=True, normalize_x=True):
        self.series = series
        self.timestamps = timestamps
        self.time_features = time_features
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.sum_target = sum_target
        self.normalize_x = normalize_x

        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X, self.y, self.y_timestamps = self.create_samples()

        if self.normalize_x:
            num_samples, length, channels = self.X.shape
            self.X = self.scaler_x.fit_transform(self.X.reshape(num_samples, -1)).reshape(num_samples, length, channels)

        self.y = self.scaler_y.fit_transform(self.y)

    def create_samples(self):
        X, y, ts = [], [], []
        for i in range(len(self.series) - self.window_size - self.prediction_horizon + 1):
            x_values = self.series[i: i + self.window_size].reshape(-1, 1)
            x_time = self.time_features[i: i + self.window_size]
            x = np.hstack([x_values, x_time])
            X.append(x)

            future = self.series[i + self.window_size: i + self.window_size + self.prediction_horizon]
            if self.sum_target:
                y.append([future.sum()])
            else:
                y.append(future)

            ts.append(self.timestamps[i + self.window_size + self.prediction_horizon - 1])
        return np.stack(X), np.stack(y), ts

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def inverse_transform_y(self, y_scaled):
        y_scaled = np.array(y_scaled).reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled)

# ==== Preprocess function for UK-DALE ====
def preprocess_ukdale_hdf5(h5_path, building_ids, window_size, prediction_horizon,
                           train_ratio=0.7, val_ratio=0.1, resample_freq="5min",
                           sum_target=True, normalize_x=True):
    train_datasets, val_datasets, test_datasets, lengths = {}, {}, {}, {}

    with h5py.File(h5_path, "r") as f:
        for bid in building_ids:
            household_id = f"building{bid}"
            print(f"Processing {household_id}...")
            data = pd.read_hdf(h5_path, key=f"{household_id}/elec/meter1")
            df = data.copy()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['power']  # Flatten the ('power', 'apparent') multiindex

            # Resample to 5min average, convert W → Wh
            df = df.resample(resample_freq).mean().dropna()
            df['Consumption(Wh)'] = df['power'] * (pd.Timedelta(resample_freq).total_seconds() / 3600.0)

            timestamps = df.index.to_list()
            series = df['Consumption(Wh)'].astype(np.float32).values

            # Time features
            hour = np.array([ts.hour for ts in timestamps]).reshape(-1, 1)
            weekday = np.array([ts.weekday() for ts in timestamps]).reshape(-1, 1)
            time_features = np.hstack([hour, weekday]).astype(np.float32)

            # Split train / val / test
            total_len = len(series)
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))

            def slice_data(series, tf, ts, start, end):
                return series[start:end], tf[start:end], ts[start:end]

            train_s, train_tf, train_ts = slice_data(series, time_features, timestamps, 0, train_end)
            val_s, val_tf, val_ts = slice_data(series, time_features, timestamps, train_end - window_size - prediction_horizon + 1, val_end)
            test_s, test_tf, test_ts = slice_data(series, time_features, timestamps, val_end - window_size - prediction_horizon + 1, total_len)

            train_dataset = PowerDataset(train_s, train_ts, train_tf, window_size, prediction_horizon, sum_target, normalize_x)
            val_dataset = PowerDataset(val_s, val_ts, val_tf, window_size, prediction_horizon, sum_target, normalize_x)
            test_dataset = PowerDataset(test_s, test_ts, test_tf, window_size, prediction_horizon, sum_target, normalize_x)

            train_datasets[household_id] = train_dataset
            val_datasets[household_id] = val_dataset
            test_datasets[household_id] = test_dataset
            lengths[household_id] = {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            }

    return train_datasets, val_datasets, test_datasets, lengths

# ==== Cached Wrapper ====
def load_or_preprocess_datasets(config, h5_path, cache_dir="data/cached_ukdale"):
    os.makedirs(cache_dir, exist_ok=True)

    sum_flag = "sum" if config.get("sum_target", True) else "seq"
    norm_flag = "normX" if config.get("normalize_x", True) else "rawX"
    suffix_flag = config.get("cache_suffix", "timefeat")

    cache_filename = (
        f"ukdale_w{config['window_size']}_h{config['prediction_horizon']}_"
        f"{sum_flag}_{norm_flag}_resample{config.get('resample_freq', 'none')}"
        f"_{suffix_flag}.pkl"
    )
    dataset_cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(dataset_cache_path):
        try:
            print("Loading cached UK-DALE dataset from:", dataset_cache_path)
            with open(dataset_cache_path, "rb") as f:
                cache = pickle.load(f)
            return cache['train'], cache['val'], cache['test'], cache['lengths']
        except Exception as e:
            print("Cache corrupted:", e)
            os.remove(dataset_cache_path)

    print("Preprocessing raw UK-DALE data (with time features)...")
    train_datasets, val_datasets, test_datasets, lengths = preprocess_ukdale_hdf5(
        h5_path=h5_path,
        building_ids=config.get("building_ids", [1, 2, 3, 4, 5]),
        window_size=config['window_size'],
        prediction_horizon=config['prediction_horizon'],
        resample_freq=config.get('resample_freq', '5min'),
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
