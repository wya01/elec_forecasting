# utils/preprocess1_time_weather.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# ==== Dataset class ====
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
            future = self.series[i + self.window_size: i + self.window_size + self.prediction_horizon]

            if np.isnan(x).any() or np.isnan(future).any():
                continue

            X.append(x)
            y.append([future.sum()] if self.sum_target else future)
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


# ==== 加载天气数据 ====
def load_weather_csv(weather_path, resample_freq=None):
    df = pd.read_csv(weather_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.set_index("date")
    df = df.sort_index()
    df.columns = df.columns.str.strip()

    for col in ["drybulb", "soltot", "rain"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if resample_freq:
        df = df.resample(resample_freq).mean().interpolate()

    return df[["drybulb", "soltot", "rain"]]


# ==== 加载住户数据 ====
def load_household_csv(path, resample_freq=None, value_col="Consumption(Wh)"):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.columns = df.columns.str.strip()
    df = df.sort_index()
    if resample_freq:
        df = df.resample(resample_freq).sum().dropna()
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in {path}. Available: {df.columns.tolist()}")
    return df


# ==== 主处理函数 ====
def preprocess_all_households(data_dir, weather_path, window_size=12, prediction_horizon=1,
                               train_ratio=0.7, val_ratio=0.1, resample_freq=None,
                               value_col="Consumption(Wh)", sum_target=True, normalize_x=True):
    weather_df = load_weather_csv(weather_path, resample_freq)

    train_datasets, val_datasets, test_datasets, lengths = {}, {}, {}, {}

    for file in sorted(os.listdir(data_dir)):
        if file.endswith('_Wh.csv'):
            household_id = file.replace(".csv", "")
            df = load_household_csv(os.path.join(data_dir, file), resample_freq, value_col)
            series = df[value_col].astype(np.float32).values
            timestamps = df.index.to_list()

            hour = np.array([ts.hour for ts in timestamps]).reshape(-1, 1)
            weekday = np.array([ts.weekday() for ts in timestamps]).reshape(-1, 1)
            weather_matched = weather_df.reindex(df.index).fillna(method='ffill').dropna().values.astype(np.float32)

            time_features = np.hstack([hour, weekday, weather_matched])

            total_len = len(series)
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))

            def slice_data(series, tf, ts, start, end):
                return series[start:end], tf[start:end], ts[start:end]

            train_s, train_tf, train_ts = slice_data(series, time_features, timestamps, 0, train_end)
            val_s, val_tf, val_ts = slice_data(series, time_features, timestamps, train_end - window_size - prediction_horizon + 1, val_end)
            test_s, test_tf, test_ts = slice_data(series, time_features, timestamps, val_end - window_size - prediction_horizon + 1, total_len)

            train_datasets[household_id] = PowerDataset(train_s, train_ts, train_tf, window_size, prediction_horizon, sum_target, normalize_x)
            val_datasets[household_id] = PowerDataset(val_s, val_ts, val_tf, window_size, prediction_horizon, sum_target, normalize_x)
            test_datasets[household_id] = PowerDataset(test_s, test_ts, test_tf, window_size, prediction_horizon, sum_target, normalize_x)
            lengths[household_id] = {'train': len(train_datasets[household_id]), 'val': len(val_datasets[household_id]), 'test': len(test_datasets[household_id])}

    return train_datasets, val_datasets, test_datasets, lengths


# ==== 主入口 ====
def load_or_preprocess_datasets(config, data_dir, cache_dir="data/cached", force_reload=False):
    os.makedirs(cache_dir, exist_ok=True)

    sum_flag = "sum" if config.get("sum_target", True) else "seq"
    norm_flag = "normX" if config.get("normalize_x", True) else "rawX"
    suffix_flag = config.get("cache_suffix", "timeweather")

    cache_filename = (
        f"dataset_w{config['window_size']}_h{config['prediction_horizon']}_"
        f"{sum_flag}_{norm_flag}_resample{config.get('resample_freq', 'none')}_"
        f"{suffix_flag}.pkl"
    )
    dataset_cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(dataset_cache_path) and not force_reload:
        try:
            print("Loading cached dataset from:", dataset_cache_path)
            with open(dataset_cache_path, "rb") as f:
                cache = pickle.load(f)
            return cache['train'], cache['val'], cache['test'], cache['lengths']
        except Exception as e:
            print("⚠ Cache corrupted:", e)
            os.remove(dataset_cache_path)

    print("⚙ Preprocessing raw data (with time + weather features)...")

    weather_path = os.path.join(data_dir, "weather.csv")
    train_datasets, val_datasets, test_datasets, lengths = preprocess_all_households(
        data_dir, weather_path,
        config["window_size"], config["prediction_horizon"],
        config.get("train_ratio", 0.7), config.get("val_ratio", 0.1),
        config.get("resample_freq", None),
        config.get("value_col", "Consumption(Wh)"),
        config.get("sum_target", True), config.get("normalize_x", True)
    )

    with open(dataset_cache_path, "wb") as f:
        pickle.dump({
            "train": train_datasets,
            "val": val_datasets,
            "test": test_datasets,
            "lengths": lengths
        }, f)
    print("✅ Cached at:", dataset_cache_path)

    return train_datasets, val_datasets, test_datasets, lengths

