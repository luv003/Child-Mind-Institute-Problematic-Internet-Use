import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Any

from config import DATA_PATH, COLS
from utils import logger

class HybridDataset(Dataset):
    def __init__(self, tabular_data: np.ndarray, ids: np.ndarray, targets: np.ndarray = None, cat_len=9, is_test=False):
        self.categorical_data = torch.LongTensor(tabular_data[:, :cat_len])
        self.numerical_data = torch.FloatTensor(tabular_data[:, cat_len:-1])
        self.ts_indicator = tabular_data[:, -1]
        self.ids = ids
        self.is_test = is_test
        if targets is not None:
            self.targets = torch.LongTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.numerical_data)

    def __getitem__(self, idx: int):
        categorical = self.categorical_data[idx]
        numerical = self.numerical_data[idx]
        if self.ts_indicator[idx] == 1:
            file_path = os.path.join(
                DATA_PATH,
                f"series_{'test' if self.is_test else 'train'}.parquet/id={self.ids[idx]}/part-0.parquet",
            )
            if not os.path.exists(file_path):
                logger.warning(f"Time-series file not found for id={self.ids[idx]}, using zeros.")
                time_series = torch.zeros((1, 7))
            else:
                time_series_df = pd.read_parquet(file_path)
                if len(time_series_df) == 0:
                    logger.warning(f"No time-series data for id={self.ids[idx]}, using zeros.")
                    time_series = torch.zeros((1, 7))
                else:
                    numerical_ts = time_series_df.iloc[:, [1, 2, 3, 4, 5]].values
                    timestamp = (time_series_df.iloc[:, 9] / 5000000000).round(3).values
                    weekday = time_series_df.iloc[:, 10].values - 1
                    weekday = np.clip(weekday, 0, 6)
                    time_series_arr = np.column_stack([numerical_ts, timestamp, weekday])
                    time_series = torch.FloatTensor(time_series_arr[::100])
        else:
            time_series = torch.zeros((1, 7))
        if self.targets is not None:
            target = self.targets[idx]
            return categorical, numerical, time_series, target
        return categorical, numerical, time_series

def collate_fn(batch: List[Any]):
    categorical, numerical, time_series, targets = zip(*batch)
    time_series_padded = pad_sequence(time_series, batch_first=True)
    return (
        torch.stack(categorical),
        torch.stack(numerical),
        time_series_padded,
        torch.stack(targets),
    )

def collate_fn_test(batch: List[Any]):
    categorical, numerical, time_series = zip(*batch)
    time_series_padded = pad_sequence(time_series, batch_first=True)
    return (
        torch.stack(categorical),
        torch.stack(numerical),
        time_series_padded,
    )
