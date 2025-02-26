import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datetime import datetime
import os


"""Data processing that runs on the cleaned dataset. Some elements are hard-coded."""


class DataTransform(nn.Module):
    def __init__(self, device, num_transform_cols):
        self.device = device
        self.num_transform_cols = num_transform_cols

    def fit(self, x: torch.Tensor):
        raise NotImplementedError

    def transform(self, x: torch.Tensor, transform_col=None):
        raise NotImplementedError

    def reverse(self, transformed: torch.Tensor, reverse_col=0):
        raise NotImplementedError

    def set_device(self, device='cuda'):
        raise NotImplementedError

    def change_transform_cols(self, new_val):
        self.num_transform_cols = new_val


class MinMaxNorm(DataTransform):
    def __init__(self, device='cuda', num_transform_cols=3):
        super().__init__(device, num_transform_cols)
        self.min_val = None
        self.max_val = None

    def fit(self, x: torch.Tensor):
        # x is expected to have shape [1, N, num_features]
        self.max_val = torch.max(
            x[..., :self.num_transform_cols], dim=1, keepdim=True).values.to(self.device).float()
        self.min_val = torch.min(
            x[..., :self.num_transform_cols], dim=1, keepdim=True).values.to(self.device).float()

    def transform(self, x: torch.Tensor, transform_col=None):
        x_transformed = x.clone()
        if transform_col is None:
            x_transformed[..., :self.num_transform_cols] = (
                x[..., :self.num_transform_cols] - self.min_val
            ) / (self.max_val - self.min_val)
        else:
            if transform_col >= self.num_transform_cols:
                raise IndexError(
                    f"transform_col ({transform_col}) must be less than num_transform_cols ({self.num_transform_cols})")
            x_transformed[..., transform_col:transform_col+1] = (
                x[..., transform_col:transform_col+1] -
                self.min_val[..., transform_col:transform_col+1]
            ) / (self.max_val[..., transform_col:transform_col+1] - self.min_val[..., transform_col:transform_col+1])
        return x_transformed

    def reverse(self, transformed: torch.Tensor, reverse_col=0):
        x_reversed = transformed.clone()
        x_reversed[..., reverse_col:reverse_col+1] = (
            transformed[..., reverse_col:reverse_col+1] *
            (self.max_val[..., reverse_col:reverse_col+1] -
             self.min_val[..., reverse_col:reverse_col+1])
        ) + self.min_val[..., reverse_col:reverse_col+1]
        return x_reversed

    def set_device(self, device='cuda'):
        self.device = device
        if self.min_val is not None and self.max_val is not None:
            if device == 'cpu':
                self.max_val = self.max_val.cpu().detach()
                self.min_val = self.min_val.cpu().detach()
            else:
                self.max_val = self.max_val.to(device)
                self.min_val = self.min_val.to(device)


class StandardScaleNorm(DataTransform):
    def __init__(self, device='cuda', num_transform_cols=3):
        self.mean = None
        self.std = None
        self.device = device
        self.num_transform_cols = num_transform_cols

    def fit(self, x: torch.Tensor):
        # x is expected to have shape [1, N, num_features]
        self.mean = x[..., :self.num_transform_cols].mean(
            dim=1, keepdim=True).to(self.device).float()
        self.std = x[..., :self.num_transform_cols].std(
            dim=1, keepdim=True).to(self.device).float()

    def transform(self, x: torch.Tensor, transform_col=None):
        x_transformed = x.clone()
        if transform_col is None:
            x_transformed[..., :self.num_transform_cols] = (
                x[..., :self.num_transform_cols] - self.mean
            ) / self.std
        else:
            if transform_col >= self.num_transform_cols:
                raise IndexError(
                    f"transform_col ({transform_col}) must be less than num_transform_cols ({self.num_transform_cols})")
            x_transformed[..., transform_col:transform_col+1] = (
                x[..., transform_col:transform_col+1] -
                self.mean[..., transform_col:transform_col+1]
            ) / self.std[..., transform_col:transform_col+1]
        return x_transformed

    def reverse(self, transformed: torch.Tensor, reverse_col=0):
        x_reversed = transformed.clone()
        x_reversed[..., reverse_col:reverse_col+1] = (
            transformed[..., reverse_col:reverse_col+1] *
            self.std[..., reverse_col:reverse_col+1]
        ) + self.mean[..., reverse_col:reverse_col+1]
        return x_reversed

    def set_device(self, device='cuda'):
        self.device = device
        if self.mean is not None and self.std is not None:
            if device == 'cpu':
                self.mean = self.mean.cpu().detach()
                self.std = self.std.cpu().detach()
            else:
                self.mean = self.mean.to(device)
                self.std = self.std.to(device)


class TransformSequence(DataTransform):
    def __init__(self, transforms, device):
        self.transforms = transforms
        self.device = device

    def fit(self, x: torch.Tensor):
        for t in self.transforms:
            t.fit(x)

    def transform(self, x: torch.Tensor):
        x = x.clone()
        for t in self.transforms:
            x = t.transform(x)

        return x

    def reverse(self, transformed: torch.Tensor):
        x = transformed.clone()
        for t in reversed(self.transforms):
            x = t.reverse(x)
        return x

    def set_device(self, new_device):
        self.device = new_device
        for t in self.transforms:
            t.set_device(new_device)


def formPairs(
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24,
        target_vars=1):
    """
    Given a time series tensor, forms sliding windows for inputs (X) and targets (Y).
    Assumes that x_tensor and y_tensor have the same length along dim=0.
    """
    assert x_tensor.shape[0] == y_tensor.shape[0]
    N = x_tensor.shape[0]
    x_start = x_start_hour - 1
    X, Y = [], []

    while (x_start + x_window + x_y_gap + y_window) < N:
        x = x_tensor[x_start: x_start + x_window, :]
        y = y_tensor[x_start + x_window + x_y_gap:
                     x_start + x_window + x_y_gap + y_window, 0]
        X.append(x)
        Y.append(y)
        x_start += step_size

    X = torch.stack(X)  # Shape: [num_samples, x_window, num_features]
    Y = torch.stack(Y)  # Shape: [num_samples, y_window, num_features]
    return X, Y


def benchmark_preprocess(
        train_start_end=(datetime(2023, 2, 10), datetime(2024, 7, 1)),
        val_start_end=(datetime(2024, 7, 1), datetime(2024, 9, 1)),
        test_start_end=(datetime(2024, 9, 1), datetime(2025, 1, 6)),
        spatial=True,
        device='cpu',
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        train_transforms=[StandardScaleNorm(device='cpu')]):
    """
    Loads the cleaned CSV data and performs time‐series splitting, transformation,
    windowing (pair formation), and creation of DataLoaders. The datasets and loaders
    are saved with a suffix indicating if the data is spatial or non-spatial.

    Returns the list of fitted transform objects.
    """
    if train_transforms is None:
        train_transforms = [StandardScaleNorm(device=device)]

    csv_path = "data/ercot_data_cleaned.csv"
    raw_df = pd.read_csv(csv_path)

    date_series = pd.to_datetime(raw_df["Unnamed: 0"])
    raw_df = raw_df.drop("Unnamed: 0", axis=1)

    if not spatial:
        raw_df = raw_df[["ACTUAL_NetLoad", "NetLoad_Error",
                         "NetLoad", "HoD", "DoW", "MoY"]]

    train_mask = (date_series >= train_start_end[0]) & (
        date_series < train_start_end[1])
    val_mask = (date_series >= val_start_end[0]) & (
        date_series < val_start_end[1])
    test_mask = (date_series >= test_start_end[0]) & (
        date_series < test_start_end[1])

    train_df = raw_df[train_mask]
    val_df = raw_df[val_mask]
    test_df = raw_df[test_mask]

    train_tensor = torch.tensor(train_df.to_numpy(), device=device).float()
    val_tensor = torch.tensor(val_df.to_numpy(), device=device).float()
    test_tensor = torch.tensor(test_df.to_numpy(), device=device).float()
    if spatial:
        for t in train_transforms:
            t.change_transform_cols(12)
    else:
        for t in train_transforms:
            t.change_transform_cols(3)
    print(train_tensor)
    for t in train_transforms:
        t.fit(train_tensor.unsqueeze(0).to(device))
        train_tensor = t.transform(train_tensor)
        print(train_tensor)

    X_train, Y_train = formPairs(
        train_tensor, train_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
    X_val, Y_val = formPairs(
        val_tensor, val_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
    X_test, Y_test = formPairs(
        test_tensor, test_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    pin_memory = True if device == 'cuda' else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    suffix = "spatial" if spatial else "non_spatial"
    print(f"{suffix} Train loader shape: {next(iter(train_loader))[0].shape}")
    print(f"{suffix} Val loader shape: {next(iter(val_loader))[0].shape}")
    print(f"{suffix} Test loader shape: {next(iter(test_loader))[0].shape}")

    output_dir = f"data/{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_dataset, os.path.join(
        output_dir, f"train_dataset_{suffix}.pt"))
    torch.save(val_dataset, os.path.join(
        output_dir, f"val_dataset_{suffix}.pt"))
    torch.save(test_dataset, os.path.join(
        output_dir, f"test_dataset_{suffix}.pt"))
    torch.save(train_loader, os.path.join(
        output_dir, f"train_loader_{suffix}.pt"))
    torch.save(val_loader, os.path.join(output_dir, f"val_loader_{suffix}.pt"))
    torch.save(test_loader, os.path.join(
        output_dir, f"test_loader_{suffix}.pt"))

    torch.save(train_transforms, os.path.join(
        output_dir, f"transforms_{suffix}.pt"))

    return train_transforms


if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    device = torch.device(device)

    spatial_transforms = benchmark_preprocess(spatial=True, device=device)
    non_spatial_transforms = benchmark_preprocess(spatial=False, device=device)
