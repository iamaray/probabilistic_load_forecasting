import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datetime import datetime


class MinMaxNorm:
    def __init__(self, device='cuda'):
        self.device = device
        self.min_val = None
        self.max_val = None

    def fit(self, x: torch.Tensor):
        self.max_val = torch.max(
            x, dim=0, keepdim=True).values.to(self.device).float()
        self.min_val = torch.min(
            x, dim=0, keepdim=True).values.to(self.device).float()

    def transform(self, x: torch.Tensor):
        return (x - self.min_val) / (self.max_val - self.min_val)

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def reverse(self, transformed: torch.Tensor):
        return ((transformed * (self.max_val - self.min_val)) + self.min_val).to(self.device)

    def set_device(self, device='cuda'):
        if device == 'cpu':
            self.max_val = self.max_val.cpu().detach()
            self.min_val = self.min_val.cpu().detach()
        else:
            self.max_val = self.max_val.to(device)
            self.min_val = self.min_val.to(device)


class StandardScaleNorm:
    def __init__(self, device='cuda', num_transform_cols=3):
        self.mean = None
        self.std = None
        self.device = device
        # Save the number of columns to transform (features in the last dim)
        self.num_transform_cols = num_transform_cols

    def fit(self, x: torch.Tensor):
        """
        Compute the mean and std only for the first num_transform_cols features.
        x: Tensor of shape [batch_size, sequence_len, num_feats]
        """
        # Compute mean and std along the sequence (dim=1) for the selected features.
        # The resulting mean and std will have shape [batch_size, 1, num_transform_cols]
        self.mean = x[..., :self.num_transform_cols].mean(
            dim=1, keepdim=True).to(self.device).float()
        self.std = x[..., :self.num_transform_cols].std(
            dim=1, keepdim=True).to(self.device).float()

    def transform(self, x: torch.Tensor, transform_col=None):
        """
        Standardize the features.

        If transform_col is None, then transform the first num_transform_cols features.
        If transform_col is provided (as an integer), then transform only the specific column
        indicated by transform_col (i.e. the feature in the last dimension at that index).

        The remaining features are left unchanged.
        """
        x_transformed = x.clone()

        if transform_col is None:
            x_transformed[..., :self.num_transform_cols] = (
                x[..., :self.num_transform_cols] - self.mean
            ) / self.std
        else:
            if transform_col >= self.num_transform_cols:
                raise IndexError(f"transform_col ({
                                 transform_col}) must be less than num_transform_cols ({self.num_transform_cols})")

            x_transformed[..., transform_col:transform_col+1] = (
                x[..., transform_col:transform_col+1] -
                self.mean[..., transform_col:transform_col+1]
            ) / self.std[..., transform_col:transform_col+1]
        return x_transformed

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def reverse(self, transformed: torch.Tensor, reverse_col=0):
        """
        Reverse the standard scaling for only one column (i.e. one feature index).
        reverse_col: integer index indicating which feature to revert.
        """
        # Clone to avoid modifying the input tensor
        x_reversed = transformed.clone()
        # Use slicing with reverse_col:reverse_col+1 to preserve dimensions.
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


def readtoFiltered(csv_path, variates=[]):
    """Returns a dataframe and a corresponding tensor with desired columns,
    plus a 'dow' column (day-of-week, Monday=1,...,Sunday=7)."""

    df = pd.read_csv(csv_path)
    date_to_index = pd.to_datetime(df['Unnamed: 0'])

    # Compute the day-of-week (Monday=0 => +1 => Monday=1, Sunday=7)
    day_of_week = date_to_index.dt.weekday + 1

    if len(variates) == 0:
        numeric_feats = [name for name in df.columns if name != 'marketday']
        filtered = df[numeric_feats].copy()
    else:
        numeric_feats = [name for name in variates if name != 'marketday']
        filtered = df[numeric_feats].copy()

    filtered = filtered.fillna(0).astype('float32')
    filtered['dow'] = day_of_week.values.astype('float32')

    numeric_feats.append('dow')

    return filtered, torch.tensor(filtered.values), date_to_index


def formPairs(
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24):

    assert x_tensor.shape[0] == y_tensor.shape[0]
    N = x_tensor.shape[0]

    x_start = x_start_hour - 1
    X, Y = [], []

    while (x_start + x_window + x_y_gap + y_window) < N:
        x = x_tensor[x_start: x_start + x_window, :]
        y = y_tensor[x_start + x_window + x_y_gap:
                     x_start + x_window + x_y_gap + y_window, :]

        # (time, features) -> (features, time) if you prefer that shape
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)

        X.append(x)
        Y.append(y)
        x_start += step_size

    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y


def formARPairs(x_tensor: torch.Tensor,
                y_tensor: torch.Tensor,
                num_targets: int = 1,
                num_auxiliary: int = 0,
                window_length=168,
                prediction_length=24,
                step_size=1):
    """Forms (X, y) pairs for autoregression where x_i = x_{0:T-1} with x_0=0, y_i = y_{1:T}"""

    assert x_tensor.shape[0] == y_tensor.shape[0]
    N = x_tensor.shape[0]

    window_end = window_length
    X, Y = [], []

    while window_end <= N:
        x = x_tensor[window_end - window_length:window_end]
        y = y_tensor[window_end - window_length:window_end]

        x_ar = x.clone()
        x_ar[1:, :num_targets] = x[:-1, :num_targets]
        x_ar[0, :num_targets] = 0

        if num_auxiliary > 0:
            x_ar[:, num_targets:] = x[:, num_targets:]

        x_ar = x_ar.transpose(0, 1)
        y = y.transpose(0, 1)

        X.append(x_ar)
        Y.append(y)

        window_end += step_size

    X = torch.stack(X)
    Y = torch.stack(Y)

    return X, Y


def computeNetLoadTensor(df: pd.DataFrame, locations=[]):
    if len(locations) == 0:
        vals = df.to_numpy()
        return vals[:, 0] - (vals[:, 1] + vals[:, 2])
    else:
        raise NotImplementedError(
            'Not implemented for multiple locations yet.')


def preprocess(
        csv_path=None,
        net_load_input=True,
        net_load_labels=True,
        auto_reg=False,
        variates=[],
        window_length=168,
        prediction_length=24,
        step_size=24,
        train_start_date=(datetime(2023, 2, 1), 0),  # date, hour
        train_end_date=(datetime(2024, 6, 30), 23),
        val_start_date=None,
        val_end_date=None,
        test_start_date=(datetime(2024, 7, 1), 0),
        test_end_date=(datetime(2025, 1, 8), 13),
        device='cuda',
        data_norm=MinMaxNorm):
    """
    Preprocessing pipeline. 
    If val_start_date and val_end_date are provided, a validation set and loader are also returned.
    Otherwise, val_loader=None.
    """

    df, data_tensor, date_to_index = readtoFiltered(
        csv_path=csv_path, variates=variates)
    print(df)

    def find_idx(x):
        """Find index in date_to_index corresponding to (date, hour)."""
        return date_to_index.index[date_to_index == x[0]][x[1]]

    train_start_idx = find_idx(train_start_date)
    train_end_idx = find_idx(train_end_date)

    test_start_idx = find_idx(test_start_date)
    test_end_idx = find_idx(test_end_date)

    val_start_idx = None
    val_end_idx = None
    have_val = (val_start_date is not None) and (val_end_date is not None)
    if have_val:
        val_start_idx = find_idx(val_start_date)
        val_end_idx = find_idx(val_end_date)

    x_tensor, y_tensor = data_tensor, data_tensor

    def compute_net_load_tensor(tensor):
        net_load = (tensor[:, 0] - (tensor[:, 1] + tensor[:, 2])).unsqueeze(-1)
        if tensor.shape[-1] > 3:
            aux_feats = tensor[:, 3:]
            if len(aux_feats.shape) <= 1:
                aux_feats = aux_feats.unsqueeze(-1)
            return torch.hstack([net_load, aux_feats])
        return net_load

    if net_load_input:
        x_tensor = compute_net_load_tensor(data_tensor)
    # Use columns [:3] for net load label or original 3 columns as needed
    if net_load_labels:
        y_tensor = compute_net_load_tensor(data_tensor[:, :3])
    else:
        y_tensor = data_tensor[:, :3]

    x_train_raw = x_tensor[train_start_idx:train_end_idx + 1, :]
    y_train_raw = y_tensor[train_start_idx:train_end_idx + 1, :]

    if have_val:
        x_val_raw = x_tensor[val_start_idx:val_end_idx + 1, :]
        y_val_raw = y_tensor[val_start_idx:val_end_idx + 1, :]
    else:
        x_val_raw, y_val_raw = None, None

    x_test_raw = x_tensor[test_start_idx:test_end_idx + 1, :]
    y_test_raw = y_tensor[test_start_idx:test_end_idx + 1, :]

    train_norm = None
    if data_norm is not None:
        train_norm = data_norm(device=device)
        train_norm.fit(y_train_raw)

    if not auto_reg:
        x_train, y_train = formPairs(
            x_tensor=x_train_raw,
            y_tensor=y_train_raw,
            x_start_hour=9,
            x_y_gap=24,
            x_window=window_length,
            y_window=prediction_length,
            step_size=step_size
        )

        if have_val:
            x_val, y_val = formPairs(
                x_tensor=x_val_raw,
                y_tensor=y_val_raw,
                x_start_hour=9,
                x_y_gap=24,
                x_window=window_length,
                y_window=prediction_length,
                step_size=step_size
            )
        else:
            x_val, y_val = None, None

        x_test, y_test = formPairs(
            x_tensor=x_test_raw,
            y_tensor=y_test_raw,
            x_start_hour=9,
            x_y_gap=24,
            x_window=window_length,
            y_window=prediction_length,
            step_size=step_size
        )

    else:
        num_targets, num_aux = 1, 1
        if net_load_input and net_load_labels:
            num_targets, num_aux = 3, 1
        elif not net_load_input and net_load_labels:
            num_targets, num_aux = 1, 3

        x_train, y_train = formARPairs(
            x_tensor=x_train_raw,
            y_tensor=y_train_raw,
            num_targets=num_targets,
            num_auxiliary=num_aux,
            window_length=window_length,
            prediction_length=prediction_length,
            step_size=step_size
        )
        if have_val:
            x_val, y_val = formARPairs(
                x_tensor=x_val_raw,
                y_tensor=y_val_raw,
                num_targets=num_targets,
                num_auxiliary=num_aux,
                window_length=window_length,
                prediction_length=prediction_length,
                step_size=step_size
            )
        else:
            x_val, y_val = None, None

        x_test, y_test = formARPairs(
            x_tensor=x_test_raw,
            y_tensor=y_test_raw,
            num_targets=num_targets,
            num_auxiliary=num_aux,
            window_length=window_length,
            prediction_length=prediction_length,
            step_size=step_size
        )

    trainset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=trainset, batch_size=128, shuffle=True)

    if have_val:
        valset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(valset, batch_size=128, shuffle=False)
    else:
        val_loader = None

    testset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    return df, train_loader, val_loader, test_loader, train_norm, date_to_index, test_start_idx
