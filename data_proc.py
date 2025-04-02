import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datetime import datetime
import os
import argparse
import json
import pytz

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

    def reverse(self, transformed: torch.Tensor, reverse_col=0, is_std=False):
        x_reversed = transformed.clone()
        if is_std:
            x_reversed[..., reverse_col:reverse_col+1] = (
                transformed[..., reverse_col:reverse_col+1] *
                self.std[..., reverse_col:reverse_col+1]
            )
        else:
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

    def transform(self, x: torch.Tensor, transform_col=None):
        x = x.clone()
        for t in self.transforms:
            x = t.transform(x, transform_col=transform_col)
        return x

    def reverse(self, transformed: torch.Tensor, reverse_col=0):
        x = transformed.clone()
        for t in reversed(self.transforms):
            x = t.reverse(x, reverse_col)
        return x

    def set_device(self, device):
        self.device = device
        for t in self.transforms:
            t.set_device(device)


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
    Y = torch.stack(Y)  # Shape: [num_samples, y_window]
    return X, Y


def formPairsAR(
        x_tensor: torch.Tensor,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24):
    """
    For autoregressive training of DeepAR.
    Returns:
      target_seq: Tensor of shape [num_samples, x_window+y_window] (first column of the combined sequence)
      covariate_seq: Tensor of shape [num_samples, x_window+y_window, num_features-1]
                      (all features except the target column)
      mask_seq: Boolean tensor of shape [num_samples, x_window+y_window] with True for observed (conditioning) and
                False for forecast period.
    """
    N = x_tensor.shape[0]
    num_features = x_tensor.shape[1]
    x_start = x_start_hour - 1
    targets, covariates, masks = [], [], []
    while (x_start + x_window + x_y_gap + y_window) < N:
        # shape (x_window, num_features)
        x_obs = x_tensor[x_start: x_start + x_window, :]
        x_fore = x_tensor[x_start + x_window + x_y_gap: x_start +
                          # shape (y_window, num_features)
                          x_window + x_y_gap + y_window, :]
        # shape (x_window+y_window, num_features)
        combined = torch.cat([x_obs, x_fore], dim=0)
        targets.append(combined[:, 0])
        covariates.append(combined[:, 1:])
        mask = torch.cat([torch.ones(x_window, dtype=torch.bool),
                          torch.zeros(y_window, dtype=torch.bool)], dim=0)
        masks.append(mask)
        x_start += step_size

    # Shape: [num_samples, x_window+y_window]
    target_seq = torch.stack(targets)
    # Shape: [num_samples, x_window+y_window, num_features-1]
    covariate_seq = torch.stack(covariates)
    # Shape: [num_samples, x_window+y_window]
    mask_seq = torch.stack(masks)
    return target_seq, covariate_seq, mask_seq


def benchmark_preprocess(
        csv_path="data/ercot_data_cleaned.csv",
        train_start_end=(datetime(2023, 2, 10), datetime(2024, 7, 1)),
        val_start_end=(datetime(2024, 7, 1), datetime(2024, 9, 1)),
        test_start_end=(datetime(2024, 9, 1), datetime(2025, 1, 6)),
        spatial=True,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        included_feats=None,
        num_transform_cols=2,
        train_transforms=[StandardScaleNorm(device='cpu')],
        ar_model: bool = False,
        output_dir=None,
        date_column="Unnamed: 0"):
    """
    Loads the cleaned CSV data and performs time-series splitting, transformation,
    windowing (pair formation), and creation of DataLoaders. The datasets and loaders
    are saved with a suffix indicating if the data is spatial or non-spatial.

    If 'ar_model' is True, the function creates datasets and loaders suitable for the DeepAR model,
    which returns for each sample a tuple of (target, covariates, mask).

    Returns the list of fitted transform objects.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train_transforms is None:
        train_transforms = [StandardScaleNorm(device=device)]

    raw_df = pd.read_csv(csv_path)

    # Find date column (try common names if specified column doesn't exist)
    if date_column in raw_df.columns:
        date_series = pd.to_datetime(raw_df[date_column])
        raw_df = raw_df.drop(date_column, axis=1)
    elif "time" in raw_df.columns:
        date_series = pd.to_datetime(raw_df["time"])
        raw_df = raw_df.drop("time", axis=1)
    elif "date" in raw_df.columns:
        date_series = pd.to_datetime(raw_df["date"])
        raw_df = raw_df.drop("date", axis=1)
    elif "datetime" in raw_df.columns:
        date_series = pd.to_datetime(raw_df["datetime"])
        raw_df = raw_df.drop("datetime", axis=1)
    else:
        # Assume first column is the date column
        date_series = pd.to_datetime(raw_df.iloc[:, 0])
        raw_df = raw_df.iloc[:, 1:]
        print(
            f"Warning: No date column found with name '{date_column}'. Using first column as date.")

    # Handle timezone differences
    is_tz_aware = date_series.dt.tz is not None

    # Convert dates for comparison based on timezone awareness
    def convert_date_for_comparison(date_obj):
        if is_tz_aware and date_obj.tzinfo is None:
            # If data has timezone but comparison date doesn't, add UTC timezone
            return pytz.UTC.localize(date_obj)
        elif not is_tz_aware and date_obj.tzinfo is not None:
            # If data has no timezone but comparison date does, remove timezone
            return date_obj.replace(tzinfo=None)
        return date_obj

    train_start = convert_date_for_comparison(train_start_end[0])
    train_end = convert_date_for_comparison(train_start_end[1])
    val_start = convert_date_for_comparison(val_start_end[0])
    val_end = convert_date_for_comparison(val_start_end[1])
    test_start = convert_date_for_comparison(test_start_end[0])
    test_end = convert_date_for_comparison(test_start_end[1])

    # Filter features if specified
    if included_feats is not None:
        raw_df = raw_df[included_feats]

    train_mask = (date_series >= train_start) & (date_series < train_end)
    val_mask = (date_series >= val_start) & (date_series < val_end)
    test_mask = (date_series >= test_start) & (date_series < test_end)

    train_df = raw_df[train_mask]
    val_df = raw_df[val_mask]
    test_df = raw_df[test_mask]

    train_tensor = torch.tensor(train_df.to_numpy(), device=device).float()
    val_tensor = torch.tensor(val_df.to_numpy(), device=device).float()
    test_tensor = torch.tensor(test_df.to_numpy(), device=device).float()

    # if spatial:
    #     for t in train_transforms:
    #         t.change_transform_cols(12)
    # else:
    #     for t in train_transforms:
    #         t.change_transform_cols(3)

    for t in train_transforms:
        t.change_transform_cols(num_transform_cols)

        t.fit(train_tensor.unsqueeze(0).to(device))
        train_tensor = t.transform(train_tensor)

    suffix = "spatial" if spatial else "non_spatial"

    if output_dir is None:
        output_dir = f"data/{suffix}"

    print(f"Saving data to directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_tensor, os.path.join(
        output_dir, f"train_tensor_{suffix}.pt"))

    # Save train_tensor to its own file
    # output_dir = f"data/{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_tensor, os.path.join(
        output_dir, f"train_tensor_{suffix}.pt"))

    if ar_model:
        suffix = f"{suffix}_AR"

        X_train_target, X_train_cov, X_train_mask = formPairsAR(
            train_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        X_val_target, X_val_cov, X_val_mask = formPairsAR(
            val_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        X_test_target, X_test_cov, X_test_mask = formPairsAR(
            test_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)

        train_dataset = TensorDataset(
            X_train_target, X_train_cov, X_train_mask)
        val_dataset = TensorDataset(X_val_target, X_val_cov, X_val_mask)
        test_dataset = TensorDataset(X_test_target, X_test_cov, X_test_mask)
    else:
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

    print(f"{suffix} Train loader shape: {next(iter(train_loader))[0].shape}")
    print(f"{suffix} Val loader shape: {next(iter(val_loader))[0].shape}")
    print(f"{suffix} Test loader shape: {next(iter(test_loader))[0].shape}")

    train_dataset_path = os.path.join(output_dir, f"train_dataset_{suffix}.pt")
    print(f"Saving train dataset to: {train_dataset_path}")
    torch.save(train_dataset, train_dataset_path)

    val_dataset_path = os.path.join(output_dir, f"val_dataset_{suffix}.pt")
    print(f"Saving validation dataset to: {val_dataset_path}")
    torch.save(val_dataset, val_dataset_path)

    test_dataset_path = os.path.join(output_dir, f"test_dataset_{suffix}.pt")
    print(f"Saving test dataset to: {test_dataset_path}")
    torch.save(test_dataset, test_dataset_path)

    train_loader_path = os.path.join(output_dir, f"train_loader_{suffix}.pt")
    print(f"Saving train loader to: {train_loader_path}")
    torch.save(train_loader, train_loader_path)

    val_loader_path = os.path.join(output_dir, f"val_loader_{suffix}.pt")
    print(f"Saving validation loader to: {val_loader_path}")
    torch.save(val_loader, val_loader_path)

    test_loader_path = os.path.join(output_dir, f"test_loader_{suffix}.pt")
    print(f"Saving test loader to: {test_loader_path}")
    torch.save(test_loader, test_loader_path)

    if len(train_transforms) > 1:
        train_transforms = TransformSequence(train_transforms, device)
        transforms_path = os.path.join(output_dir, f"transforms_{suffix}.pt")
        print(f"Saving transforms to: {transforms_path}")
        torch.save(train_transforms, transforms_path)
    else:
        transforms_path = os.path.join(output_dir, f"transforms_{suffix}.pt")
        print(f"Saving transforms to: {transforms_path}")
        torch.save(train_transforms[0], transforms_path)

    # Generate report with data processing summary
    config_dict = {
        'csv_path': csv_path,
        'train_start': train_start_end[0].strftime("%Y-%m-%d") if hasattr(train_start_end[0], 'strftime') else str(train_start_end[0]),
        'train_end': train_start_end[1].strftime("%Y-%m-%d") if hasattr(train_start_end[1], 'strftime') else str(train_start_end[1]),
        'val_start': val_start_end[0].strftime("%Y-%m-%d") if hasattr(val_start_end[0], 'strftime') else str(val_start_end[0]),
        'val_end': val_start_end[1].strftime("%Y-%m-%d") if hasattr(val_start_end[1], 'strftime') else str(val_start_end[1]),
        'test_start': test_start_end[0].strftime("%Y-%m-%d") if hasattr(test_start_end[0], 'strftime') else str(test_start_end[0]),
        'test_end': test_start_end[1].strftime("%Y-%m-%d") if hasattr(test_start_end[1], 'strftime') else str(test_start_end[1]),
        'x_start_hour': x_start_hour,
        'x_y_gap': x_y_gap,
        'x_window': x_window,
        'y_window': y_window,
        'step_size': step_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'included_feats': included_feats,
        'num_transform_cols': num_transform_cols,
        'spatial': spatial,
        'ar_model': ar_model,
        'date_column': date_column
    }

    if ar_model:
        generate_data_report(
            output_dir=output_dir,
            suffix=suffix,
            config=config_dict,
            train_tensor=train_tensor,
            val_tensor=val_tensor,
            test_tensor=test_tensor,
            X_train=X_train_target,
            Y_train=None,  # No separate Y for AR models
            X_val=X_val_target,
            Y_val=None,
            X_test=X_test_target,
            Y_test=None,
            train_transforms=train_transforms,
            included_feats=included_feats,
            ar_model=ar_model
        )
    else:
        generate_data_report(
            output_dir=output_dir,
            suffix=suffix,
            config=config_dict,
            train_tensor=train_tensor,
            val_tensor=val_tensor,
            test_tensor=test_tensor,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            train_transforms=train_transforms,
            included_feats=included_feats,
            ar_model=ar_model
        )

    return train_transforms


def generate_data_report(
        output_dir,
        suffix,
        config,
        train_tensor,
        val_tensor,
        test_tensor,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        train_transforms,
        included_feats=None,
        ar_model=False):
    """Generate a report summarizing the data processing and save it to a text file."""
    report_path = os.path.join(output_dir, f"data_report_{suffix}.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DATA PREPROCESSING REPORT - {suffix.upper()}\n")
        f.write("=" * 80 + "\n\n")

        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Source CSV: {config.get('csv_path', 'Unknown')}\n")
        f.write(f"Date column: {config.get('date_column', 'Unknown')}\n")

        # Date ranges
        f.write("\nDATE RANGES\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Training period: {config.get('train_start', 'Unknown')} to {config.get('train_end', 'Unknown')}\n")
        f.write(
            f"Validation period: {config.get('val_start', 'Unknown')} to {config.get('val_end', 'Unknown')}\n")
        f.write(
            f"Testing period: {config.get('test_start', 'Unknown')} to {config.get('test_end', 'Unknown')}\n")

        # Data dimensions
        f.write("\nDATA DIMENSIONS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Raw train tensor shape: {tuple(train_tensor.shape)}\n")
        f.write(f"Raw validation tensor shape: {tuple(val_tensor.shape)}\n")
        f.write(f"Raw test tensor shape: {tuple(test_tensor.shape)}\n")

        # Sliding window parameters
        f.write("\nSLIDING WINDOW PARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Input window length (x_window): {config.get('x_window', 'Unknown')} hours\n")
        f.write(
            f"Gap between input and forecast (x_y_gap): {config.get('x_y_gap', 'Unknown')} hours\n")
        f.write(
            f"Forecast horizon (y_window): {config.get('y_window', 'Unknown')} hours\n")
        f.write(
            f"Sliding window step size: {config.get('step_size', 'Unknown')} hours\n")
        f.write(
            f"Starting hour offset: {config.get('x_start_hour', 'Unknown')}\n")

        # Dataset samples
        f.write("\nDATASET SAMPLES\n")
        f.write("-" * 50 + "\n")
        if ar_model:
            f.write(f"Number of training samples: {X_train.shape[0]}\n")
            f.write(f"Number of validation samples: {X_val.shape[0]}\n")
            f.write(f"Number of test samples: {X_test.shape[0]}\n")
            f.write(f"AR sequence length: {X_train.shape[1]}\n")
        else:
            f.write(f"Number of training samples: {X_train.shape[0]}\n")
            f.write(f"Number of validation samples: {X_val.shape[0]}\n")
            f.write(f"Number of test samples: {X_test.shape[0]}\n")
            f.write(f"Input sequence length (X): {X_train.shape[1]}\n")
            if Y_train is not None:
                f.write(f"Target sequence length (Y): {Y_train.shape[1]}\n")

        # Feature information
        f.write("\nFEATURE INFORMATION\n")
        f.write("-" * 50 + "\n")
        if included_feats:
            f.write(f"Number of features: {len(included_feats)}\n")
            f.write("Selected features:\n")
            for feat in included_feats:
                f.write(f"  - {feat}\n")
        else:
            f.write(f"Using all available features in the dataset.\n")

        # Model type info
        f.write("\nMODEL TYPE INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Prepared for autoregressive model: {ar_model}\n")
        f.write(f"Spatial data: {config.get('spatial', False)}\n")

        # Normalization information
        f.write("\nNORMALIZATION INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Number of transformed columns: {config.get('num_transform_cols', 'Unknown')}\n")

        # Describe transforms
        if isinstance(train_transforms, list):
            transform = train_transforms[0]
        else:
            transform = train_transforms

        transform_type = transform.__class__.__name__
        f.write(f"Transformation type: {transform_type}\n")

        try:
            if transform_type == "StandardScaleNorm":
                if hasattr(transform, 'mean') and hasattr(transform, 'std'):
                    f.write("\nMEAN AND STD VALUES\n")
                    for i in range(min(5, config.get('num_transform_cols', 3))):
                        f.write(
                            f"Feature {i}: Mean={transform.mean[0, 0, i].item():.4f}, Std={transform.std[0, 0, i].item():.4f}\n")
                    if config.get('num_transform_cols', 3) > 5:
                        f.write("(Only showing first 5 features)\n")
            elif transform_type == "MinMaxNorm":
                if hasattr(transform, 'min_val') and hasattr(transform, 'max_val'):
                    f.write("\nMIN AND MAX VALUES\n")
                    for i in range(min(5, config.get('num_transform_cols', 3))):
                        f.write(
                            f"Feature {i}: Min={transform.min_val[0, 0, i].item():.4f}, Max={transform.max_val[0, 0, i].item():.4f}\n")
                    if config.get('num_transform_cols', 3) > 5:
                        f.write("(Only showing first 5 features)\n")
        except (IndexError, AttributeError) as e:
            f.write(
                f"\nCould not extract detailed transform statistics: {str(e)}\n")

        # Output information
        f.write("\nOUTPUT INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(
            f"DataLoader batch size: {config.get('batch_size', 'Unknown')}\n")
        f.write(
            f"DataLoader workers: {config.get('num_workers', 'Unknown')}\n\n")

        f.write("Generated files:\n")
        f.write(f"  - train_tensor_{suffix}.pt\n")
        f.write(f"  - train_dataset_{suffix}.pt\n")
        f.write(f"  - val_dataset_{suffix}.pt\n")
        f.write(f"  - test_dataset_{suffix}.pt\n")
        f.write(f"  - train_loader_{suffix}.pt\n")
        f.write(f"  - val_loader_{suffix}.pt\n")
        f.write(f"  - test_loader_{suffix}.pt\n")
        f.write(f"  - transforms_{suffix}.pt\n")
        f.write(f"  - data_report_{suffix}.txt (this file)\n\n")

        # Timestamp
        from datetime import datetime
        f.write(
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Data report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Data preprocessing for power consumption dataset')
    parser.add_argument('--config_path', type=str, default='cfgs/data_proc/spain_data/spain_dataset_non_spatial_ar.json',
                        help='Path to the config file containing preprocessing parameters')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    def parse_date(date_value, default_date):
        if not date_value:
            return default_date
        if isinstance(date_value, str):
            return datetime.strptime(date_value, "%Y-%m-%d")
        return default_date

    train_start = parse_date(config.get("train_start"), datetime(2023, 2, 10))
    train_end = parse_date(config.get("train_end"), datetime(2024, 7, 1))
    val_start = parse_date(config.get("val_start"), datetime(2024, 7, 1))
    val_end = parse_date(config.get("val_end"), datetime(2024, 9, 1))
    test_start = parse_date(config.get("test_start"), datetime(2024, 9, 1))
    test_end = parse_date(config.get("test_end"), datetime(2025, 1, 6))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transforms = [StandardScaleNorm(device=device)]

    transforms = benchmark_preprocess(
        csv_path=config.get("csv_path", "data/ercot_data_cleaned.csv"),
        train_start_end=(train_start, train_end),
        val_start_end=(val_start, val_end),
        test_start_end=(test_start, test_end),
        spatial=config.get("spatial", False),
        x_start_hour=config.get("x_start_hour", 9),
        x_y_gap=config.get("x_y_gap", 15),
        x_window=config.get("x_window", 168),
        y_window=config.get("y_window", 24),
        step_size=config.get("step_size", 24),
        batch_size=config.get("batch_size", 64),
        num_workers=config.get("num_workers", 1),
        included_feats=config.get("included_feats", None),
        num_transform_cols=config.get("num_transform_cols", 2),
        train_transforms=train_transforms,
        ar_model=config.get("ar_model", False),
        output_dir=config.get("output_dir", None),
        date_column=config.get("date_column", "Unnamed: 0")
    )

    print(f"Data preprocessing completed successfully. Transforms saved.")
