import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader


def readtoFiltered(csv_path, variates=[]):
    """Returns a dataframe and a corresponding tensor with desired columns"""
    df = pd.read_csv(csv_path)

    if len(variates) == 0:
        numeric_feats = [name for name in list(
            df.columns) if name != 'marketday']
        filtered = df[numeric_feats]
        filtered.fillna(0)
        filtered.astype('float32')
        return filtered[numeric_feats], torch.Tensor(filtered.values)

    filtered = df[variates]

    filtered = filtered.fillna(0)
    filtered = filtered.astype('float32')

    return filtered, torch.Tensor(filtered.values)


def formPairs(
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        net_load=True,
        window_length=168,
        prediction_length=24,
        step_size=1):
    """Takes a tensor with raw data and forms (X, y) pairs with a sliding window"""

    assert x_tensor.shape[0] == y_tensor.shape[0]
    N = x_tensor.shape[0]

    window_end = window_length
    prediction_end = window_length + prediction_length

    X = []
    Y = []

    while prediction_end <= N:
        x = x_tensor[window_end - window_length: window_end]
        y = y_tensor[window_end:prediction_end]

        x = x.unsqueeze(0) if x.ndim == 1 else x.transpose(0, 1)
        y = y.unsqueeze(0) if y.ndim == 1 else y.transpose(0, 1)

        X.append(x)
        Y.append(y)

        window_end += step_size
        prediction_end += step_size

    X = torch.stack(X)
    Y = torch.stack(Y)

    return X, Y


def computeNetLoadTensor(df: pd.DataFrame, locations=[]):
    # TODO: generalize
    if len(locations) == 0:
        vals = df.to_numpy()
        return vals[:, 0] - (vals[:, 1] + vals[:, 2])
    else:
        raise NotImplementedError('Not implemented.')


def preprocess(
        csv_path=None,
        net_load_input=True,
        net_load_labels=True,
        variates=[],
        window_length=168,
        prediction_length=24,
        step_size=1,
        train_proportion=0.8):
    """Preprocessing pipeline"""

    df, data_tensor = readtoFiltered(csv_path=csv_path, variates=variates)

    train_cutoff = int(train_proportion * data_tensor.shape[0])

    # TODO: generalize this for multiple locations
    x_tensor, y_tensor = data_tensor, data_tensor
    if net_load_input:
        x_tensor = data_tensor[:, 0] - (data_tensor[:, 1] + data_tensor[:, 2])
    if net_load_labels:
        y_tensor = data_tensor[:, 0] - (data_tensor[:, 1] + data_tensor[:, 2])

    x_train, y_train = formPairs(
        x_tensor=x_tensor, y_tensor=y_tensor,
        window_length=window_length, prediction_length=prediction_length, step_size=step_size)

    x_test, y_test = formPairs(
        x_tensor=x_tensor, y_tensor=y_tensor,
        window_length=window_length, prediction_length=prediction_length, step_size=step_size)

    trainset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)

    testset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    return df, train_loader, test_loader


class MinMaxNorm:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, x: torch.Tensor):
        self.max_val = torch.max(x, dim=1, keepdim=True).values
        self.min_val = torch.min(x, dim=1, keepdim=True).values

    def transform(self, x: torch.Tensor):
        return (x - self.min_val) / (self.max_val - self.min_val)

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def reverse(self, transformed: torch.Tensor):
        return (transformed * (self.max_val - self.min_val)) + self.min_val


class StandardScaleNorm:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True)

    def transform(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def reverse(self, transformed: torch.Tensor):
        return (transformed * self.std) + self.mean
