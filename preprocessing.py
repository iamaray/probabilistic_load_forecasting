import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


def readtoFiltered(csv_path, variates=[]):
    """Returns a dataframe and a corresponding tensor with desired columns"""
    df = pd.read_csv(csv_path)

    if len(variates) == 0:
        return df

    filtered = df[variates]
    return filtered, torch.Tensor(filtered.values)


def formPairs(data_tensor: torch.Tensor, window_length=168, prediction_length=24, step_size=1):
    """Takes a tensor with raw data and forms (X, y) pairs with a sliding window"""

    window_end = window_length - 1
    prediction_end = window_length + prediction_length - 1

    X = []
    Y = []

    while prediction_end < data_tensor.shape[0]:
        x = data_tensor[window_end - window_length: window_end, :]
        y = data_tensor[window_end:prediction_end, :]

        X.append(x)
        Y.append(y)

        window_end += step_size
        prediction_end += step_size


def preprocess(csv_path, variates=[], window_length=168, prediction_length=24, step_size=1, train_proportion=0.8):
    df, data_tensor = readtoFiltered(csv_path=csv_path, variates=variates)

    train_cutoff = int(train_proportion * data_tensor.shape[0])

    x_train, y_train = formPairs(data_tensor=data_tensor[:train_cutoff, :],
                                 window_length=window_length, prediction_length=prediction_length, step_size=step_size)

    x_test, y_test = formPairs(data_tensor=data_tensor[train_cutoff:, :],
                               window_length=window_length, prediction_length=prediction_length, step_size=step_size)

    trainset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)

    testset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    return df, train_loader, test_loader
