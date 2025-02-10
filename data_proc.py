import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datetime import datetime


"""Data processing that runs on the cleaned dataset. Some elements are hard-coded."""


def benchmark_preprocess(
        train_start_end,
        val_start_end,
        test_start_end,
        spatial,
        device,
        input_window,
        train_norms=[]):

    train_loader = None
    val_loader = None
    test_loader = None



    return train_loader, val_loader, test_loader
