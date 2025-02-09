import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

############################
# 0) Strategically refill the dataframe
############################
def strategic_fill(df):
    """
    Fills missing values in the DataFrame based on the following policy:
    - For each missing hour, check the same hour from the previous day.
      If it exists, use it to fill the missing value.
    - If the previous day's same hour is also missing, check the next day's same hour.
      If it exists, use it to fill the missing value.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()

    # Iterate over each missing hour
    for time in df_filled.index[df_filled.isnull().any(axis=1)]:
        # Extract the hour and day
        hour = time.time()
        previous_day = time - pd.Timedelta(days=1)
        next_day = time + pd.Timedelta(days=1)

        # Check if the previous day's same hour exists and is not missing
        if previous_day in df_filled.index and not df_filled.loc[previous_day].isnull().any():
            df_filled.loc[time] = df_filled.loc[previous_day]
        # If previous day is missing, check the next day's same hour
        elif next_day in df_filled.index and not df_filled.loc[next_day].isnull().any():
            df_filled.loc[time] = df_filled.loc[next_day]

    return df_filled




############################
# 1) Read CSV and convert to DataFrame
############################
def load_data(csv_path, date_col='marketday', hour_col='hourending'):
    """
    Reads the CSV file, combines date_col + hour_col into a single datetime,
    and sets it as the index. Returns the DataFrame.
    """
    df = pd.read_csv(csv_path)
    # Adjust the following lines as needed based on your CSV structure
    df['time'] = (pd.to_datetime(df[date_col])
                  + pd.to_timedelta(df[hour_col] - 1, unit='h'))
    df.set_index('time', inplace=True)
    df['ERC_Wind'] = df['ERC_CWind'] + df['ERC_NWind'] + df['ERC_PWind'] + df['ERC_SWind'] + df['ERC_WWind']
    df["ACTUAL_NetLoad"] = df["ACTUAL_ERC_Load"] - df["ACTUAL_ERC_Wind"] - df["ACTUAL_ERC_Solar"]
    df["NetLoad"] = df["ERC_Load"] - df["ERC_Wind"] - df["ERC_Solar"]

    df["Load_Error"] = df["ACTUAL_ERC_Load"] - df["ERC_Load"]
    df["Wind_Error"] = df["ACTUAL_ERC_Wind"] - df["ERC_Wind"]
    df["Solar_Error"] = df["ACTUAL_ERC_Solar"] - df["ERC_Solar"]
    df["NetLoad_Error"] = df["ACTUAL_NetLoad"] - df["NetLoad"]


    # Generate a complete hourly range for the period
    start_time = df.index.min()
    end_time = df.index.max()
    complete_range = pd.date_range(start=start_time, end=end_time, freq='H')
    # Find missing hours and then insert these times to make the time consecutive
    missing_hours = complete_range.difference(df.index)
    print(missing_hours)
    missing_df = pd.DataFrame(index=missing_hours)
    df = pd.concat([df, missing_df])
    df = df.sort_index()
    # df = df.ffill()

    df = strategic_fill(df)

    # Check if there are still missing values
    remaining_missing = df.index[df.isnull().any(axis=1)]
    if remaining_missing.empty:
        print("\nNo more remaining missing hours after strategic fill!")

    # Add time-based features
    df['HoD'] = df.index.hour  # Hour of the Day (0 to 23)
    df['DoW'] = df.index.dayofweek + 1  # Day of the Week (1=Monday to 7=Sunday)
    df['MoY'] = df.index.month
    return df


############################
# 2) Shift forecast columns by -24 hours
############################
def shift_forecast_columns(df, forecast_cols, shift_hours=-24):
    """
    Applies df[col].shift(-24) to each column in forecast_cols.
    In other words, for any given index t, the forecast becomes
    the forecast that was originally at t+24 in the raw data.
    """
    df_shifted = df.copy()
    for col in forecast_cols:
        if col in df_shifted.columns:
            df_shifted[col] = df_shifted[col].shift(shift_hours)
    # Optionally drop rows with NaNs that result from the shift
    df_shifted.dropna(subset=forecast_cols, inplace=True)
    return df_shifted


def standardize_df(df, train_start, train_end, val_start, val_end, columns):
    """
    Standardize the given columns of df based on mean & std from the training period only.

    Args:
        df (pd.DataFrame): The full dataset with a DateTimeIndex.
        train_start (pd.Timestamp): Start date for training data.
        train_end (pd.Timestamp): End date for training data.
        val_start (pd.Timestamp): Start date for validation data.
        val_end (pd.Timestamp): End date for validation data.
        columns (list of str): The columns to be standardized.

    Returns:
        tuple:
            - df_scaled (pd.DataFrame): The standardized DataFrame.
            - means (pd.Series): Mean values computed over the training period.
            - stds (pd.Series): Standard deviation values computed over the training period.
            - df_train (pd.DataFrame): Training data.
            - df_val (pd.DataFrame): Validation data.
            - df_test (pd.DataFrame): Test data.
    """
    # 1) Extract the training set
    df_train = df.loc[train_start:train_end, columns].copy()

    # 2) Compute mean & std based on training set only
    means = df_train.mean()
    stds = df_train.std().replace(0, 1e-8)  # Avoid division by zero

    # 3) Standardize the entire dataset using training stats
    df_scaled = df.copy()
    df_scaled[columns] = (df_scaled[columns] - means) / stds

    # 4) Extract validation & test sets
    df_val = df_scaled.loc[val_start:val_end]
    df_test = df_scaled.loc[val_end:]  # Everything after val_end

    return df_scaled, means, stds, df_train, df_val, df_test


# def new_formPairs(
#     df,
#     start_date,
#     end_date,
#     lookback_hours=168,
#     forecast_hours=24,
#     forecast_deadline_hour=9,
#     actual_cols = ["ACTUAL_NetLoad", "ACTUAL_ERC_Load", "ACTUAL_ERC_Wind", "ACTUAL_ERC_Solar"],
#     forecast_cols =["NetLoad", "ERC_Load", "ERC_Wind", "ERC_Solar"],
#     step_size = 24
# ):
#     """
#     Builds one (X, y) sample per forecast day D in [start_date, end_date).
#
#     - X is the 168-hour window [D - 168h, D), ending at (D - 1) hour.
#       The last 15 hours are masked for actual columns if they exceed (D-1) 9:00.
#     - y is the next 24-hour actual data [D, D+24).
#
#     Args:
#         df (pd.DataFrame): DataFrame with a DateTimeIndex at hourly resolution.
#         start_date (str or pd.Timestamp): Start date for day-by-day sampling.
#         end_date (str or pd.Timestamp): End date for day-by-day sampling (exclusive).
#         lookback_hours (int): How many hours to look back for X (default 168 for 7 days).
#         forecast_hours (int): How many hours to predict for y (default 24).
#         forecast_deadline_hour (int): The hour of D-1 after which actual data is unknown (default 9).
#         actual_cols (list of str): Columns containing actual measurements.
#         forecast_cols (list of str): Columns containing forecast or other features.
#
#     Returns:
#         (samples_X, samples_y):
#             - samples_X: list of torch.FloatTensor, each of shape (lookback_hours, #features).
#             - samples_y: list of torch.FloatTensor, each of shape (forecast_hours,).
#
#     Note:
#         - The final 15 hours of actual columns in df_window are set to 0.0 if they are
#           after forecast_deadline for that day.
#         - y is defined here as ACTUAL_NetLoad over the forecast_hours window [D, D+24).
#     """
#
#     samples_X, samples_y = [], []
#
#     current_day = pd.to_datetime(start_date)
#     final_day = pd.to_datetime(end_date)
#
#     while current_day < final_day:
#         # 1) Forecast day is 'current_day'
#         #    Forecast deadline is (current_day - 1 day) at 'forecast_deadline_hour' (e.g. 9:00)
#         forecast_deadline = (current_day - pd.Timedelta(days=1)).replace(
#             hour=forecast_deadline_hour, minute=0, second=0
#         )
#
#         # 2) The lookback window ends one hour before current_day (e.g., if current_day is 2024-07-17 00:00,
#         #    we end at 2024-07-16 23:00). Then we go back 'lookback_hours'.
#         lookback_end = current_day - pd.Timedelta(hours=1)
#         lookback_start = lookback_end - pd.Timedelta(hours=lookback_hours-1)
#
#         # Slice the DataFrame for [lookback_start, lookback_end]
#         df_window = df.loc[lookback_start:lookback_end].copy()
#
#         # 3) Mask actual data after forecast_deadline (if it falls within the window)
#         overlap_start = max(forecast_deadline, lookback_start)
#         overlap_end = min(lookback_end, df_window.index.max())
#
#         if overlap_start < overlap_end:
#             mask_cond = (df_window.index >= overlap_start) & (df_window.index < overlap_end)
#             # Replace masked actual columns with 0.0
#             df_window.loc[mask_cond, actual_cols] = 0.0
#
#         # 4) Define y = actual data for [current_day, current_day+24)
#         forecast_start = current_day
#         forecast_end = current_day + pd.Timedelta(hours=forecast_hours)
#         df_future = df.loc[forecast_start:forecast_end - pd.Timedelta(hours=1)]
#
#         y_df = df_future["ACTUAL_NetLoad"].copy()
#
#         # 5) Convert X and y to tensors
#         #    Select the columns we want: actual_cols + forecast_cols
#         select_cols = actual_cols + forecast_cols
#
#         # Ensure numeric dtype
#         df_window[select_cols] = df_window[select_cols].apply(pd.to_numeric, errors='coerce')
#         # df_window[select_cols].fillna(0.0, inplace=True)
#
#         X_tensor = torch.tensor(df_window[select_cols].values, dtype=torch.float32)
#         y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
#
#         samples_X.append(X_tensor)  # shape: (lookback_hours, number_of_features)
#         samples_y.append(y_tensor)  # shape: (forecast_hours,)
#
#         # Move to next day
#         current_day += pd.Timedelta(hours=step_size)
#
#     return samples_X, samples_y


import pandas as pd
import torch

import pandas as pd
import torch


def new_formPairs(
        df,
        start_date,
        end_date,
        lookback_hours=168,
        forecast_hours=24,
        forecast_deadline_hour=9,
        actual_cols=["ACTUAL_NetLoad", "ACTUAL_ERC_Load", "ACTUAL_ERC_Wind", "ACTUAL_ERC_Solar"],
        forecast_cols=["NetLoad", "ERC_Load", "ERC_Wind", "ERC_Solar"],
        error_cols=["NetLoad_Error", "Load_Error", "Wind_Error", "Solar_Error"],
        aux_cols=["HoD", "DoW", "MoY"],
        step_size=24
):
    """
    Builds (X, y) samples for each forecast day D in [start_date, end_date).

    - X includes:
        1. actual_cols + error_cols + aux_cols: [D-1 09:00 to D-1 09:00 - 168h]
        2. forecast_cols: [D-1 23:00 to D-1 23:00 - 168h]
    - y is the next 24-hour actual data [D, D+24).
    """
    if error_cols is None:
        error_cols = []
    if aux_cols is None:
        aux_cols = []

    selected_actual_cols = actual_cols + error_cols + aux_cols
    samples_X, samples_y = [], []

    current_day = pd.to_datetime(start_date)
    final_day = pd.to_datetime(end_date)

    while current_day < final_day:
        forecast_start = current_day
        forecast_end = current_day + pd.Timedelta(hours=forecast_hours)
        df_future = df.loc[forecast_start:forecast_end - pd.Timedelta(hours=1)]
        y_tensor = torch.tensor(df_future["ACTUAL_NetLoad"].values, dtype=torch.float32)

        actual_lookback_end = (current_day - pd.Timedelta(hours=24 - forecast_deadline_hour))
        actual_lookback_start = actual_lookback_end - pd.Timedelta(hours=lookback_hours) + pd.Timedelta(hours=1)
        df_actual = df.loc[actual_lookback_start:actual_lookback_end, selected_actual_cols]

        forecast_lookback_end = (current_day - pd.Timedelta(hours=1))
        forecast_lookback_start = forecast_lookback_end - pd.Timedelta(hours=lookback_hours) + pd.Timedelta(hours=1)
        df_forecast = df.loc[forecast_lookback_start:forecast_lookback_end, forecast_cols]

        df_window = pd.concat([df_actual.reset_index(drop=True), df_forecast.reset_index(drop=True)], axis=1)

        if df_window.shape[0] != lookback_hours:
            print(f"Warning: Lookback window shape mismatch at {current_day}")
            print(f"Expected: {lookback_hours}, Got: {df_window.shape[0]}")
            continue

        X_tensor = torch.tensor(df_window.values, dtype=torch.float32)

        samples_X.append(X_tensor)
        samples_y.append(y_tensor)

        current_day += pd.Timedelta(hours=step_size)

    return samples_X, samples_y
