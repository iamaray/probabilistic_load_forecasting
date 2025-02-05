import numpy as np
import pandas as pd
# from preprocessing import preprocess
from new_preprocessing import load_data, shift_forecast_columns, build_day_ahead_samples_with_mask,standardize_df
from deepar.model import Net
from deepar.train import DeepARTrainer
from deepar.train import grid_search_torch_model
from datetime import datetime, timedelta
import torch

if __name__ == "__main__":
	# 1) Load & shift & Add Columns & Handle missing values etc (see the load_data and shift_forecast_columns functions)
	df = load_data(csv_path='data/ercot_data_2025_Jan.csv')
	df = shift_forecast_columns(df, forecast_cols=["NetLoad", "ERC_Load","ERC_Wind","ERC_Solar" ])

	actual_cols = ["ACTUAL_NetLoad", "ACTUAL_ERC_Load", "ACTUAL_ERC_Wind", "ACTUAL_ERC_Solar"]
	forecast_cols = ["NetLoad", "ERC_Load", "ERC_Wind", "ERC_Solar"]

	# 2) Define date splits
	train_start_date = datetime(2023, 2, 10)
	train_end_date = datetime(2024, 7, 1)  # up to (but not including) 2024-07-01
	test_start_date = train_end_date
	test_end_date = datetime(2025, 1, 6)


	# 3) Standardize the DF (only certain columns)
	cols_to_scale = actual_cols + forecast_cols
	df_scaled, means, stds = standardize_df(
		df=df,
		train_start=train_start_date,
		train_end=train_end_date,
		columns=cols_to_scale
	)

	print(df_scaled[actual_cols+forecast_cols].head())

	# 4a) TRAIN: build samples from [train_start_date, train_end_date)
	samples_X_train, samples_y_train = build_day_ahead_samples_with_mask(
		df=df_scaled,
		start_date=train_start_date,
		end_date=train_end_date,
		actual_cols=actual_cols,
		forecast_cols=forecast_cols,
		lookback_hours=168,
		forecast_hours=24,
		forecast_deadline_hour=9,
	)

	# 4b) TEST: build samples from [test_start_date, test_end_date)
	samples_X_test, samples_y_test = build_day_ahead_samples_with_mask(
		df=df_scaled,
		start_date=test_start_date,
		end_date=test_end_date,
		actual_cols=actual_cols,
		forecast_cols=forecast_cols,
		lookback_hours=168,
		forecast_hours=24,
		forecast_deadline_hour=9,
	)

	print("Train samples:", len(samples_X_train))
	print("Test samples:", len(samples_X_test))

	# 5) Stack them for model input
	X_train_tensor = torch.stack(samples_X_train, dim=0)
	y_train_tensor = torch.stack(samples_y_train, dim=0)

	X_test_tensor = torch.stack(samples_X_test, dim=0)
	y_test_tensor = torch.stack(samples_y_test, dim=0)

	print("X_train_tensor:", X_train_tensor.shape)
	print("y_train_tensor:", y_train_tensor.shape)
	print("X_test_tensor:", X_test_tensor.shape)
	print("y_test_tensor:", y_test_tensor.shape)

	# 6) Build DataLoaders
	train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

	test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

	# Now you're ready for training
	for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
		# training step ...
		pass

