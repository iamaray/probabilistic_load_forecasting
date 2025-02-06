import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # <-- Add this line
import torch.optim as optim
from datetime import datetime, timedelta

# Import model and trainer
from mm_stlf.models import MM_STLF
from mm_stlf.trainer import Trainer
from new_preprocessing import load_data, shift_forecast_columns, build_day_ahead_samples_with_mask, standardize_df



if __name__ == "__main__":
	# 1) Load & shift & Add Columns & Handle missing values etc (see the load_data and shift_forecast_columns functions)
	df = load_data(csv_path='data/ercot_data_2025_Jan.csv')
	df = shift_forecast_columns(df, forecast_cols=["NetLoad", "ERC_Load","ERC_Wind","ERC_Solar" ])

	actual_cols = ["ACTUAL_NetLoad", "ACTUAL_ERC_Load", "ACTUAL_ERC_Wind", "ACTUAL_ERC_Solar"]
	forecast_cols = ["NetLoad", "ERC_Load", "ERC_Wind", "ERC_Solar"]

	# 2) Define date splits
	train_start_date = datetime(2023, 2, 10)
	train_end_date = datetime(2024, 7, 1)  # Training ends here
	val_start_date = train_end_date  # Validation starts right after training
	val_end_date = datetime(2024, 9, 1)  # Validation ends before test set
	test_start_date = val_end_date  # Test set starts after validation
	test_end_date = datetime(2025, 1, 6)

	# 3) Standardize the DF
	df_scaled, means, stds, df_train, df_val, df_test = standardize_df(
		df=df,
		train_start=train_start_date,
		train_end=train_end_date,
		val_start=val_start_date,
		val_end=val_end_date,
		columns=actual_cols + forecast_cols
	)

	print(df_scaled[actual_cols + forecast_cols].head())

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

	# 4b) VALIDATION: build samples from [val_start_date, val_end_date)
	samples_X_val, samples_y_val = build_day_ahead_samples_with_mask(
		df=df_scaled,
		start_date=val_start_date,
		end_date=val_end_date,
		actual_cols=actual_cols,
		forecast_cols=forecast_cols,
		lookback_hours=168,
		forecast_hours=24,
		forecast_deadline_hour=9,
	)

	# 4c) TEST: build samples from [test_start_date, test_end_date)
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
	print("Validation samples:", len(samples_X_val))
	print("Test samples:", len(samples_X_test))

	# 5) Stack them for model input
	X_train_tensor = torch.stack(samples_X_train, dim=0)
	y_train_tensor = torch.stack(samples_y_train, dim=0)

	X_val_tensor = torch.stack(samples_X_val, dim=0)
	y_val_tensor = torch.stack(samples_y_val, dim=0)

	X_test_tensor = torch.stack(samples_X_test, dim=0)
	y_test_tensor = torch.stack(samples_y_test, dim=0)

	print("X_train_tensor:", X_train_tensor.shape)
	print("y_train_tensor:", y_train_tensor.shape)
	print("X_val_tensor:", X_val_tensor.shape)
	print("y_val_tensor:", y_val_tensor.shape)
	print("X_test_tensor:", X_test_tensor.shape)
	print("y_test_tensor:", y_test_tensor.shape)

	# 6) Build DataLoaders
	train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

	val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

	test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

	# 6) Create PyTorch DataLoaders
	batch_size = 32


	# 7) Define the Model, Loss, and Optimizer
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Ensure argument order aligns with MM_STLF(seq_length, num_features, d_emb, num_mixers, dropout_rate)
	model = MM_STLF(
		seq_length=168,  # L: sequence length (past 168 hours)
		num_features=len(actual_cols + forecast_cols),  # D: total number of input features
		d_emb=64,  # d_emb: embedding dimension
		num_mixers=4,  # Number of mixer layers
		dropout_rate=0.1  # Dropout rate
	).to(device)
	criterion = nn.MSELoss()  # Mean Squared Error loss
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# 8) Initialize and Train Model
	trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device)

	trainer.train(num_epochs=50, save_path="./mm_stlf/results/best_model.pth")

	# 9) Evaluate on the Test Set
	trainer.test()