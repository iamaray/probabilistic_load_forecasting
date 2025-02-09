"""
This file is constructed on Feb 6, 2025 after the discussion with Arya.

The first goal of this script is to construct the cleaned-version of the excel file, i.e.,
meaning it is consecutive in dates and hours, no missing values, error columns calculated correctly, columns shifted correctly,
and feature columns (such hour of the day, day of the week, and etc. We name the cleaned-version "ercot_data_cleaned.csv"

The second goal is then form the (X,y) pairs and eventually get to the point where we have the train_loader, val_loader, and test_loader.
Once we have these three loaders, Arya and I will use exactly the same three loaders (train_loader, val_loader, and test_loader) to compare
different benchmark models.

The third goal of is, once we have the result (y_predict tensor), we will need to write it to the f"{model_name}_results.csv" where the
index is time (datetime), and columns (ACTUAL_NetLoad and Pred_NetLoadQ10, Pred_NetLoadQ50, Pred_NetLoadQ90). So that later, we can have
a python script called performance_analysis.py which iteratively pulls in these f"{model_name}_results.csv" files to conduct MAPE, coverage rate,
Interval Length, Pinball Loss, Energy Score and etc comparison.

"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # <-- Add this line
import torch.optim as optim
from datetime import datetime, timedelta

# Import model and trainer
from mm_stlf.models import MM_STLF
from mm_stlf.trainer import Trainer
from new_preprocessing import load_data, shift_forecast_columns, new_formPairs, standardize_df



if __name__ == "__main__":
	"""1) Create the cleaned dataframe and save it into './data/ercot_data_cleaned.csv'"""
	# Define 4 different categories of columns, actual_cols and error_cols behave the same
	# while forecast_cols a little different (3rd party)
	actual_cols = ["ACTUAL_NetLoad", "ACTUAL_ERC_Load", "ACTUAL_ERC_Wind", "ACTUAL_ERC_Solar"]
	forecast_cols = ["NetLoad", "ERC_Load", "ERC_Wind", "ERC_Solar"]
	error_cols = ["NetLoad_Error", "Load_Error", "Wind_Error", "Solar_Error"]
	aux_cols = ["HoD", "DoW", "MoY"]
	df = load_data(csv_path='data/ercot_data_2025_Jan.csv')
	df = shift_forecast_columns(df, forecast_cols=["NetLoad", "ERC_Load","ERC_Wind","ERC_Solar"],shift_hours=-24)
	df_cleaned = df[actual_cols+error_cols+forecast_cols+aux_cols]

	# df_cleaned.to_csv('./data/ercot_data_cleaned.csv')


	"""2) Define date splits, and standardize data"""
	train_start_date = datetime(2023, 2, 10)
	train_end_date = datetime(2024, 7, 1)  # Training ends here
	val_start_date = train_end_date  # Validation starts right after training
	val_end_date = datetime(2024, 9, 1)  # Validation ends before test set
	test_start_date = val_end_date  # Test set starts after validation
	test_end_date = datetime(2025, 1, 6)


	"""3) Standardize the df_clean"""
	df_scaled, means, stds, df_train, df_val, df_test = standardize_df(
		df=df_cleaned,
		train_start=train_start_date,
		train_end=train_end_date,
		val_start=val_start_date,
		val_end=val_end_date,
		columns=actual_cols + error_cols + forecast_cols
	)

	print(df_scaled.head())

	"""4) Prepare the (X,y) pairs for the train, validation and test"""
	# Training Data
	samples_X_train, samples_y_train = new_formPairs(
		df=df_scaled,
		start_date=train_start_date,
		end_date=train_end_date,
		actual_cols=actual_cols,
		forecast_cols=forecast_cols,
		error_cols=["NetLoad_Error", "Load_Error", "Wind_Error", "Solar_Error"],  # Optional if default
		aux_cols=["HoD", "DoW", "MoY"],  # Optional if default
		lookback_hours=168,
		forecast_hours=24,
		forecast_deadline_hour=9,  # Starting from the 9 AM of the previous day
		step_size=24  # Moving 1 day at a time
	)

	# Validation Data
	samples_X_val, samples_y_val = new_formPairs(
		df=df_scaled,
		start_date=val_start_date,
		end_date=val_end_date,
		actual_cols=actual_cols,
		forecast_cols=forecast_cols,
		error_cols=["NetLoad_Error", "Load_Error", "Wind_Error", "Solar_Error"],
		aux_cols=["HoD", "DoW", "MoY"],
		lookback_hours=168,
		forecast_hours=24,
		forecast_deadline_hour=9,
		step_size=24
	)

	# Test Data
	samples_X_test, samples_y_test = new_formPairs(
		df=df_scaled,
		start_date=test_start_date,
		end_date=test_end_date,
		actual_cols=actual_cols,
		forecast_cols=forecast_cols,
		error_cols=["NetLoad_Error", "Load_Error", "Wind_Error", "Solar_Error"],
		aux_cols=["HoD", "DoW", "MoY"],
		lookback_hours=168,
		forecast_hours=24,
		forecast_deadline_hour=9,
		step_size=24
	)

	print("Train samples:", len(samples_X_train))
	print("Validation samples:", len(samples_X_val))
	print("Test samples:", len(samples_X_test))

	"""5) Build the DataLoaders using the (X,y) pairs"""
	X_train_tensor = torch.stack(samples_X_train, dim=0)
	y_train_tensor = torch.stack(samples_y_train, dim=0)

	X_val_tensor = torch.stack(samples_X_val, dim=0)
	y_val_tensor = torch.stack(samples_y_val, dim=0)

	X_test_tensor = torch.stack(samples_X_test, dim=0)
	y_test_tensor = torch.stack(samples_y_test, dim=0)

	print(X_train_tensor[1].shape)


	train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

	val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

	test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

	torch.save(train_dataset, './data/train_dataset.pt')
	torch.save(val_dataset, './data/val_dataset.pt')
	torch.save(test_dataset, './data/test_dataset.pt')


	"""Use the following code to laod the dataloader"""
	# # Load the saved datasets
	# actual_cols = ["ACTUAL_NetLoad", "ACTUAL_ERC_Load", "ACTUAL_ERC_Wind", "ACTUAL_ERC_Solar"]
	# forecast_cols = ["NetLoad", "ERC_Load", "ERC_Wind", "ERC_Solar"]
	# error_cols = ["NetLoad_Error", "Load_Error", "Wind_Error", "Solar_Error"]
	# aux_cols = ["HoD", "DoW", "MoY"]
	#
	# # Load the saved datasets
	# train_dataset = torch.load('./data/train_dataset.pt', weights_only=False)
	# val_dataset = torch.load('./data/val_dataset.pt', weights_only=False)
	# test_dataset = torch.load('./data/test_dataset.pt', weights_only=False)
	#
	# # Recreate the DataLoaders
	# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
	# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
	# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
