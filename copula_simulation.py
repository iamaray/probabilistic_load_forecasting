"""
This file is created on Jan 8, 2025
Using the latest data from Jensen,
We have data ranging from Feb 2023 to Jan 2025.
This file serves as a template moving forward to set up benchmarks, comparing different models
by Moyi and Arya.
"""

import sys
import os
# sys.path.append(r"C:\Users\WP6298\Documents")
from scipy.stats import norm, shapiro, kstest, anderson, t
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
pd.options.mode.chained_assignment = None  # default='warn'
import random
current_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if current_project_path not in sys.path:
    sys.path.insert(0, current_project_path)
from utils.helper import shuffle
import warnings
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from copulas.visualization import compare_1d, compare_2d, compare_3d
from copulas.univariate import (
    Univariate,
    StudentTUnivariate,
    GaussianKDE,
    GaussianUnivariate,
    BetaUnivariate,
    GammaUnivariate,
    UniformUnivariate,
    LogLaplace,
    TruncatedGaussian
)
from copulas.multivariate import GaussianMultivariate, VineCopula
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
seed = 750
np.random.seed(seed)
random.seed(seed)
np.set_printoptions(precision=3)

import matplotlib.pyplot as plt
import seaborn as sns


def df_for_error_learning(df, columns, start_time, end_time, interested_hours):
    # Filter by the specified time range
    filtered_df = df.loc[start_time:end_time + timedelta(hours=23), columns]

    # Filter by the specified hours
    filtered_df = filtered_df[filtered_df.index.hour.isin(interested_hours)]

    # Create a new DataFrame to hold the transformed data
    transformed_data = []

    # Iterate over the DataFrame and reshape the data
    for date, group in filtered_df.groupby(filtered_df.index.date):
        row = {}
        for column in columns:
            for hour in interested_hours:
                value = group.loc[group.index.hour == hour, column]
                if not value.empty:
                    row[f"{column}_hour{hour}"] = value.iloc[0]
                else:
                    row[f"{column}_hour{hour}"] = None
        transformed_data.append(row)

    # Convert the transformed data into a DataFrame
    unique_dates = np.unique(filtered_df.index.date)
    transformed_df = pd.DataFrame(transformed_data, index=pd.to_datetime(unique_dates))
    return transformed_df


### let's try this new method, first use copula univariate to calculate the best parametric models and corresponding parameters for each variable
def find_best_univariate_models(df, columns, test_time, interested_hours, learning_period, num_scenarios,
                                copula_choice="GaussianMultivariate"):
    if learning_period == 30:
        start_time = test_time - timedelta(days=365) - timedelta(days=15)
        end_time = start_time + timedelta(days=30)
        df_el = df_for_error_learning(df, columns, start_time, end_time, interested_hours)
        print(f"df_el head is given as \n {df_el.head()}")

        er_array = df_el.values
        er_mu = np.mean(er_array, axis=0)
        er_std = np.std(er_array, axis=0)
        zscore_array = (er_array - er_mu) / er_std

        parameters_dict = {}
        for idx, column in enumerate(df_el):
            data = zscore_array[:, idx]
            univariate = Univariate()
            univariate.fit(data)
            parameters = univariate.to_dict()
            parameters_dict[column] = parameters

        # Extract univariate distribution types from parameters_dict
        distribution_types = [params['type'] for params in parameters_dict.values()]
        # print(f"distribution_types is given as\n {distribution_types}")

        # Create GaussianMultivariate copula with the specified univariate distributions
        if copula_choice == "GaussianMultivariate":
            joint_dist = GaussianMultivariate(distribution=distribution_types)
            joint_dist.fit(zscore_array)

        # If copula_choice is CVine or other VineCopula types, convert zscore_array back to DataFrame
        elif copula_choice in ["CVine", "DVine", "RVine"]:
            zscore_df = pd.DataFrame(zscore_array, columns=df_el.columns)  # Convert back to DataFrame
            if copula_choice == "CVine":
                joint_dist = VineCopula('center')
            elif copula_choice == "DVine":
                joint_dist = VineCopula('direct')
            elif copula_choice == "RVine":
                joint_dist = VineCopula('regular')
            joint_dist.fit(zscore_df)  # Fit the vine copula with DataFrame input

        # Get the rank marix for later shaake shuffle
        sampled = joint_dist.sample(num_scenarios)
        u_array = sampled  # just to align with the version 5 code, there the samples is a pandaframe
        sorted_indices = np.argsort(u_array, axis=0)
        ranks_array = np.argsort(sorted_indices, axis=0) + 1

        return parameters_dict, er_mu, er_std, joint_dist, ranks_array


def range_for_interested_hours(df_snap_shot_next24, parameters_dict, interested_hours, er_mu, er_std, num_scenarios, ranks_array):
    y_prediction = df_snap_shot_next24[interested_hours,:].flatten('F')
    # print(f'y_prediction is of shape {y_prediction}')
    vector = np.arange(1, num_scenarios+1)/(num_scenarios+1)

    # scenarios = np.zeros((num_scenarios, df_snap_shot_next24.shape[1]))
    scenarios = np.zeros((num_scenarios, len(er_mu)))

    # Valid univariate types
    valid_univariate_types = {
        'copulas.univariate.student_t.StudentTUnivariate',
        'copulas.univariate.gaussian_kde.GaussianKDE',
        'copulas.univariate.gaussian.GaussianUnivariate',
        'copulas.univariate.beta.BetaUnivariate',
        'copulas.univariate.gamma.GammaUnivariate',
        'copulas.univariate.uniform.UniformUnivariate',
        'copulas.univariate.log_laplace.LogLaplace',
        'copulas.univariate.truncated_gaussian.TruncatedGaussian'
    }

    for idx, column in enumerate(parameters_dict.keys()):
        params = parameters_dict[column]
        # print(f"params is {params}")

        if params['type'] not in valid_univariate_types:
            raise ValueError(f"Unsupported univariate type: {params['type']}")

        # Load the best model and corresponding params to the copula
        univariate = Univariate()
        univariate = Univariate.from_dict(params)

        inverse_samples = univariate.ppf(vector)
        # print(f"inverse_samples is given as\n {inverse_samples}")
        scenarios[:, idx] = y_prediction[idx] + er_mu[idx] + inverse_samples * er_std[idx]
        ss_scenarios = shuffle(scenarios, ranks_array)

    return y_prediction, scenarios, ss_scenarios



def evaluate_copula_metrics(df, columns, start_time, end_time, hour_blocks, copula_choices, num_scenarios,
                            learning_period=30):
    """This function is created Oct,2024 to calculate the performance of multiple copulas."""
    # Make a deep copy of the df, so that we do not modify df in the following
    df_copula_performance = copy.deepcopy(df)

    # Add columns for each copula and each metric
    for copula_choice in copula_choices:
        df_copula_performance[f"{copula_choice}_CR"] = "NA"
        df_copula_performance[f"{copula_choice}_IL"] = "NA"

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    current_time = start_time

    while current_time <= end_time:
        print(f"we are currently standing at {current_time}")
        for hour_block in hour_blocks:
            interested_hour = hour_block
            # Loop through each copula choice and compute metrics
            for copula_choice in copula_choices:
                try:
                    # Find the best univariate models and copula distribution
                    parameters_dict, er_mu, er_std, joint_dist, ranks_array = find_best_univariate_models(
                        df, columns, current_time, interested_hour, learning_period, num_scenarios, copula_choice
                    )

                    # Point forecast from 3rd party (not fully defined in your script, but assuming it's handled here)
                    df_snap_shot_next24 = df.loc[current_time:current_time + timedelta(hours=23),
                                          ["ERC_Load", "ERC_Wind", "ERC_Solar"]].to_numpy()

                    # Generate prediction and scenarios
                    y_prediction, scenarios, ss_scenarios = range_for_interested_hours(
                        df_snap_shot_next24, parameters_dict, interested_hour, er_mu, er_std, num_scenarios,
                        ranks_array
                    )

                    assert ss_scenarios.shape == (num_scenarios, 3*len(hour_block))


                    ### The code below needs to be written in a loop, loop through the hour in the hour_block
                    for i, hour in enumerate(hour_block):
                        # print(f"we are at index {i}, at hour {hour} within {hour_block} using copula {copula_choice}")
                        # Calculate NetLoad forecast (point and scenarios)
                        netload_point_forecasting = y_prediction[i] - y_prediction[i+len(hour_block)] - y_prediction[i+len(hour_block)*2]
                        netload_ss_scenarios = ss_scenarios[:, i] - ss_scenarios[:, i+len(hour_block)] - ss_scenarios[:, i+len(hour_block)*2]

                        # Check if the actual net load is within the forecasted 10-90th percentile range
                        condition1 = df_copula_performance.loc[
                                         current_time + timedelta(hours=hour), "ACTUAL_NetLoad"] >= np.percentile(
                            netload_ss_scenarios, 10)
                        condition2 = df_copula_performance.loc[
                                         current_time + timedelta(hours=hour), "ACTUAL_NetLoad"] <= np.percentile(
                            netload_ss_scenarios, 90)

                        # Calculate Coverage Rate (CR)
                        if condition1 and condition2:
                            df_copula_performance.loc[current_time + timedelta(hours=hour), f"{copula_choice}_CR"] = 1
                        else:
                            df_copula_performance.loc[current_time + timedelta(hours=hour), f"{copula_choice}_CR"] = 0

                        # Calculate Interval Length (IL)
                        df_copula_performance.loc[
                            current_time + timedelta(hours=hour), f"{copula_choice}_IL"] = np.percentile(
                            netload_ss_scenarios, 90) - np.percentile(netload_ss_scenarios, 10)

                        # print(f"The Interval length here is {np.percentile(netload_ss_scenarios, 90) - np.percentile(netload_ss_scenarios, 10)}")

                except Exception as e:
                    # Catch the exception and print a message, then continue with the next iteration
                    print(f"Error encountered for copula {copula_choice} at {current_time} hour block {hour_block}: {e}")
                    continue

        # Move to the next day
        current_time += timedelta(days=1)

    return df_copula_performance

def plot_2d_correlation(df, variables, hour_block, title_prefix="Correlation Map"):
	"""
	Plots a 2D correlation map for specified variables and hour blocks.
	Args:
	- df (pd.DataFrame): The data source to use for plotting.
	- variables (list): A list containing two column names, e.g., ["var1", "var2"].
	- hour_block (list): A list of integers specifying the hours to filter, e.g., [18, 19, 20].
	- title_prefix (str): Prefix for the plot title. Will be combined with variable names and hour block.

	Returns:
	- None: Displays a plot.
	"""
	# Filter the dataframe for the specified hour block
	filtered_df = df[df['hourending'].isin(hour_block)]

	# Extract the variables for plotting
	x, y = variables
	x_data = filtered_df[x]
	y_data = filtered_df[y]

	# Plot the 2D correlation map
	plt.figure(figsize=(8, 6))
	sns.kdeplot(x=x_data, y=y_data, cmap="Blues", fill=True)

	# Construct the plot title
	title = f"{title_prefix}: {x} vs {y} for Hours {hour_block}"
	plt.title(title)
	plt.xlabel(x)
	plt.ylabel(y)
	plt.grid(True)
	plt.show()


if __name__ == '__main__':
	# Find the path for the data file
	src_dir_path = os.path.dirname(__file__)
	data_dir_path = os.path.join(src_dir_path, '..', 'data')
	data_path = os.path.join(data_dir_path, "ercot_data_2025_Jan.csv")
	data = pd.read_csv(data_path)

	data['time'] = pd.to_datetime(data['marketday']) + pd.to_timedelta(data['hourending'] - 1, unit='h')
	data.set_index('time', inplace=True)


	### It is been verified that we have the following:
	# For Total Actual Load Columns in (MW): X = Y + Z + AA + AB
	# For Total Actual Wind Columns in (MW): AG = AC + AD + AE + AF + AH
	# For Total Actual Solar Columns in (MW): AI (Just one single column)

	# For Total Forecasted Load Columns in (MW): M (Just one single column)
	# For total Forecasted Wind Columns in (MW):  S + T + U + V + W
	# For total Forecasted Solar Columns in (MW): N (Just one single column)

	data['ERC_Wind'] = data['ERC_CWind'] + data['ERC_NWind'] + data['ERC_PWind'] + data['ERC_SWind'] + data['ERC_WWind']

	df_refilled = data.copy()
	print(f"The columns of df_refilled are {df_refilled.columns}")

	######Learn the error distribution for a set of specific hours (hour blocks)
	df_refilled["load_error"] = df_refilled["ACTUAL_ERC_Load"] - df_refilled["ERC_Load"]
	df_refilled["wind_error"] = df_refilled["ACTUAL_ERC_Wind"] - df_refilled["ERC_Wind"]
	df_refilled["solar_error"] = df_refilled["ACTUAL_ERC_Solar"] - df_refilled["ERC_Solar"]

	df_refilled["ACTUAL_NetLoad"] = df_refilled["ACTUAL_ERC_Load"] - df_refilled["ACTUAL_ERC_Wind"] - df_refilled["ACTUAL_ERC_Solar"]
	df_refilled["NetLoad"] = df_refilled["ERC_Load"] - df_refilled["ERC_Wind"] - df_refilled["ERC_Solar"]
	df_refilled["netload_error"] = df_refilled["ACTUAL_NetLoad"] - df_refilled["NetLoad"]
	df_show_off = df_refilled[["ACTUAL_ERC_Load", "ERC_Load",
							   "ACTUAL_ERC_Wind", "ERC_Wind",
							   "ACTUAL_ERC_Solar", "ERC_Solar",
							   "ACTUAL_NetLoad","NetLoad",
							   "load_error", "wind_error", "solar_error",
							   "netload_error"]]


	"""
	Now we calculate z-scores and cdfs of columns
	"""

	# Calculate z-score and empirical CDF for specified columns
	error_columns = ["load_error", "wind_error", "solar_error", "netload_error"]

	for col in error_columns:
		# Calculate the mean and standard deviation for the column
		mean = df_refilled[col].mean()
		std = df_refilled[col].std()

		# Calculate the z-score
		zscore_col = f"{col}_zscore"
		df_refilled[zscore_col] = (df_refilled[col] - mean) / std

		# Calculate the empirical CDF
		cdf_col = f"{col}_cdf"
		df_refilled[cdf_col] = df_refilled[zscore_col].rank(method="average", pct=True)

	# Display the modified dataframe
	columns_to_display = error_columns + [f"{col}_zscore" for col in error_columns] + [f"{col}_cdf" for col in
																					   error_columns]
	print(df_refilled[columns_to_display].head(5))
	print(df_refilled[columns_to_display].tail(5))



	"""
	Then drawing contour maps to study correlations
		among load_error, wind_error, and solar_error
	"""
	plot_2d_correlation(
		df=df_refilled,
		variables=["wind_error_cdf", "solar_error_cdf"],
		hour_block=[18, 19, 20],
		title_prefix="Empirical Contour Map"
	)

	copula_choices = ["GaussianMultivariate", "CVine", "DVine", "RVine"]

	df_copula_performance = evaluate_copula_metrics(
		df=df_refilled,
		columns=["load_error", "wind_error", "solar_error"],
		start_time=datetime(2024, 2, 1),
		end_time=datetime(2024, 2, 1),
		hour_blocks=[[0, 1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19, 20],
					 [21, 22, 23]],
		copula_choices=copula_choices,  # Pass multiple copulas here
		num_scenarios=30,
		learning_period=30
	)

	print(f"df_copula_performance slicing is \n {df_copula_performance[datetime(2024,2,1,0):datetime(2024,2,1,23)]}")

