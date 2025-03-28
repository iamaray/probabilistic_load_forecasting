# Data Preprocessing for Load Forecasting

This document explains how to use the generalized data preprocessing code (`data_proc.py`) with JSON configuration files.

## Overview

The preprocessing script loads time series data from a CSV file, performs feature selection, creates training/validation/test splits, and generates sliding window examples for forecasting models. It supports both standard sequence-to-sequence forecasting and autoregressive (AR) model training.

## Configuration Files

The preprocessing is controlled through JSON configuration files. Example configurations can be found in the `cfgs/data_proc/` directory:

- `spain_dataset_non_spatial.json`: Basic configuration for non-spatial data (load and a few time features)
- `spain_dataset_spatial.json`: Configuration for spatial data (includes additional features like wind, solar, etc.)
- `spain_dataset_ar.json`: Configuration for autoregressive models

## Usage

Run the preprocessing script with a configuration file:

```bash
python data_proc.py --config_path cfgs/data_proc/spain_dataset_non_spatial.json
```

## Configuration Parameters

| Parameter            | Description                                       | Default Value                   |
| -------------------- | ------------------------------------------------- | ------------------------------- |
| `csv_path`           | Path to the input CSV file                        | `"data/ercot_data_cleaned.csv"` |
| `train_start`        | Training data start date (YYYY-MM-DD)             | `"2023-02-10"`                  |
| `train_end`          | Training data end date (YYYY-MM-DD)               | `"2024-07-01"`                  |
| `val_start`          | Validation data start date (YYYY-MM-DD)           | `"2024-07-01"`                  |
| `val_end`            | Validation data end date (YYYY-MM-DD)             | `"2024-09-01"`                  |
| `test_start`         | Test data start date (YYYY-MM-DD)                 | `"2024-09-01"`                  |
| `test_end`           | Test data end date (YYYY-MM-DD)                   | `"2025-01-06"`                  |
| `ar_model`           | Whether to prepare data for autoregressive models | `false`                         |
| `x_start_hour`       | Starting hour for the first example               | `9`                             |
| `x_y_gap`            | Gap between input and forecast in hours           | `15`                            |
| `x_window`           | Input window length in hours                      | `168` (1 week)                  |
| `y_window`           | Forecast horizon in hours                         | `24` (1 day)                    |
| `step_size`          | Step size for sliding window in hours             | `24` (1 day)                    |
| `batch_size`         | Batch size for DataLoaders                        | `64`                            |
| `num_workers`        | Number of workers for DataLoaders                 | `1`                             |
| `included_feats`     | List of feature columns to include                | `null` (all features)           |
| `num_transform_cols` | Number of columns to normalize                    | `2`                             |
| `spatial`            | Whether the data includes spatial features        | `false`                         |
| `output_dir`         | Directory to save processed data                  | `null` (auto-generated)         |
| `date_column`        | Name of the date/time column                      | `"Unnamed: 0"`                  |

## Output Files

The script produces the following output files in the specified output directory:

- `train_tensor_{suffix}.pt`: Raw normalized training data tensor
- `train_dataset_{suffix}.pt`: Training dataset with sliding windows
- `val_dataset_{suffix}.pt`: Validation dataset with sliding windows
- `test_dataset_{suffix}.pt`: Test dataset with sliding windows
- `train_loader_{suffix}.pt`: Training DataLoader
- `val_loader_{suffix}.pt`: Validation DataLoader
- `test_loader_{suffix}.pt`: Test DataLoader
- `transforms_{suffix}.pt`: Data normalization transforms
- `data_report_{suffix}.txt`: Detailed report of data processing

Where `{suffix}` is either `spatial` or `non_spatial`, possibly with `_AR` appended for autoregressive models.

## Data Report

A comprehensive data report is automatically generated in the output directory. This report includes:

- Dataset information (source file, date ranges)
- Data dimensions and sample counts
- Sliding window parameters
- Feature information
- Model type settings
- Normalization details (including mean/std or min/max values)
- Output file information
- Timestamp of processing

This report is useful for reproducibility and documenting the exact preprocessing steps and data characteristics.

## Example: Creating a New Configuration

To process a new dataset, create a new JSON configuration file:

```json
{
  "csv_path": "data/my_new_dataset.csv",
  "train_start": "2020-01-01",
  "train_end": "2021-01-01",
  "val_start": "2021-01-01",
  "val_end": "2021-07-01",
  "test_start": "2021-07-01",
  "test_end": "2021-12-31",
  "included_feats": [
    "load",
    "temperature",
    "humidity",
    "hour",
    "day_of_week",
    "month"
  ],
  "num_transform_cols": 3,
  "spatial": false,
  "output_dir": "data/my_processed_data",
  "date_column": "timestamp"
}
```

Then run the preprocessing script with your configuration:

```bash
python data_proc.py --config_path path/to/your/config.json
```
