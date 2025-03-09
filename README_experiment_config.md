# Experiment Configuration for BNN Prior Ablation

This README explains how to use the experiment configuration feature for the Bayesian Neural Network (BNN) prior ablation experiments.

## Overview

The `bnn_prior_ablation.py` script now accepts an experiment configuration JSON file that allows you to customize various aspects of the experiments without modifying the code. This makes it easier to run different experiments with different parameters.

## Usage

Run the script with the `--experiment_config_path` argument:

```bash
python bsmdet_experiments/bnn_prior_ablation.py --spatial='non_spatial' --prior_param_grid_path='cfgs/bmdet/mixture_model_grid.json' --experiment_config_path='cfgs/bmdet/experiment_config.json'
```

## Configuration File Format

The experiment configuration file should be a JSON file with the following structure:

```json
{
  "training": {
    "train_epochs": 1,
    "save_model": true
  },
  "inference": {
    "num_samples": 40,
    "test_start_date": "2024-09-01",
    "test_end_date": "2025-01-06"
  },
  "paths": {
    "model_save_dir": "modelsave/bmdet",
    "results_dir": "results",
    "data_dir": "data"
  },
  "device": "cuda"
}
```

### Configuration Options

#### Training

- `train_epochs`: Number of epochs to train the model (default: 1)
- `save_model`: Whether to save the trained model (default: true)

#### Inference

- `num_samples`: Number of samples to generate during inference (default: 40)
- `test_start_date`: Start date for the test period in ISO format (default: "2024-09-01")
- `test_end_date`: End date for the test period in ISO format (default: "2025-01-06")

#### Paths

- `model_save_dir`: Directory to save the trained models (default: "modelsave/bmdet")
- `results_dir`: Directory to save the results (default: "results")
- `data_dir`: Directory containing the data (default: "data")

#### Device

- `device`: Device to use for training and inference (default: "cuda" if available, otherwise "cpu")

## Example

Here's an example of a configuration file that trains for 5 epochs and uses 100 samples for inference:

```json
{
  "training": {
    "train_epochs": 5,
    "save_model": true
  },
  "inference": {
    "num_samples": 100,
    "test_start_date": "2024-09-01",
    "test_end_date": "2025-01-06"
  },
  "paths": {
    "model_save_dir": "modelsave/bmdet",
    "results_dir": "results",
    "data_dir": "data"
  },
  "device": "cuda"
}
```

## Notes

- If a parameter is not specified in the configuration file, the default value will be used.
- The `device` parameter will be overridden if CUDA is not available, regardless of what is specified in the configuration file.
- Dates must be in ISO format (YYYY-MM-DD).
