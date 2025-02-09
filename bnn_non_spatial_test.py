from bayes_transformer.model import BayesianMDeT, BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from preprocessing import preprocess
from metrics import compute_metrics
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
matplotlib.use('Agg')


def plot_samples(
    model_wrapper,
    test_loader,
    train_norm,
    date_to_index,
    test_start_idx,
    window_len=168,
    num_samples=20
):
    """
    For three test samples (near the start, middle, and end of the test set):
      1) Generate a 24-hour forecast.
      2) Create a plot showing ground truth, the mean forecast, and a 10-90% prediction interval.
      3) Save the plot as a PNG file.
      4) Print and return the corresponding global start/end indices and dates.

    The function assumes that:
      - The test dataset was built using a sliding window (with step_size=1),
      - Each sample’s forecast corresponds to a 24-hour period starting at index (test_start_idx + window_len + sample_index) in the original data,
      - The original data’s timestamps are stored in `date_to_index` (a pandas Series or DatetimeIndex),
      - The model’s test() method returns a tensor of shape [1, 24, 1, num_samples] (if using a batch of 1).
    """
    testset_len = len(test_loader.dataset)
    sample_indices = [0, testset_len // 5, testset_len //
                      2, testset_len - 200, testset_len - 1]
    results = []

    for idx in sample_indices:
        # Get one (X, y) pair from the test set.
        # Note: from formPairs, x is returned as (features, window_length) and y as (1, prediction_length)
        x_samp, y_samp = test_loader.dataset[idx]
        # x_samp has shape (features, window_length); add batch dimension.
        x_samp = x_samp.unsqueeze(0)  # now shape: [1, features, window_length]

        # Get the model's forecast. The test() method transposes input to [batch, window_length, features],
        # applies any scaling, and returns a tensor of shape [1, 24, 1, num_samples].
        out = model_wrapper.test(
            in_test=x_samp,
            samples=num_samples,
            scaler=train_norm,
            force_cpu=True
        )
        # Remove the batch and channel dimensions.
        # We expect out originally of shape [1, 24, 1, num_samples];
        # squeeze(dim=0) removes the batch, and squeeze(dim=1) removes the singleton channel.
        out = out.squeeze(dim=0).squeeze(dim=1)  # now shape: [24, num_samples]

        # Compute the mean forecast and the 10th/90th percentiles along the sample dimension.
        pred_mean = out.mean(dim=1)     # shape [24]
        p10 = torch.quantile(out, 0.10, dim=1)
        p90 = torch.quantile(out, 0.90, dim=1)

        # Convert predictions to numpy arrays.
        pred_mean = pred_mean.numpy()  # shape (24,)
        p10 = p10.numpy()
        p90 = p90.numpy()

        # Ground truth y_samp comes from formPairs as shape [1, 24].
        # Squeeze to get shape (24,).
        y_samp = y_samp.squeeze().numpy()

        # Compute the global forecast start index in the original (unbatched) data.
        # The first sample in the test set corresponds to a forecast starting at:
        #    global_index = test_start_idx + window_len
        # and then each subsequent sliding window shifts this forecast by 1.
        global_start_idx = test_start_idx + window_len + idx
        global_end_idx = global_start_idx + 24 - 1

        # Retrieve the original dates using the computed global indices.
        # (We assume date_to_index is a pandas Series or DatetimeIndex.)
        start_date = date_to_index.iloc[global_start_idx]
        end_date = date_to_index.iloc[global_end_idx]

        # Instead of using the raw date_to_index (which may only include dates), we create a proper hourly time axis.
        # We assume the forecast is hourly; adjust freq if needed.
        time_axis = pd.date_range(start=start_date, periods=24, freq='H')

        # Create the plot.
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, y_samp, 'ko-', label='Ground Truth', linewidth=2)
        plt.plot(time_axis, pred_mean, 'b^-', label='Mean Prediction')
        plt.fill_between(time_axis, p10, p90, color='blue',
                         alpha=0.2, label='10-90% Interval')
        plt.title(f'24-hr Forecast: {start_date} to {end_date}')
        plt.xlabel('Date/Hour')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as a PNG file.
        plot_name = f"forecast_sample_{idx}.png"
        plt.savefig(plot_name)
        plt.close()

        print(f"Sample idx={idx} => global indices: {
              global_start_idx} to {global_end_idx}")
        print(f"Dates: {start_date} to {end_date}. Saved plot as {plot_name}.")

        results.append((global_start_idx, global_end_idx,
                       str(start_date), str(end_date)))

    return results


def main():
    df, train_loader, test_loader, train_norm, test_norm, date_to_index, test_start_idx = preprocess(
        csv_path='data/ercot_data_2025_Jan.csv',
        net_load_input=True,
        variates=['marketday', 'ACTUAL_ERC_Load',
                  'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'],
        device='cpu'
    )

    weights_path = 'modelsave/bmdet_model/bmdet.pt'
    model_wrapper = torch.load(weights_path, map_location='cpu')
    model_wrapper.model.eval()

    metrics = []
    outs = []
    # for i, (x, y) in enumerate(test_loader):
    #     out = model_wrapper.test(in_test=x, samples=100, scaler=train_norm)
    #     outs.append(out)

    #     y = y.transpose(1, 2)
    #     metrics.append(compute_metrics(out, y))
    #     print(out.shape)
    #     print(y.shape)
    #     print(len(metrics))
    #     print(f"ACR: {np.mean([m['avg_coverage_rate'] for m in metrics])},\n"
    #           f"AIL: {np.mean([m['avg_interval_length'] for m in metrics])},\n"
    #           f"AES: {np.mean([m['energy_score'] for m in metrics])}")

    results = plot_samples(
        model_wrapper=model_wrapper,
        test_loader=test_loader,
        train_norm=train_norm,
        date_to_index=date_to_index,
        test_start_idx=test_start_idx,
        window_len=168,
        num_samples=100
    )

    print("\nPlot indices summary (start_idx, end_idx, start_date, end_date):")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
