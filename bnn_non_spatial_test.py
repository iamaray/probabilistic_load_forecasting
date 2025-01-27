import torch
import numpy as np
from bayes_transformer.model import BayesianMDeT, BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from preprocessing import preprocess
from metrics import compute_metrics
from matplotlib import pyplot as plt

df, train_loader, test_loader, train_norm, test_norm = preprocess(
    csv_path='data/ercot_data_2025_Jan.csv',
    net_load_input=True,
    variates=['marketday', 'ACTUAL_ERC_Load',
              'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'],
    device='cpu')

weights_path = 'modelsave/bmdet_model/bmdet.pt'

model_wrapper = torch.load(weights_path, map_location='cpu')

model_wrapper.model.eval()

metrics = []
outs = []
for i, (x, y) in enumerate(test_loader):
    # if i == 0:
    #     # Detach and move input to CPU
    #     x = x.detach().cpu()
    #     print(x.shape)
    #     # Transform and reverse-transform the input
    #     x_scaled = test_norm.transform(x)       # Scaled version of x
    #     x_rev = test_norm.reverse(x_scaled)     # Reverse-scaled version of x

    #     x = x.numpy()
    #     x_rev = x_rev.numpy()
    #     # Plot the original input vs reverse-scaled input
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(
    #         range(x.shape[-1]), x[0, 0, :], label=f"Original Feature ", alpha=0.8
    #     )
    #     plt.plot(
    #         range(x_rev.shape[-1]), x_rev[0, 0, :], '--', label=f"Reversed Feature ", alpha=0.7
    #     )

    #     # Add plot details
    #     plt.title("Original vs Reverse-Scaled Input")
    #     plt.xlabel("Time Steps")
    #     plt.ylabel("Values")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    #     break  # Stop after the first batch

    # Shape: [batch_size, 24, 1, 20]
    out = model_wrapper.test(in_test=x, samples=20, scaler=train_norm)
    outs.append(out)
    y = y.transpose(1, 2)
    metrics.append(compute_metrics(out, y))

    print(out.shape)
    print(y.shape)
    print(len(metrics))

    print(f"ACR: {np.mean([m['avg_coverage_rate'] for m in metrics])},\n"
          f"AIL: {np.mean([m['avg_interval_length'] for m in metrics])},\n"
          f"AES: {np.mean([m['energy_score'] for m in metrics])}")

    if i == 0:
        forecasts = out[0, :, 0, :].detach().numpy()
        actual = y[0, :, 0].detach().numpy()

        # Plot all samples
        plt.figure(figsize=(10, 6))
        for hour in range(forecasts.shape[0]):
            plt.scatter(
                [hour + 1] * forecasts.shape[1],
                forecasts[hour, :],
                alpha=0.5, label="Samples" if hour == 0 else None, color='blue'
            )

        plt.plot(range(1, 25), actual, 'r-', label="Actual", linewidth=2)

        plt.title("Forecast Samples vs Actual (24-hour Period)")
        plt.xlabel("Hour")
        plt.ylabel("Net Load")
        plt.legend()
        plt.grid(True)
        plt.show()

    if i == 5:
        break