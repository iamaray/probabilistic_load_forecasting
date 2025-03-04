import torch
import torch.nn.functional as F
from torch.distributions import Normal
from deepar.model import DeepAR
from data_proc import MinMaxNorm, StandardScaleNorm
import json
import itertools
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    model_weights = torch.load(
        "modelsave/deepar_best_model_nonspatial.pt", map_location=torch.device(device))
    # Load the config file
    with open("modelsave/deepar_best_model_nonspatial_cfg.json", "r") as f:
        config = json.load(f)

    model = DeepAR(
        covariate_size=config["covariate_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        embedding_dim=config["embedding_dim"]
    )

    model.load_state_dict(model_weights)

    test_loader = torch.load(
        "data/non_spatial_AR/test_loader_non_spatial_AR.pt")

    for (x, y, z) in test_loader:
        print(x.shape, y.shape, z.shape)

    data_norm = torch.load(
        "data/non_spatial_AR/transforms_non_spatial_AR.pt")[0]
    data_norm.set_device(device)

    model.eval()

    # We'll accumulate results for all batches here.
    # Will hold samples: shape (num_samples, total_batch, seq_len)
    all_samples = []
    # Will hold mean predictions: shape (total_batch, seq_len)
    all_means = []
    # Will hold sigma predictions: shape (total_batch, seq_len)
    all_sigmas = []

    num_sampling = 40  # number of samples per time step

    with torch.no_grad():
        for batch in test_loader:
            # Each batch is a tuple: (target, covariates, mask)
            # target: [batch_size, seq_len]
            # covariates: [batch_size, seq_len, covariate_size]
            # mask: [batch_size, seq_len] (True where observed)
            target, covariates, mask = batch
            target = data_norm.transform(
                transform_col=0, x=target.unsqueeze(-1).to(device)).squeeze()
            covariates = data_norm.transform(covariates.to(device))
            # ensure we're on the same device
            device = next(model.parameters()).device
            target = target.to(device)
            covariates = covariates.to(device)
            mask = mask.to(device)

            batch_size, seq_len = target.shape

            # Prepare tensors to accumulate predictions for this batch.
            # We'll generate 40 sample runs (each run produces a full sequence of predictions).
            samples_batch = torch.zeros(
                num_sampling, batch_size, seq_len, device=device)
            # The deterministic predictions (mean and sigma) will be taken from one run (they are fixed given the inputs).
            means_batch = torch.zeros(batch_size, seq_len, device=device)
            sigmas_batch = torch.zeros(batch_size, seq_len, device=device)

            # For each sample run:
            for i in range(num_sampling):
                # Initialize the LSTM state and autoregressive input.
                h = model.init_hidden(batch_size)
                c = model.init_cell(batch_size)
                input_prev = torch.zeros(batch_size, 1, device=device)
                # Create an id tensor for embeddings. Here we assume each sample's id is its index.
                idx = torch.arange(batch_size, device=device).unsqueeze(
                    0)  # shape: (1, batch_size)

                # We'll collect predictions for each time step in this run.
                sample_sequence = torch.zeros(
                    batch_size, seq_len, device=device)
                mean_sequence = []
                sigma_sequence = []

                # Autoregressive loop over the time steps.
                for t in range(seq_len):
                    # Extract covariate information for time step t: shape (batch_size, covariate_size)
                    cov_t = covariates[:, t]
                    # Get the embedding for each time series (same for all time steps)
                    # shape: (batch_size, embedding_dim)
                    emb = model.embedding(idx).squeeze(0)
                    # Concatenate previous target value, current covariates, and embedding.
                    # input_prev: (batch_size, 1), cov_t: (batch_size, covariate_size), emb: (batch_size, embedding_dim)
                    lstm_in = torch.cat([input_prev, cov_t, emb], dim=1).unsqueeze(
                        0)  # shape: (1, batch_size, input_size)

                    # Run one step through the LSTM.
                    out, (h, c) = model.lstm(lstm_in, (h, c))
                    out = out.squeeze(0)  # shape: (batch_size, hidden_size)
                    params = model.fc(out)  # shape: (batch_size, 2)
                    mu = params[:, 0]       # predicted mean: (batch_size,)
                    # predicted sigma, ensure > 0
                    sigma = F.softplus(params[:, 1]) + 1e-6

                    mean_sequence.append(mu)
                    sigma_sequence.append(sigma)

                    # Create a Normal distribution and sample a prediction.
                    dist = Normal(mu, sigma)
                    pred_sample = dist.sample()  # shape: (batch_size,)

                    # For autoregressive input: if the target is observed at time t, use the ground truth.
                    # Otherwise, use the sample from the model.
                    observed = mask[:, t]
                    next_input = torch.where(
                        observed, target[:, t], pred_sample)
                    input_prev = next_input.unsqueeze(
                        1)  # shape: (batch_size, 1)

                    # Save the sample prediction for time t.
                    sample_sequence[:, t] = pred_sample

                # Stack the collected means and sigmas along time dimension.
                # shape: (batch_size, seq_len)
                means_stack = torch.stack(mean_sequence, dim=1)
                # shape: (batch_size, seq_len)
                sigmas_stack = torch.stack(sigma_sequence, dim=1)

                samples_batch[i] = sample_sequence
                # Save means and sigmas from the first run (they are deterministic given the inputs)
                if i == 0:
                    means_batch = means_stack
                    sigmas_batch = sigmas_stack

            # Append the results from this batch.
            # shape: (num_sampling, batch_size, seq_len)
            all_samples.append(samples_batch)
            all_means.append(means_batch)        # shape: (batch_size, seq_len)
            # shape: (batch_size, seq_len)
            all_sigmas.append(sigmas_batch)

    # Optionally, combine results from all batches.
    # shape: (num_sampling, total_samples, seq_len)
    all_samples = torch.cat(all_samples, dim=1)
    all_samples = data_norm.reverse(all_samples.unsqueeze(-1)).squeeze(-1)
    # shape: (total_samples, seq_len)
    all_means = torch.cat(all_means, dim=0)
    # shape: (total_samples, seq_len)
    all_sigmas = torch.cat(all_sigmas, dim=0)

    print("Mean predictions shape: ", all_means.shape)
    print("Sigma predictions shape: ", all_sigmas.shape)
    print("Sampled trajectories shape: ", all_samples.shape)

    T = next(iter(test_loader))[0]
    unobserved_start = 167  # Time steps 167 to 191 are unobserved.
    time_axis = torch.arange(unobserved_start, T.shape[1]).cpu().numpy()

    # For the first test sample (index 0):
    gt_unobserved = T[0, unobserved_start:]  # Shape: [192 - 167] (e.g., 25)

    # Extract the sampled trajectories for the first test sample over the unobserved period.
    # S is of shape [40, 119, 192] where the second dimension corresponds to batch samples.
    # Shape: [40, 192 - 167]
    samples_first = all_samples[:, 0, unobserved_start:]

    # Compute the 10th and 90th percentiles along the sample dimension.
    p10 = torch.quantile(samples_first, 0.1, dim=0)
    p90 = torch.quantile(samples_first, 0.9, dim=0)

    # Plot the ground truth and the 10th/90th percentile bands.
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, gt_unobserved.cpu().numpy(),
             label="Ground Truth", color="black", linewidth=2)
    plt.plot(time_axis, p10.cpu().numpy(),
             label="10th Percentile", color="blue", linestyle="--")
    plt.plot(time_axis, p90.cpu().numpy(),
             label="90th Percentile", color="red", linestyle="--")
    plt.fill_between(time_axis, p10.cpu().numpy(), p90.cpu().numpy(),
                     color="gray", alpha=0.3, label="10-90 Percentile Band")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(
        "Unobserved Period: Ground Truth vs 10th/90th Percentile of Sampled Trajectories")
    plt.legend()
    plt.show()
