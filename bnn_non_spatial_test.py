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

    if i == 5:
        break
