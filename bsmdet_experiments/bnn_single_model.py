import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_proc import StandardScaleNorm, MinMaxNorm
import argparse
import json
import torch
from bayes_transformer.trainer import BayesTrainer
from bayes_transformer.model import BSMDeTWrapper


def main(
        terminal_args=True,
        hyperparam_path=None,
        train_epochs=None,
        device=None,
        suffix=None,
        save_suff=None,
        model=None):

    args = None

    if terminal_args:
        parser = argparse.ArgumentParser(description='Train BNN model')
        parser.add_argument('--hyperparam_path', type=str,
                            default='modelsave/bmdet/best_hyperparams_non_spatial.json',
                            help='Path to hyperparameter JSON file')
        parser.add_argument('--train_epochs', type=int, default=100,
                            help='Number of training epochs')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to train on (cuda/cpu)')
        parser.add_argument('--suffix', type=str, default='non_spatial',
                            help='Whether to use spatial features')

        args = parser.parse_args()

    if args is not None:
        hyperparam_path = args.hyperparam_path
        train_epochs = args.train_epochs
        device = args.device
        suffix = args.suffix
        # _, standard_scale_norm, train_loader, _, _ = preprocess(
        #     spatial=args.spatial, device=args.device)

        # train_loader = torch.load('data/non_spatial/train_loader_non_spatial.pt')
        # val_loader = torch.load('data/non_spatial/val_loader_non_spatial.pt')
        # test_loader = torch.load('data/non_spatial/test_loader_non_spatial.pt')
        # norms = torch.load(
        #     os.path.join("data/non_spatial", "transforms_non_spatial.pt"))

    train_loader = torch.load(
        f'data/{suffix}/train_loader_{suffix}.pt')
    val_loader = torch.load(f'data/{suffix}/val_loader_{suffix}.pt')
    test_loader = torch.load(
        f'data/{suffix}/test_loader_{suffix}.pt')
    norms = torch.load(
        os.path.join(f"data/{suffix}", f"transforms_{suffix}.pt"))

    for (x, y) in train_loader:
        print(x.shape, y.shape)
        print(x.max(), x.min())
        print(y.max(), y.min())
    device = torch.device(device)

    if model is None:
        with open(hyperparam_path, 'r') as f:
            hyperparams = json.load(f)
        model = BSMDeTWrapper(**hyperparams)

    savename = suffix
    if save_suff is not None:
        savename = suffix + "_" + save_suff

    trainer = BayesTrainer(model_wrapper=model, train_loader=train_loader,
                           device=device, modelsave=True, savename=f'bmdet_{savename}')

    # print(f"TRAINING ON {hyperparams}\n")

    trained_model = trainer.train(epochs=train_epochs)

    return trained_model

# if __name__ == "__main__":
#     main()
