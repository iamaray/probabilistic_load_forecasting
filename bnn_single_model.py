from final_data_prep import preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
import torch
import json
import argparse
import os
from data_proc import StandardScaleNorm, MinMaxNorm


def main():
    # Set up argument parser
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

    # _, standard_scale_norm, train_loader, _, _ = preprocess(
    #     spatial=args.spatial, device=args.device)

    # train_loader = torch.load('data/non_spatial/train_loader_non_spatial.pt')
    # val_loader = torch.load('data/non_spatial/val_loader_non_spatial.pt')
    # test_loader = torch.load('data/non_spatial/test_loader_non_spatial.pt')
    # norms = torch.load(
    #     os.path.join("data/non_spatial", "transforms_non_spatial.pt"))

    train_loader = torch.load(
        f'data/{args.suffix}/train_loader_{args.suffix}.pt')
    val_loader = torch.load(f'data/{args.suffix}/val_loader_{args.suffix}.pt')
    test_loader = torch.load(
        f'data/{args.suffix}/test_loader_{args.suffix}.pt')
    norms = torch.load(
        os.path.join(f"data/{args.suffix}", f"transforms_{args.suffix}.pt"))

    for (x, y) in train_loader:
        print(x.shape, y.shape)

    device = torch.device(args.device)

    with open(args.hyperparam_path, 'r') as f:
        hyperparams = json.load(f)

    model = BSMDeTWrapper(**hyperparams)
    trainer = BayesTrainer(model_wrapper=model, train_loader=train_loader,
                           device=device, modelsave=True, savename=f'bmdet_{args.suffix}')

    print(f"TRAINING ON {hyperparams}\n")

    trainer.train(epochs=args.train_epochs)


if __name__ == "__main__":
    main()
