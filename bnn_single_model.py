from final_data_prep import preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
import torch
import json
import argparse


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
    parser.add_argument('--spatial', action='store_false',
                        help='Whether to use spatial features')

    args = parser.parse_args()

    _, standard_scale_norm, train_loader, _, _ = preprocess(
        spatial=args.spatial, device=args.device)

    device = torch.device(args.device)

    with open(args.hyperparam_path, 'r') as f:
        hyperparams = json.load(f)

    model = BSMDeTWrapper(**hyperparams)
    trainer = BayesTrainer(model_wrapper=model, train_loader=train_loader,
                           train_norm=standard_scale_norm, device=device,
                           modelsave=True, savename='bmdet_non_spatial')

    print(f"TRAINING ON {hyperparams}\n")

    try:
        trainer.train(epochs=args.train_epochs)
    except Exception as e:
        print(f"Training Failed: {str(e)}")


if __name__ == "__main__":
    main()
