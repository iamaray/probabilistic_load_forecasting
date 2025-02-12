# from preprocessing import readtoFiltered, preprocess
from final_data_prep import preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from grid_search import grid_search_torch_model
import torch
import json


def main(hyperparam_path: str, train_epochs=100, device='cuda'):
    minmax_norm, standard_scale_norm, train_loader, val_loader, test_loader = preprocess(
        spatial=False)

    device = torch.device(device)

    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)

    model = BSMDeTWrapper(**hyperparams)
    trainer = BayesTrainer(model_wrapper=model, train_loader=train_loader,
                           train_norm=standard_scale_norm, device=device,
                           modelsave=True, savename='bmdet_non_spatial')
    try:
        trainer.train(epochs=train_epochs)
    except Exception as e:
        print(f"Training Failed: {str(e)}")


if __name__ == "__main__":
    # df, train_loader, test_loader, train_norm, test_norm = preprocess(
    #     csv_path='data/ercot_data_2025_Jan.csv',
    #     net_load_input=True,
    #     variates=['marketday', 'ACTUAL_ERC_Load',
    #               'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'])

    # model_wrapper = BSMDeTWrapper(num_targets=1, num_aux_feats=1)
    # trainer = BayesTrainer(
    #     model_wrapper=model_wrapper,
    #     train_loader=train_loader,
    #     train_norm=train_norm,
    #     test_norm=train_norm,
    #     modelsave=True)

    # trainer.train()
    main('modelsave/bmdet/best_hyperparams_non_spatial.json')
