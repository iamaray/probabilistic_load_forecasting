from preprocessing import readtoFiltered, preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from grid_search import grid_search_torch_model
import torch


if __name__ == "__main__":
    df, train_loader, test_loader, train_norm, test_norm = preprocess(
        csv_path='data/ercot_data_2025_Jan.csv',
        net_load_input=True,
        variates=['marketday', 'ACTUAL_ERC_Load',
                  'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'])

    model_wrapper = BSMDeTWrapper(num_targets=1, num_aux_feats=1)
    trainer = BayesTrainer(
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        train_norm=train_norm,
        test_norm=train_norm,
        modelsave=True)

    trainer.train()
