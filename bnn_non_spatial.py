from preprocessing import readtoFiltered, preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
# from grid_search import grid_search_torch_model
from bayes_transformer.trainer import grid_search_torch_model
import torch
from preprocessing import MinMaxNorm
from datetime import datetime
import torch.multiprocessing as mp


def main():
    df, train_loader, test_loader, train_norm, date_to_index, test_start_idx = preprocess(
        csv_path='data/ercot_data_2025_Jan.csv',
        net_load_input=True,
        variates=['marketday', 'ACTUAL_ERC_Load',
                  'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'],
        device='cuda')

    # print(type(test_norm), type(train_norm))
    # print(test_norm.min_val, train_norm.min_val)
    # print(test_norm.max_val, train_norm.max_val)

    for x, y in train_loader:
        print(x.shape, y.shape)

    for x, y in test_loader:
        print(x.shape, y.shape)

    # model_wrapper = BSMDeTWrapper(cuda=False, num_targets=1, num_aux_feats=3)
    # trainer = BayesTrainer(model_wrapper=model_wrapper,
    #                        train_loader=train_loader, train_norm=train_norm, test_norm=test_norm)

    param_grid = {
        'num_targets': [1],
        'num_aux_feats': [2],
        'd_model': [32],
        'encoder_layers': [2, 3],
        'encoder_d_ff': [128],
        'encoder_sublayers': [2, 3],
        'encoder_h': [8],
        'encoder_dropout': [0.1],
        'decoder_layers': [2, 3],
        'decoder_dropout': [0.1],
        'decoder_h': [8],
        'decoder_d_ff': [128],
        'decoder_sublayers': [2, 3]
    }

    # param_grid = {
    #     'num_targets': [1],
    #     'num_aux_feats': [1],
    #     'd_model': [32],
    #     'encoder_layers': [2],
    #     'encoder_d_ff': [128],
    #     'encoder_sublayers': [2],
    #     'encoder_h': [8],
    #     'encoder_dropout': [0.1],
    #     'decoder_layers': [2],
    #     'decoder_dropout': [0.1],
    #     'decoder_h': [8],
    #     'decoder_d_ff': [128],
    #     'decoder_sublayers': [3],
    #     'cuda': [True]
    # }

    training_args = {'epochs': 100}

    grid_search_torch_model(
        model_class=BSMDeTWrapper,
        trainer_class=BayesTrainer,
        param_grid=param_grid,
        training_args=training_args,
        train_loader=train_loader,
        test_loader=test_loader,
        savename='bmdet_best_non_spatial.pt',
        train_norm=train_norm,
        test_norm=train_norm,
        max_workers=2)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
