from preprocessing import readtoFiltered, preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from grid_search import grid_search_torch_model
import torch
from preprocessing import MinMaxNorm


if __name__ == "__main__":
    df, train_loader, test_loader, train_norm, test_norm = preprocess(
        csv_path='data/ercot_data_2025_Jan.csv',
        net_load_input=True,
        variates=['marketday', 'ACTUAL_ERC_Load',
                  'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'])

    # print(type(test_norm), type(train_norm))
    # print(test_norm.min_val, train_norm.min_val)
    # print(test_norm.max_val, train_norm.max_val)

    # for x, y in train_loader:
    #     print(x.shape, y.shape)

    # model_wrapper = BSMDeTWrapper(cuda=False, num_targets=1, num_aux_feats=3)
    # trainer = BayesTrainer(model_wrapper=model_wrapper,
    #                        train_loader=train_loader, train_norm=train_norm, test_norm=test_norm)

    param_grid = {
        'num_targets': [1],
        'num_aux_feats': [1],
        'd_model': [32, 64, 128],
        'encoder_layers': [2, 3, 4],
        'encoder_d_ff': [128, 256],
        'encoder_sublayers': [2, 3],
        'encoder_h': [8, 16],
        'encoder_dropout': [0.1],
        'decoder_layers': [2, 3, 4],
        'decoder_dropout': [0.1, 0.2],
        'decoder_h': [8, 16, 32],
        'decoder_d_ff': [128, 256],
        'decoder_sublayers': [3, 4],
        'cuda': [True]
    }

    training_args = {'epochs': 50}

    grid_search_torch_model(
        model_class=BSMDeTWrapper,
        trainer_class=BayesTrainer,
        param_grid=param_grid,
        training_args=training_args,
        train_loader=train_loader,
        test_loader=test_loader,
        train_norm=train_norm,
        test_norm=test_norm,
        device='cuda')
