import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from bayes_transformer.trainer import grid_search_torch_model
import torch
from preprocessing import MinMaxNorm
from final_data_prep import preprocess
from datetime import datetime
import torch.multiprocessing as mp


def main():
    minmax_norm, standard_scale_norm, train_loader, val_loader, test_loader = preprocess(
        spatial=True)

    print("TRAIN LOADER SAMPLES:")

    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nVAL LOADER SAMPLES:")

    for x, y in val_loader:
        print(x.shape, y.shape)

    param_grid = {
        'num_targets': [1],
        'num_aux_feats': [14],
        'd_model': [32],
        'encoder_layers': [2, 3, 4, 5],
        'encoder_d_ff': [128],
        'encoder_sublayers': [2],
        'encoder_h': [8],
        'encoder_dropout': [0.1],
        'decoder_layers': [2, 3, 4, 5],
        'decoder_dropout': [0.1],
        'decoder_h': [8],
        'decoder_d_ff': [128],
        'decoder_sublayers': [2]
    }

    training_args = {'epochs': 70}

    grid_search_torch_model(
        model_class=BSMDeTWrapper,
        trainer_class=BayesTrainer,
        param_grid=param_grid,
        training_args=training_args,
        train_loader=train_loader,
        test_loader=val_loader,
        savename='bmdet_best_spatial.pt',
        train_norm=standard_scale_norm,
        test_norm=standard_scale_norm,
        max_workers=2)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
