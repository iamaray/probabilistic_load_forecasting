# from preprocessing import readtoFiltered, preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
# from grid_search import grid_search_torch_model
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
        standard_scale_norm.transform(x.to('cuda'))
        standard_scale_norm.reverse(x.to('cuda'))

    print("\nVAL LOADER SAMPLES:")

    for x, y in val_loader:
        print(x.shape, y.shape)

    param_grid = {
        'num_targets': [1],
        'num_aux_feats': [14],
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

    training_args = {'epochs': 1}

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

    # model = BSMDeTWrapper(num_aux_feats=5)
    # trainer = BayesTrainer(model_wrapper=model, train_loader=train_loader,
    #                        device='cuda', train_norm=standard_scale_norm, modelsave=True)
    # trainer.train(epochs=100)
    # trainer.test(test_loader=train_loader, samples=20)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
