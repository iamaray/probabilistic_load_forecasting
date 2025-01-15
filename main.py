from preprocessing import readtoFiltered, preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
from grid_search import grid_search_torch_model
import torch
from preprocessing import MinMaxNorm

df, train_loader, test_loader, train_norm, test_norm = preprocess(
    'data/ercot_data_2025_Jan.csv', variates=['marketday', 'ACTUAL_ERC_Load', 'ACTUAL_ERC_Solar', 'ACTUAL_ERC_Wind', 'hourending'])

print(type(test_norm), type(train_norm))
print(test_norm.min_val, train_norm.min_val)
print(test_norm.max_val, train_norm.max_val)

model_wrapper = BSMDeTWrapper(cuda=False, num_targets=1, num_aux_feats=1)
trainer = BayesTrainer(model_wrapper=model_wrapper,
                       train_loader=train_loader, train_norm=train_norm, test_norm=test_norm)

trainer.train()

param_grid = {
    'd_model': [16, 32, 64, 128, 256],
    'encoder_layers': [2, 3, 4],
    'encoder_d_ff': [64, 128, 256],
    'encoder_sublayers': [2, 3],
    'encoder_h': [8, 16, 32],
    'encoder_dropout': [0.1],
    'decoder_layers': [2, 3, 4],
    'decoder_dropout': [0.1, 0.2],
    'decoder_h': [8, 16, 32],
    'decoder_d_ff': [64, 128, 256],
    'decoder_sublayers': [3, 4],
}
training_args = {'epochs': 1}

grid_search_torch_model(model_class=BSMDeTWrapper, trainer_class=BayesTrainer,
                        param_grid=param_grid, training_args=training_args, train_loader=train_loader, test_loader=test_loader)
